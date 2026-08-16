"""Stride-invariant evidence aggregation for sampled-frame feature inference.

WHY THIS EXISTS
---------------
Door and Damage spend ~97% of their wall clock inside YOLO, so the only lever
that meaningfully reduces runtime is issuing fewer inference calls -- i.e.
sampling frames.  Sampling on its own is unsafe: the legacy DoorTracker
confirms a track with ABSOLUTE hit counts (``n_init=3``,
``min_hits_for_decision=3``).  Halving the sample rate halves the hits, so
marginal-but-real doors stop confirming.  Measured at stride 2:

    GW_21  hits 6 -> 2   LEFT_UP track lost, state fell back to CLOSED/0.000
    GW_22  hits 8 -> 2   LEFT_UP track lost, state fell back to CLOSED/0.000

This module replaces that count-based confirmation with a *rate-based* one.
Support is expressed as a FRACTION of the frames actually sampled, so the same
physical evidence clears the bar at any stride.  Nothing is loosened: at
stride 1 the default rate is deliberately at least as strict as the tracker's
3-hit rule over a typical ~50-frame wagon span.

WHAT IT IS NOT
--------------
This is not a tracker and not a replacement for one.  It performs no Kalman
prediction and no Hungarian assignment.  It answers a narrower question --
"which detections across these sampled frames are the same physical object,
and what does their combined evidence say?" -- which is all the wagon-level
verdict actually needs.

It is also NOT "highest confidence wins".  A single high-confidence outlier
cannot outvote repeated, spatially-consistent evidence; see
``Candidate.resolve_state``.

DESIGN
------
1. SPATIAL GROUPING.  Detections are chained into candidates in frame order.
   The train is moving, so a candidate's box translates between observations;
   the gate therefore scales with the frame gap (``max_drift_frac_per_frame``)
   instead of requiring raw overlap.  IoU is accepted as an alternative gate
   for slow/stationary objects.

2. EVIDENCE.  Each candidate accumulates, per observed state, the number of
   DISTINCT frames supporting it and the confidences seen.

3. RESOLUTION.  Frame support is the primary key, mean confidence the
   secondary.  A state must clear ``min_support_frames`` AND
   ``min_support_fraction`` of the candidate's observation span to win.

4. SNAPSHOT.  One best representative frame per candidate, chosen by the
   caller's score (callers pass the same snapshot_score they already use), so
   snapshot semantics are unchanged.

Pure stdlib + dataclasses: no cv2, no torch.  Fully unit-testable without
models, weights or video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

Bbox = Sequence[float]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregationConfig:
    """Thresholds for grouping and evidence resolution.

    Defaults are chosen to be stride-invariant and, at stride 1, no weaker
    than the legacy tracker's 3-hits-in-~50-frames confirmation rule.
    """

    #: Max centre drift BETWEEN CONSECUTIVE SAMPLED FRAMES, as a fraction of
    #: frame width, before two detections are considered different objects.
    #: Scales with the frame gap, so stride 2 tolerates twice the drift of
    #: stride 1 -- this is what keeps grouping stride-invariant.
    max_drift_frac_per_frame: float = 0.045

    #: Alternative gate: boxes overlapping at least this much are the same
    #: object regardless of drift (covers slow or stationary objects).
    min_iou_for_same: float = 0.30

    #: Hard ceiling on drift for one link, as a fraction of frame width.
    #: Prevents a long blind gap from chaining two unrelated objects.
    max_total_drift_frac: float = 0.60

    #: A candidate is dropped if no state reaches this many DISTINCT frames.
    min_support_frames: int = 2

    #: ...and that state must also hold this fraction of the candidate's own
    #: observed frames.  Guards against one stray label inside a long run.
    min_support_fraction: float = 0.34

    #: Detections below this are ignored entirely.  Callers normally pass the
    #: model threshold they already use, so this changes nothing by default.
    min_confidence: float = 0.0

    #: Frames may be missed between observations; a gap wider than this (in
    #: SAMPLED steps, not raw frames) starts a new candidate.
    max_frame_gap_steps: int = 4


DEFAULT_AGGREGATION = AggregationConfig()


# -----------------------------------------------------------------------------
# Observation + candidate
# -----------------------------------------------------------------------------

@dataclass
class Observation:
    """One detection on one sampled frame, in ORIGINAL cache frame numbering."""
    frame_idx: int
    state: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    score: float = 0.0          # caller's snapshot score; picks the best frame
    payload: Optional[Dict[str, Any]] = None

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou(a: Bbox, b: Bbox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Candidate:
    """One physical object, as evidenced across several sampled frames."""
    observations: List[Observation] = field(default_factory=list)

    # -- construction ----------------------------------------------------
    def add(self, obs: Observation) -> None:
        self.observations.append(obs)

    @property
    def last(self) -> Observation:
        return self.observations[-1]

    @property
    def frames(self) -> List[int]:
        return sorted({o.frame_idx for o in self.observations})

    @property
    def frame_support(self) -> int:
        """DISTINCT frames, so two boxes on one frame are not double-counted."""
        return len(self.frames)

    @property
    def span_frames(self) -> int:
        f = self.frames
        return (f[-1] - f[0] + 1) if f else 0

    # -- evidence --------------------------------------------------------
    def state_evidence(self) -> Dict[str, Dict[str, Any]]:
        """`{state: {frames, mean_conf, max_conf}}`, counting distinct frames."""
        by: Dict[str, Dict[str, Any]] = {}
        for o in self.observations:
            e = by.setdefault(o.state, {"_frames": set(), "_confs": []})
            e["_frames"].add(o.frame_idx)
            e["_confs"].append(float(o.confidence))
        out: Dict[str, Dict[str, Any]] = {}
        for st, e in by.items():
            confs = e["_confs"]
            out[st] = {
                "frames": len(e["_frames"]),
                "mean_conf": sum(confs) / len(confs) if confs else 0.0,
                "max_conf": max(confs) if confs else 0.0,
            }
        return out

    def resolve_state(
        self, cfg: AggregationConfig = DEFAULT_AGGREGATION,
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Winning state for this candidate, or `(None, 0.0, ...)`.

        Frame support is PRIMARY, mean confidence only breaks ties.  This is
        what stops one high-confidence outlier from overriding repeated
        evidence: a single OPEN frame among many CLOSED frames has support 1
        and loses regardless of its confidence.
        """
        ev = self.state_evidence()
        if not ev:
            return None, 0.0, {}
        total = self.frame_support or 1
        ranked = sorted(
            ev.items(),
            key=lambda kv: (-kv[1]["frames"], -kv[1]["mean_conf"], kv[0]),
        )
        st, info = ranked[0]
        if info["frames"] < cfg.min_support_frames:
            return None, 0.0, {"reason": "insufficient_frame_support", "evidence": ev}
        if info["frames"] / total < cfg.min_support_fraction:
            return None, 0.0, {"reason": "insufficient_support_fraction", "evidence": ev}
        return st, float(info["mean_conf"]), {"evidence": ev, "support": info["frames"],
                                              "total_frames": total}

    def best_observation(self, state: Optional[str] = None) -> Optional[Observation]:
        """Highest-scoring observation, optionally restricted to one state."""
        pool = [o for o in self.observations
                if state is None or o.state == state] or self.observations
        return max(pool, key=lambda o: (o.score, o.confidence), default=None)


# -----------------------------------------------------------------------------
# Aggregator
# -----------------------------------------------------------------------------

class EvidenceAggregator:
    """Groups sampled-frame detections into candidates and resolves evidence.

    Usage mirrors a tracker's, minus the prediction step::

        agg = EvidenceAggregator(frame_width=W, frame_height=H)
        for frame_idx, dets in sampled_frames:      # ORIGINAL indices
            agg.add_frame(frame_idx, dets)
        result = agg.finalize()
    """

    def __init__(
        self,
        *,
        frame_width: int,
        frame_height: int = 0,
        config: AggregationConfig = DEFAULT_AGGREGATION,
        stride: int = 1,
    ) -> None:
        self.w = max(1, int(frame_width))
        self.h = max(1, int(frame_height))
        self.cfg = config
        self.stride = max(1, int(stride))
        self.candidates: List[Candidate] = []
        self._sampled_frames: List[int] = []

    # -- ingestion -------------------------------------------------------

    def add_frame(self, frame_idx: int, detections: Sequence[Observation]) -> None:
        """Ingest one SAMPLED frame.  `frame_idx` is the original cache index."""
        self._sampled_frames.append(int(frame_idx))
        for obs in sorted(detections, key=lambda o: (-o.confidence, o.bbox[0])):
            if obs.confidence < self.cfg.min_confidence:
                continue
            target = self._match(obs)
            if target is None:
                c = Candidate()
                c.add(obs)
                self.candidates.append(c)
            else:
                target.add(obs)

    def _match(self, obs: Observation) -> Optional[Candidate]:
        """Nearest compatible candidate, gated by drift-scaled distance or IoU."""
        best: Optional[Candidate] = None
        best_cost = float("inf")
        for c in self.candidates:
            prev = c.last
            gap_frames = obs.frame_idx - prev.frame_idx
            if gap_frames <= 0:
                continue                      # same frame -> a different object
            gap_steps = max(1, gap_frames // self.stride)
            if gap_steps > self.cfg.max_frame_gap_steps:
                continue
            if iou(obs.bbox, prev.bbox) >= self.cfg.min_iou_for_same:
                cost = 0.0
            else:
                dx = abs(obs.center[0] - prev.center[0]) / self.w
                allowed = min(
                    self.cfg.max_drift_frac_per_frame * max(1, gap_frames),
                    self.cfg.max_total_drift_frac,
                )
                if dx > allowed:
                    continue
                cost = dx
            if cost < best_cost:
                best, best_cost = c, cost
        return best

    # -- resolution ------------------------------------------------------

    def finalize(self) -> Dict[str, Any]:
        """Resolve every candidate.  Returns groups + the surviving evidence."""
        groups: List[Dict[str, Any]] = []
        for i, c in enumerate(self.candidates, start=1):
            state, conf, detail = c.resolve_state(self.cfg)
            best = c.best_observation(state)
            groups.append({
                "candidate_id": i,
                "state": state,
                "confidence": conf,
                "frame_support": c.frame_support,
                "span_frames": c.span_frames,
                "first_frame": c.frames[0] if c.frames else -1,
                "last_frame": c.frames[-1] if c.frames else -1,
                "accepted": state is not None,
                "detail": detail,
                "best": best,
            })
        return {
            "groups": groups,
            "accepted": [g for g in groups if g["accepted"]],
            "sampled_frame_count": len(self._sampled_frames),
            "sampled_frames": list(self._sampled_frames),
            "stride": self.stride,
        }
