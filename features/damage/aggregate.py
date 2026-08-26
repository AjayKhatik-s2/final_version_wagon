"""Damage aggregation, separated from damage inference.

Phase 2 of the unified architecture. `_run_sampled_one_camera` currently does
two unrelated jobs in one loop: it scores frames with the model, and it votes
those detections into per-wagon damage tracks. Only the first needs the video.
The second is arithmetic over detections, and keeping it welded to the frame
reader is what forces feature inference to wait for a materialized wagon cache.

This module is the second job alone. It consumes detections already collected
from the raw video and returns the SAME per-wagon records the production path
returns -- same aggregator, same acceptance rule, same fields.

Nothing here opens a video, loads a model, runs inference, or reads a wagon
cache. `tests/test_damage_aggregation_equivalence.py` proves the output is
identical to the pre-refactor implementation on a non-empty corpus, and
`TestNoInferenceInAggregation` proves the absence of inference structurally
rather than by inspection.

Scope, stated honestly: this extracts the SAMPLED path, which is production's
default (`inference_mode="sampled"`, stride 3). The legacy tracker path calls
`DamageTracker.update(detections, frame, ...)`, and that tracker consumes the
frame itself to build snapshots -- it cannot be made pure without changing
DamageTracker, which is out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core import constants as C

#: One detection, exactly the fields the aggregator and the output contract
#: need. This is what Phase 1 hands over -- no frame, no model.
@dataclass
class DamageDetection:
    frame_idx: int
    class_name: str
    confidence: float
    bbox: Sequence[float]

    def to_dict(self) -> Dict[str, Any]:
        return {"frame_idx": int(self.frame_idx),
                "class_name": str(self.class_name),
                "confidence": float(self.confidence),
                "bbox": [float(v) for v in self.bbox]}


def detections_from_observations(observations: Iterable[Any]
                                 ) -> List[DamageDetection]:
    """`core.timeline_evidence.Observation` -> aggregator input.

    Uses `local_frame`, not the timestamp: the aggregator groups by frame and
    measures support as a fraction of SAMPLED frames, so it must see the same
    indices the scorer used.
    """
    out: List[DamageDetection] = []
    for o in observations or []:
        if getattr(o, "kind", None) != "damage":
            continue
        if o.local_frame is None or o.bbox is None:
            continue
        out.append(DamageDetection(
            frame_idx=int(o.local_frame), class_name=str(o.label or ""),
            confidence=float(o.confidence or 0.0), bbox=list(o.bbox)))
    return out


def aggregate_damage_from_detections(
    detections: Sequence[DamageDetection],
    *,
    camera_id: str,
    frame_width: int,
    frame_height: int,
    stride: int,
    scored_frames: Optional[Sequence[int]] = None,
    snapshots: Optional[Dict[int, Any]] = None,
) -> List[Dict[str, Any]]:
    """Per-wagon damage records from already-collected detections.

    Byte-for-byte the tail of `_run_sampled_one_camera`: the same
    `EvidenceAggregator`, the same `add_frame` per scored frame, the same
    `finalize()["accepted"]` filter and the same record fields.

    `scored_frames` is the list of frames the model actually looked at,
    including those where it found nothing, because that is what the production
    loop feeds the aggregator -- `agg.add_frame(fi, [])` on an empty frame.
    Supplying it reproduces the production call shape exactly.

    Measured, rather than assumed: declaring the empty frames does NOT change
    the accepted groups, at any hit-to-frame ratio tried -- `add_frame(fi, [])`
    is behaviourally inert in `EvidenceAggregator`, whose support is measured
    over the frames carrying observations. It is passed because production
    passes it and the call shapes must not drift apart, not because a
    difference was observed. The same holds for Door.

    `snapshots` maps frame index -> image, for the evidence crop. Absent, the
    record carries `_snapshot=None` and the caller resolves the image later;
    every other field is unchanged.
    """
    from features.evidence_aggregator import EvidenceAggregator, Observation

    if not detections and not scored_frames:
        return []

    agg = EvidenceAggregator(frame_width=frame_width,
                             frame_height=frame_height, stride=stride)

    by_frame: Dict[int, List[Observation]] = {}
    for d in detections:
        bb = [float(v) for v in d.bbox]
        by_frame.setdefault(int(d.frame_idx), []).append(Observation(
            frame_idx=int(d.frame_idx), state=str(d.class_name).lower(),
            confidence=float(d.confidence),
            bbox=(bb[0], bb[1], bb[2], bb[3]), score=float(d.confidence)))

    frames = (sorted(set(int(f) for f in scored_frames))
              if scored_frames is not None else sorted(by_frame))
    for fi in frames:
        agg.add_frame(fi, by_frame.get(fi, []))

    snaps = snapshots or {}
    out: List[Dict[str, Any]] = []
    for g in agg.finalize()["accepted"]:
        best = g.get("best")
        if best is None:
            continue
        out.append({
            "camera_id":   camera_id,
            "track_id":    int(g["candidate_id"]),
            "class_name":  str(g["state"]).lower(),
            "confidence":  float(g["confidence"]),
            "best_confidence": float(best.confidence),
            "total_hits":  int(g["frame_support"]),
            "first_frame": int(g["first_frame"]),
            "last_frame":  int(g["last_frame"]),
            "best_frame_idx": int(best.frame_idx),
            "bbox":        list(best.bbox),
            "_snapshot":   snaps.get(int(best.frame_idx)),
        })
    return out


def aggregate_damage_from_observations(
    observations: Iterable[Any],
    *,
    camera_id: str,
    frame_width: int,
    frame_height: int,
    stride: int,
    scored_frames: Optional[Sequence[int]] = None,
    snapshots: Optional[Dict[int, Any]] = None,
) -> List[Dict[str, Any]]:
    """The same, taking `Observation` records straight from Phase 1."""
    return aggregate_damage_from_detections(
        detections_from_observations(observations), camera_id=camera_id,
        frame_width=frame_width, frame_height=frame_height, stride=stride,
        scored_frames=scored_frames, snapshots=snapshots)
