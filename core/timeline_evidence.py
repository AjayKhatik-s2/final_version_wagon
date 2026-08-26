"""Timestamped evidence, and the one place it becomes a wagon assignment.

The pipeline collects observations from four cameras -- gaps, classification,
door, damage, load, OCR -- and every one of them is fundamentally a thing that
happened at a TIME. The canonical roster is also a set of time windows. So
assignment should be arithmetic on timestamps, and nothing else.

It has not been. Features ran after the roster existed, reading frames from
`wagon_cache/<GW_n>/`, so an observation was assigned by which directory it was
read out of. That works only because the materializer bucketed the frames
first, and it silently encodes the roster into the filesystem: a wrong bucket
is indistinguishable from a right one afterwards, because the evidence no
longer carries the time it came from.

This module makes the assignment explicit and auditable:

    Observation(camera, t_start, t_end, kind, confidence, geometry, model)
        -> assign_observations(...) -> GW_n + the REASON it landed there

Two policies, both configurable and both recorded per assignment, because both
are real decisions rather than implementation details:

    on_boundary   an observation exactly on a canonical gap. Default "next":
                  the boundary is the first instant of the following wagon.
    span          an observation that straddles a gap. Default "center": the
                  wagon containing its midpoint. "overlap" instead gives it to
                  the wagon it spends longest inside. Neither is obviously
                  right, so the choice is named in the audit.

Shared by BOTH modes. Batch and sequential differ in when evidence is
collected, never in how it is assigned -- there is one implementation and both
call it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core import constants as C
from core.master_timeline import (
    BoundaryPolicy, CameraClock, DEFAULT_BOUNDARY_POLICY,
    master_interval_to_local,
)

SCHEMA = "wagon_eye.timeline_evidence.v1"
ARTIFACT_NAME = "timeline_evidence.json"

# Observation kinds. Gap and classification are Stage-1 evidence; the rest are
# feature evidence. All are assigned by the same arithmetic.
KIND_GAP = "gap"
KIND_CLASSIFICATION = "classification"
KIND_DOOR = "door"
KIND_DAMAGE = "damage"
KIND_LOAD = "load"
KIND_OCR = "ocr"
FEATURE_KINDS = (KIND_DOOR, KIND_DAMAGE, KIND_LOAD, KIND_OCR)

# How an observation that straddles a canonical gap is resolved.
SPAN_CENTER = "center"
SPAN_OVERLAP = "overlap"

# Why an observation landed where it did.
REASON_CONTAINED = "contained"
REASON_BOUNDARY = "on_boundary"
REASON_SPAN_CENTER = "span_center"
REASON_SPAN_OVERLAP = "span_overlap"
REASON_OUTSIDE = "outside_wagon_region"
REASON_NO_TIME = "no_timestamp"


@dataclass(frozen=True)
class AssignmentPolicy:
    """The two real decisions, named so they appear in the audit."""
    boundary: BoundaryPolicy = DEFAULT_BOUNDARY_POLICY
    span: str = SPAN_CENTER

    def __post_init__(self):
        if self.span not in (SPAN_CENTER, SPAN_OVERLAP):
            raise ValueError(f"span must be {SPAN_CENTER!r} or "
                             f"{SPAN_OVERLAP!r}, got {self.span!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {"on_boundary": self.boundary.on_boundary,
                "boundary_epsilon": self.boundary.epsilon,
                "span": self.span}


DEFAULT_ASSIGNMENT_POLICY = AssignmentPolicy()


@dataclass
class Observation:
    """One thing a model saw, at a time, on a camera.

    Times are MASTER seconds. A collector working in a camera's own clock
    projects with `CameraClock.to_master_time()` before recording, so
    everything downstream compares like with like -- frame numbers are never
    comparable across cameras and are kept only as provenance.
    """
    camera_id: str
    kind: str
    t_start: float
    t_end: Optional[float] = None
    confidence: float = 0.0
    local_frame: Optional[int] = None
    bbox: Optional[Sequence[float]] = None
    model: str = ""
    label: str = ""
    detected: bool = True            # False = projected from the master
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def end(self) -> float:
        return self.t_end if self.t_end is not None else self.t_start

    @property
    def center(self) -> float:
        return (self.t_start + self.end) / 2.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.t_start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id, "kind": self.kind,
            "t_start": round(self.t_start, 4), "t_end": round(self.end, 4),
            "center_time": round(self.center, 4),
            "confidence": round(self.confidence, 4),
            "local_frame": self.local_frame,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "model": self.model, "label": self.label,
            "detected": self.detected,
            "payload": dict(self.payload),
        }


@dataclass
class Assignment:
    """One observation, the wagon it belongs to, and why."""
    observation: Observation
    global_id: Optional[str]
    reason: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = self.observation.to_dict()
        d.update({"assigned_gw": self.global_id, "assignment_reason":
                  self.reason, "assignment_detail": self.detail})
        return d


# --- assignment -------------------------------------------------------------

def _windows(wagons: Sequence[Any]) -> List[Tuple[Any, float, float]]:
    return sorted(((w, float(w.start_time), float(w.end_time))
                   for w in wagons), key=lambda x: x[1])


def assign_observation(obs: Observation, wagons: Sequence[Any], *,
                       policy: AssignmentPolicy = DEFAULT_ASSIGNMENT_POLICY
                       ) -> Assignment:
    """Which canonical wagon owns this observation, by absolute time.

    Never by segment index, array position or a camera's own wagon numbering --
    those are all camera-local and cannot survive a clock offset.
    """
    if obs.t_start is None:
        return Assignment(obs, None, REASON_NO_TIME,
                          "observation carries no timestamp")
    wins = _windows(wagons)
    if not wins:
        return Assignment(obs, None, REASON_OUTSIDE, "no canonical wagons")

    eps = policy.boundary.epsilon
    lo, hi = wins[0][1], wins[-1][2]

    # An instant (or a span short enough to sit inside one wagon).
    if obs.duration <= eps:
        t = obs.t_start
        for i, (w, s, e) in enumerate(wins):
            if abs(t - s) <= eps:
                if policy.boundary.on_boundary == "next":
                    return Assignment(obs, w.global_id, REASON_BOUNDARY,
                                      f"on the boundary at {s:.3f}s -> next")
                prev = wins[i - 1][0] if i > 0 else w
                return Assignment(obs, prev.global_id, REASON_BOUNDARY,
                                  f"on the boundary at {s:.3f}s -> previous")
        if abs(t - hi) <= eps:
            return Assignment(obs, wins[-1][0].global_id, REASON_BOUNDARY,
                              "on the final boundary -> last wagon")
        for w, s, e in wins:
            if s <= t <= e:
                return Assignment(obs, w.global_id, REASON_CONTAINED,
                                  f"{t:.3f}s inside {s:.3f}-{e:.3f}s")
        return Assignment(obs, None, REASON_OUTSIDE,
                          f"{t:.3f}s outside the wagon region "
                          f"{lo:.3f}-{hi:.3f}s")

    # A span. Which wagons does it touch?
    touched = [(w, s, e, min(obs.end, e) - max(obs.t_start, s))
               for w, s, e in wins
               if min(obs.end, e) > max(obs.t_start, s)]
    if not touched:
        return Assignment(obs, None, REASON_OUTSIDE,
                          f"{obs.t_start:.3f}-{obs.end:.3f}s outside the "
                          f"wagon region {lo:.3f}-{hi:.3f}s")
    if len(touched) == 1:
        w, s, e, _ov = touched[0]
        return Assignment(obs, w.global_id, REASON_CONTAINED,
                          f"span {obs.t_start:.3f}-{obs.end:.3f}s within "
                          f"{s:.3f}-{e:.3f}s")

    if policy.span == SPAN_OVERLAP:
        w, s, e, ov = max(touched, key=lambda t: (t[3], -t[1]))
        return Assignment(obs, w.global_id, REASON_SPAN_OVERLAP,
                          f"spans {len(touched)} wagons; longest overlap "
                          f"{ov:.3f}s with {s:.3f}-{e:.3f}s")
    mid = obs.center
    for w, s, e, _ov in touched:
        if s <= mid <= e:
            return Assignment(obs, w.global_id, REASON_SPAN_CENTER,
                              f"spans {len(touched)} wagons; centre "
                              f"{mid:.3f}s falls in {s:.3f}-{e:.3f}s")
    w, s, e, ov = max(touched, key=lambda t: (t[3], -t[1]))
    return Assignment(obs, w.global_id, REASON_SPAN_OVERLAP,
                      f"spans {len(touched)} wagons; centre outside them all, "
                      f"fell back to longest overlap {ov:.3f}s")


def assign_observations(observations: Iterable[Observation],
                        wagons: Sequence[Any], *,
                        policy: AssignmentPolicy = DEFAULT_ASSIGNMENT_POLICY
                        ) -> List[Assignment]:
    """Assign every observation. Order-independent and deterministic."""
    return [assign_observation(o, wagons, policy=policy)
            for o in sorted(observations,
                            key=lambda o: (o.t_start, o.camera_id, o.kind))]


# --- the collected evidence -------------------------------------------------

@dataclass
class TimelineEvidence:
    """Everything observed, everything assigned, and the reasoning."""
    observations: List[Observation] = field(default_factory=list)
    assignments: List[Assignment] = field(default_factory=list)
    policy: AssignmentPolicy = DEFAULT_ASSIGNMENT_POLICY
    wagon_active: Optional[Dict[str, Any]] = None
    canonical_gaps: List[float] = field(default_factory=list)
    camera_offsets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    roster: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = ""

    def add(self, obs: Observation) -> None:
        self.observations.append(obs)

    def extend(self, obs: Iterable[Observation]) -> None:
        self.observations.extend(obs)

    def fuse(self, wagons: Sequence[Any]) -> List[Assignment]:
        """Phase 2. Assign every collected observation to a canonical wagon."""
        self.assignments = assign_observations(self.observations, wagons,
                                               policy=self.policy)
        self.roster = [{"global_id": w.global_id,
                        "start_time": round(float(w.start_time), 4),
                        "end_time": round(float(w.end_time), 4),
                        "classification": getattr(w, "classification", "")}
                       for w in _win_order(wagons)]
        return self.assignments

    def by_wagon(self) -> Dict[str, List[Assignment]]:
        out: Dict[str, List[Assignment]] = {}
        for a in self.assignments:
            out.setdefault(a.global_id or "UNASSIGNED", []).append(a)
        return out

    def counts_by_kind(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for o in self.observations:
            out[o.kind] = out.get(o.kind, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        per_wagon = {gw: len(v) for gw, v in self.by_wagon().items()}
        return {
            "schema": SCHEMA,
            "mode": self.mode,
            "assignment_policy": self.policy.to_dict(),
            "wagon_active": self.wagon_active,
            "canonical_gaps": [round(t, 4) for t in self.canonical_gaps],
            "camera_offsets": self.camera_offsets,
            "roster": self.roster,
            "observation_counts": self.counts_by_kind(),
            "assignments_per_wagon": per_wagon,
            "unassigned": per_wagon.get("UNASSIGNED", 0),
            "observations": [a.to_dict() for a in self.assignments],
        }

    def summary_lines(self) -> List[str]:
        out = [f"timeline evidence ({self.mode or 'unknown mode'}): "
               f"{len(self.observations)} observation(s) "
               f"{self.counts_by_kind()}",
               f"  policy: {self.policy.to_dict()}",
               f"  roster: {len(self.roster)} wagon(s), "
               f"{len(self.canonical_gaps)} canonical gap(s)"]
        per = self.by_wagon()
        unassigned = len(per.get("UNASSIGNED", []))
        if unassigned:
            out.append(f"  UNASSIGNED: {unassigned} observation(s) outside "
                       f"the wagon region")
        return out


def _win_order(wagons: Sequence[Any]) -> List[Any]:
    return sorted(wagons, key=lambda w: float(w.start_time))


def write_artifact(ev: TimelineEvidence, output_dir: str) -> str:
    """Persist the audit. One file explains every assignment in the run."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, ARTIFACT_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ev.to_dict(), f, indent=2, default=str)
    return path


# --- adapters: existing outputs -> timestamped observations -----------------
#
# The feature processors currently emit per-wagon JSON keyed by GW id, having
# read frames the materializer had already bucketed. These adapters recover the
# TIME from what those payloads already carry -- `best_frame_idx` plus the
# camera's fps and offset -- so the unified model works on today's outputs.
#
# When a processor is inverted to emit observations directly from raw video, it
# replaces the adapter and nothing downstream changes: same Observation, same
# fusion, same audit.

def observations_from_feature_json(payload: Dict[str, Any], *,
                                   feature: str,
                                   clocks: Dict[str, CameraClock]
                                   ) -> List[Observation]:
    """Timestamped observations from one wagon's feature JSON.

    Each record carries its own `camera_id`, so its frame index is converted
    with THAT camera's clock -- never the master's.
    """
    out: List[Observation] = []
    if not isinstance(payload, dict):
        return out

    def _obs(rec: Dict[str, Any], label: str, conf: float) -> Optional[Observation]:
        cam = rec.get("camera_id")
        clock = clocks.get(cam) if cam else None
        frame = rec.get("best_frame_idx", rec.get("frame_idx"))
        if clock is None or frame is None or clock.fps <= 0:
            return None
        t = clock.to_master_time(float(frame) / clock.fps)
        return Observation(
            camera_id=cam, kind=feature, t_start=t, t_end=t,
            confidence=float(conf or 0.0), local_frame=int(frame),
            bbox=rec.get("bbox"), model=str(payload.get("model") or feature),
            label=label, payload={k: v for k, v in rec.items()
                                  if k not in ("bbox",)})

    if feature == KIND_DAMAGE:
        for tr in (payload.get("top_damage_details") or []):
            if isinstance(tr, dict):
                o = _obs(tr, str(tr.get("class_name") or "damage"),
                         tr.get("best_confidence"))
                if o:
                    out.append(o)
    elif feature == KIND_DOOR:
        for tr in (payload.get("tracks") or []):
            if isinstance(tr, dict):
                o = _obs(tr, str(tr.get("state") or ""), tr.get("confidence"))
                if o:
                    out.append(o)
    elif feature == KIND_LOAD:
        for cam, side in (payload.get("per_camera") or {}).items():
            if not isinstance(side, dict):
                continue
            rec = dict(side)
            rec["camera_id"] = cam
            o = _obs(rec, str(payload.get("load_status") or ""),
                     payload.get("load_confidence"))
            if o:
                out.append(o)
    elif feature == KIND_OCR:
        rec = {"camera_id": payload.get("supporting_cameras", [None])[0]
               if payload.get("supporting_cameras") else None,
               "best_frame_idx": payload.get("best_frame"),
               "bbox": payload.get("best_bbox")}
        o = _obs(rec, str(payload.get("wagon_identifier") or ""),
                 payload.get("wagon_identifier_confidence"))
        if o:
            out.append(o)
    return out


def observations_from_gaps(gaps: Iterable[Any], camera_id: str, *,
                           clock: Optional[CameraClock] = None,
                           detected: bool = True,
                           model: str = "gap") -> List[Observation]:
    """Gap events as observations. Support gaps are evidence, never authority."""
    out: List[Observation] = []
    for g in gaps or []:
        t = getattr(g, "center_time", None)
        if t is None and isinstance(g, dict):
            t = g.get("center_time")
        if t is None:
            continue
        t = float(t)
        if clock is not None:
            t = clock.to_master_time(t)
        out.append(Observation(
            camera_id=camera_id, kind=KIND_GAP, t_start=t, t_end=t,
            confidence=float(getattr(g, "confidence", 0.0) or 0.0),
            local_frame=getattr(g, "center_frame", None),
            model=model, detected=detected,
            payload={"track_id": getattr(g, "track_id", None)}))
    return out


def observations_from_classification(spans: Iterable[Any], camera_id: str
                                     ) -> List[Observation]:
    """Classified spans as observations, on the master clock already."""
    return [Observation(
        camera_id=camera_id, kind=KIND_CLASSIFICATION,
        t_start=float(s.start_time), t_end=float(s.end_time),
        confidence=float(getattr(s, "confidence", 0.0) or 0.0),
        label=str(getattr(s, "label", "") or ""), model="classifier")
        for s in spans or []]
