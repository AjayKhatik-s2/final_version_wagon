"""Camera-local evidence bundle + local->global mapping (sequential mode).

Sequential mode lets each camera run its COMPLETE existing Stage-1 chain
(YOLO -> filtering -> gap validation -> GapTracker -> temporal validation ->
fragment stitching -> temporal classification -> segments_from_gaps) and then
its Door/Damage/Load features, WITHOUT waiting for any other camera.  The
results are persisted here as a self-contained bundle keyed by CAMERA-LOCAL
segment ids.

Global wagon ids do not exist until every required bundle is sealed.  Global
assembly then maps each local segment onto a `GW_n` using the offsets the
existing fixed-master fusion already computes, and RELABELS the persisted
evidence.  No detector is ever re-run during assembly.

Nothing in this module performs inference, reads a video, or touches
`wagon_count/`.  It is pure bookkeeping over artefacts the proven pipeline
already produces.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "wagon_eye.camera_evidence.v1"

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

#: Ordered camera lifecycle.  A bundle may resume from the last completed
#: stage; SEALED is terminal and immutable.  FAILED is terminal too and must
#: never block another camera.
LIFECYCLE: Tuple[str, ...] = (
    "PENDING",       # nothing done yet
    "TRACKING",      # GapTracker finished (raw + tracked candidates)
    "VALIDATED",     # fragment stitching + gap validation finished
    "SEGMENTED",     # segments_from_gaps + classification finished
    "MATERIALIZED",  # per-local-segment frame cache written
    "FEATURES",      # Door / Damage / Load finished for this camera
    "REPORTED",      # camera-only report written
    "SEALED",        # immutable; eligible for global assembly
)
FAILED = "FAILED"
TERMINAL = ("SEALED", FAILED)


class CameraEvidenceError(RuntimeError):
    """Invalid bundle state or an illegal lifecycle transition."""


def next_state(current: str) -> str:
    if current == FAILED:
        raise CameraEvidenceError("FAILED is terminal")
    if current == "SEALED":
        raise CameraEvidenceError("SEALED is terminal")
    return LIFECYCLE[LIFECYCLE.index(current) + 1]


def can_advance(current: str, target: str) -> bool:
    """Only forward, one step at a time; FAILED reachable from anywhere."""
    if target == FAILED:
        return current not in TERMINAL
    if current in TERMINAL or target not in LIFECYCLE:
        return False
    return LIFECYCLE.index(target) == LIFECYCLE.index(current) + 1


def local_segment_id(camera_id: str, index: int) -> str:
    """Camera-local wagon id.  Deliberately NOT `GW_n` -- it is not global."""
    return f"L_{camera_id}_{int(index)}"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class LocalSegment:
    """One camera-local wagon span, in that camera's ORIGINAL frame numbering."""
    local_id: str
    index: int                 # 1-based within this camera
    start_frame: int
    end_frame: int
    start_time: float          # camera-local seconds
    end_time: float
    label: str = "UNKNOWN"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LocalSegment":
        return LocalSegment(
            local_id=str(d["local_id"]), index=int(d["index"]),
            start_frame=int(d["start_frame"]), end_frame=int(d["end_frame"]),
            start_time=float(d["start_time"]), end_time=float(d["end_time"]),
            label=str(d.get("label", "UNKNOWN")),
            confidence=float(d.get("confidence", 0.0)),
        )


@dataclass
class CameraManifest:
    camera_id: str
    state: str = "PENDING"
    fps: float = 0.0
    total_frames: int = 0
    width: int = 0
    height: int = 0
    video_path: str = ""
    failure_reason: str = ""
    schema: str = SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Mapping local -> global
# ---------------------------------------------------------------------------

#: Every mapping outcome is recorded.  Nothing is silently dropped or merged.
MAP_EXACT       = "EXACT"          # 1:1, unambiguous
MAP_MANY_TO_ONE = "MANY_TO_ONE"    # several local segments -> one GW
MAP_ONE_TO_MANY = "ONE_TO_MANY"    # one local segment spans several GWs
MAP_UNMATCHED   = "UNMATCHED"      # no global wagon overlaps this segment
MAP_UNRESOLVED  = "UNRESOLVED_OFFSET"   # camera clock never resolved


@dataclass
class SegmentMapping:
    local_id: str
    camera_id: str
    global_id: Optional[str]
    kind: str
    overlap_seconds: float = 0.0
    overlap_fraction: float = 0.0        # of the LOCAL segment's duration
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    offset_applied: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def map_segments_to_global(
    segments: Sequence[LocalSegment],
    global_wagons: Sequence[Any],
    *,
    camera_id: str,
    offset: float = 0.0,
    offset_resolved: bool = True,
    min_overlap_fraction: float = 0.10,
) -> List[SegmentMapping]:
    """Map camera-local segments onto global wagons by temporal overlap.

    `global_wagons` are the finalized roster entries (anything exposing
    `global_id`, `start_time`, `end_time` in MASTER seconds).

    A local segment's master-clock window is `local_time + offset`, matching
    the convention the existing fusion uses (`t_global = t_local + delta`).
    `offset` MUST come from the fusion's resolved camera offsets; an
    unresolved camera passes `offset_resolved=False` and is mapped with
    `offset=0.0` (today's shared-`t=0` assumption) and flagged, never guessed.

    Ambiguity is REPORTED, never resolved silently:
      * several locals landing on one GW      -> MANY_TO_ONE on each
      * one local spanning several GWs        -> ONE_TO_MANY, with every
                                                 candidate retained
      * no overlap at all                     -> UNMATCHED
    The winner is always the maximum-overlap GW so downstream has a usable
    assignment, but `kind` and `candidates` preserve the full picture.
    """
    out: List[SegmentMapping] = []
    eff_offset = float(offset) if offset_resolved else 0.0

    for seg in segments:
        g0 = seg.start_time + eff_offset
        g1 = seg.end_time + eff_offset
        duration = max(1e-9, g1 - g0)

        cands: List[Dict[str, Any]] = []
        for w in global_wagons:
            ov = _overlap(g0, g1, float(w.start_time), float(w.end_time))
            if ov > 0:
                cands.append({
                    "global_id": w.global_id,
                    "overlap_seconds": round(ov, 4),
                    "overlap_fraction": round(ov / duration, 4),
                })
        cands.sort(key=lambda c: (-c["overlap_seconds"], c["global_id"]))

        if not cands:
            out.append(SegmentMapping(
                local_id=seg.local_id, camera_id=camera_id, global_id=None,
                kind=MAP_UNMATCHED, offset_applied=eff_offset,
                note="no global wagon overlaps this segment"))
            continue

        best = cands[0]
        significant = [c for c in cands
                       if c["overlap_fraction"] >= min_overlap_fraction]
        kind = MAP_EXACT
        note = ""
        if not offset_resolved:
            kind = MAP_UNRESOLVED
            note = "camera clock offset unresolved; mapped with offset=0.0"
        elif len(significant) > 1:
            kind = MAP_ONE_TO_MANY
            note = (f"segment spans {len(significant)} global wagons "
                    f"(support camera likely missed a gap)")

        out.append(SegmentMapping(
            local_id=seg.local_id, camera_id=camera_id,
            global_id=best["global_id"], kind=kind,
            overlap_seconds=best["overlap_seconds"],
            overlap_fraction=best["overlap_fraction"],
            candidates=cands, offset_applied=eff_offset, note=note))

    # Second pass: flag many-to-one collisions (several locals -> one GW).
    counts: Dict[str, int] = {}
    for m in out:
        if m.global_id:
            counts[m.global_id] = counts.get(m.global_id, 0) + 1
    for m in out:
        if m.global_id and counts[m.global_id] > 1 and m.kind == MAP_EXACT:
            m.kind = MAP_MANY_TO_ONE
            m.note = (f"{counts[m.global_id]} local segments map to "
                      f"{m.global_id} (support camera likely saw an extra gap)")
    return out


def mapping_summary(mappings: Sequence[SegmentMapping]) -> Dict[str, Any]:
    """Audit rollup written alongside the per-segment records."""
    by_kind: Dict[str, int] = {}
    for m in mappings:
        by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
    return {
        "total": len(mappings),
        "by_kind": by_kind,
        "ambiguous": sum(v for k, v in by_kind.items() if k != MAP_EXACT),
        "unmatched_local_ids": [m.local_id for m in mappings
                                if m.kind == MAP_UNMATCHED],
    }


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------

class CameraEvidenceBundle:
    """On-disk `camera_evidence/<CAMERA>/` tree.  Pure bookkeeping."""

    def __init__(self, root: str, camera_id: str) -> None:
        self.camera_id = camera_id
        self.dir = os.path.join(root, camera_id)
        self.manifest_path = os.path.join(self.dir, "manifest.json")

    # -- manifest ---------------------------------------------------------

    def load_manifest(self) -> CameraManifest:
        if not os.path.exists(self.manifest_path):
            return CameraManifest(camera_id=self.camera_id)
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        m = CameraManifest(camera_id=d.get("camera_id", self.camera_id))
        for k, v in d.items():
            if hasattr(m, k):
                setattr(m, k, v)
        return m

    def save_manifest(self, m: CameraManifest) -> None:
        os.makedirs(self.dir, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(m.to_dict(), f, indent=2)

    def advance(self, target: str, **fields: Any) -> CameraManifest:
        m = self.load_manifest()
        if not can_advance(m.state, target):
            raise CameraEvidenceError(
                f"{self.camera_id}: illegal transition {m.state} -> {target}")
        m.state = target
        for k, v in fields.items():
            if hasattr(m, k):
                setattr(m, k, v)
        self.save_manifest(m)
        return m

    def fail(self, reason: str) -> CameraManifest:
        m = self.load_manifest()
        m.state = FAILED
        m.failure_reason = str(reason)
        self.save_manifest(m)
        return m

    @property
    def is_sealed(self) -> bool:
        return self.load_manifest().state == "SEALED"

    @property
    def is_terminal(self) -> bool:
        return self.load_manifest().state in TERMINAL

    # -- payloads ---------------------------------------------------------

    def write_json(self, name: str, payload: Any) -> str:
        os.makedirs(self.dir, exist_ok=True)
        p = os.path.join(self.dir, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return p

    def read_json(self, name: str) -> Optional[Any]:
        p = os.path.join(self.dir, name)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_segments(self, segments: Sequence[LocalSegment]) -> str:
        return self.write_json("segments.json",
                               [s.to_dict() for s in segments])

    def read_segments(self) -> List[LocalSegment]:
        raw = self.read_json("segments.json") or []
        return [LocalSegment.from_dict(d) for d in raw]


def ready_for_global_assembly(
    root: str, master_camera: str, all_cameras: Sequence[str],
) -> Tuple[bool, str]:
    """Global assembly requires a SEALED master; others sealed OR failed.

    The master alone determines the count under fixed-master fusion, so a
    failed master aborts. A failed support camera degrades evidence only and
    must never block assembly.
    """
    master = CameraEvidenceBundle(root, master_camera).load_manifest()
    if master.state == FAILED:
        return False, f"master {master_camera} FAILED: {master.failure_reason}"
    if master.state != "SEALED":
        return False, f"master {master_camera} is {master.state}, not SEALED"
    pending = [c for c in all_cameras
               if c != master_camera
               and CameraEvidenceBundle(root, c).load_manifest().state
               not in TERMINAL]
    if pending:
        return False, f"waiting on {pending}"
    return True, "all required camera evidence available"
