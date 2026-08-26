"""Prove, from persisted evidence, that only RIGHT_UP shapes the roster.

The master-fixed architecture says support cameras observe the canonical
timeline and never redefine it. That is a claim about behaviour, and a claim
worth checking against what a real run actually wrote rather than against the
docstrings that assert it.

Everything needed is already on disk. `assemble_global_train_state_master_fixed`
persists `global_gaps` (minted from RIGHT_UP alone), `camera_offsets`,
`support_alignment_summary` (matched / missing / extra per camera),
`extra_support_observations` (candidate gaps that did NOT become boundaries),
`wagon_window` and its own `invariant_checks`. This module reads those and
answers two separate questions that are easy to conflate:

    Did a support camera CHANGE the canonical roster?      -> violation
    Did a support camera DISAGREE with the canonical roster? -> expected

The second is normal and frequent. A support camera sees the train from a
different angle, so it misses gaps, invents gaps, splits a wagon across two
local segments, merges two into one, and occasionally classifies a wagon as a
brake van. All of that is recorded and none of it is a fault. Only the first is
a fault, and conflating them makes a healthy run look broken -- or hides a real
break in the noise.

Reads only. Computes no gaps, no wagons, no offsets, and changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C
from core.master_timeline import (
    AVAILABLE, CameraClock, DEFAULT_BOUNDARY_POLICY, BoundaryPolicy,
    LocalWindow, master_interval_to_local,
)

#: Support-camera outcomes that are DISAGREEMENTS, not violations. Each says
#: the camera saw something different; none says the roster moved.
EXPECTED_DISAGREEMENTS = (
    "ONE_TO_MANY",        # one local segment spans several canonical wagons
    "MANY_TO_ONE",        # several local segments fall in one canonical wagon
    "UNMATCHED",          # a local segment matches no canonical wagon
    "UNRESOLVED_OFFSET",  # this camera's clock never aligned
    "GAP_MISSED",         # canonical gap this camera did not detect
    "GAP_EXTRA",          # gap this camera detected that RIGHT_UP did not
    "CLASS_DISAGREEMENT",
)


@dataclass
class CameraObservation:
    """One camera's view of ONE canonical wagon. Never an identity of its own."""
    camera_id: str
    matched_master_wagon: str
    mapping_status: str = "UNMATCHED"
    local_segment_ids: List[str] = field(default_factory=list)
    local_start_frame: Optional[int] = None
    local_end_frame: Optional[int] = None
    local_start_time: Optional[float] = None
    local_end_time: Optional[float] = None
    classification: Optional[str] = None
    classification_confidence: float = 0.0
    alignment_offset: float = 0.0
    offset_status: str = ""
    coverage_status: str = ""            # from master_timeline
    coverage_reason: str = ""
    gap_before_detected: Optional[bool] = None
    gap_after_detected: Optional[bool] = None
    provenance: str = ""

    @property
    def has_evidence(self) -> bool:
        return bool(self.local_segment_ids) or self.coverage_status == AVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "matched_master_wagon": self.matched_master_wagon,
            "mapping_status": self.mapping_status,
            "local_segment_ids": list(self.local_segment_ids),
            "local_start_frame": self.local_start_frame,
            "local_end_frame": self.local_end_frame,
            "local_start_time": self.local_start_time,
            "local_end_time": self.local_end_time,
            "classification": self.classification,
            "classification_confidence": round(self.classification_confidence, 4),
            "alignment_offset": round(self.alignment_offset, 4),
            "offset_status": self.offset_status,
            "coverage_status": self.coverage_status,
            "coverage_reason": self.coverage_reason,
            "gap_before_detected": self.gap_before_detected,
            "gap_after_detected": self.gap_after_detected,
            "provenance": self.provenance,
        }


@dataclass
class CanonicalWagonView:
    """A canonical wagon plus what each camera saw of it.

    `canonical_*` comes from RIGHT_UP and is the identity. `observations` are
    evidence hung off it. The two are deliberately NOT flattened: a camera
    saying BRAKE_VAN does not make the wagon a brake van, and the view has to
    be able to show both.
    """
    canonical_wagon_id: str
    wagon_index: int
    canonical_class: str
    classification_confidence: float
    master_start_time: float
    master_end_time: float
    master_start_frame: int
    master_end_frame: int
    master_gap_before: Optional[Dict[str, Any]] = None
    master_gap_after: Optional[Dict[str, Any]] = None
    provenance: str = ""
    observations: Dict[str, CameraObservation] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.master_end_time - self.master_start_time

    def classification_disagreements(self) -> Dict[str, str]:
        """Cameras whose class differs from canonical. Reported, never applied."""
        out = {}
        for cam, obs in self.observations.items():
            if obs.classification and obs.classification != self.canonical_class:
                out[cam] = obs.classification
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_wagon_id": self.canonical_wagon_id,
            "wagon_index": self.wagon_index,
            "canonical_class": self.canonical_class,
            "classification_confidence": round(self.classification_confidence, 4),
            "master_start_time": round(self.master_start_time, 4),
            "master_end_time": round(self.master_end_time, 4),
            "master_start_frame": self.master_start_frame,
            "master_end_frame": self.master_end_frame,
            "duration": round(self.duration, 4),
            "master_gap_before": self.master_gap_before,
            "master_gap_after": self.master_gap_after,
            "provenance": self.provenance,
            "observations": {c: o.to_dict()
                             for c, o in self.observations.items()},
            "classification_disagreements": self.classification_disagreements(),
        }


@dataclass
class Violation:
    """A support camera actually changed the canonical roster."""
    kind: str
    camera_id: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "camera_id": self.camera_id,
                "detail": self.detail}


@dataclass
class AuditResult:
    master_camera: str = C.MASTER_CAMERA
    canonical_wagon_count: int = 0
    canonical_gap_count: int = 0
    timeline_start: float = 0.0
    timeline_end: float = 0.0
    wagons: List[CanonicalWagonView] = field(default_factory=list)
    camera_offsets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_camera_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    disagreements: Dict[str, List[str]] = field(default_factory=dict)
    violations: List[Violation] = field(default_factory=list)
    engine_invariant_checks: Dict[str, Any] = field(default_factory=dict)

    @property
    def invariant_holds(self) -> bool:
        return not self.violations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "master_camera": self.master_camera,
            "canonical_wagon_count": self.canonical_wagon_count,
            "canonical_gap_count": self.canonical_gap_count,
            "timeline_start": round(self.timeline_start, 4),
            "timeline_end": round(self.timeline_end, 4),
            "invariant_holds": self.invariant_holds,
            "violations": [v.to_dict() for v in self.violations],
            "camera_offsets": self.camera_offsets,
            "per_camera_stats": self.per_camera_stats,
            "disagreements": {k: list(v) for k, v in self.disagreements.items()},
            "engine_invariant_checks": dict(self.engine_invariant_checks),
            "wagons": [w.to_dict() for w in self.wagons],
        }

    def summary_lines(self) -> List[str]:
        out = [
            f"canonical roster : {self.canonical_wagon_count} wagon(s) from "
            f"{self.master_camera}, {self.canonical_gap_count} canonical gap(s)",
            f"timeline         : {self.timeline_start:.2f}s -> "
            f"{self.timeline_end:.2f}s",
        ]
        for cam, st in sorted(self.per_camera_stats.items()):
            out.append(
                f"  {cam:<13} offset={st.get('offset', 0.0):+7.3f}s "
                f"({st.get('offset_status', '?')})  "
                f"matched={st.get('matched_gaps', 0)} "
                f"missed={st.get('missed_gaps', 0)} "
                f"extra={st.get('extra_gaps', 0)} "
                f"unmatched_segments={st.get('unmatched_segments', 0)} "
                f"class_disagreements={st.get('class_disagreements', 0)}")
        out.append(f"invariant        : "
                   f"{'HOLDS' if self.invariant_holds else 'VIOLATED'}")
        for v in self.violations:
            out.append(f"  VIOLATION {v.kind} [{v.camera_id}] {v.detail}")
        return out


# --- invariant checks -------------------------------------------------------

def check_invariant(state: Any) -> List[Violation]:
    """Every way a support camera could have moved the roster, checked.

    Each check is phrased so that PASSING means the canonical structure came
    from RIGHT_UP alone. A support camera disagreeing is not tested here --
    that is expected and is reported separately.
    """
    out: List[Violation] = []
    master = getattr(state, "master_camera", C.MASTER_CAMERA)

    # 1. Every canonical gap was minted from a master observation.
    for g in (getattr(state, "global_gaps", None) or []):
        if not isinstance(g, dict):
            continue
        src = g.get("master_camera") or (
            (g.get("master_observation") or {}).get("camera_id"))
        if src and src != master:
            out.append(Violation(
                "GAP_FROM_SUPPORT_CAMERA", str(src),
                f"global gap {g.get('global_gap_id')} claims master camera "
                f"{src!r}, not {master!r}"))

    # 2. The engine's own count check: RIGHT_UP validated gaps == global gaps.
    checks = getattr(state, "invariant_checks", None) or {}
    ru = checks.get("right_up_final_gap_count")
    gg = checks.get("global_gap_count")
    if ru is not None and gg is not None and int(ru) != int(gg):
        out.append(Violation(
            "GAP_COUNT_DIVERGED", master,
            f"RIGHT_UP validated {ru} gap(s) but the global sequence has {gg}"))

    # 3. No support camera's EXTRA observation became a canonical gap.
    extras = getattr(state, "extra_support_observations", None) or {}
    canonical_times = {round(float((g.get("master_observation") or {})
                                   .get("center_time", -1.0)), 3)
                       for g in (getattr(state, "global_gaps", None) or [])
                       if isinstance(g, dict)}
    for cam, obs_list in extras.items():
        for o in (obs_list or []):
            if not isinstance(o, dict):
                continue
            t = round(float(o.get("global_time", o.get("center_time", -1.0))), 3)
            if t in canonical_times:
                out.append(Violation(
                    "EXTRA_BECAME_CANONICAL", cam,
                    f"{cam} extra observation at {t}s coincides with a "
                    f"canonical gap -- a support detection must never mint one"))

    # 4. Ids are contiguous GW_1..GW_N in ascending master time.
    wagons = list(getattr(state, "wagons", ()) or ())
    for i, w in enumerate(wagons, start=1):
        if w.global_id != f"GW_{i}":
            out.append(Violation(
                "ROSTER_RENUMBERED", master,
                f"position {i} holds {w.global_id!r}; the roster must be "
                f"contiguous GW_1..GW_N in master order"))
            break
    times = [float(w.start_time) for w in wagons]
    if times != sorted(times):
        out.append(Violation("ROSTER_REORDERED", master,
                             "canonical wagons are not in ascending master time"))

    # 5. The roster size matches the master's own wagon-window count.
    mwc = getattr(state, "master_wagon_count", None)
    if mwc and int(mwc) != len(wagons):
        out.append(Violation(
            "COUNT_DIVERGED_FROM_MASTER_WINDOW", master,
            f"master wagon window says {mwc} wagon(s), roster holds "
            f"{len(wagons)}"))
    return out


# --- the audit --------------------------------------------------------------

def audit(state: Any, *,
          local_segments: Optional[Dict[str, Sequence[Any]]] = None,
          per_camera_meta: Optional[Dict[str, Dict[str, Any]]] = None,
          policy: BoundaryPolicy = DEFAULT_BOUNDARY_POLICY,
          all_cameras: Sequence[str] = C.ALL_CAMERAS) -> AuditResult:
    """Build the canonical view and check the invariant, from persisted state.

    `local_segments` maps camera -> that camera's own LocalSegment records, when
    available (sequential bundles, or a batch tracking dump). Without them the
    view still reports the canonical roster, the offsets, coverage per camera
    and the engine's gap statistics -- just not local segment ids.
    """
    res = AuditResult(master_camera=getattr(state, "master_camera",
                                            C.MASTER_CAMERA))
    wagons = list(getattr(state, "wagons", ()) or ())
    res.canonical_wagon_count = len(wagons)
    res.canonical_gap_count = len(getattr(state, "global_gaps", None) or [])
    res.camera_offsets = dict(getattr(state, "camera_offsets", None) or {})
    res.engine_invariant_checks = dict(getattr(state, "invariant_checks", None)
                                       or {})
    if wagons:
        res.timeline_start = float(wagons[0].start_time)
        res.timeline_end = float(wagons[-1].end_time)

    meta = per_camera_meta or {}
    try:
        resolved = state.camera_time_offsets()
    except Exception:
        resolved = {}
    clocks = {
        cam: CameraClock(
            camera_id=cam,
            fps=float((meta.get(cam) or {}).get("fps") or 0.0),
            total_frames=int((meta.get(cam) or {}).get("total_frames") or 0),
            offset=float(resolved.get(cam, 0.0) or 0.0),
            offset_status=str((res.camera_offsets.get(cam) or {}).get(
                "status") or "UNRESOLVED"))
        for cam in all_cameras
    }

    # Per-camera gap agreement, straight from what fusion already recorded.
    align = dict(getattr(state, "support_alignment_summary", None) or {})
    for cam in all_cameras:
        a = align.get(cam) or {}
        res.per_camera_stats[cam] = {
            "offset": clocks[cam].offset,
            "offset_status": clocks[cam].offset_status,
            "matched_gaps": len(a.get("matches") or {}) or a.get("matched", 0),
            "missed_gaps": len(a.get("missing_global_gap_ids") or []),
            "extra_gaps": len(a.get("extra_observations") or []),
            "unmatched_segments": 0,
            "class_disagreements": 0,
            "is_master": cam == res.master_camera,
        }

    mapped = _map_local_segments(local_segments or {}, wagons, resolved,
                                 res.camera_offsets)

    for w in wagons:
        view = CanonicalWagonView(
            canonical_wagon_id=w.global_id,
            wagon_index=w.wagon_index,
            canonical_class=w.classification,
            classification_confidence=float(w.classification_confidence or 0.0),
            master_start_time=float(w.start_time),
            master_end_time=float(w.end_time),
            master_start_frame=int(w.start_frame_master),
            master_end_frame=int(w.end_frame_master),
            master_gap_before=getattr(w, "leading_gap", None),
            master_gap_after=getattr(w, "trailing_gap", None),
            provenance=f"boundary from {res.master_camera} validated gaps")
        for cam in all_cameras:
            clock = clocks[cam]
            win = master_interval_to_local(clock, w.start_time, w.end_time)
            hits = mapped.get(cam, {}).get(w.global_id, [])
            obs = CameraObservation(
                camera_id=cam,
                matched_master_wagon=w.global_id,
                mapping_status=_status_for(hits, cam, clock),
                local_segment_ids=[h["local_id"] for h in hits],
                local_start_frame=(win.start_frame if win.available else None),
                local_end_frame=(win.end_frame if win.available else None),
                local_start_time=(round(win.start_time, 4)
                                  if win.available else None),
                local_end_time=(round(win.end_time, 4) if win.available else None),
                classification=(hits[0].get("label") if hits else None),
                classification_confidence=float(
                    hits[0].get("confidence") or 0.0) if hits else 0.0,
                alignment_offset=clock.offset,
                offset_status=clock.offset_status,
                coverage_status=win.status,
                coverage_reason=win.reason,
                provenance=("local segment" if hits else
                            "projected from master timeline"))
            view.observations[cam] = obs
        for cam, other in view.classification_disagreements().items():
            res.disagreements.setdefault("CLASS_DISAGREEMENT", []).append(
                f"{view.canonical_wagon_id}: {res.master_camera}="
                f"{view.canonical_class} vs {cam}={other}")
            res.per_camera_stats[cam]["class_disagreements"] += 1
        res.wagons.append(view)

    for cam, per_gw in mapped.items():
        res.per_camera_stats.setdefault(cam, {})["unmatched_segments"] = \
            len(per_gw.get(None, []))
    for cam, st in res.per_camera_stats.items():
        if st.get("missed_gaps"):
            res.disagreements.setdefault("GAP_MISSED", []).append(
                f"{cam}: {st['missed_gaps']} canonical gap(s) not detected")
        if st.get("extra_gaps"):
            res.disagreements.setdefault("GAP_EXTRA", []).append(
                f"{cam}: {st['extra_gaps']} extra gap(s), none canonical")

    res.violations = check_invariant(state)
    return res


def _status_for(hits: List[Dict[str, Any]], cam: str,
                clock: CameraClock) -> str:
    if not clock.offset_resolved and cam != C.MASTER_CAMERA:
        return "UNRESOLVED_OFFSET"
    if not hits:
        return "UNMATCHED"
    if len(hits) > 1:
        return "MANY_TO_ONE"
    return hits[0].get("kind") or "EXACT"


def _map_local_segments(local_segments: Dict[str, Sequence[Any]],
                        wagons: Sequence[Any],
                        resolved: Dict[str, float],
                        offsets_meta: Dict[str, Dict[str, Any]]
                        ) -> Dict[str, Dict[Optional[str], List[Dict[str, Any]]]]:
    """Group each camera's local segments under the canonical wagon they hit.

    Reuses `core.camera_evidence.map_segments_to_global`, the existing mapper,
    rather than adding a second matching rule. Segments matching nothing are
    filed under None so they stay visible as UNMATCHED.
    """
    from core.camera_evidence import map_segments_to_global

    out: Dict[str, Dict[Optional[str], List[Dict[str, Any]]]] = {}
    for cam, segs in (local_segments or {}).items():
        if not segs:
            continue
        is_resolved = (offsets_meta.get(cam, {}) or {}).get("status") in (
            "REFERENCE", "RESOLVED")
        maps = map_segments_to_global(
            list(segs), list(wagons), camera_id=cam,
            offset=float(resolved.get(cam, 0.0) or 0.0),
            offset_resolved=bool(is_resolved))
        bucket: Dict[Optional[str], List[Dict[str, Any]]] = {}
        by_local = {getattr(s, "local_id", ""): s for s in segs}
        for m in maps:
            seg = by_local.get(m.local_id)
            bucket.setdefault(m.global_id or None, []).append({
                "local_id": m.local_id,
                "kind": m.kind,
                "label": getattr(seg, "label", None),
                "confidence": getattr(seg, "confidence", 0.0),
            })
        out[cam] = bucket
    return out
