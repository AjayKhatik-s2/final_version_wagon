"""The wagon-active region: audit and gate around the EXISTING master window.

The region itself is not new. `wagon_count/train_structure.WagonWindow` already
defines it -- "The counted region of the train: first WAGON .. last WAGON" --
already denies a GW id to anything outside it:

    leading_non_wagon_objects   "Segments before the first WAGON.
                                 Outside the window -> NO GW id."
    trailing_non_wagon_objects  "Segments after the last WAGON.
                                 Outside the window -> NO GW id."

already renumbers the survivors GW_1..GW_N, and already refuses to invent one:
"If no segment is classified WAGON, the window is empty and the wagon count is 0.
Nothing is invented." Fusion enforces it through `wagon_only=True`, and the same
derivation runs in both pipelines -- `tests/test_camera_pipeline_equivalence.py`
asserts the sequential and batch functions are equivalent.

So this module does not compute the region a second time. Building a second
counting system is exactly what must not happen here. What it does:

  * states the region as an explicit lifecycle -- BEFORE_WAGON_REGION ->
    WAGON_REGION_ACTIVE -> AFTER_WAGON_REGION -- with a transition per boundary;
  * gathers TOP-camera boundary evidence as CORROBORATION, never authority;
  * records every prediction that fell outside the region, and why it was
    ignored, so "why did this object get no GW id" is answerable from the log;
  * asserts the gate actually held: nothing outside the region carries a GW id.

Why top cameras corroborate but cannot move a boundary
------------------------------------------------------
`wagon_start_frame` / `wagon_end_frame` come from the RIGHT_UP master, which is
the authority for identity and order. Letting a top camera move them would
change which objects are wagons -- and `total_wagons` counts WAGON units, so it
would change the count from a camera whose vehicle-type classifier is known to
call an ENGINE a WAGON. The requirement that top cameras "cannot independently
create or extend global wagons", and that trailing predictions must not extend
`active_end`, is met by giving them no write access at all.

They still contribute: their agreement raises the recorded confidence in a
boundary, their disagreement is logged, and a sustained top-camera run that
disagrees with the master boundary is exactly the diagnostic an operator wants.

Sustained evidence
------------------
Already guaranteed upstream, not re-implemented here. `WagonWindow` is derived
from classifications that `apply_temporal_classification` has already smoothed:
a confidence-weighted vote within each segment, then hysteresis measured IN
SECONDS of train, specifically so a 0.33 s burst cannot move a boundary while a
genuine 3.87 s single-segment brake van survives. A single frame therefore
cannot open or close the region before this module ever sees it.

`min_sustain_sec` here governs only whether TOP-camera evidence is strong enough
to be recorded as corroboration -- never whether the region opens or closes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("core.active_region")

_TAG = "[ACTIVE-REGION]"
_GW_TAG = "[GLOBAL-WAGON]"

BEFORE = "BEFORE_WAGON_REGION"
ACTIVE = "WAGON_REGION_ACTIVE"
AFTER = "AFTER_WAGON_REGION"

#: A top-camera run must cover at least this much train to be recorded as
#: corroborating a boundary. Expressed in SECONDS, matching how
#: `temporal_classification` measures persistence -- the measured noise bursts
#: on this project's data are ~0.33 s, the genuine short brake van 3.87 s.
DEFAULT_MIN_SUSTAIN_SEC = 1.0

#: How far a top camera's own first/last wagon evidence may sit from the
#: master's boundary and still count as agreeing with it.
DEFAULT_BOUNDARY_TOLERANCE_SEC = 2.0

REASON_CLOSED = "region_already_closed"
REASON_NOT_OPEN = "region_not_yet_open"
REASON_NO_WAGON = "no_wagon_evidence_anywhere"


@dataclass
class Boundary:
    """One edge of the region, and everything behind the decision."""

    kind: str = "start"
    frame: Optional[int] = None
    time: Optional[float] = None
    source_camera: str = ""
    evidence: str = ""
    confidence: float = 0.0
    reason: str = ""
    corroborated_by: List[str] = field(default_factory=list)
    dissent: Dict[str, Any] = field(default_factory=dict)
    #: START only: the first canonical wagon's own frame and id, plus the delta.
    #: A non-zero delta means the region opened AFTER its first wagon, which
    #: would be the "lost first wagon" failure -- reported, never inferred.
    master_first_wagon_frame: Optional[int] = None
    first_wagon_global_id: str = ""
    frames_after_first_wagon: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "frame": self.frame,
            "master_first_wagon_frame": self.master_first_wagon_frame,
            "first_wagon_global_id": self.first_wagon_global_id,
            "frames_after_first_wagon": self.frames_after_first_wagon,
            "time": (round(self.time, 4) if self.time is not None else None),
            "source_camera": self.source_camera, "evidence": self.evidence,
            "confidence": round(float(self.confidence or 0.0), 4),
            "reason": self.reason,
            "corroborated_by": list(self.corroborated_by),
            "dissent": dict(self.dissent),
        }

    def render(self) -> str:
        bits = [f"{_TAG} {self.kind.upper()}"]
        if self.frame is not None:
            bits.append(f"frame={self.frame}")
        if self.time is not None:
            bits.append(f"time={self.time:.2f}")
        bits.append(f"camera={self.source_camera or '-'}")
        bits.append(f"evidence={self.evidence or '-'}")
        bits.append(f"confidence={float(self.confidence or 0.0):.3f}")
        if self.kind == "start" and self.master_first_wagon_frame is not None:
            bits.append(f"master_first_wagon_frame="
                        f"{self.master_first_wagon_frame}")
            bits.append(f"first_wagon={self.first_wagon_global_id}")
            bits.append(f"frames_after_first_wagon="
                        f"{self.frames_after_first_wagon}")
        bits.append(f"reason={self.reason or '-'}")
        return "  ".join(bits)


#: A leading/trailing non-wagon segment this many times the median canonical
#: wagon length is long enough to plausibly CONTAIN one or more wagons. Derived
#: from the train's own wagons, not a fixed frame count, so it scales with wagon
#: length, camera fps and train speed.
DEFAULT_SUSPECT_LENGTH_RATIO = 1.8

REASON_SUSPECT_MISSED_GAP = "SUSPECT_MISSED_GAP"


def _suspect_merged_segments(
    win: Dict[str, Any], wagons: Sequence[Any],
    ratio: float = DEFAULT_SUSPECT_LENGTH_RATIO,
) -> List[Dict[str, Any]]:
    """Leading/trailing non-wagon objects long enough to hide a real wagon.

    The failure this surfaces: if the gap between the locomotive and the first
    wagon is never DETECTED, there is no segment boundary there, so the loco and
    wagon 1 are ONE segment. That segment classifies as ENGINE (the loco
    dominates), lands in `leading_non_wagon_objects`, and the wagon inside it is
    lost -- while the region correctly reports "not active", because by the
    master's own structure the wagon sequence has not started yet.

    No leading-edge LABEL rule can recover that wagon: it is not a separate
    segment to relabel. Only a boundary the gap model did not find would split
    it, and inventing one here would mean creating a GW_n from classification
    instead of from a validated master gap -- a different architecture, and not
    this module's decision to make.

    What IS decidable from persisted evidence is that the segment is suspiciously
    long. A locomotive is roughly one wagon-length; a leading segment measuring
    two or three canonical wagons is consistent with a missed gap. This reports
    that, with the numbers, and changes no count.

    Measured against the median canonical wagon of THIS train, so it needs no
    absolute threshold and adapts to fps and train speed.
    """
    out: List[Dict[str, Any]] = []
    durations = []
    for w in wagons:
        st, en = getattr(w, "start_time", None), getattr(w, "end_time", None)
        if st is not None and en is not None and float(en) > float(st):
            durations.append(float(en) - float(st))
    if not durations:
        return out
    durations.sort()
    median = durations[len(durations) // 2]
    if median <= 0:
        return out

    for key, position in (("leading_non_wagon_objects", "leading"),
                          ("trailing_non_wagon_objects", "trailing")):
        for o in (win.get(key) or []):
            if not isinstance(o, dict):
                continue
            st, en = o.get("start_time"), o.get("end_time")
            if st is None or en is None:
                continue
            dur = float(en) - float(st)
            if dur < median * ratio:
                continue
            out.append({
                "position": position,
                "classification": o.get("classification"),
                "classification_confidence": o.get("classification_confidence"),
                "start_frame": o.get("start_frame"),
                "end_frame": o.get("end_frame"),
                "duration_sec": round(dur, 3),
                "median_wagon_sec": round(median, 3),
                "wagon_lengths": round(dur / median, 2),
                "reason": REASON_SUSPECT_MISSED_GAP,
                "note": ("this non-wagon segment is long enough to contain "
                         "canonical wagon(s); a missed gap between the "
                         "non-wagon object and the first/last wagon would "
                         "produce exactly this. NOT counted -- reported only."),
            })
    return out


@dataclass
class ActiveRegionResult:
    found: bool = False
    reason: str = ""
    start: Boundary = field(default_factory=lambda: Boundary(kind="start"))
    end: Boundary = field(default_factory=lambda: Boundary(kind="end"))
    transitions: List[str] = field(default_factory=list)
    eligible_global_ids: List[str] = field(default_factory=list)
    excluded_leading: List[Dict[str, Any]] = field(default_factory=list)
    excluded_trailing: List[Dict[str, Any]] = field(default_factory=list)
    interior_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    ignored_predictions: List[Dict[str, Any]] = field(default_factory=list)
    gate_held: bool = True
    gate_violations: List[str] = field(default_factory=list)
    suspect_merged_segments: List[Dict[str, Any]] = field(default_factory=list)

    #: Which cameras were allowed to define the canonical structure, stated so
    #: the invariant is auditable from the output rather than only from prose.
    timeline_master: str = ""
    side_support: List[str] = field(default_factory=list)

    @property
    def top_predictions_ignored(self) -> int:
        """Top-camera WAGON predictions that would have moved a boundary.

        These are exactly the classifications the side-camera authority rule
        exists to reject: a top camera calling WAGON over a leading engine or a
        trailing brake van. Counting them makes the rule's effect visible --
        "zero" means the top cameras agreed, not that nothing was checked.
        """
        return len([p for p in self.ignored_predictions
                    if p.get("camera") in C.TOP_CAMERAS])

    def structure_authority(self) -> Dict[str, Any]:
        tops = [p for p in self.ignored_predictions
                if p.get("camera") in C.TOP_CAMERAS]
        per_cam: Dict[str, Dict[str, int]] = {}
        for cam in C.TOP_CAMERAS:
            mine = [p for p in tops if p.get("camera") == cam]
            per_cam[cam] = {
                "would_have_moved_start": len(
                    [p for p in mine if p.get("reason") == REASON_NOT_OPEN]),
                "would_have_moved_end": len(
                    [p for p in mine if p.get("reason") == REASON_CLOSED]),
                "total_ignored": len(mine),
            }
        return {
            "timeline_master": self.timeline_master or C.CAMERA_RIGHT_UP,
            "side_support": list(self.side_support),
            "non_authoritative_cameras": list(C.TOP_CAMERAS),
            "top_camera_classification_is_read_only": True,
            "top_predictions_ignored_total": len(tops),
            "would_have_moved_start": sum(
                v["would_have_moved_start"] for v in per_cam.values()),
            "would_have_moved_end": sum(
                v["would_have_moved_end"] for v in per_cam.values()),
            "per_top_camera": per_cam,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found, "reason": self.reason,
            "state_sequence": list(self.transitions),
            "start": self.start.to_dict(), "end": self.end.to_dict(),
            "eligible_global_ids": list(self.eligible_global_ids),
            "eligible_count": len(self.eligible_global_ids),
            "excluded_leading": list(self.excluded_leading),
            "excluded_trailing": list(self.excluded_trailing),
            "interior_anomalies": list(self.interior_anomalies),
            "interior_anomalies_are_still_counted": True,
            "ignored_predictions": list(self.ignored_predictions),
            "gate_held": self.gate_held,
            "gate_violations": list(self.gate_violations),
            "suspect_merged_segments": list(self.suspect_merged_segments),
            "start_reason": self.start.reason,
            "end_reason": self.end.reason,
            "structure_authority": self.structure_authority(),
        }


def _runs(records: Sequence[Any], fps: float,
          wanted: str = C.CLASS_WAGON) -> List[Tuple[float, float, float]]:
    """Contiguous `wanted` runs in one camera's local clock, as
    `(start_s, end_s, mean_confidence)`. Consecutive records merge, so a
    sustained sequence is measured as one run rather than many segments."""
    if fps <= 0:
        return []
    out: List[Tuple[float, float, float]] = []
    cur: Optional[List[float]] = None
    n = 0
    for r in sorted(records, key=lambda x: (getattr(x, "start_frame", 0) or 0)):
        sf, ef = getattr(r, "start_frame", None), getattr(r, "end_frame", None)
        if sf is None or ef is None:
            continue
        s, e = float(sf) / fps, (float(ef) + 1.0) / fps
        conf = float(getattr(r, "confidence", 0.0) or 0.0)
        if str(getattr(r, "label", "") or "") != wanted:
            if cur is not None:
                out.append((cur[0], cur[1], cur[2] / max(1, n)))
                cur, n = None, 0
            continue
        if cur is None:
            cur, n = [s, e, conf], 1
        else:
            cur[1] = max(cur[1], e)
            cur[2] += conf
            n += 1
    if cur is not None:
        out.append((cur[0], cur[1], cur[2] / max(1, n)))
    return out


def resolve(
    state: Any,
    *,
    top_classifications: Optional[Dict[str, Sequence[Any]]] = None,
    camera_fps: Optional[Dict[str, float]] = None,
    camera_offsets: Optional[Dict[str, float]] = None,
    min_sustain_sec: float = DEFAULT_MIN_SUSTAIN_SEC,
    boundary_tolerance_sec: float = DEFAULT_BOUNDARY_TOLERANCE_SEC,
    verbose: bool = True,
) -> ActiveRegionResult:
    """Audit and gate the region. ONE function, both pipelines. Never raises.

    Reads the master-derived `state.wagon_window`; it does not recompute it.
    """
    tops = top_classifications or {}
    fps_map = camera_fps or {}
    offsets = camera_offsets or {}
    res = ActiveRegionResult()

    win = dict(getattr(state, "wagon_window", None) or {})
    wagons = list(getattr(state, "wagons", None) or [])

    res.transitions.append(BEFORE)
    if verbose:
        log.info("%s state=%s", _TAG, BEFORE)

    if not win.get("found"):
        # No wagon anywhere. Nothing is invented -- the window already refuses
        # to, and this records that refusal rather than papering over it.
        res.found = False
        res.reason = str(win.get("reason") or REASON_NO_WAGON)
        res.start.reason = res.end.reason = REASON_NO_WAGON
        res.excluded_leading = list(win.get("leading_non_wagon_objects") or [])
        for cam, recs in tops.items():
            for s, e, conf in _runs(recs, float(fps_map.get(cam) or 0.0)):
                if (e - s) < min_sustain_sec:
                    continue
                res.ignored_predictions.append({
                    "camera": cam, "type": C.CLASS_WAGON,
                    "start_time": round(s, 3), "end_time": round(e, 3),
                    "confidence": round(conf, 4), "reason": REASON_NOT_OPEN})
        if verbose:
            log.info("%s no active region -- %s", _TAG, res.reason)
            for p in res.ignored_predictions:
                log.info("%s IGNORE %s WAGON prediction %.2f-%.2fs reason=%s",
                         _TAG, p["camera"], p["start_time"], p["end_time"],
                         p["reason"])
        return res

    res.found = True
    res.reason = "master wagon window: first WAGON .. last WAGON"
    a_start = win.get("wagon_start_time")
    a_end = win.get("wagon_end_time")

    # The START is the first canonical wagon's own start frame -- there is no
    # hysteresis or offset between them, and `train_structure` sets it as
    # `wagon_units[0].start_frame_master`. The delta is reported anyway so a
    # future regression that introduced a delay would be visible rather than
    # inferred.
    _gw1_frame = None
    if wagons:
        _gw1_frame = getattr(wagons[0], "start_frame_master", None)
    res.start = Boundary(
        kind="start", frame=win.get("wagon_start_frame"), time=a_start,
        source_camera=str(getattr(state, "master_camera", "") or C.CAMERA_RIGHT_UP),
        evidence=f"first WAGON segment index {win.get('first_wagon_segment_index')}",
        confidence=1.0,
        reason="first_canonical_wagon")
    res.start.master_first_wagon_frame = _gw1_frame
    res.start.first_wagon_global_id = (
        str(getattr(wagons[0], "global_id", "")) if wagons else "")
    try:
        res.start.frames_after_first_wagon = (
            int(win.get("wagon_start_frame")) - int(_gw1_frame)
            if _gw1_frame is not None
            and win.get("wagon_start_frame") is not None else None)
    except (TypeError, ValueError):
        res.start.frames_after_first_wagon = None
    res.end = Boundary(
        kind="end", frame=win.get("wagon_end_frame"), time=a_end,
        source_camera=str(getattr(state, "master_camera", "") or C.CAMERA_RIGHT_UP),
        evidence=f"last WAGON segment index {win.get('last_wagon_segment_index')}",
        confidence=1.0,
        reason="RIGHT_UP master is the authority for identity and order")

    # ---- TOP-camera corroboration. Read-only, by design. ----------------
    for cam in C.TOP_CAMERAS:
        recs = tops.get(cam) or []
        fps = float(fps_map.get(cam) or 0.0)
        delta = float(offsets.get(cam) or 0.0)
        runs = [(s, e, c) for (s, e, c) in _runs(recs, fps)
                if (e - s) >= min_sustain_sec]
        if not runs:
            if verbose and recs:
                log.info("%s waiting_for_sustained_wagon_evidence camera=%s "
                         "(no run >= %.2fs)", _TAG, cam, min_sustain_sec)
            continue
        # Local -> global: global = local + delta.
        first_g = runs[0][0] + delta
        last_g = runs[-1][1] + delta
        for bnd, cam_t, master_t in (("start", first_g, a_start),
                                     ("end", last_g, a_end)):
            b = res.start if bnd == "start" else res.end
            if master_t is None:
                continue
            if abs(cam_t - float(master_t)) <= boundary_tolerance_sec:
                if cam not in b.corroborated_by:
                    b.corroborated_by.append(cam)
                b.confidence = min(1.0, b.confidence)
            else:
                b.dissent[cam] = {
                    "camera_time": round(cam_t, 3),
                    "master_time": round(float(master_t), 3),
                    "delta_sec": round(cam_t - float(master_t), 3),
                    "applied": False,
                }
        # Any sustained top-camera WAGON run outside the region is recorded and
        # ignored. A trailing one must not reopen the region; a leading one must
        # not move the start backward.
        # A run is CLIPPED against the region, not required to lie wholly
        # outside it. The real failure mode is a top camera calling WAGON
        # continuously through the trailing brake vans: one long run that starts
        # inside and never stops. The part past `active_end` is precisely the
        # prediction that must not reopen the region, so that PORTION is what
        # gets recorded and ignored.
        for s, e, conf in runs:
            gs, ge = s + delta, e + delta
            if a_end is not None and ge > float(a_end):
                seg_s = max(gs, float(a_end))
                if (ge - seg_s) >= min_sustain_sec:
                    res.ignored_predictions.append({
                        "camera": cam, "type": C.CLASS_WAGON,
                        "start_time": round(seg_s, 3), "end_time": round(ge, 3),
                        "confidence": round(conf, 4),
                        "reason": REASON_CLOSED})
            if a_start is not None and gs < float(a_start):
                seg_e = min(ge, float(a_start))
                if (seg_e - gs) >= min_sustain_sec:
                    res.ignored_predictions.append({
                        "camera": cam, "type": C.CLASS_WAGON,
                        "start_time": round(gs, 3), "end_time": round(seg_e, 3),
                        "confidence": round(conf, 4),
                        "reason": REASON_NOT_OPEN})

    # Record WHO was allowed to define this structure. RIGHT_UP is the master
    # timeline; LEFT_UP is side-camera corroboration; the two top cameras are
    # evidence only. Stated in the output so the rule is checkable from a
    # delivered artifact and not only from reading this module.
    res.timeline_master = str(getattr(state, "master_camera", "")
                              or C.CAMERA_RIGHT_UP)
    res.side_support = [c for c in C.SIDE_CAMERAS if c != res.timeline_master]

    res.transitions.append(ACTIVE)
    res.eligible_global_ids = [str(getattr(w, "global_id", "") or "")
                               for w in wagons]
    res.excluded_leading = list(win.get("leading_non_wagon_objects") or [])
    res.excluded_trailing = list(win.get("trailing_non_wagon_objects") or [])
    res.interior_anomalies = list(win.get("interior_non_wagon_objects") or [])
    res.suspect_merged_segments = _suspect_merged_segments(win, wagons)
    res.transitions.append(AFTER)

    # ---- Did the gate actually hold? ------------------------------------
    # Asserted rather than assumed: fusion's `wagon_only=True` should already
    # have kept every out-of-region object out of the roster, and this is what
    # catches it if that ever stops being true.
    # Compared against the object's OWN edges, with a small tolerance, because
    # a leading engine ends exactly where the region starts: `end < a_start` is
    # false for it, while `start < a_start` is true, and the first real wagon's
    # start EQUALS a_start so it is not caught. Same logic mirrored at the end.
    _EPS = 1e-6
    for w in wagons:
        st = getattr(w, "start_time", None)
        en = getattr(w, "end_time", None)
        gw = str(getattr(w, "global_id", "") or "")
        if a_start is not None and st is not None and float(st) < float(a_start) - _EPS:
            res.gate_violations.append(
                f"{gw} starts at {float(st):.3f}s, before the region "
                f"({float(a_start):.3f}s)")
        if a_end is not None and en is not None and float(en) > float(a_end) + _EPS:
            res.gate_violations.append(
                f"{gw} ends at {float(en):.3f}s, after the region "
                f"({float(a_end):.3f}s)")
    res.gate_held = not res.gate_violations

    if verbose:
        log.info("%s", res.start.render())
        log.info("%s state=%s", _TAG, ACTIVE)
        for cam in res.start.corroborated_by:
            log.info("%s START corroborated_by=%s", _TAG, cam)
        for cam, d in res.start.dissent.items():
            log.info("%s START dissent=%s delta=%.2fs applied=False", _TAG,
                     cam, d.get("delta_sec", 0.0))
        log.info("%s", res.end.render())
        for cam in res.end.corroborated_by:
            log.info("%s END corroborated_by=%s", _TAG, cam)
        log.info("%s state=%s", _TAG, AFTER)
        for p in res.ignored_predictions:
            log.info("%s IGNORE trailing/leading WAGON prediction camera=%s "
                     "%.2f-%.2fs reason=%s", _TAG, p["camera"],
                     p["start_time"], p["end_time"], p["reason"])
        for gw in res.eligible_global_ids:
            log.info("%s %s created inside_active_region=true", _GW_TAG, gw)
        for pos, objs in (("leading", res.excluded_leading),
                          ("trailing", res.excluded_trailing)):
            for o in objs:
                log.info("%s non-wagon position=%s type=%s "
                         "outside_active_region=true gw_id=None", _GW_TAG, pos,
                         o.get("classification"))
        for o in res.interior_anomalies:
            log.info("%s interior anomaly type=%s STILL COUNTED "
                     "(master gaps are authoritative)", _GW_TAG,
                     o.get("classification"))
        for sm in res.suspect_merged_segments:
            log.warning("%s SUSPECT_MISSED_GAP %s %s frames %s-%s spans "
                        "%.2fs = %.2f canonical wagon lengths (median %.2fs) "
                        "-- may CONTAIN a real wagon; not counted, reported",
                        _TAG, sm["position"], sm["classification"],
                        sm["start_frame"], sm["end_frame"],
                        sm["duration_sec"], sm["wagon_lengths"],
                        sm["median_wagon_sec"])
        if not res.gate_held:
            log.error("%s SEVERE: gate did not hold -- %s", _TAG,
                      res.gate_violations)
        log.info("%s %d eligible wagon(s); excluded leading=%d trailing=%d; "
                 "interior anomalies=%d; ignored predictions=%d", _TAG,
                 len(res.eligible_global_ids), len(res.excluded_leading),
                 len(res.excluded_trailing), len(res.interior_anomalies),
                 len(res.ignored_predictions))
    return res
