"""Where the WAGONS are, from sustained classification across four cameras.

The canonical timeline contains WAGON activity and nothing else. ENGINE,
BRAKE_VAN and UNKNOWN are real parts of the train, and they are still reported,
but they are not regions of this timeline and can never become a `GW_n`.

The failure this replaces: the head of the rake is where classifiers are least
reliable. The locomotive shares a lot of appearance with a wagon, and a gap
detector fires on its leading face, so a single early WAGON frame next to a gap
was enough to start the timeline on the engine -- inventing GW_1 out of the
locomotive and shifting every subsequent id by one.

Two ideas fix that, and both are about EVIDENCE rather than thresholds:

  sustained, not instantaneous
    A camera enters wagon-active only after WAGON persists for
    `min_active_duration`, and leaves only after WAGON has been absent for
    `min_inactive_duration`. Hysteresis, so one misread frame neither starts
    nor ends the timeline. A lone WAGON prediction on the engine is recorded as
    a rejected blip, not a boundary.

  four cameras, not one
    Each camera produces its own wagon-active interval, projected onto the
    master clock by its measured offset, and the common interval is the MEDIAN
    of the four. Median rather than earliest, because earliest is precisely the
    failure mode -- one camera mistaking the locomotive for a wagon would drag
    the start onto the engine. With four opinions, one wrong one cannot.

RIGHT_UP stays the gap authority throughout. The other three corroborate the
INTERVAL; they never create, move or renumber a gap, and they never renumber a
wagon. Once the common interval is known, RIGHT_UP's canonical gaps INSIDE it
produce GW_1..GW_N.

    4-camera classification -> per-camera WAGON-active regions
      -> common WAGON-active interval -> RIGHT_UP gaps inside it
      -> GW_1..GW_N -> projected to all cameras
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C

#: The ONLY class that makes the timeline active. Stated as a constant so the
#: rule is greppable: nothing else may extend a wagon-active interval.
ACTIVE_CLASS = C.CLASS_WAGON

#: Present on the train, never on this timeline, never a GW_n.
NON_WAGON_CLASSES: Tuple[str, ...] = (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN,
                                      C.CLASS_UNKNOWN)

METHOD_MEDIAN = "median_of_cameras"
METHOD_MASTER_ONLY = "master_only"
METHOD_NONE = "no_wagon_evidence"


@dataclass(frozen=True)
class ActivationPolicy:
    """How much evidence is enough. All explicit, all testable.

    `min_active_duration` is the sustained WAGON time needed to OPEN a region.
    It has to exceed the longest plausible misread at the head of the train --
    an engine glimpsed as a wagon for a fraction of a second -- while staying
    well under a real wagon, which passes in several seconds.

    `min_inactive_duration` is the sustained non-WAGON time needed to CLOSE
    one. Larger than the active threshold on purpose: an inter-wagon gap
    briefly classifies as something other than WAGON, and closing the region
    there would chop the rake into pieces.
    """
    min_active_duration: float = 2.0
    min_inactive_duration: float = 6.0
    min_confidence: float = 0.0
    prefer_master: bool = False

    def __post_init__(self):
        for name in ("min_active_duration", "min_inactive_duration"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")


DEFAULT_POLICY = ActivationPolicy()


@dataclass
class WagonActiveInterval:
    """One sustained stretch of WAGON activity, on the master clock."""
    start_time: float
    end_time: float
    segment_count: int = 0
    mean_confidence: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return {"start_time": round(self.start_time, 4),
                "end_time": round(self.end_time, 4),
                "duration": round(self.duration, 4),
                "segment_count": self.segment_count,
                "mean_confidence": round(self.mean_confidence, 4)}


@dataclass
class RejectedBlip:
    """WAGON evidence too short to open a region. Kept so it stays auditable."""
    start_time: float
    end_time: float
    label: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"start_time": round(self.start_time, 4),
                "end_time": round(self.end_time, 4),
                "label": self.label, "reason": self.reason}


@dataclass
class CameraWagonActivity:
    """One camera's wagon-active view, projected onto the master clock."""
    camera_id: str
    wagon_active_start: Optional[float] = None
    wagon_active_end: Optional[float] = None
    intervals: List[WagonActiveInterval] = field(default_factory=list)
    rejected_blips: List[RejectedBlip] = field(default_factory=list)
    offset: float = 0.0
    offset_status: str = ""
    non_wagon_before: List[str] = field(default_factory=list)
    non_wagon_after: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def usable(self) -> bool:
        return (self.wagon_active_start is not None
                and self.wagon_active_end is not None)

    @property
    def duration(self) -> float:
        return ((self.wagon_active_end - self.wagon_active_start)
                if self.usable else 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "wagon_active_start": (round(self.wagon_active_start, 4)
                                   if self.wagon_active_start is not None
                                   else None),
            "wagon_active_end": (round(self.wagon_active_end, 4)
                                 if self.wagon_active_end is not None
                                 else None),
            "duration": round(self.duration, 4),
            "interval_count": len(self.intervals),
            "intervals": [i.to_dict() for i in self.intervals],
            "rejected_blips": [b.to_dict() for b in self.rejected_blips],
            "offset": round(self.offset, 4),
            "offset_status": self.offset_status,
            "non_wagon_before": list(self.non_wagon_before),
            "non_wagon_after": list(self.non_wagon_after),
            "usable": self.usable,
            "reason": self.reason,
        }


@dataclass
class CommonWagonWindow:
    """The one WAGON-active interval the canonical roster is built inside."""
    found: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    method: str = METHOD_NONE
    contributing: List[str] = field(default_factory=list)
    per_camera: Dict[str, CameraWagonActivity] = field(default_factory=dict)
    master_start: Optional[float] = None
    master_end: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time) if self.found else 0.0

    def contains(self, t: float) -> bool:
        return self.found and self.start_time <= t <= self.end_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "wagon_active_start": round(self.start_time, 4) if self.found else None,
            "wagon_active_end": round(self.end_time, 4) if self.found else None,
            "duration_seconds": round(self.duration, 4),
            "method": self.method,
            "contributing_cameras": list(self.contributing),
            "master_start": (round(self.master_start, 4)
                             if self.master_start is not None else None),
            "master_end": (round(self.master_end, 4)
                           if self.master_end is not None else None),
            "per_camera": {c: a.to_dict() for c, a in self.per_camera.items()},
            "notes": list(self.notes),
            "reason": self.reason,
        }

    def summary_lines(self) -> List[str]:
        if not self.found:
            return [f"wagon-active : NOT FOUND -- {self.reason}"]
        out = [f"wagon-active : {self.start_time:.2f}s -> {self.end_time:.2f}s "
               f"({self.duration:.2f}s) via {self.method} "
               f"from {self.contributing}"]
        if self.master_start is not None:
            out.append(f"  master said  {self.master_start:.2f} -> "
                       f"{self.master_end:.2f}  "
                       f"(delta {self.start_time - self.master_start:+.2f} / "
                       f"{self.end_time - self.master_end:+.2f})")
        for cam, a in sorted(self.per_camera.items()):
            if a.usable:
                out.append(f"  {cam:<13} {a.wagon_active_start:7.2f} -> "
                           f"{a.wagon_active_end:7.2f}  "
                           f"intervals={len(a.intervals)} "
                           f"rejected={len(a.rejected_blips)}")
            else:
                out.append(f"  {cam:<13} no sustained WAGON -- {a.reason}")
        for n in self.notes:
            out.append(f"  note: {n}")
        return out


# --- per-camera detection ---------------------------------------------------

def camera_wagon_activity(spans: Sequence[Any], camera_id: str, *,
                          offset: float = 0.0, offset_status: str = "",
                          policy: ActivationPolicy = DEFAULT_POLICY
                          ) -> CameraWagonActivity:
    """One camera's sustained WAGON regions, on the master clock.

    `spans` carry `.start_time`, `.end_time`, `.label`, `.confidence` in the
    camera's LOCAL clock; `offset` projects them (`t_master = t_local + delta`).

    Hysteresis is what makes this robust: a run of WAGON must last
    `min_active_duration` before it opens a region, so an engine misread for a
    fraction of a second is rejected outright and recorded as a blip. Once
    open, a region survives short non-WAGON stretches -- an inter-wagon gap
    reads as something other than WAGON, and closing there would split the rake.
    """
    act = CameraWagonActivity(camera_id=camera_id, offset=float(offset),
                              offset_status=offset_status)
    if not spans:
        act.reason = "no classified segments"
        return act

    ordered = sorted(spans, key=lambda s: float(s.start_time))
    projected = [(float(s.start_time) + offset, float(s.end_time) + offset,
                  str(getattr(s, "label", "") or C.CLASS_UNKNOWN),
                  float(getattr(s, "confidence", 0.0) or 0.0))
                 for s in ordered]

    def is_active(label: str, conf: float) -> bool:
        return label == ACTIVE_CLASS and conf >= policy.min_confidence

    # 1. contiguous runs of WAGON / non-WAGON
    runs: List[Tuple[bool, float, float, List[Tuple]]] = []
    for s, e, label, conf in projected:
        a = is_active(label, conf)
        if runs and runs[-1][0] == a:
            k, rs, _re, members = runs[-1]
            members.append((s, e, label, conf))
            runs[-1] = (k, rs, e, members)
        else:
            runs.append((a, s, e, [(s, e, label, conf)]))

    # 2. an active run only OPENS a region if it is sustained
    for a, rs, re_, members in runs:
        if not a:
            continue
        if (re_ - rs) < policy.min_active_duration:
            act.rejected_blips.append(RejectedBlip(
                rs, re_, ACTIVE_CLASS,
                f"WAGON for {re_ - rs:.2f}s, under the "
                f"{policy.min_active_duration:.2f}s needed to open a region"))

    sustained = [(rs, re_, members) for a, rs, re_, members in runs
                 if a and (re_ - rs) >= policy.min_active_duration]
    if not sustained:
        seen = sorted({label for _s, _e, label, _c in projected})
        act.reason = (f"no WAGON run reached {policy.min_active_duration:.2f}s"
                      f" -- saw {seen}")
        return act

    # 3. merge across short non-WAGON stretches, so a gap does not close a region
    merged: List[Tuple[float, float, List[Tuple]]] = [sustained[0]]
    for rs, re_, members in sustained[1:]:
        prev_s, prev_e, prev_m = merged[-1]
        if (rs - prev_e) < policy.min_inactive_duration:
            merged[-1] = (prev_s, re_, prev_m + members)
        else:
            merged.append((rs, re_, members))

    for rs, re_, members in merged:
        confs = [c for _s, _e, _l, c in members] or [0.0]
        act.intervals.append(WagonActiveInterval(
            start_time=rs, end_time=re_, segment_count=len(members),
            mean_confidence=sum(confs) / len(confs)))

    act.wagon_active_start = act.intervals[0].start_time
    act.wagon_active_end = act.intervals[-1].end_time
    act.non_wagon_before = [l for s, _e, l, _c in projected
                            if s < act.wagon_active_start
                            and l in NON_WAGON_CLASSES]
    act.non_wagon_after = [l for _s, e, l, _c in projected
                           if e > act.wagon_active_end
                           and l in NON_WAGON_CLASSES]
    return act


# --- combining the four -----------------------------------------------------

def common_wagon_window(
    activities: Dict[str, CameraWagonActivity], *,
    master_camera: str = C.MASTER_CAMERA,
    policy: ActivationPolicy = DEFAULT_POLICY,
) -> CommonWagonWindow:
    """One WAGON-active interval, corroborated across the cameras.

    The median of the per-camera starts and ends, NOT the earliest. Earliest is
    exactly the failure being fixed: a camera that mistakes the locomotive for
    a wagon reports an early start, and taking the minimum would let that one
    camera put the boundary on the engine. A median needs a majority to be
    wrong before it moves.

    With one camera the median is that camera. With two the median of the pair
    is their midpoint, which is defensible but weakly corroborated, so it is
    noted. RIGHT_UP's own figures are always recorded alongside the result, so
    a reviewer can see how far corroboration moved the boundary and why.
    """
    win = CommonWagonWindow(per_camera=dict(activities))
    usable = {c: a for c, a in activities.items() if a.usable}
    if not usable:
        win.reason = "no camera produced a sustained WAGON region"
        return win

    master = activities.get(master_camera)
    if master is not None and master.usable:
        win.master_start = master.wagon_active_start
        win.master_end = master.wagon_active_end

    if policy.prefer_master and master is not None and master.usable:
        win.start_time, win.end_time = win.master_start, win.master_end
        win.method = METHOD_MASTER_ONLY
        win.contributing = [master_camera]
    else:
        starts = [a.wagon_active_start for a in usable.values()]
        ends = [a.wagon_active_end for a in usable.values()]
        win.start_time = float(statistics.median(starts))
        win.end_time = float(statistics.median(ends))
        win.method = METHOD_MEDIAN
        win.contributing = sorted(usable)
        if len(usable) == 1:
            win.notes.append(
                f"only {win.contributing[0]} produced sustained WAGON "
                f"evidence; the interval is uncorroborated")
        elif len(usable) == 2:
            win.notes.append(
                "two cameras only; the median is their midpoint and a single "
                "bad camera can still move it")

    if win.master_start is not None:
        d0 = win.start_time - win.master_start
        if abs(d0) > 1e-6:
            win.notes.append(
                f"corroboration moved the start {d0:+.2f}s from "
                f"{master_camera}'s own {win.master_start:.2f}s")

    win.found = win.end_time > win.start_time
    if not win.found:
        win.reason = (f"degenerate interval {win.start_time:.2f} -> "
                      f"{win.end_time:.2f}")
    return win


def gaps_inside_window(gap_times: Sequence[float],
                       window: CommonWagonWindow) -> List[float]:
    """RIGHT_UP's canonical gap centres that fall inside the WAGON-active span.

    Subtractive and master-only. Support cameras corroborated the INTERVAL;
    they contribute no gap here and cannot renumber one.
    """
    if not window.found:
        return list(gap_times)
    return [t for t in sorted(gap_times)
            if window.start_time <= t <= window.end_time]


def build_wagon_timeline(window: CommonWagonWindow,
                         gap_times: Sequence[float],
                         *, number_from: int = 1
                         ) -> List[Dict[str, Any]]:
    """GW_1..GW_N from RIGHT_UP's gaps inside the corroborated interval.

    The interval's own edges bound the first and last wagon, so the roster
    covers wagon activity and stops there. ENGINE and BRAKE_VAN lie outside it
    by construction and receive no id.
    """
    if not window.found:
        return []
    inside = gaps_inside_window(gap_times, window)
    bounds = [window.start_time] + list(inside) + [window.end_time]
    out: List[Dict[str, Any]] = []
    n = number_from
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e <= s:
            continue
        out.append({
            "global_id": f"GW_{n}", "wagon_index": n,
            "start_time": round(s, 4), "end_time": round(e, 4),
            "duration": round(e - s, 4),
            "gap_before": (round(bounds[i], 4) if i > 0 else None),
            "gap_after": (round(bounds[i + 1], 4)
                          if i + 1 < len(bounds) - 1 else None),
            "classification": ACTIVE_CLASS,
        })
        n += 1
    return out


def audit_payload(window: CommonWagonWindow,
                  gap_times: Sequence[float] = ()) -> Dict[str, Any]:
    """The block published into global_state, so the boundary is auditable.

    Answers, from persisted data alone: where each camera saw wagons, what
    evidence it used, what it rejected, how the four were combined, and which
    canonical gaps ended up inside the result.
    """
    inside = gaps_inside_window(gap_times, window)
    payload = window.to_dict()
    payload.update({
        "schema": "wagon_eye.wagon_active.v1",
        "canonical_gaps_total": len(list(gap_times)),
        "canonical_gaps_inside": len(inside),
        "canonical_gaps_excluded": len(list(gap_times)) - len(inside),
        "gap_times_inside": [round(t, 4) for t in inside],
        "non_wagon_classes_excluded": list(NON_WAGON_CLASSES),
        "active_class": ACTIVE_CLASS,
    })
    return payload
