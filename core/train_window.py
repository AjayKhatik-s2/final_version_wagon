"""Where the train actually starts and ends, from classification not gaps.

A gap detector answers "did something pass between two vehicles". It cannot
answer "is a train here at all", and at the two ends of a run that difference
matters. The first detection of a pass often is not an inter-wagon gap: it is
empty track before the train arrives, or the leading face of the ENGINE
crossing the frame. Taking the first gap as TRAIN_START therefore starts the
train early, on nothing; taking the last as TRAIN_END can cut a real brake van
off the back.

So the boundary here is set by CLASSIFICATION, corroborated across cameras, and
gaps are used only to segment and to be reported as rejected candidates. A
region counts as train only when a classifier called it ENGINE, WAGON or
BRAKE_VAN. UNKNOWN does not qualify at an edge -- inside the train it means "a
vehicle we could not name", which is why `get_master_wagon_window` counts it,
but at the boundary it is exactly the ambiguity that would let empty track in.

This is deliberately SEPARATE from the wagon window. `get_master_wagon_window`
finds the counted region, first WAGON to last WAGON, excluding engine and brake
van because they never receive a `GW_n`. This finds the PHYSICAL train,
including them. Both are needed and they are not the same interval.

Produces one canonical `TRAIN_START -> TRAIN_END -> duration` for the next
stage to consume. Changes nothing about gap detection, wagon counting, fusion
or the roster; it only reports an interval and the evidence behind it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C

#: A region is "train present" only if a classifier named it one of these.
#: UNKNOWN is excluded ON PURPOSE -- see the module docstring.
TRAIN_CLASSES: Tuple[str, ...] = (C.CLASS_ENGINE, C.CLASS_WAGON,
                                  C.CLASS_BRAKE_VAN)

SOURCE_MASTER = "master_classification"
SOURCE_SUPPORT = "support_classification"
SOURCE_NONE = "no_classified_evidence"


@dataclass(frozen=True)
class TrainWindowPolicy:
    """Tunables, all explicit so a boundary is never a floating-point accident.

    `max_discontinuity` bridges a short unlabelled stretch inside the train --
    one segment the classifier could not read must not split the train into two
    runs. It is deliberately smaller than the time between trains and larger
    than any single misread segment.

    `edge_tolerance` is how far apart two cameras' idea of an edge may sit
    before they count as disagreeing. Camera clocks are resolved to within a
    fraction of a second, so a couple of seconds is generous.

    `min_corroborating_cameras` is 1 by default: RIGHT_UP alone is sufficient,
    matching the master-fixed architecture. Raising it makes the boundary
    require agreement, at the cost of failing when only one camera classified.
    """
    max_discontinuity: float = 8.0
    edge_tolerance: float = 2.0
    min_corroborating_cameras: int = 1
    require_classification: bool = True


DEFAULT_POLICY = TrainWindowPolicy()


@dataclass
class LabelledSpan:
    """One classified stretch of one camera's timeline, in MASTER seconds."""
    camera_id: str
    start_time: float
    end_time: float
    label: str
    confidence: float = 0.0
    segment_id: str = ""

    @property
    def is_train(self) -> bool:
        return self.label in TRAIN_CLASSES


@dataclass
class CameraTrainEvidence:
    """What one camera says about where the train is, on the master clock."""
    camera_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    offset: float = 0.0
    offset_status: str = ""
    train_segments: int = 0
    labels: List[str] = field(default_factory=list)
    first_label: str = ""
    last_label: str = ""
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.start_time is not None and self.end_time is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "start_time": (round(self.start_time, 4)
                           if self.start_time is not None else None),
            "end_time": (round(self.end_time, 4)
                         if self.end_time is not None else None),
            "offset": round(self.offset, 4),
            "offset_status": self.offset_status,
            "train_segments": self.train_segments,
            "labels": list(self.labels),
            "first_label": self.first_label,
            "last_label": self.last_label,
            "usable": self.usable,
            "reason": self.reason,
        }


@dataclass
class RejectedBoundary:
    """A gap that looked like an edge but classification did not support."""
    position: str                    # "leading" | "trailing"
    time: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"position": self.position, "time": round(self.time, 4),
                "reason": self.reason}


@dataclass
class TrainWindow:
    """The canonical train presence interval, with its provenance."""
    found: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    start_source: str = SOURCE_NONE
    end_source: str = SOURCE_NONE
    start_camera: str = ""
    end_camera: str = ""
    start_corroborating: List[str] = field(default_factory=list)
    end_corroborating: List[str] = field(default_factory=list)
    per_camera: Dict[str, CameraTrainEvidence] = field(default_factory=dict)
    rejected_boundaries: List[RejectedBoundary] = field(default_factory=list)
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
            "train_start": round(self.start_time, 4) if self.found else None,
            "train_end": round(self.end_time, 4) if self.found else None,
            "duration": round(self.duration, 4),
            "start_source": self.start_source,
            "end_source": self.end_source,
            "start_camera": self.start_camera,
            "end_camera": self.end_camera,
            "start_corroborating": list(self.start_corroborating),
            "end_corroborating": list(self.end_corroborating),
            "per_camera": {c: e.to_dict() for c, e in self.per_camera.items()},
            "rejected_boundaries": [r.to_dict()
                                    for r in self.rejected_boundaries],
            "notes": list(self.notes),
            "reason": self.reason,
        }

    def summary_lines(self) -> List[str]:
        if not self.found:
            return [f"train window : NOT FOUND -- {self.reason}"]
        out = [
            f"train window : {self.start_time:.2f}s -> {self.end_time:.2f}s "
            f"({self.duration:.2f}s)",
            f"  start      : {self.start_source} via {self.start_camera}"
            f"  corroborated by {self.start_corroborating or ['-']}",
            f"  end        : {self.end_source} via {self.end_camera}"
            f"  corroborated by {self.end_corroborating or ['-']}",
        ]
        for cam, e in sorted(self.per_camera.items()):
            if e.usable:
                out.append(f"  {cam:<13} {e.start_time:7.2f} -> "
                           f"{e.end_time:7.2f}  segments={e.train_segments} "
                           f"({e.first_label} .. {e.last_label})")
            else:
                out.append(f"  {cam:<13} no classified train evidence "
                           f"-- {e.reason}")
        for r in self.rejected_boundaries:
            out.append(f"  rejected {r.position} boundary at {r.time:.2f}s "
                       f"-- {r.reason}")
        return out


# --- evidence normalisation -------------------------------------------------

def spans_from_master_classifications(classifications: Sequence[Any],
                                      fps: float,
                                      camera_id: str = C.MASTER_CAMERA
                                      ) -> List[LabelledSpan]:
    """`_MasterClassification` records -> spans on the master clock.

    The master IS the reference clock, so no offset is applied.
    """
    out: List[LabelledSpan] = []
    if fps <= 0:
        return out
    for c in classifications or []:
        try:
            out.append(LabelledSpan(
                camera_id=camera_id,
                start_time=float(c.start_frame) / fps,
                end_time=float(c.end_frame + 1) / fps,
                label=str(c.label), confidence=float(c.confidence or 0.0),
                segment_id=f"seg_{c.segment_index}"))
        except (AttributeError, TypeError, ValueError):
            continue
    out.sort(key=lambda s: s.start_time)
    return out


def spans_from_local_segments(segments: Sequence[Any], camera_id: str,
                              offset: float = 0.0) -> List[LabelledSpan]:
    """`LocalSegment` records -> spans PROJECTED onto the master clock.

    `t_master = t_local + offset`, the same convention the counting engine
    resolves offsets in.
    """
    out: List[LabelledSpan] = []
    for s in segments or []:
        try:
            out.append(LabelledSpan(
                camera_id=camera_id,
                start_time=float(s.start_time) + float(offset),
                end_time=float(s.end_time) + float(offset),
                label=str(getattr(s, "label", "") or C.CLASS_UNKNOWN),
                confidence=float(getattr(s, "confidence", 0.0) or 0.0),
                segment_id=str(getattr(s, "local_id", "") or "")))
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda s: s.start_time)
    return out


def _longest_train_run(spans: Sequence[LabelledSpan],
                       policy: TrainWindowPolicy
                       ) -> Optional[Tuple[float, float, List[LabelledSpan]]]:
    """The longest continuous stretch of train-labelled spans.

    Runs are joined across a stretch shorter than `max_discontinuity`, so one
    unreadable segment in the middle of a rake does not split the train in two.
    Taking the LONGEST run rather than the first is what discards an isolated
    misclassification out on empty track.
    """
    train = [s for s in spans if s.is_train]
    if not train:
        return None
    runs: List[List[LabelledSpan]] = [[train[0]]]
    for s in train[1:]:
        if s.start_time - runs[-1][-1].end_time <= policy.max_discontinuity:
            runs[-1].append(s)
        else:
            runs.append([s])
    best = max(runs, key=lambda r: r[-1].end_time - r[0].start_time)
    return (best[0].start_time, best[-1].end_time, best)


def camera_evidence(spans: Sequence[LabelledSpan], camera_id: str,
                    *, offset: float = 0.0, offset_status: str = "",
                    policy: TrainWindowPolicy = DEFAULT_POLICY
                    ) -> CameraTrainEvidence:
    """One camera's train interval, or why it has none."""
    ev = CameraTrainEvidence(camera_id=camera_id, offset=offset,
                             offset_status=offset_status)
    if not spans:
        ev.reason = "no classified segments"
        return ev
    run = _longest_train_run(spans, policy)
    if run is None:
        ev.reason = (f"no segment classified "
                     f"{'/'.join(TRAIN_CLASSES)} -- "
                     f"saw {sorted({s.label for s in spans})}")
        return ev
    start, end, members = run
    ev.start_time, ev.end_time = start, end
    ev.train_segments = len(members)
    ev.labels = [s.label for s in members]
    ev.first_label, ev.last_label = members[0].label, members[-1].label
    return ev


# --- the detector -----------------------------------------------------------

def detect_train_window(
    *,
    master_spans: Sequence[LabelledSpan],
    support_spans: Optional[Dict[str, Sequence[LabelledSpan]]] = None,
    master_gap_times: Sequence[float] = (),
    camera_offsets: Optional[Dict[str, Tuple[float, str]]] = None,
    policy: TrainWindowPolicy = DEFAULT_POLICY,
    master_camera: str = C.MASTER_CAMERA,
) -> TrainWindow:
    """One canonical TRAIN_START -> TRAIN_END, from classification + gaps.

    The master's classified train run sets the boundary when it exists, which
    keeps this consistent with RIGHT_UP being the temporal authority. Support
    cameras corroborate it, and stand in for it only when the master classified
    nothing at all.

    `master_gap_times` are the validated RIGHT_UP gap centres. They never set a
    boundary. Any that fall outside the classified train are reported as
    rejected candidates -- that is the empty-track or engine-face detection
    this stage exists to refuse.
    """
    win = TrainWindow()
    offsets = camera_offsets or {}

    win.per_camera[master_camera] = camera_evidence(
        master_spans, master_camera, offset=0.0, offset_status="REFERENCE",
        policy=policy)
    for cam, spans in (support_spans or {}).items():
        off, status = offsets.get(cam, (0.0, ""))
        win.per_camera[cam] = camera_evidence(
            spans, cam, offset=off, offset_status=status, policy=policy)

    usable = {c: e for c, e in win.per_camera.items() if e.usable}
    if not usable:
        win.reason = ("no camera classified any region as "
                      f"{'/'.join(TRAIN_CLASSES)}")
        win.notes.append("gap evidence alone cannot establish a train window")
        return win

    master_ev = win.per_camera.get(master_camera)
    if master_ev is not None and master_ev.usable:
        win.start_time, win.end_time = master_ev.start_time, master_ev.end_time
        win.start_source = win.end_source = SOURCE_MASTER
        win.start_camera = win.end_camera = master_camera
    else:
        # The master classified nothing; fall back to the widest corroborated
        # support interval rather than to gaps.
        win.start_time = min(e.start_time for e in usable.values())
        win.end_time = max(e.end_time for e in usable.values())
        win.start_source = win.end_source = SOURCE_SUPPORT
        starter = min(usable.values(), key=lambda e: e.start_time)
        ender = max(usable.values(), key=lambda e: e.end_time)
        win.start_camera, win.end_camera = starter.camera_id, ender.camera_id
        win.notes.append(
            f"{master_camera} produced no classified train region; boundary "
            f"taken from support classification")

    tol = policy.edge_tolerance
    win.start_corroborating = sorted(
        c for c, e in usable.items()
        if c != win.start_camera and abs(e.start_time - win.start_time) <= tol)
    win.end_corroborating = sorted(
        c for c, e in usable.items()
        if c != win.end_camera and abs(e.end_time - win.end_time) <= tol)

    need = max(0, policy.min_corroborating_cameras - 1)
    if need and len(win.start_corroborating) < need:
        win.notes.append(
            f"TRAIN_START corroborated by {len(win.start_corroborating)} "
            f"camera(s), policy wants {need}")
    if need and len(win.end_corroborating) < need:
        win.notes.append(
            f"TRAIN_END corroborated by {len(win.end_corroborating)} "
            f"camera(s), policy wants {need}")

    # Gaps outside the classified train are refused as boundaries, and said so.
    for t in sorted(master_gap_times):
        if t < win.start_time:
            win.rejected_boundaries.append(RejectedBoundary(
                "leading", t,
                "gap precedes the first classified train region -- empty track "
                "or the leading face of the ENGINE, not an inter-wagon gap"))
        elif t > win.end_time:
            win.rejected_boundaries.append(RejectedBoundary(
                "trailing", t,
                "gap follows the last classified train region -- the train had "
                "already left the frame"))

    win.found = win.end_time > win.start_time
    if not win.found:
        win.reason = (f"degenerate window "
                      f"{win.start_time:.2f}s -> {win.end_time:.2f}s")
    return win


def train_window_from_state(state: Any, *,
                            master_classifications: Sequence[Any] = (),
                            camera_segments: Optional[Dict[str, Sequence[Any]]] = None,
                            policy: TrainWindowPolicy = DEFAULT_POLICY
                            ) -> TrainWindow:
    """Convenience wrapper over a parsed GlobalTrainState.

    Uses the offsets the counting engine already resolved; estimates nothing.
    """
    try:
        resolved = state.camera_time_offsets()
    except Exception:
        resolved = {}
    meta = getattr(state, "camera_offsets", None) or {}
    offsets = {cam: (float(resolved.get(cam, 0.0) or 0.0),
                     str((meta.get(cam) or {}).get("status") or ""))
               for cam in C.ALL_CAMERAS}

    master = getattr(state, "master_camera", C.MASTER_CAMERA)
    master_spans = spans_from_master_classifications(
        master_classifications, float(getattr(state, "master_fps", 0.0) or 0.0),
        camera_id=master)
    support = {
        cam: spans_from_local_segments(segs, cam, offsets.get(cam, (0.0, ""))[0])
        for cam, segs in (camera_segments or {}).items() if cam != master
    }
    gap_times = [float((g.get("master_observation") or {}).get("center_time", 0.0))
                 for g in (getattr(state, "global_gaps", None) or [])
                 if isinstance(g, dict)]
    return detect_train_window(
        master_spans=master_spans, support_spans=support,
        master_gap_times=gap_times, camera_offsets=offsets,
        policy=policy, master_camera=master)


# --- persisted artifact -----------------------------------------------------

ARTIFACT_NAME = "train_window.json"
ARTIFACT_SCHEMA = "wagon_eye.train_window.v1"


def write_artifact(window: TrainWindow, output_dir: str) -> str:
    """Persist the canonical train window next to the global state.

    This is the hand-off between the two stages: the train-window detector
    writes it, and the master-gap stage reads it to decide which of RIGHT_UP's
    validated gaps lie inside the physical train.
    """
    import json
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, ARTIFACT_NAME)
    payload = {"schema": ARTIFACT_SCHEMA}
    payload.update(window.to_dict())
    # Spelled-out keys, so a consumer never has to guess the clock.
    payload["train_start_global_time"] = payload.get("train_start")
    payload["train_end_global_time"] = payload.get("train_end")
    payload["duration_seconds"] = payload.get("duration")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def read_artifact(output_dir: str) -> Optional[Dict[str, Any]]:
    """Read back the persisted window, or None when the stage did not run."""
    import json
    path = os.path.join(output_dir, ARTIFACT_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    return doc if doc.get("found") else None


# --- the hand-off to the master-gap stage -----------------------------------

@dataclass
class GapFilterResult:
    """What restricting the master's gaps to the physical train removed."""
    kept: List[Any] = field(default_factory=list)
    dropped_leading: List[Any] = field(default_factory=list)
    dropped_trailing: List[Any] = field(default_factory=list)
    applied: bool = False
    reason: str = ""

    @property
    def dropped(self) -> int:
        return len(self.dropped_leading) + len(self.dropped_trailing)

    def summary(self) -> str:
        if not self.applied:
            return f"train-window gap filter NOT applied -- {self.reason}"
        return (f"train-window gap filter: kept {len(self.kept)}, dropped "
                f"{len(self.dropped_leading)} leading + "
                f"{len(self.dropped_trailing)} trailing")


def filter_gaps_to_window(gaps: Sequence[Any], window: Optional[TrainWindow],
                          *, fps: float = 0.0,
                          margin: float = 0.0) -> GapFilterResult:
    """Keep only the master gaps that fall inside the physical train.

    This is the ONLY thing the train window does to the counting pipeline, and
    it is deliberately subtractive: it can remove a gap that lies outside the
    classified train, and it can never add, move, split or renumber one. The
    master-fixed invariant is untouched -- `build_global_gap_sequence` still
    mints the global sequence from whatever RIGHT_UP gaps it is handed, and
    still consults no support camera.

    What it removes is the failure this stage exists for: a leading detection
    on empty track or across the ENGINE's front face, and a trailing detection
    after the train has left, both of which would otherwise become inter-wagon
    boundaries and add a phantom wagon at one end of the rake.

    A window that was not found leaves the gaps exactly as they were. Refusing
    to filter is always safer than filtering on a boundary nobody confirmed.
    """
    res = GapFilterResult()
    if window is None or not window.found:
        res.kept = list(gaps)
        res.reason = ("no validated train window; master gaps used unchanged"
                      if window is None else window.reason)
        return res

    lo = window.start_time - margin
    hi = window.end_time + margin
    for g in gaps:
        t = _gap_time(g, fps)
        if t is None:
            res.kept.append(g)          # cannot place it -> never discard it
            continue
        if t < lo:
            res.dropped_leading.append(g)
        elif t > hi:
            res.dropped_trailing.append(g)
        else:
            res.kept.append(g)
    res.applied = True
    return res


def _gap_time(gap: Any, fps: float) -> Optional[float]:
    """A gap's centre on the master clock, however the object spells it."""
    for attr in ("center_time", "global_time"):
        v = getattr(gap, attr, None)
        if v is not None:
            return float(v)
    cf = getattr(gap, "center_frame", None)
    if cf is not None and fps > 0:
        return float(cf) / float(fps)
    if isinstance(gap, dict):
        for key in ("center_time", "global_time"):
            if gap.get(key) is not None:
                return float(gap[key])
    return None


# ===========================================================================
# The complete-train timeline -- the master temporal coordinate system
# ===========================================================================
#
# Order matters, and it is the reverse of how the pipeline grew:
#
#   1. the COMPLETE PHYSICAL TRAIN   TRAIN_START -> TRAIN_END, from
#      classification + continuity. Includes ENGINE and BRAKE_VAN, because
#      they are part of the train even though they are never counted.
#   2. canonical GAPS                RIGHT_UP's validated gaps, restricted to
#      that interval. RIGHT_UP alone; no support camera may mint one.
#   3. counted ROSTER                GW_1..GW_N, derived from those gaps and
#      numbered from the FIRST WAGON. Engine and brake van get no id.
#   4. projection                    every camera reads the same coordinates.
#
# Deriving the train from the wagons has the boundary problem backwards: the
# first counted wagon is not the front of the train, so anchoring on it starts
# the coordinate system behind the locomotive and puts the ENGINE at a negative
# offset from a timeline that claims to describe the whole train.

COUNTED_CLASS = C.CLASS_WAGON


@dataclass
class TrainRegion:
    """One classified stretch of the physical train, counted or not."""
    index: int                       # position in the complete train, 0-based
    kind: str                        # ENGINE | WAGON | BRAKE_VAN | UNKNOWN
    start_time: float
    end_time: float
    confidence: float = 0.0
    source_camera: str = ""
    global_id: Optional[str] = None  # GW_n, only ever on a counted WAGON

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def counted(self) -> bool:
        return self.global_id is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "kind": self.kind,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "duration": round(self.duration, 4),
            "confidence": round(self.confidence, 4),
            "source_camera": self.source_camera,
            "global_id": self.global_id, "counted": self.counted,
        }


@dataclass
class TrainTimeline:
    """The complete physical train, as one ordered coordinate system."""
    window: TrainWindow
    regions: List[TrainRegion] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.window.found

    @property
    def start_time(self) -> float:
        return self.window.start_time

    @property
    def end_time(self) -> float:
        return self.window.end_time

    @property
    def duration(self) -> float:
        return self.window.duration

    @property
    def counted_regions(self) -> List[TrainRegion]:
        return [r for r in self.regions if r.counted]

    @property
    def non_counted_regions(self) -> List[TrainRegion]:
        """ENGINE / BRAKE_VAN / UNKNOWN -- on the timeline, off the roster."""
        return [r for r in self.regions if not r.counted]

    @property
    def counted_wagon_count(self) -> int:
        return len(self.counted_regions)

    def region_at(self, t: float) -> Optional[TrainRegion]:
        for r in self.regions:
            if r.start_time <= t <= r.end_time:
                return r
        return None

    def global_id_at(self, t: float) -> Optional[str]:
        r = self.region_at(t)
        return r.global_id if r is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_start_global_time": (round(self.start_time, 4)
                                        if self.found else None),
            "train_end_global_time": (round(self.end_time, 4)
                                      if self.found else None),
            "duration_seconds": round(self.duration, 4),
            "region_count": len(self.regions),
            "counted_wagon_count": self.counted_wagon_count,
            "regions": [r.to_dict() for r in self.regions],
            "window": self.window.to_dict(),
        }

    def summary_lines(self) -> List[str]:
        if not self.found:
            return self.window.summary_lines()
        out = [f"complete train : {self.start_time:.2f}s -> "
               f"{self.end_time:.2f}s ({self.duration:.2f}s), "
               f"{len(self.regions)} region(s), "
               f"{self.counted_wagon_count} counted wagon(s)"]
        for r in self.regions:
            tag = r.global_id or "--"
            out.append(f"  [{r.index:>2}] {r.kind:<10} {r.start_time:7.2f} -> "
                       f"{r.end_time:7.2f}  {tag}")
        return out


def build_train_timeline(window: TrainWindow,
                         master_spans: Sequence[LabelledSpan],
                         *, number_from: int = 1) -> TrainTimeline:
    """The complete train as ordered regions, with GW ids on the WAGONs only.

    Regions are the master's classified spans clipped to the train window, so
    the coordinate system covers the whole physical train -- ENGINE and BRAKE
    VAN included. Only `WAGON` regions receive a `GW_n`, numbered from the
    FIRST WAGON, which keeps the counted roster identical to what
    `train_structure.get_master_wagon_window()` produces from the same
    classification. This does not replace that function; it is the timeline
    view that surrounds it.
    """
    tl = TrainTimeline(window=window)
    if not window.found:
        return tl

    lo, hi = window.start_time, window.end_time
    idx = 0
    for span in sorted(master_spans, key=lambda s: s.start_time):
        s, e = max(span.start_time, lo), min(span.end_time, hi)
        if e <= s:
            continue                    # entirely outside the physical train
        tl.regions.append(TrainRegion(
            index=idx, kind=span.label, start_time=s, end_time=e,
            confidence=span.confidence, source_camera=span.camera_id))
        idx += 1

    n = number_from
    for r in tl.regions:
        if r.kind == COUNTED_CLASS:
            r.global_id = f"GW_{n}"
            n += 1
    return tl
