"""What to draw on a processed video, so it audits the canonical timeline.

The processed videos are the only place a person can SEE whether alignment
worked. That makes them worth treating as an audit surface rather than
decoration: if a camera's overlay says GW_7 while the frame plainly shows the
neighbouring wagon, the misalignment is visible in seconds instead of being
inferred from JSON.

Everything here is read from the canonical structures and nothing is
recomputed. Regions and `GW_n` come from `TrainTimeline`; boundaries come from
RIGHT_UP's canonical gap sequence; the projection onto a camera comes from
`core.master_timeline`. This module owns no detector, no threshold and no
matching rule, so it cannot disagree with the pipeline it is drawing.

The distinction the overlay exists to make is DETECTED versus PROJECTED. A
boundary drawn on a support camera is, by default, a master boundary projected
onto that camera's clock -- the local detector may never have seen it. Drawing
both the same way would let a viewer read agreement into a picture that only
shows arithmetic, which is precisely the confusion this pipeline keeps having
to unpick. So a projected boundary is labelled as one.

Split from the drawing code on purpose: this half is pure data and is what the
tests exercise. Placing a marker on the wrong frame is a bug you can assert on;
a rectangle two pixels off is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C
from core.master_timeline import (
    CameraClock, master_interval_to_local, master_time_to_local_frame,
)

#: A boundary this camera's own detector found, at the projected instant.
DETECTED = "DETECTED"
#: A canonical boundary projected from RIGHT_UP; this camera did not see it.
PROJECTED = "PROJECTED"
#: Canonical, and outside this camera's footage entirely.
OUT_OF_COVERAGE = "OUT_OF_COVERAGE"

#: How close a local detection must be to the projected instant to count as
#: the same physical gap. One third of a second absorbs clock-offset residue
#: and frame quantisation without merging two genuinely different gaps -- real
#: inter-wagon gaps are seconds apart.
DETECTION_TOLERANCE = 1.0 / 3.0

#: Half-width, in frames, of the drawn band. Narrow on purpose: a wide band
#: hides the wagon it is supposed to delimit.
MARKER_HALF_WIDTH_FRAMES = 1


@dataclass(frozen=True)
class GapMarker:
    """One canonical boundary, placed on one camera's timeline."""
    global_gap_id: int
    master_time: float
    status: str
    local_frame: Optional[int] = None
    local_time: Optional[float] = None
    detected_local_time: Optional[float] = None
    center_x: Optional[float] = None      # tracked geometry, when detected
    gw_before: Optional[str] = None
    gw_after: Optional[str] = None

    @property
    def drawable(self) -> bool:
        return self.local_frame is not None and self.status != OUT_OF_COVERAGE

    @property
    def label(self) -> str:
        arrow = f"{self.gw_before or '--'}|{self.gw_after or '--'}"
        return f"GAP {self.global_gap_id} {arrow} [{self.status}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_gap_id": self.global_gap_id,
            "master_time": round(self.master_time, 4),
            "status": self.status,
            "local_frame": self.local_frame,
            "local_time": (round(self.local_time, 4)
                           if self.local_time is not None else None),
            "detected_local_time": (round(self.detected_local_time, 4)
                                    if self.detected_local_time is not None
                                    else None),
            "center_x": self.center_x,
            "gw_before": self.gw_before, "gw_after": self.gw_after,
        }


@dataclass(frozen=True)
class FrameOverlay:
    """Everything the renderer draws on ONE frame."""
    camera_id: str
    frame_idx: int
    local_time: float
    master_time: float
    in_train: bool
    region_kind: str = ""
    region_confidence: float = 0.0
    global_id: Optional[str] = None
    gap_before: Optional[GapMarker] = None
    gap_after: Optional[GapMarker] = None
    markers_here: Tuple[GapMarker, ...] = ()

    @property
    def wagon_label(self) -> str:
        """What the wagon line reads. ENGINE and BRAKE_VAN have no GW id."""
        if not self.in_train:
            return "OUTSIDE TRAIN"
        if self.global_id:
            return f"{self.global_id}  {self.region_kind}"
        return f"{self.region_kind} (not counted)"

    def lines(self) -> List[str]:
        """The stable text block, in a fixed order so it does not jitter."""
        out = [
            f"CAMERA {self.camera_id}",
            f"t_local {self.local_time:7.2f}s   t_master {self.master_time:7.2f}s",
            self.wagon_label,
        ]
        if self.in_train and self.region_kind:
            out.append(f"class conf {self.region_confidence:.2f}")
        out.append(self._gap_line("GAP_BEFORE", self.gap_before))
        out.append(self._gap_line("GAP_AFTER", self.gap_after))
        return out

    @staticmethod
    def _gap_line(name: str, m: Optional[GapMarker]) -> str:
        if m is None:
            return f"{name}: -- (train end)"
        return (f"{name}: {m.master_time:.2f}s master  "
                f"gap {m.global_gap_id} [{m.status}]")


@dataclass
class OverlayPlan:
    """Per-camera overlay data for a whole video. Built once, read per frame."""
    camera_id: str
    clock: CameraClock
    markers: List[GapMarker] = field(default_factory=list)
    regions: List[Any] = field(default_factory=list)     # TrainRegion
    train_start: float = 0.0
    train_end: float = 0.0
    found: bool = False

    def marker_frames(self) -> Dict[int, List[GapMarker]]:
        out: Dict[int, List[GapMarker]] = {}
        for m in self.markers:
            if m.drawable:
                out.setdefault(int(m.local_frame), []).append(m)
        return out

    def summary_lines(self) -> List[str]:
        counts: Dict[str, int] = {}
        for m in self.markers:
            counts[m.status] = counts.get(m.status, 0) + 1
        return [
            f"[OVERLAY/{self.camera_id}] {len(self.markers)} canonical gap(s): "
            + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            f"[OVERLAY/{self.camera_id}] {len(self.regions)} train region(s), "
            f"train {self.train_start:.2f}-{self.train_end:.2f}s master",
        ]


# --- building the plan ------------------------------------------------------

def build_overlay_plan(
    *,
    timeline: Any,
    canonical_gap_times: Sequence[float],
    clock: CameraClock,
    detected_gap_times: Sequence[float] = (),
    detected_center_x: Optional[Dict[float, float]] = None,
    tolerance: float = DETECTION_TOLERANCE,
) -> OverlayPlan:
    """Place the canonical boundaries on one camera, and say how each was known.

    `canonical_gap_times` are RIGHT_UP's gap centres on the master clock -- the
    immutable sequence. `detected_gap_times` are THIS camera's own gap centres,
    already on the master clock, used only to decide DETECTED vs PROJECTED.
    Nothing here creates, moves or merges a gap.
    """
    plan = OverlayPlan(camera_id=clock.camera_id, clock=clock,
                       found=bool(getattr(timeline, "found", False)))
    if plan.found:
        plan.train_start = float(timeline.start_time)
        plan.train_end = float(timeline.end_time)
        plan.regions = list(timeline.regions)

    local_detected = sorted(float(t) for t in (detected_gap_times or ()))
    centers = detected_center_x or {}

    for i, t in enumerate(sorted(float(x) for x in canonical_gap_times),
                          start=1):
        frame = master_time_to_local_frame(clock, t)
        near = _closest(local_detected, t, tolerance)
        if frame is None:
            status = OUT_OF_COVERAGE
        else:
            status = DETECTED if near is not None else PROJECTED
        before, after = _wagons_around(timeline, t)
        plan.markers.append(GapMarker(
            global_gap_id=i, master_time=t, status=status,
            local_frame=frame,
            local_time=(clock.to_local_time(t) if frame is not None else None),
            detected_local_time=(clock.to_local_time(near)
                                 if near is not None else None),
            center_x=centers.get(near) if near is not None else None,
            gw_before=before, gw_after=after))
    return plan


def _closest(times: Sequence[float], t: float,
             tolerance: float) -> Optional[float]:
    best, best_d = None, tolerance
    for x in times:
        d = abs(x - t)
        if d <= best_d:
            best, best_d = x, d
    return best


def _wagons_around(timeline: Any, t: float) -> Tuple[Optional[str], Optional[str]]:
    """The counted wagons either side of a boundary, when there are any.

    ENGINE and BRAKE_VAN sit on the timeline without a `GW_n`, so a boundary
    beside one legitimately reports None on that side rather than borrowing the
    next wagon's id.
    """
    if not getattr(timeline, "found", False):
        return (None, None)
    before = after = None
    for r in timeline.regions:
        if r.end_time <= t + 1e-9:
            before = r.global_id if r.global_id else before
        if r.start_time >= t - 1e-9 and after is None:
            after = r.global_id
    return (before, after)


def overlay_at(plan: OverlayPlan, frame_idx: int) -> FrameOverlay:
    """The overlay for one frame. Deterministic, so the text does not flicker."""
    fps = plan.clock.fps or 1.0
    local_t = frame_idx / fps
    master_t = plan.clock.to_master_time(local_t)

    region = None
    if plan.found:
        for r in plan.regions:
            if r.start_time <= master_t <= r.end_time:
                region = r
                break

    before = after = None
    for m in plan.markers:
        if m.master_time <= master_t:
            before = m
        elif after is None:
            after = m

    here = tuple(m for m in plan.markers
                 if m.drawable
                 and abs(int(m.local_frame) - frame_idx)
                 <= MARKER_HALF_WIDTH_FRAMES)

    return FrameOverlay(
        camera_id=plan.camera_id, frame_idx=frame_idx, local_time=local_t,
        master_time=master_t,
        in_train=bool(region is not None),
        region_kind=(region.kind if region is not None else ""),
        region_confidence=(float(region.confidence) if region is not None
                           else 0.0),
        global_id=(region.global_id if region is not None else None),
        gap_before=before, gap_after=after, markers_here=here)


# --- drawing ----------------------------------------------------------------

_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_DETECTED_COLOR = (0, 220, 0)      # green: this camera really saw it
_PROJECTED_COLOR = (0, 200, 255)   # amber: master boundary, projected here


def draw_overlay(frame, ov: FrameOverlay, *, band_px: int = 6) -> None:
    """Draw one frame's overlay. Thin by design -- the decisions are in the plan.

    A canonical boundary is a narrow BLACK vertical band, placed at the tracked
    gap geometry when this camera detected it and at frame centre when it is
    only projected. The band is outlined in green for DETECTED and amber for
    PROJECTED, and captioned, so the two are never confused at a glance.
    """
    import cv2

    h, w = frame.shape[:2]
    for m in ov.markers_here:
        x = int(m.center_x) if m.center_x is not None else w // 2
        x = max(band_px, min(w - band_px - 1, x))
        colour = _DETECTED_COLOR if m.status == DETECTED else _PROJECTED_COLOR
        cv2.rectangle(frame, (x - band_px, 0), (x + band_px, h), _BLACK, -1)
        cv2.rectangle(frame, (x - band_px, 0), (x + band_px, h), colour, 2)
        cv2.putText(frame, m.label, (max(4, x - 140), h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    lines = ov.lines()
    pad, lh = 8, 22
    box_h = pad * 2 + lh * len(lines)
    cv2.rectangle(frame, (8, 8), (430, 8 + box_h), _BLACK, -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (16, 8 + pad + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1,
                    lineType=cv2.LINE_AA)
