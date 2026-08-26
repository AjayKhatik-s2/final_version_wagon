"""The canonical master timeline, and the one way to project it onto a camera.

RIGHT_UP owns the train's temporal structure. `wagon_count/global_fusion.py`
mints the global gap sequence from RIGHT_UP alone and treats it as immutable;
every other camera is an OBSERVER of that timeline. This module is the single
place that answers the observer's question:

    "which of my local frames correspond to this master time window?"

Three implementations of that arithmetic existed, and they disagreed. Asked for
a wagon at master t=100-104s against a camera holding 1350 frames of 90s
footage -- i.e. a camera that stopped recording ten seconds before the wagon
existed:

    global_fusion.project_global_time_to_local  -> None        (correct)
    materializer._wagon_local_range             -> (0, -1)     (correct)
    reporting._evidence_lookup.wagon_local_frames -> (1349, 1349)

The third clamps. It returns the camera's FINAL frame for every wagon that
occurs after the footage ends, so a report shows one still image as evidence
for a dozen different wagons, each labelled with a different wagon id. Nothing
about the picture reveals the error. global_fusion's own docstring names the
hazard: "NEVER clamps -- clamping to the last frame is what fabricates
evidence."

The distinction that matters, and which the correct implementations already
made, is between PARTIAL overlap and NO overlap:

    partial  a wagon half-inside the footage yields the frames that exist,
             clamped to the boundary. Those frames really do show that wagon.
    none     the wagon is outside the footage entirely. There is no such frame,
             so the answer is "unavailable", never a substitute.

This module preserves the first and enforces the second, and reports WHICH of
the two happened so a caller can say "camera ended early" instead of silently
showing nothing.

Nothing here detects, counts, classifies or renumbers anything. It converts
time to frames and back, under one boundary policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C

# --- availability -----------------------------------------------------------
#
# Why an enumeration rather than Optional[range]: "no frames" has several
# causes and they are not interchangeable in a report. A camera that ended
# early is a coverage fact; a camera whose offset never resolved is an
# alignment failure; a camera with no metadata is a pipeline failure. Collapsing
# them into None loses the reason at exactly the point someone needs it.
AVAILABLE = "AVAILABLE"
PARTIAL = "PARTIAL"                  # overlaps the footage, but not fully
BEFORE_START = "BEFORE_START"        # camera started late; wagon precedes it
AFTER_END = "AFTER_END"              # camera ended early; wagon follows it
NO_METADATA = "NO_METADATA"          # no fps / frame count for this camera
UNRESOLVED_OFFSET = "UNRESOLVED_OFFSET"

#: Statuses that carry usable frames.
USABLE = (AVAILABLE, PARTIAL)

#: Offset statuses the counting engine considers trustworthy.
RESOLVED_OFFSET_STATES = ("REFERENCE", "RESOLVED")


# --- boundary policy --------------------------------------------------------

@dataclass(frozen=True)
class BoundaryPolicy:
    """How a detection sitting on a canonical gap boundary is assigned.

    A gap has real duration, but the canonical boundary between GW_n and
    GW_n+1 is a single instant, so a detection can land exactly on it -- and
    with floating-point time it can land a hair either side of it for reasons
    that have nothing to do with the train. Left to chance, two modules
    comparing the same timestamp against the same boundary can disagree.

    `epsilon` is the half-width, in seconds, of the band treated as "on the
    boundary". Default is one third of a frame at 15 fps: wide enough to absorb
    rounding through a float multiply, far narrower than any real gap.

    `on_boundary` decides that band, and defaults to NEXT. A boundary IS the
    start of the following wagon -- `build_global_wagons` builds segments as
    [prev, b-1] and starts the next at b -- so assigning the instant forward
    matches how the roster itself was cut.
    """
    epsilon: float = 1.0 / 45.0
    on_boundary: str = "next"        # "next" | "previous"

    def __post_init__(self):
        if self.on_boundary not in ("next", "previous"):
            raise ValueError(
                f"on_boundary must be 'next' or 'previous', got "
                f"{self.on_boundary!r}")
        if self.epsilon < 0:
            raise ValueError("epsilon must not be negative")


DEFAULT_BOUNDARY_POLICY = BoundaryPolicy()


# --- camera clock -----------------------------------------------------------

@dataclass(frozen=True)
class CameraClock:
    """One camera's timebase, relative to the master.

    `offset` is the camera's clock delta in the sense the counting engine
    resolves it: `t_master = t_local + offset`, so `t_local = t_master - offset`.
    An UNRESOLVED camera keeps offset 0.0, which is the historical shared-t0
    projection, but its status is carried so a caller can refuse to trust it.
    """
    camera_id: str
    fps: float = 0.0
    total_frames: int = 0
    offset: float = 0.0
    offset_status: str = "REFERENCE"

    @property
    def has_metadata(self) -> bool:
        return self.fps > 0 and self.total_frames > 0

    @property
    def offset_resolved(self) -> bool:
        return self.offset_status in RESOLVED_OFFSET_STATES

    @property
    def duration(self) -> float:
        """Footage length in LOCAL seconds."""
        return (self.total_frames / self.fps) if self.has_metadata else 0.0

    def to_local_time(self, t_master: float) -> float:
        return float(t_master) - float(self.offset)

    def to_master_time(self, t_local: float) -> float:
        return float(t_local) + float(self.offset)

    @classmethod
    def from_state(cls, state: Any, camera_id: str,
                   fps: Optional[float] = None,
                   total_frames: Optional[int] = None) -> "CameraClock":
        """Build from a parsed GlobalTrainState's offset metadata.

        Reuses the offsets the counting engine already resolved rather than
        re-estimating anything.
        """
        meta = (getattr(state, "camera_offsets", None) or {}).get(camera_id) or {}
        status = str(meta.get("status") or "UNRESOLVED")
        try:
            resolved = state.camera_time_offsets()
        except Exception:
            resolved = {}
        return cls(camera_id=camera_id,
                   fps=float(fps or 0.0),
                   total_frames=int(total_frames or 0),
                   offset=float(resolved.get(camera_id, 0.0) or 0.0),
                   offset_status=status)


# --- projection result ------------------------------------------------------

@dataclass(frozen=True)
class LocalWindow:
    """A master interval expressed in one camera's frames, or why it cannot be."""
    camera_id: str
    status: str
    start_frame: int = 0
    end_frame: int = -1              # inclusive; end < start means empty
    start_time: float = 0.0          # LOCAL seconds
    end_time: float = 0.0
    master_start: float = 0.0
    master_end: float = 0.0
    offset: float = 0.0
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.status in USABLE and self.end_frame >= self.start_frame

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)

    def as_range(self) -> Tuple[int, int]:
        """`(start, end)` inclusive, or the empty `(0, -1)`.

        Matches what `materializer._wagon_local_range` has always returned, so
        an existing caller can adopt this without changing its own logic.
        """
        return (self.start_frame, self.end_frame) if self.available else (0, -1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id, "status": self.status,
            "start_frame": self.start_frame, "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "master_start": round(self.master_start, 4),
            "master_end": round(self.master_end, 4),
            "offset": round(self.offset, 4),
            "available": self.available, "reason": self.reason,
        }


def _unavailable(clock: CameraClock, status: str, reason: str,
                 t_start: float, t_end: float) -> LocalWindow:
    return LocalWindow(camera_id=clock.camera_id, status=status,
                       start_frame=0, end_frame=-1,
                       master_start=float(t_start), master_end=float(t_end),
                       offset=clock.offset, reason=reason)


# --- the API ----------------------------------------------------------------

def master_time_to_local_frame(clock: CameraClock, t_master: float,
                               *, allow_unresolved: bool = True
                               ) -> Optional[int]:
    """A master instant as this camera's frame index, or None if outside.

    Never clamps. Equivalent to `global_fusion.project_global_time_to_local`,
    which is the behaviour the other two implementations should always have had.
    """
    if not clock.has_metadata:
        return None
    if not allow_unresolved and not clock.offset_resolved:
        return None
    frame = int(round(clock.to_local_time(t_master) * clock.fps))
    if frame < 0 or frame > clock.total_frames - 1:
        return None
    return frame


def master_interval_to_local(clock: CameraClock, t_start: float, t_end: float,
                             *, allow_unresolved: bool = True) -> LocalWindow:
    """Project a master interval onto one camera.

    PARTIAL overlap is clamped to the frames that exist -- they genuinely show
    this wagon. NO overlap is refused with the reason, so the caller can report
    "this camera had stopped recording" instead of showing its last frame.
    """
    if not clock.has_metadata:
        return _unavailable(clock, NO_METADATA,
                            f"no fps/frame-count for {clock.camera_id}",
                            t_start, t_end)
    if not allow_unresolved and not clock.offset_resolved:
        return _unavailable(clock, UNRESOLVED_OFFSET,
                            f"{clock.camera_id} clock offset is "
                            f"{clock.offset_status}", t_start, t_end)

    local_start = clock.to_local_time(t_start)
    local_end = clock.to_local_time(t_end)
    sf = int(round(local_start * clock.fps))
    ef = int(round(local_end * clock.fps)) - 1      # inclusive
    last = clock.total_frames - 1

    if ef < 0:
        return _unavailable(
            clock, BEFORE_START,
            f"master {t_start:.2f}-{t_end:.2f}s precedes {clock.camera_id}'s "
            f"footage (starts at master {clock.to_master_time(0.0):.2f}s)",
            t_start, t_end)
    if sf > last:
        return _unavailable(
            clock, AFTER_END,
            f"master {t_start:.2f}-{t_end:.2f}s follows the end of "
            f"{clock.camera_id}'s footage (ends at master "
            f"{clock.to_master_time(clock.duration):.2f}s)",
            t_start, t_end)

    partial = sf < 0 or ef > last
    sf = max(0, min(last, sf))
    ef = max(0, min(last, ef))
    if ef < sf:
        ef = sf
    return LocalWindow(
        camera_id=clock.camera_id,
        status=PARTIAL if partial else AVAILABLE,
        start_frame=sf, end_frame=ef,
        start_time=sf / clock.fps, end_time=(ef + 1) / clock.fps,
        master_start=float(t_start), master_end=float(t_end),
        offset=clock.offset,
        reason=("clipped to this camera's coverage" if partial else ""))


def camera_covers(clock: CameraClock, t_master: float) -> bool:
    """Is this master instant inside the camera's real footage?"""
    return master_time_to_local_frame(clock, t_master) is not None


# --- boundary-aware assignment ---------------------------------------------

def assign_master_time(t_master: float, wagons: Sequence[Any], *,
                       policy: BoundaryPolicy = DEFAULT_BOUNDARY_POLICY
                       ) -> Optional[str]:
    """Which canonical wagon owns a detection at this master instant.

    The wagons are the canonical roster and are not modified, reordered or
    extended. A time before the first wagon or after the last returns None --
    that is the engine / brake-van region, which by design has no GW id.

    Boundary handling follows `policy`: inside the epsilon band around a
    shared boundary the detection goes to the next wagon by default, because
    a boundary is the first instant of the following wagon.
    """
    if not wagons:
        return None
    eps = policy.epsilon
    ordered = sorted(wagons, key=lambda w: float(w.start_time))

    # Boundaries are resolved BEFORE containment, and that order is load-
    # bearing. A boundary at t=4.0 is simultaneously the end of GW_1 and the
    # start of GW_2, so an instant a hair below it -- 4.0 minus a rounding
    # error -- is still inside GW_1 by plain comparison. Checking containment
    # first would hand it to GW_1 and quietly make the epsilon band one-sided,
    # which is exactly the floating-point accident the policy exists to remove.
    for i, w in enumerate(ordered):
        if abs(t_master - float(w.start_time)) <= eps:
            if policy.on_boundary == "next":
                return w.global_id
            prev = ordered[i - 1] if i > 0 else None
            return prev.global_id if prev is not None else w.global_id

    # The last wagon's trailing edge has no following wagon, so it belongs
    # there whatever the policy says.
    last = ordered[-1]
    if abs(t_master - float(last.end_time)) <= eps:
        return last.global_id

    for w in ordered:
        if float(w.start_time) <= t_master <= float(w.end_time):
            return w.global_id
    return None


def wagon_master_window(wagon: Any) -> Tuple[float, float]:
    """`(start_time, end_time)` on the master clock. The canonical window."""
    return (float(wagon.start_time), float(wagon.end_time))


def project_roster(clocks: Dict[str, CameraClock], wagons: Sequence[Any],
                   *, allow_unresolved: bool = True
                   ) -> Dict[str, Dict[str, LocalWindow]]:
    """`{global_id: {camera_id: LocalWindow}}` for the whole roster.

    Every canonical wagon appears for every camera, including the cameras that
    cannot see it -- those carry an unavailable status and a reason. A wagon is
    never dropped because a camera missed it.
    """
    out: Dict[str, Dict[str, LocalWindow]] = {}
    for w in wagons:
        s, e = wagon_master_window(w)
        out[w.global_id] = {
            cam: master_interval_to_local(clock, s, e,
                                          allow_unresolved=allow_unresolved)
            for cam, clock in clocks.items()
        }
    return out
