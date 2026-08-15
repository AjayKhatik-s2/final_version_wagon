"""
gap_validation.py  --  turn raw gap tracks into VALIDATED wagon boundaries
=========================================================================

A raw YOLO gap detection is a CANDIDATE, not a wagon boundary.

This module sits between the existing tracker and the fusion stage:

    GapTracker.process_video()        (UNCHANGED -- detection + tracking)
            |
            v   List[GapEvent]  with center_x_trajectory + hit_frames + bbox_history
            |
    validate_gap_events()            <-- THIS MODULE
            |
            +--> accepted : List[GapEvent]      -> wagon-boundary candidates
            +--> rejected : List[GapRejection]  -> diagnostics with reasons
            |
            v
    wagon window / fusion / global ids

WHY A SEPARATE LAYER
--------------------
The tracker is deliberately untouched: no detection threshold, Kalman parameter,
association gate, `min_hits` or `max_miss` value changes. Everything this module
needs is already recorded on each emitted `GapEvent`:

    center_x_trajectory   Kalman-smoothed bbox centre per hit
    hit_frames            the frame index of each hit (parallel array)
    bbox_history          the raw bbox per hit
    start_frame/end_frame track extent
    confidence            mean detection confidence over the track
    hit_count             number of frames the gap was actually detected

So validation is a pure, deterministic function of data the pipeline already
produces. No new model, no optical flow, no extra video pass -- which matters on
the CPU-only EC2 target.

THE PHYSICAL PRINCIPLE
----------------------
The train moves, so a real inter-wagon gap sweeps across the image. A detection
that keeps firing at the same pixel column is far more likely to be track
furniture, a shadow, a pole or a lighting artefact than a gap between two
wagons.

But "moves => valid" is NOT sufficient on its own: a moving false positive is
possible, and perspective makes apparent speed vary a lot between cameras and
across the frame. So several independent signals are combined, each with its own
recorded rejection reason:

    1. temporal persistence    enough hits, over enough frames
    2. detection continuity    no excessively long blind stretch inside the track
    3. motion                  enough absolute displacement
    4. speed plausibility      apparent speed inside a plausible band
    5. trajectory consistency  mostly one direction, not jittering back and forth
    6. confidence              mean and floor
    7. train-motion context    speed comparable to the other gaps in this camera
    8. duplicate suppression   one physical gap yields at most one GapEvent

Nothing is silently dropped: every rejection is returned with the reason and the
measured features, so `RAW -> TRACKED -> VALID` is fully auditable.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from global_train_state import GapEvent

# =============================================================================
# Rejection reasons
# =============================================================================

REJECTED_TOO_SHORT = "REJECTED_TOO_SHORT"
REJECTED_LOW_CONFIDENCE = "REJECTED_LOW_CONFIDENCE"
REJECTED_STATIC = "REJECTED_STATIC"
REJECTED_LOW_MOTION = "REJECTED_LOW_MOTION"
REJECTED_IMPLAUSIBLE_SPEED = "REJECTED_IMPLAUSIBLE_SPEED"
REJECTED_INCONSISTENT_TRAJECTORY = "REJECTED_INCONSISTENT_TRAJECTORY"
REJECTED_DETECTION_GAP = "REJECTED_DETECTION_GAP"
REJECTED_DUPLICATE = "REJECTED_DUPLICATE"
REJECTED_TRAIN_MOTION_MISMATCH = "REJECTED_TRAIN_MOTION_MISMATCH"
REJECTED_WRONG_DIRECTION = "REJECTED_WRONG_DIRECTION"
REJECTED_NO_TRAJECTORY = "REJECTED_NO_TRAJECTORY"
REJECTED_MIN_SEPARATION = "REJECTED_MIN_SEPARATION"

# Assigned later in the pipeline (wagon window), listed here so the vocabulary
# lives in one place.
REJECTED_OUTSIDE_WAGON_WINDOW = "REJECTED_OUTSIDE_WAGON_WINDOW"
REJECTED_NON_WAGON_REGION = "REJECTED_NON_WAGON_REGION"

ALL_REJECTION_REASONS = (
    REJECTED_TOO_SHORT, REJECTED_LOW_CONFIDENCE, REJECTED_STATIC,
    REJECTED_LOW_MOTION, REJECTED_IMPLAUSIBLE_SPEED,
    REJECTED_INCONSISTENT_TRAJECTORY, REJECTED_DETECTION_GAP,
    REJECTED_DUPLICATE, REJECTED_TRAIN_MOTION_MISMATCH,
    REJECTED_WRONG_DIRECTION, REJECTED_NO_TRAJECTORY,
    REJECTED_OUTSIDE_WAGON_WINDOW, REJECTED_NON_WAGON_REGION,
    REJECTED_MIN_SEPARATION,
)

# =============================================================================
# HARD vs SOFT rejection reasons
#
# Inside a CONFIRMED WAGON_ACTIVE region the train is physically in its wagon
# run, so a correctly tracked, correctly directed, temporally valid candidate is
# far more likely to be a real wagon boundary than the same candidate would be
# before the train arrives or after it leaves. That context is used to relax the
# SOFT gates only -- the HARD gates are exactly the protections that must never
# depend on context.
# =============================================================================

HARD_REJECTION_REASONS = frozenset({
    REJECTED_NO_TRAJECTORY,          # not a usable track at all
    REJECTED_TOO_SHORT,              # insufficient tracker confirmation
    REJECTED_DETECTION_GAP,          # corrupted / mostly-blind track
    REJECTED_STATIC,                 # isolated pinned artefact
    REJECTED_WRONG_DIRECTION,        # travels against the train
    REJECTED_DUPLICATE,              # same physical gap already accepted
    REJECTED_MIN_SEPARATION,         # resolved as a duplicate/fragment
})
"""Never relaxed, in any train state. These are the false-positive defences."""

SOFT_REJECTION_REASONS = frozenset({
    REJECTED_TRAIN_MOTION_MISMATCH,  # speed differs from the local reference
    REJECTED_IMPLAUSIBLE_SPEED,      # speed outside the plausible band
    REJECTED_LOW_MOTION,             # moved, but less than the floor
    REJECTED_LOW_CONFIDENCE,         # weaker detector confidence
    REJECTED_INCONSISTENT_TRAJECTORY,  # noisier trajectory than expected
})
"""Relaxable inside a confirmed WAGON_ACTIVE region, subject to the safety
floors in `GapValidationConfig` -- a soft reason is not a free pass."""


# =============================================================================
# Configuration -- every threshold named, documented and CLI-reachable
# =============================================================================

@dataclass
class ResolvedThresholds:
    """Config thresholds resolved into this camera's own pixels and frames.

    Produced by ``GapValidationConfig.resolve(frame_width, fps)``. Every value
    here is camera-specific and computed at runtime -- nothing is baked in from
    any particular train or camera geometry.
    """
    frame_width: int
    fps: float
    min_track_frames: int
    max_detection_gap_frames: int
    min_motion_px: float
    static_max_motion_px: float
    min_motion_px_per_sec: float
    max_motion_px_per_sec: float
    duplicate_max_center_px: float
    min_separation_frames: int

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


@dataclass
class GapValidationConfig:
    """Thresholds for turning gap tracks into wagon-boundary candidates.

    GENERALIZATION: thresholds are stored in CAMERA-INDEPENDENT units -- fractions
    of frame width for distances, seconds for durations, and dimensionless ratios
    -- and are resolved to this camera's pixels and frames at runtime by
    ``resolve(frame_width, fps)``.

    This matters because the same pipeline processes many trains on differing
    hardware. An absolute "4 px" static threshold calibrated at 848x480 becomes
    2.3x more permissive at 1920x1080 (the same physical jitter spans more
    pixels), so a stationary artefact would stop being rejected. An absolute
    "4 frame" minimum becomes half the wall-clock time at 30 fps. Relative units
    remove both failure modes.

    The numeric defaults below were MEASURED on the local development train and
    are initial defaults, not production invariants. They are expressed relative
    to that train's 848x480 @ 15 fps geometry so they carry over to other
    geometries; they still need confirming against additional trains before being
    treated as settled.

    Defaults were chosen from MEASURED behaviour of the real gap tracks on this
    project's videos (848x480 @ 15 fps). The measurement ran the EXISTING tracker
    over the first 1500 frames of three cameras:

      camera        tracks  |displacement| px    speed px/s      monotonic  conf   coverage
      RIGHT_UP        16     401 / 468 / 547     317 / 470 / 554   1.00      0.81+  1.00
      RIGHT_UP_TOP    18     111 / 340 / 614      74 / 387 / 546   0.61+     0.72+  0.58+
      LEFT_UP_TOP      8       0 / 248 / 354       0 / 237 / 356   0.52+     0.51+  0.43+
                             (min / median / max)

    The three zero-displacement LEFT_UP_TOP tracks are real false positives:
    30 hits each, coverage 1.00, confidence **0.93**, lasting 2.0 s, centre
    pinned at x=436, x=487 and x=283 respectively. Confidence, hit-count and
    coverage filters ALL pass them. Only motion rejects them -- which is why this
    layer exists.

    Every threshold below therefore has a measured margin against real gaps. The
    intent is to remove physically implausible artefacts, NOT to reach any
    particular wagon count.
    """

    enabled: bool = True
    """Master switch. False = emit every tracked gap (previous behaviour)."""

    # ---- 1. temporal persistence ----------------------------------------
    min_track_seconds: float = 0.27
    """Minimum track extent in SECONDS (fps-independent). The tracker already
    requires min_hits=3 to confirm a track, so this mostly guards against a
    track that is confirmed but collapses into almost no time span.
    Measured default: 4 frames at 15 fps."""

    min_hits: int = 3
    """Minimum number of frames the gap was actually detected. Matches the
    tracker's own confirmation rule, restated here so the validation layer is
    self-contained and auditable."""

    # ---- 2. detection continuity ----------------------------------------
    max_detection_gap_seconds: float = 1.33
    """Longest MISSED stretch tolerated inside one track, in SECONDS. A track may
    legitimately look like HIT HIT HIT MISS HIT HIT; a mostly-blind track is
    treated cautiously. Measured default: 20 frames at 15 fps."""

    min_coverage: float = 0.20
    """hits / track_extent. Guards the HIT MISS MISS MISS MISS HIT shape."""

    # ---- 3. motion ------------------------------------------------------
    min_motion_frac: float = 0.0142
    """Minimum centre displacement over the track, as a FRACTION OF FRAME WIDTH.
    Measured: the smallest displacement of any real gap was 110.7 px of 848
    (13.1% of width), so this ~1.4% floor leaves roughly a 9x margin while still
    excluding near-stationary detections."""

    static_max_motion_frac: float = 0.0047
    """At or below this displacement (FRACTION OF FRAME WIDTH) the track is
    reported as REJECTED_STATIC rather than REJECTED_LOW_MOTION, so pinned
    artefacts (rails, sleepers, poles, markings, shadows) are distinguishable in
    the diagnostics. Measured: the three confirmed false positives moved <=0.2 px
    of 848 while the smallest real gap moved 110.7 px, so anything in between
    separates them; ~0.5% of width absorbs Kalman jitter without approaching a
    real gap's motion. Being width-relative is essential -- a fixed 4 px would
    stop rejecting static objects at higher resolutions."""

    # ---- 4. speed plausibility ------------------------------------------
    min_motion_frac_per_sec: float = 0.0094
    max_motion_frac_per_sec: float = 2.36
    """Plausible band for apparent speed, in FRACTIONS OF FRAME WIDTH per second.
    Wide on purpose: perspective makes the same physical gap move at very
    different rates on side versus top cameras and across the frame, and trains
    accelerate. The band excludes the physically absurd, not a specific expected
    speed. Measured defaults: 8 and 2000 px/s at 848 px width."""

    # ---- 5. trajectory consistency --------------------------------------
    min_monotonic_fraction: float = 0.60
    """Fraction of consecutive inter-hit steps that must share the dominant
    direction. A real gap carried by the train moves consistently one way;
    a detection that jitters left-right-left is not tracking a passing object.
    Not 1.0, because Kalman smoothing plus detection noise legitimately produces
    the occasional backward step."""

    min_steps_for_trajectory: int = 3
    """Below this many inter-hit steps the direction statistic is meaningless,
    so the trajectory test is skipped rather than guessed."""

    # ---- 6. confidence ---------------------------------------------------
    min_mean_confidence: float = 0.45
    """Mean detection confidence over the track. Note the detector's own
    threshold (0.4, UNCHANGED) already applies per frame; this asks the track as
    a whole to be a little better than the per-frame floor."""

    # ---- 7. train-motion context ----------------------------------------
    motion_reference_window: int = 5
    """Neighbouring tracks each side used for the ROLLING LOCAL motion reference.

    The reference speed a candidate is compared against is the median of its
    temporal neighbours, not the whole video's median. Trains accelerate and
    decelerate: on the development train the validated speed falls from 560 to
    312 px/s across one pass (a 1.80x range), and a local reference tracks that
    far better -- worst deviation 1.24x local versus 1.61x global.

    MEASURED NEUTRALITY: switching from the global to this local reference changed
    ZERO accept/reject decisions on the full development video (0 candidates were
    rejected on any speed reason to begin with, and simulating the local rule over
    all 52 candidates produced 0 differing decisions). It is adopted because it is
    the physically correct reference under changing speed, not to recover any gap
    here -- there were none to recover.

    0 disables the local reference and falls back to the global median."""

    stopped_speed_fraction: float = 0.10
    """A track is 'stalled' when its speed falls below this fraction of the
    camera's median gap speed. Dimensionless, so it transfers across geometries.
    Measured: the one isolated static artefact on the development video sat at
    12 px/s against a 501 px/s reference = 0.024, while the slowest genuine gap
    sat at 0.622 -- so 0.10 separates them with a wide margin on both sides."""

    stop_corroboration_min_tracks: int = 2
    """Near-zero-motion tracks needed, overlapping in time, before a stall is
    treated as the TRAIN having stopped rather than as static artefacts.

    An ISOLATED near-zero track stays REJECTED_STATIC -- that is the measured
    false-positive signature (the development video contains exactly one such
    track, at 12 px/s against a 501 px/s reference, and it is correctly rejected).
    Only when several confirmed tracks stall together, having previously moved, is
    a genuine stop the better explanation.

    Requiring corroboration is what keeps this from becoming a hole: one static
    artefact can never excuse itself. Set to 0 to disable."""

    # ---- WAGON_ACTIVE recovery policy -----------------------------------
    wagon_active_recovery_enabled: bool = True
    """Inside a CONFIRMED wagon region, re-examine candidates that failed only a
    SOFT gate and accept them if they still clear every HARD requirement.

    Rationale: validation runs before classification (it produces the segments
    classification needs), so the first pass cannot know the train state. Once
    the wagon window is derived, the candidates that fell inside it are
    re-examined with that context. Two independent real trains showed genuine
    wagon gaps lost to soft speed/trajectory gates inside the wagon run.

    This is NOT 'accept everything in WAGON_ACTIVE': the HARD gates
    (untracked, too short, blind track, isolated static, wrong direction,
    duplicate, separation) still reject, and the floors below still apply."""

    wagon_active_min_monotonic: float = 0.45
    """Trajectory floor for recovery. Above the normal threshold a candidate is
    accepted outright; between this floor and the threshold its trajectory is
    merely noisier than expected and is recoverable; at or below this floor the
    direction reverses about as often as it holds, which is noise rather than an
    object crossing the frame, so it stays rejected. Dimensionless."""

    wagon_active_min_path_efficiency: float = 0.40
    """Path-efficiency floor for recovery: |net displacement| / path travelled.

    Measured on synthetic shapes matching real behaviour -- a noisy-but-genuine
    sweep scores ~0.71, while an oscillation that drifts scores ~0.13. 0.40 sits
    between them. This is the gate that keeps 'noisy trajectory' recoverable
    while 'impossible/noise trajectory' stays rejected."""

    wagon_active_min_confidence: float = 0.30
    """Confidence floor for recovery. A weaker detection inside the wagon run can
    still be a real gap, but a near-chance one cannot. Model-scale, not
    train-scale, so it transfers between trains."""

    wagon_active_min_motion_frac: float = 0.0
    """Optional extra displacement floor for recovery, as a FRACTION OF FRAME
    WIDTH. 0.0 means 'inherit static_max_motion_frac', i.e. anything the static
    gate did not already reject. Raise it to demand more movement before a
    soft-failed candidate is recovered."""

    train_motion_check_enabled: bool = True
    min_tracks_for_train_reference: int = 5
    """A per-camera reference speed is only computed when at least this many
    tracks survived the earlier tests -- otherwise the median is not meaningful
    and the check is skipped."""

    train_motion_tolerance: float = 4.0
    """A track's speed may differ from the camera's median gap speed by up to
    this FACTOR (either direction) before it is rejected.

    Measured: within one camera the real gaps span only ~1.9x (RIGHT_UP_TOP
    285-546 px/s), and the train's own deceleration across a full pass adds
    roughly another 1.8x, so ~2.5x is the realistic worst case. 4.0 keeps a
    margin above that while still catching gross outliers -- e.g. the measured
    RIGHT_UP_TOP track at 74 px/s against that camera's 387 px/s median, which
    is a tracker latch onto something other than a passing gap.

    Erring toward rejection here is deliberate: an under-count is a reported
    number, whereas a fabricated wagon is a wrong one."""

    direction_check_enabled: bool = True
    """Reject a track that travels against the camera's dominant gap direction.

    Measured: gap motion direction is per-camera, not global -- RIGHT_UP and
    RIGHT_UP_TOP gaps move in -x, LEFT_UP_TOP gaps move in +x. The dominant
    direction is therefore derived from each camera's own surviving tracks, never
    assumed."""

    min_tracks_for_direction_reference: int = 5
    """Below this many survivors the dominant direction is not established and
    the check is skipped rather than guessed."""

    # ---- 8. duplicate suppression ---------------------------------------
    duplicate_suppression_enabled: bool = True
    duplicate_min_time_overlap: float = 0.30
    """Two tracks are candidates for being the SAME physical gap only when their
    frame ranges genuinely OVERLAP by at least this fraction of the shorter
    track. Time overlap (rather than mere proximity) is used deliberately: two
    distinct inter-wagon gaps are temporally disjoint, so this rule cannot merge
    two real wagons."""

    duplicate_max_center_frac: float = 0.1415
    """...and their centre columns must also be within this distance (FRACTION OF
    FRAME WIDTH), i.e. the two tracks follow the same object in the same part of
    the image. Measured default: 120 px at 848 px width."""

    min_separation_seconds: float = 0.67
    """Minimum time between two consecutive VALIDATED physical gap events.

    A measured physical constraint of the observed train: consecutive real wagon
    gaps were never closer than ~10 frames at 15 fps. Stored in SECONDS so it
    transfers to other frame rates, and applied ONLY to final validated events --
    never to raw detections, which legitimately cluster because several belong to
    one track.

    Treated as an initial measured default, not a production invariant: a shorter
    wagon or a faster train could in principle produce closer boundaries, so a
    violation is resolved as a suspected duplicate/fragmentation WITH diagnostics
    rather than silently deleted."""

    def resolve(self, frame_width: int, fps: float,
                absolute_overrides: Optional[Dict[str, float]] = None,
                ) -> ResolvedThresholds:
        """Resolve camera-independent thresholds into this camera's units.

        Falls back to the development geometry only when a camera reports no
        usable width or fps, so a malformed stream cannot silently disable
        validation.

        Parameters
        ----------
        absolute_overrides :
            Optional ``{resolved_field_name: absolute_value}`` applied AFTER
            resolution. This is the compatibility channel for deprecated
            absolute-unit CLI flags (``--gap-min-motion-px``,
            ``--gap-min-track-frames``, ...): the operator's pixel/frame value is
            honoured verbatim for this camera, while the config itself keeps
            storing only normalized units. Nothing absolute is ever persisted on
            the config, so a single absolute override cannot silently become the
            default for a differently-shaped camera.
        """
        w = int(frame_width) if frame_width and frame_width > 0 else 848
        f = float(fps) if fps and fps > 0 else 15.0
        resolved = ResolvedThresholds(
            frame_width=w, fps=f,
            min_track_frames=max(2, int(round(self.min_track_seconds * f))),
            max_detection_gap_frames=max(
                1, int(round(self.max_detection_gap_seconds * f))),
            min_motion_px=self.min_motion_frac * w,
            static_max_motion_px=self.static_max_motion_frac * w,
            min_motion_px_per_sec=self.min_motion_frac_per_sec * w,
            max_motion_px_per_sec=self.max_motion_frac_per_sec * w,
            duplicate_max_center_px=self.duplicate_max_center_frac * w,
            min_separation_frames=max(1, int(round(self.min_separation_seconds * f))),
        )
        for name, value in (absolute_overrides or {}).items():
            if value is None:
                continue
            if not hasattr(resolved, name):
                raise ValueError(
                    f"unknown gap-validation threshold override {name!r}; "
                    f"expected one of {sorted(ResolvedThresholds.__dataclass_fields__)}")
            current = getattr(resolved, name)
            setattr(resolved, name, type(current)(value))
        return resolved

    def describe(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_GAP_VALIDATION = GapValidationConfig()


# =============================================================================
# Motion features
# =============================================================================

@dataclass
class GapMotionFeatures:
    """Deterministic motion description of one gap track."""
    track_id: int
    camera_id: str
    frame_start: int
    frame_end: int
    time_start: float
    time_end: float
    duration_s: float
    track_frames: int
    hits: int
    coverage: float
    max_detection_gap: int
    center_start: float
    center_end: float
    displacement_px: float
    abs_displacement_px: float
    velocity_px_per_sec: float
    direction: int                     # +1, -1 or 0
    monotonic_fraction: float
    n_steps: int
    step_velocity_median: Optional[float]
    mean_confidence: float
    min_confidence: Optional[float]
    bbox_height_median: Optional[float]
    bbox_width_median: Optional[float]
    path_efficiency: float = 0.0
    """|net displacement| / total path travelled, in [0, 1].

    1.0 is a straight sweep; a value near 0 means the centre wandered back and
    forth far more than it progressed. This separates 'noisier trajectory than
    expected' (still ~0.7 on real data) from 'oscillating noise' (~0.1), which
    monotonic fraction alone does not: an oscillation that drifts can score 0.55
    monotonic while travelling 8x its own net displacement. Dimensionless, so it
    transfers across geometries."""
    motion_reference_speed: Optional[float] = None
    """The reference speed this candidate was judged against (px/s)."""
    motion_reference_kind: str = ""
    """'local' (rolling median of temporal neighbours) or 'global' fallback."""
    motion_paused: bool = False
    """True when the track stalled but a corroborated TRAIN STOP explained it."""

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


@dataclass
class GapRejection:
    """One rejected candidate, with the reason and the measured evidence."""
    reason: str
    detail: str
    features: GapMotionFeatures
    source_event: Optional[GapEvent] = None
    """The GapEvent that was rejected. Retained so a later, better-informed pass
    (WAGON_ACTIVE recovery) can re-admit it without re-deriving anything."""

    @property
    def is_hard(self) -> bool:
        return self.reason in HARD_REJECTION_REASONS

    @property
    def is_soft(self) -> bool:
        return self.reason in SOFT_REJECTION_REASONS

    def to_dict(self) -> Dict[str, Any]:
        return {"reason": self.reason, "detail": self.detail,
                "hard": self.is_hard, "soft": self.is_soft,
                "features": self.features.to_dict()}


@dataclass
class GapValidationResult:
    """Everything one camera's validation pass produced."""
    camera_id: str
    accepted: List[GapEvent] = field(default_factory=list)
    rejected: List[GapRejection] = field(default_factory=list)
    features: List[GapMotionFeatures] = field(default_factory=list)
    raw_detection_count: int = 0
    tracked_candidate_count: int = 0
    train_reference_speed: Optional[float] = None
    config_used: Dict[str, Any] = field(default_factory=dict)
    resolved_thresholds: Dict[str, Any] = field(default_factory=dict)
    """The camera-independent config resolved into this camera's px/frames."""
    separation_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    """Minimum-separation violations that were investigated and KEPT, with the
    evidence. Recorded so a train whose gaps genuinely sit closer than the
    observed norm is visible rather than silently trimmed."""
    train_stopped_detected: bool = False
    """True when several confirmed tracks stalled together, i.e. the TRAIN
    stopped rather than individual detections being static artefacts."""

    @property
    def train_motion_state(self) -> str:
        """Runtime motion state derived from MULTIPLE reliable tracks.

        Diagnostic only -- it never gates a decision. A single candidate can
        never define train motion: the state comes from the ordered speeds of the
        accepted population.

        MOVING / ACCELERATING / DECELERATING / STOPPED / UNKNOWN
        """
        speeds = [f.velocity_px_per_sec for f in self.features
                  if f.velocity_px_per_sec is not None]
        if len(speeds) < 3:
            return "UNKNOWN"
        if self.train_stopped_detected:
            return "STOPPED"
        ref = statistics.median(speeds)
        if ref <= 0:
            return "UNKNOWN"
        ordered = [f.velocity_px_per_sec for f in
                   sorted(self.features, key=lambda x: x.frame_start)]
        half = max(1, len(ordered) // 2)
        first, second = statistics.median(ordered[:half]), statistics.median(ordered[-half:])
        if first <= 0:
            return "UNKNOWN"
        change = (second - first) / first
        if change > 0.15:
            return "ACCELERATING"
        if change < -0.15:
            return "DECELERATING"
        return "MOVING"

    @property
    def rejection_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.rejected:
            out[r.reason] = out.get(r.reason, 0) + 1
        return out

    def to_dict(self, include_rejections: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "camera_id": self.camera_id,
            "raw_detections": self.raw_detection_count,
            "tracked_candidates": self.tracked_candidate_count,
            "valid_gap_events": len(self.accepted),
            "rejected_total": len(self.rejected),
            "rejection_counts": self.rejection_counts,
            "train_reference_speed_px_per_sec": (
                round(self.train_reference_speed, 2)
                if self.train_reference_speed is not None else None),
            "resolved_thresholds": dict(self.resolved_thresholds),
            "train_stopped_detected": self.train_stopped_detected,
            "separation_diagnostics": list(self.separation_diagnostics),
            "train_motion_state": self.train_motion_state,
            "motion_paused_tracks": sum(1 for f in self.features if f.motion_paused),
        }
        if include_rejections:
            d["rejections"] = [r.to_dict() for r in self.rejected]
        return d


# =============================================================================
# Feature extraction
# =============================================================================

def compute_motion_features(gap: GapEvent) -> Optional[GapMotionFeatures]:
    """Derive motion features from data the tracker already recorded.

    Returns None when the track carries no usable trajectory (fewer than two
    hits), which the caller reports as REJECTED_NO_TRAJECTORY.
    """
    traj = list(gap.center_x_trajectory or [])
    hits = list(gap.hit_frames or [])
    fps = gap.fps or 0.0

    if len(traj) < 2 or len(hits) < 2 or fps <= 0:
        return None

    n = min(len(traj), len(hits))
    traj, hits = traj[:n], hits[:n]

    track_frames = max(1, gap.end_frame - gap.start_frame + 1)
    duration = track_frames / fps

    # Longest run of consecutive missed frames inside the track.
    max_gap = 0
    for a, b in zip(hits, hits[1:]):
        max_gap = max(max_gap, b - a - 1)

    # Per-step apparent velocity between consecutive detections.
    steps: List[float] = []
    for (f0, x0), (f1, x1) in zip(zip(hits, traj), zip(hits[1:], traj[1:])):
        df = f1 - f0
        if df > 0:
            steps.append((x1 - x0) / (df / fps))

    signs = [1 if s > 0 else (-1 if s < 0 else 0) for s in steps]
    n_pos = sum(1 for s in signs if s > 0)
    n_neg = sum(1 for s in signs if s < 0)
    dominant = 0
    if n_pos > n_neg:
        dominant = 1
    elif n_neg > n_pos:
        dominant = -1
    monotonic = (max(n_pos, n_neg) / len(signs)) if signs else 0.0

    displacement = traj[-1] - traj[0]
    abs_disp = abs(displacement)
    path_len = sum(abs(b - a) for a, b in zip(traj, traj[1:]))
    efficiency = (abs_disp / path_len) if path_len > 0 else 0.0

    heights = [b[3] - b[1] for b in (gap.bbox_history or []) if len(b) >= 4]
    widths = [b[2] - b[0] for b in (gap.bbox_history or []) if len(b) >= 4]

    return GapMotionFeatures(
        track_id=gap.track_id, camera_id=gap.camera_id,
        frame_start=gap.start_frame, frame_end=gap.end_frame,
        time_start=gap.start_time, time_end=gap.end_time,
        duration_s=duration, track_frames=track_frames,
        hits=gap.hit_count, coverage=min(1.0, gap.hit_count / track_frames),
        max_detection_gap=max_gap,
        center_start=traj[0], center_end=traj[-1],
        displacement_px=displacement, abs_displacement_px=abs_disp,
        velocity_px_per_sec=(abs_disp / duration) if duration > 0 else 0.0,
        path_efficiency=efficiency,
        direction=dominant, monotonic_fraction=monotonic, n_steps=len(steps),
        step_velocity_median=(statistics.median(steps) if steps else None),
        mean_confidence=gap.confidence, min_confidence=None,
        bbox_height_median=(statistics.median(heights) if heights else None),
        bbox_width_median=(statistics.median(widths) if widths else None),
    )


# =============================================================================
# Validation
# =============================================================================

def _time_overlap_fraction(a: GapEvent, b: GapEvent) -> float:
    """Overlap of two frame ranges as a fraction of the shorter range."""
    lo = max(a.start_frame, b.start_frame)
    hi = min(a.end_frame, b.end_frame)
    overlap = max(0, hi - lo + 1)
    shorter = min(a.end_frame - a.start_frame + 1, b.end_frame - b.start_frame + 1)
    return (overlap / shorter) if shorter > 0 else 0.0


def validate_gap_events(
    gaps: Sequence[GapEvent],
    camera_id: str,
    cfg: GapValidationConfig = DEFAULT_GAP_VALIDATION,
    raw_detection_count: int = 0,
    verbose: bool = True,
    frame_width: int = 0,
    fps: float = 0.0,
    absolute_overrides: Optional[Dict[str, float]] = None,
) -> GapValidationResult:
    """Filter tracked gap candidates down to physically plausible wagon boundaries.

    Deterministic and order-independent: the same input always yields the same
    accepted/rejected split. Nothing is discarded silently -- every rejection
    carries its reason and its measured features.
    """
    # Resolve camera-independent thresholds into THIS camera's pixels/frames.
    # Geometry comes from the caller, or from the gaps themselves as a fallback,
    # so nothing is assumed about resolution or frame rate.
    if not fps:
        fps = next((g.fps for g in gaps if g.fps), 0.0)
    res_thr = cfg.resolve(frame_width, fps, absolute_overrides)

    result = GapValidationResult(
        camera_id=camera_id,
        raw_detection_count=raw_detection_count,
        tracked_candidate_count=len(gaps),
        config_used=cfg.describe(),
        resolved_thresholds=res_thr.to_dict(),
    )

    if not cfg.enabled:
        result.accepted = list(gaps)
        for g in gaps:
            f = compute_motion_features(g)
            if f:
                result.features.append(f)
        if verbose:
            print(f"  [GAPVAL/{camera_id}] validation disabled -- "
                  f"passing all {len(gaps)} tracked candidate(s) through")
        return result

    # ---- pass 0: stall / train-stop pre-analysis --------------------------
    #
    # This must run BEFORE the per-track static check, because during a genuine
    # train stop a gap's displacement really is ~0 and pass 1 would reject it as
    # a static artefact. The distinction is corroboration: several confirmed
    # tracks stalling TOGETHER means the train stopped, whereas an isolated
    # stalled track is the measured false-positive signature.
    prelim = [(g, compute_motion_features(g)) for g in gaps]
    prelim = [(g, f) for g, f in prelim if f is not None]
    prelim_speeds = [f.velocity_px_per_sec for _, f in prelim
                     if f.velocity_px_per_sec > 0]
    stall_ceiling = 0.0
    paused_keys: set = set()
    if prelim_speeds and cfg.stop_corroboration_min_tracks:
        prelim_ref = statistics.median(prelim_speeds)
        stall_ceiling = prelim_ref * cfg.stopped_speed_fraction
        stalled = [(g, f) for g, f in prelim
                   if f.velocity_px_per_sec <= stall_ceiling]
        # A stop requires several stalled tracks OVERLAPPING IN TIME. Stalls
        # scattered across the video are independent artefacts, not one stop.
        for i, (g1, f1) in enumerate(stalled):
            group = [(g1, f1)] + [
                (g2, f2) for j, (g2, f2) in enumerate(stalled)
                if j != i and not (f2.frame_end < f1.frame_start
                                   or f2.frame_start > f1.frame_end)]
            if len(group) >= cfg.stop_corroboration_min_tracks:
                result.train_stopped_detected = True
                for gg, ff in group:
                    paused_keys.add((ff.frame_start, ff.frame_end))
    if result.train_stopped_detected and verbose:
        print(f"  [GAPVAL/{camera_id}] train-stop detected: "
              f"{len(paused_keys)} track(s) stalled together below "
              f"{stall_ceiling:.1f} px/s -> treated as MOTION_PAUSED, not static")

    # ---- pass 1: per-track tests, independent of the other tracks ----
    survivors: List[Tuple[GapEvent, GapMotionFeatures]] = []
    for g in sorted(gaps, key=lambda x: (x.center_frame, x.track_id)):
        f = compute_motion_features(g)
        if f is None:
            result.rejected.append(GapRejection(
                REJECTED_NO_TRAJECTORY,
                "fewer than two tracked hits, or no fps: no motion can be measured",
                GapMotionFeatures(
                    track_id=g.track_id, camera_id=g.camera_id,
                    frame_start=g.start_frame, frame_end=g.end_frame,
                    time_start=g.start_time, time_end=g.end_time,
                    duration_s=0.0, track_frames=0, hits=g.hit_count,
                    coverage=0.0, max_detection_gap=0, center_start=0.0,
                    center_end=0.0, displacement_px=0.0, abs_displacement_px=0.0,
                    velocity_px_per_sec=0.0, direction=0, monotonic_fraction=0.0,
                    n_steps=0, step_velocity_median=None,
                    mean_confidence=g.confidence, min_confidence=None,
                    bbox_height_median=None, bbox_width_median=None), g))
            continue

        result.features.append(f)
        reason: Optional[str] = None
        detail = ""

        # 1. temporal persistence
        if f.track_frames < res_thr.min_track_frames:
            reason = REJECTED_TOO_SHORT
            detail = (f"track spans {f.track_frames} frame(s) "
                      f"< min_track_frames={res_thr.min_track_frames}")
        elif f.hits < cfg.min_hits:
            reason = REJECTED_TOO_SHORT
            detail = f"only {f.hits} hit(s) < min_hits={cfg.min_hits}"

        # 2. detection continuity
        elif f.max_detection_gap > res_thr.max_detection_gap_frames:
            reason = REJECTED_DETECTION_GAP
            detail = (f"longest blind run {f.max_detection_gap} frame(s) "
                      f"> max_detection_gap_frames={res_thr.max_detection_gap_frames}")
        elif f.coverage < cfg.min_coverage:
            reason = REJECTED_DETECTION_GAP
            detail = (f"coverage {f.coverage:.2f} < min_coverage={cfg.min_coverage} "
                      f"({f.hits} hits over {f.track_frames} frames)")

        # 3. confidence
        elif f.mean_confidence < cfg.min_mean_confidence:
            reason = REJECTED_LOW_CONFIDENCE
            detail = (f"mean confidence {f.mean_confidence:.2f} "
                      f"< min_mean_confidence={cfg.min_mean_confidence}")

        # 4. motion: static artefacts are called out explicitly.
        # A track stalled as part of a CORROBORATED train stop is exempt: its
        # physical gap exists, the train simply is not moving.
        elif (f.frame_start, f.frame_end) in paused_keys:
            f.motion_paused = True
        elif f.abs_displacement_px <= res_thr.static_max_motion_px:
            reason = REJECTED_STATIC
            detail = (f"centre moved {f.abs_displacement_px:.1f} px over "
                      f"{f.duration_s:.2f}s (<= static_max_motion_px="
                      f"{res_thr.static_max_motion_px}); the train is moving, so a "
                      f"pinned detection is background, not a wagon gap")
        elif f.abs_displacement_px < res_thr.min_motion_px:
            reason = REJECTED_LOW_MOTION
            detail = (f"centre moved only {f.abs_displacement_px:.1f} px "
                      f"< min_motion_px={res_thr.min_motion_px}")

        # 5. speed plausibility
        elif f.velocity_px_per_sec < res_thr.min_motion_px_per_sec:
            reason = REJECTED_IMPLAUSIBLE_SPEED
            detail = (f"apparent speed {f.velocity_px_per_sec:.1f} px/s "
                      f"< min_motion_px_per_sec={res_thr.min_motion_px_per_sec}")
        elif f.velocity_px_per_sec > res_thr.max_motion_px_per_sec:
            reason = REJECTED_IMPLAUSIBLE_SPEED
            detail = (f"apparent speed {f.velocity_px_per_sec:.1f} px/s "
                      f"> max_motion_px_per_sec={res_thr.max_motion_px_per_sec}")

        # 6. trajectory consistency
        elif (f.n_steps >= cfg.min_steps_for_trajectory
                and f.monotonic_fraction < cfg.min_monotonic_fraction):
            reason = REJECTED_INCONSISTENT_TRAJECTORY
            detail = (f"only {f.monotonic_fraction:.2f} of {f.n_steps} steps share "
                      f"the dominant direction "
                      f"< min_monotonic_fraction={cfg.min_monotonic_fraction}")

        if reason:
            result.rejected.append(GapRejection(reason, detail, f, g))
        else:
            survivors.append((g, f))

    # ---- pass 2a: dominant direction, derived per camera (never assumed) ----
    if (cfg.direction_check_enabled
            and len(survivors) >= cfg.min_tracks_for_direction_reference):
        n_pos = sum(1 for _, f in survivors if f.direction > 0)
        n_neg = sum(1 for _, f in survivors if f.direction < 0)
        dominant = 1 if n_pos > n_neg else (-1 if n_neg > n_pos else 0)
        if dominant != 0:
            kept: List[Tuple[GapEvent, GapMotionFeatures]] = []
            for g, f in survivors:
                if f.direction != 0 and f.direction != dominant:
                    result.rejected.append(GapRejection(
                        REJECTED_WRONG_DIRECTION,
                        f"track travels in {'+x' if f.direction > 0 else '-x'} but "
                        f"this camera's gaps travel in "
                        f"{'+x' if dominant > 0 else '-x'} "
                        f"({max(n_pos, n_neg)} of {len(survivors)} tracks); a gap "
                        f"moving against the train is not a wagon boundary", f, g))
                else:
                    kept.append((g, f))
            survivors = kept

    # ---- pass 2b: train-motion context, using a ROLLING LOCAL reference ----
    #
    # A candidate is compared against the median speed of its temporal
    # NEIGHBOURS, not the whole video's median, because trains accelerate and
    # decelerate during a pass. The global median is still computed and reported
    # for diagnostics, and is used as the fallback when a candidate has too few
    # neighbours to form a local estimate.
    if (cfg.train_motion_check_enabled
            and len(survivors) >= cfg.min_tracks_for_train_reference):
        survivors.sort(key=lambda t: (t[1].frame_start, t[0].track_id))
        speeds = [f.velocity_px_per_sec for _, f in survivors
                  if f.velocity_px_per_sec > 0]
        if speeds:
            ref_global = statistics.median(speeds)
            result.train_reference_speed = ref_global

            # The stop verdict was already established in pass 0.
            stall_ceiling = ref_global * cfg.stopped_speed_fraction

            kept: List[Tuple[GapEvent, GapMotionFeatures]] = []
            for idx, (g, f) in enumerate(survivors):
                v = f.velocity_px_per_sec

                # local reference: median speed of the temporal neighbours
                ref = ref_global
                ref_kind = "global"
                w = cfg.motion_reference_window
                if w > 0:
                    lo_i, hi_i = max(0, idx - w), min(len(survivors), idx + w + 1)
                    neigh = [survivors[j][1].velocity_px_per_sec
                             for j in range(lo_i, hi_i)
                             if j != idx and survivors[j][1].velocity_px_per_sec > 0]
                    if len(neigh) >= 2:
                        ref = statistics.median(neigh)
                        ref_kind = "local"
                f.motion_reference_speed = ref
                f.motion_reference_kind = ref_kind

                lo = ref / cfg.train_motion_tolerance
                hi = ref * cfg.train_motion_tolerance
                if lo <= v <= hi:
                    kept.append((g, f))
                    continue

                # Outside the band. A stalled track survives ONLY when a stop is
                # corroborated by other simultaneously-stalled confirmed tracks.
                if v <= stall_ceiling and (f.frame_start, f.frame_end) in paused_keys:
                    f.motion_paused = True
                    kept.append((g, f))
                    continue

                detail = (f"apparent speed {v:.1f} px/s is outside "
                          f"[{lo:.1f}, {hi:.1f}] px/s, i.e. more than "
                          f"{cfg.train_motion_tolerance}x from the {ref_kind} "
                          f"reference speed {ref:.1f} px/s")
                if v <= stall_ceiling:
                    detail += (f"; the track is stalled and no other confirmed "
                               f"track stalls with it, so this is an isolated "
                               f"static artefact rather than a train stop")
                result.rejected.append(GapRejection(
                    REJECTED_TRAIN_MOTION_MISMATCH, detail, f, g))
            survivors = kept

    # ---- pass 3: duplicate suppression -- one physical gap, one GapEvent ----
    if cfg.duplicate_suppression_enabled:
        deduped: List[Tuple[GapEvent, GapMotionFeatures]] = []
        for g, f in survivors:
            clash = None
            for kg, kf in deduped:
                if (_time_overlap_fraction(g, kg) >= cfg.duplicate_min_time_overlap
                        and abs(f.center_start - kf.center_start)
                        <= res_thr.duplicate_max_center_px):
                    clash = (kg, kf)
                    break
            if clash is None:
                deduped.append((g, f))
            else:
                kg, _kf = clash
                # Keep the better-evidenced track: more hits, then higher conf.
                if (f.hits, f.mean_confidence) > (kg.hit_count, kg.confidence):
                    deduped = [(x, y) for x, y in deduped if x is not kg]
                    deduped.append((g, f))
                    loser, loser_f = kg, _kf
                else:
                    loser, loser_f = g, f
                result.rejected.append(GapRejection(
                    REJECTED_DUPLICATE,
                    f"track {loser.track_id} overlaps track "
                    f"{(kg if loser is g else g).track_id} in time and position: "
                    f"same physical gap, so only one GapEvent is kept",
                    loser_f, loser))
        survivors = sorted(deduped, key=lambda t: (t[0].center_frame, t[0].track_id))

    # ---- pass 4: minimum physical separation between VALIDATED events ------
    #
    # Applied to final events only, never to raw detections (which legitimately
    # cluster because several belong to one track). A violation is investigated
    # as a suspected duplicate/fragment rather than deleted on sight: the weaker
    # of the pair goes, and the decision is recorded with both tracks' evidence.
    if res_thr.min_separation_frames > 1:
        survivors.sort(key=lambda t: (t[0].center_frame, t[0].track_id))
        kept: List[Tuple[GapEvent, GapMotionFeatures]] = []
        for g, f in survivors:
            if not kept:
                kept.append((g, f))
                continue
            pg, pf = kept[-1]
            sep = int(round(g.center_frame)) - int(round(pg.center_frame))
            if sep >= res_thr.min_separation_frames:
                kept.append((g, f))
                continue
            # Close in TIME. That alone does not make them one gap: a wide view
            # can show two couplings at once, and those are distinct physical
            # boundaries. Only when they are ALSO close in image position is a
            # duplicate/fragment the better explanation.
            if abs(f.center_start - pf.center_start) > res_thr.duplicate_max_center_px:
                result.separation_diagnostics.append({
                    "track_id": g.track_id,
                    "other_track_id": pg.track_id,
                    "separation_frames": sep,
                    "min_separation_frames": res_thr.min_separation_frames,
                    "center_distance_px": round(
                        abs(f.center_start - pf.center_start), 2),
                    "verdict": "kept: close in time but far apart in the image, "
                               "so these are two simultaneous physical gaps "
                               "rather than one fragmented track",
                })
                kept.append((g, f))
                continue
            # Too close in time AND position to be two distinct boundaries. Keep
            # whichever track is better evidenced and record the other.
            loser, loser_f, winner = ((g, f, pg) if (pf.hits, pf.mean_confidence)
                                      >= (f.hits, f.mean_confidence)
                                      else (pg, pf, g))
            if loser is pg:
                kept[-1] = (g, f)
            result.rejected.append(GapRejection(
                REJECTED_MIN_SEPARATION,
                f"only {sep} frame(s) from validated track {winner.track_id} "
                f"(< min_separation_frames={res_thr.min_separation_frames}, "
                f"= {cfg.min_separation_seconds}s at {res_thr.fps:g} fps). Two "
                f"physical wagon boundaries cannot be this close, so this is a "
                f"duplicate or a tracker fragment; the better-evidenced track "
                f"({winner.track_id}: {max(pf.hits, f.hits)} hits) is kept",
                loser_f, loser))
        survivors = kept

    result.accepted = [g for g, _ in survivors]

    if verbose:
        rc = result.rejection_counts
        print(f"  [GAPVAL/{camera_id}] raw_detections={result.raw_detection_count}  "
              f"tracked_candidates={result.tracked_candidate_count}  "
              f"valid={len(result.accepted)}  rejected={len(result.rejected)}")
        if rc:
            for reason in ALL_REJECTION_REASONS:
                if reason in rc:
                    print(f"      {reason:<36} {rc[reason]}")
        if result.train_reference_speed is not None:
            print(f"      train reference speed: "
                  f"{result.train_reference_speed:.1f} px/s (median of survivors)")

    return result


@dataclass
class RecoveryResult:
    """Outcome of the WAGON_ACTIVE second pass."""
    camera_id: str
    recovered: List[GapEvent] = field(default_factory=list)
    still_rejected: List[GapRejection] = field(default_factory=list)
    considered: int = 0
    outside_window: int = 0
    hard_blocked: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "candidates_considered": self.considered,
            "outside_wagon_window": self.outside_window,
            "blocked_by_hard_gate": self.hard_blocked,
            "recovered": len(self.recovered),
            "still_rejected": len(self.still_rejected),
            "details": list(self.details),
        }


def recover_wagon_active_candidates(
    rejected: Sequence[GapRejection],
    accepted: Sequence[GapEvent],
    wagon_start_frame: Optional[int],
    wagon_end_frame: Optional[int],
    camera_id: str,
    cfg: GapValidationConfig = DEFAULT_GAP_VALIDATION,
    frame_width: int = 0,
    fps: float = 0.0,
    absolute_overrides: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> RecoveryResult:
    """Second pass: recover SOFT-failed candidates inside the wagon window.

    WHY A SECOND PASS. Validation must run before classification, because
    classification needs the segments that validated gaps define. The first pass
    therefore cannot know the train state. Once the wagon window has been derived
    the candidates that fell inside it are re-examined WITH that context.

    WHAT IS RELAXED. Only the SOFT gates -- speed vs the local reference,
    absolute speed band, sub-floor displacement, weaker confidence, noisier
    trajectory. Every HARD gate still rejects:

        untracked / no trajectory      insufficient tracker confirmation
        mostly-blind track             isolated static artefact
        wrong direction                duplicate of an accepted gap
        minimum-separation duplicate

    WHAT STILL APPLIES. Recovery is not a free pass. A candidate must clear the
    trajectory floor, the confidence floor, the displacement floor, must not
    duplicate an already-accepted gap in time and position, and must respect the
    minimum separation from its accepted neighbours.

    Deterministic and side-effect free: `accepted` is not mutated.
    """
    result = RecoveryResult(camera_id=camera_id)
    if not cfg.enabled or not cfg.wagon_active_recovery_enabled:
        result.still_rejected = list(rejected)
        return result
    if wagon_start_frame is None or wagon_end_frame is None:
        result.still_rejected = list(rejected)
        if verbose:
            print(f"  [RECOVER/{camera_id}] no wagon window -> nothing considered")
        return result

    if not fps:
        fps = next((g.fps for g in accepted if g.fps), 0.0)
    thr = cfg.resolve(frame_width, fps, absolute_overrides)
    motion_floor = (cfg.wagon_active_min_motion_frac * thr.frame_width
                    if cfg.wagon_active_min_motion_frac > 0
                    else thr.static_max_motion_px)

    # Work against a growing accepted set so recovered gaps also protect each
    # other from duplicates and separation violations.
    live: List[GapEvent] = sorted(accepted, key=lambda g: g.center_frame)

    for rej in sorted(rejected, key=lambda r: r.features.frame_start):
        f = rej.features
        centre = int(round((f.frame_start + f.frame_end) / 2))

        if not (wagon_start_frame <= centre <= wagon_end_frame):
            result.outside_window += 1
            result.still_rejected.append(rej)
            continue

        result.considered += 1

        if rej.reason in HARD_REJECTION_REASONS:
            result.hard_blocked += 1
            result.still_rejected.append(rej)
            result.details.append({
                "track_id": f.track_id, "frame": centre, "outcome": "blocked",
                "reason": rej.reason,
                "note": "hard gate -- never relaxed by train state"})
            continue

        if rej.reason not in SOFT_REJECTION_REASONS:
            result.still_rejected.append(rej)
            continue

        # ---- safety floors: a soft reason is not a free pass ----
        blocked_by = None
        if f.abs_displacement_px <= motion_floor:
            blocked_by = (f"displacement {f.abs_displacement_px:.1f}px <= floor "
                          f"{motion_floor:.1f}px")
        elif (f.n_steps >= cfg.min_steps_for_trajectory
                and f.path_efficiency < cfg.wagon_active_min_path_efficiency):
            blocked_by = (f"path efficiency {f.path_efficiency:.2f} < floor "
                          f"{cfg.wagon_active_min_path_efficiency} -- the centre "
                          f"wandered far more than it progressed, which is noise "
                          f"rather than an object crossing the frame")
        elif (f.n_steps >= cfg.min_steps_for_trajectory
                and f.monotonic_fraction < cfg.wagon_active_min_monotonic):
            blocked_by = (f"monotonic {f.monotonic_fraction:.2f} < floor "
                          f"{cfg.wagon_active_min_monotonic}")
        elif f.mean_confidence < cfg.wagon_active_min_confidence:
            blocked_by = (f"confidence {f.mean_confidence:.2f} < floor "
                          f"{cfg.wagon_active_min_confidence}")
        elif f.direction == 0:
            blocked_by = "no dominant direction"

        if blocked_by:
            result.still_rejected.append(rej)
            result.details.append({
                "track_id": f.track_id, "frame": centre, "outcome": "blocked",
                "reason": rej.reason, "note": f"soft, but {blocked_by}"})
            continue

        # ---- must not duplicate, or crowd, an already-accepted gap ----
        clash = None
        for ag in live:
            a_centre = int(round(ag.center_frame))
            overlap = not (ag.end_frame < f.frame_start
                           or ag.start_frame > f.frame_end)
            a_x = (ag.center_x_trajectory[0] if ag.center_x_trajectory else None)
            close_x = (a_x is not None
                       and abs(a_x - f.center_start) <= thr.duplicate_max_center_px)
            if overlap and close_x:
                clash = (ag, "overlaps an accepted gap in time and position")
                break
            if (abs(a_centre - centre) < thr.min_separation_frames and close_x):
                clash = (ag, f"only {abs(a_centre - centre)} frame(s) from "
                             f"accepted track {ag.track_id} at a similar position")
                break
        if clash:
            result.still_rejected.append(rej)
            result.details.append({
                "track_id": f.track_id, "frame": centre, "outcome": "blocked",
                "reason": rej.reason,
                "note": f"soft, but {clash[1]} -- duplicate protection holds"})
            continue

        # ---- recovered ----
        result.details.append({
            "track_id": f.track_id, "frame": centre, "outcome": "recovered",
            "original_reason": rej.reason,
            "displacement_px": round(f.abs_displacement_px, 1),
            "speed_px_per_sec": round(f.velocity_px_per_sec, 1),
            "reference_speed": (round(f.motion_reference_speed, 1)
                                if f.motion_reference_speed else None),
            "monotonic": round(f.monotonic_fraction, 2),
            "confidence": round(f.mean_confidence, 3),
            "note": "inside the confirmed wagon region and clears every hard "
                    "gate, so the soft failure alone does not reject it",
        })
        result.recovered.append(rej.source_event)
        live.append(rej.source_event)
        live.sort(key=lambda g: g.center_frame)

    if verbose:
        print(f"  [RECOVER/{camera_id}] wagon-window candidates="
              f"{result.considered}  recovered={len(result.recovered)}  "
              f"hard-blocked={result.hard_blocked}  "
              f"outside-window={result.outside_window}")
        for d in result.details:
            if d["outcome"] == "recovered":
                print(f"      + trk {d['track_id']} @f{d['frame']}: was "
                      f"{d['original_reason']}, speed={d['speed_px_per_sec']} "
                      f"px/s vs ref {d['reference_speed']}")
    return result


def renumber_gap_events(gaps: Sequence[GapEvent]) -> List[GapEvent]:
    """Re-assign track_id 1..N in temporal order, as the tracker does.

    Validation removes tracks, which would otherwise leave holes in the id
    sequence. The pipeline's downstream code treats `track_id` as a temporal
    rank within the camera, so it is restored here.
    """
    out: List[GapEvent] = []
    for new_id, g in enumerate(sorted(gaps, key=lambda x: (x.center_frame, x.track_id)),
                               start=1):
        g.track_id = new_id
        out.append(g)
    return out
