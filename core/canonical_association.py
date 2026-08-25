"""Assign a feature detection to the canonical global wagon that owns it.

The problem this replaces
-------------------------
Until now a damage detection's wagon was decided *implicitly*, by which
directory the materializer had copied its frame into:

    wagon_cache/GW_25/right_up_top/frame_004120.jpg   -> "this is GW_25"

The materializer computes that bucket as
`round((GW.start_time - delta) * local_fps) .. round((GW.end_time - delta) *
local_fps) - 1`, so the wagon boundary reaching the damage processor is a
*rounded frame index* derived from a wagon time window -- and `delta` is 0.0 for
any camera whose offset the counter could not resolve.  On a camera whose
footage is displaced that bucket lands on the wrong wagon, and the damage
processor has no way to know: it sees only a directory name.  The four cameras
have four different clocks and four different frame rates, so a local frame
number is not an identity that can be compared across them.

The rule implemented here
-------------------------
The canonical physical gaps divide the normalized (master-clock) timeline into
wagon intervals.  A detection is owned by the interval its normalized time falls
in::

        GW_25            GAP_25            GW_26
    ...............|=================|...............
       BEFORE_GAP        ambiguous        AFTER_GAP
        -> GW_25         -> neither        -> GW_26

For a detection on camera X at local frame F::

    1.  t_local  = F / fps(X)                    the camera's own clock
    2.  t_global = t_local + delta(X)            the canonical master clock
    3.  locate the canonical gaps bracketing t_global
    4.  the nearer of the two is the ASSOCIATED gap; which side of it
        t_global falls on gives BEFORE_GAP / AFTER_GAP, and therefore the wagon

Step 2 is the whole point: two cameras that saw the same physical defect
normalize to the same master time and therefore to the same `GW_n`, which
comparing local frame numbers could never achieve.

What this module is not
-----------------------
It runs no detector, counts no wagons and finds no gaps.  Both inputs are
canonical and read-only:

    `state.wagons`         the finalized RIGHT_UP-mastered roster (frozen)
    `global_gaps`          the authoritative gap sequence from the counting
                           engine, each with its `master_time`

It therefore cannot create, delete, renumber or shift a `GW_n`.  The only thing
it produces is an opinion about which existing wagon a detection belongs to,
carried with enough provenance to audit that opinion.

Honest failure over a confident guess
-------------------------------------
Three situations are reported rather than guessed:

    BOUNDARY_AMBIGUOUS          within tolerance of the gap; the two candidate
                                wagons are both recorded and neither is chosen
    UNRESOLVED_NO_GAP_TIMING    a gap in the relevant stretch has no usable
                                `master_time`, so the interval that stretch
                                describes may be two wagons wide
    UNRESOLVED_*                no fps, no canonical gaps, no frame, or a
                                normalized time outside every wagon interval

`RESOLVED_ASSUMED_OFFSET` is a fourth, softer flag: the assignment stands, but
the camera's clock offset was never resolved, so `t_global = t_local + 0.0` is
the historical shared-`t=0` assumption rather than a measurement.  The tolerance
band is widened for those cameras, because a less certain time base should fall
to AMBIGUOUS more readily, not less.

Reuse for Door / Load / OCR
---------------------------
Nothing here is damage-specific.  `Detection` carries a feature name purely so
the provenance says what was associated; the resolver's inputs are a camera, a
local frame and the canonical timeline.  A future door/load/ocr pass builds the
same `CanonicalTimeline` once and calls `assign()` per detection -- which is why
the timeline is a separate object from the assignment.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Vocabulary
# -----------------------------------------------------------------------------

#: Assigned, on a camera whose clock offset was resolved.
STATUS_RESOLVED = "RESOLVED"

#: Assigned, but `t_global = t_local + 0.0` because this camera's offset was
#: never resolved.  A real assignment on an assumed time base -- distinct from
#: both a measured one and a failure, and it must not read as either.
STATUS_RESOLVED_ASSUMED_OFFSET = "RESOLVED_ASSUMED_OFFSET"

#: Too close to a canonical gap to attribute.  Both candidates are recorded.
STATUS_BOUNDARY_AMBIGUOUS = "BOUNDARY_AMBIGUOUS"

#: A gap in the relevant stretch has no usable timing, so the interval cannot
#: be trusted to be one wagon wide.
STATUS_UNRESOLVED_NO_GAP_TIMING = "UNRESOLVED_NO_GAP_TIMING"

STATUS_UNRESOLVED_NO_CAMERA_FPS = "UNRESOLVED_NO_CAMERA_FPS"
STATUS_UNRESOLVED_NO_CANONICAL_GAPS = "UNRESOLVED_NO_CANONICAL_GAPS"
STATUS_UNRESOLVED_OUTSIDE_WAGON_REGION = "UNRESOLVED_OUTSIDE_WAGON_REGION"
STATUS_UNRESOLVED_NO_FRAME = "UNRESOLVED_NO_FRAME"

#: Statuses that carry a usable `global_wagon_id`.
RESOLVED_STATUSES = (STATUS_RESOLVED, STATUS_RESOLVED_ASSUMED_OFFSET)

#: Statuses where the detection is preserved but deliberately not attributed.
UNRESOLVED_STATUSES = (
    STATUS_BOUNDARY_AMBIGUOUS,
    STATUS_UNRESOLVED_NO_GAP_TIMING,
    STATUS_UNRESOLVED_NO_CAMERA_FPS,
    STATUS_UNRESOLVED_NO_CANONICAL_GAPS,
    STATUS_UNRESOLVED_OUTSIDE_WAGON_REGION,
    STATUS_UNRESOLVED_NO_FRAME,
)

METHOD_BEFORE_GAP = "BEFORE_GAP"
METHOD_AFTER_GAP = "AFTER_GAP"
METHOD_UNASSIGNED = "UNASSIGNED"

#: Offset statuses the counting engine treats as decisive
#: (`wagon_count.global_fusion.OFFSET_REFERENCE` / `OFFSET_RESOLVED`).
OFFSET_RESOLVED_STATUSES = ("REFERENCE", "RESOLVED")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AssociationConfig:
    """Tunables.  Defaults prefer AMBIGUOUS over a confident wrong wagon."""

    tolerance_floor_s: float = 0.10
    tolerance_cap_s: float = 1.00
    """Clamp on the ambiguity half-band.

    Deliberately the same floor and cap as
    `wagon_count.global_fusion.FusionConfig.sigma_floor_s / sigma_cap_s`, whose
    `GapObservation.sigma()` already derives a timing tolerance from how long a
    gap was actually visible.  Two different tolerance conventions for the same
    physical quantity would be one convention too many.
    """

    default_tolerance_s: float = 0.25
    """Used when the gap's visible span cannot be recovered.  A gap is visible
    for roughly a dozen frames (~0.4 s at 15 fps), so half of that, rounded
    down, is a conservative band: wide enough to catch a detection sitting in
    the coupling, far narrower than the seconds-long wagon intervals it sits
    between.
    """

    assumed_offset_tolerance_factor: float = 2.0
    """Tolerance multiplier for a camera whose clock offset was never resolved.
    Its normalized time is an assumption, so the band in which we decline to
    call ownership is correspondingly wider.
    """

    min_overlap_s: float = 1e-6
    """A wagon must overlap a gap-delimited slot by more than this to own it.
    Guards against a zero-width touch at a shared boundary claiming a slot.
    """


DEFAULT_CONFIG = AssociationConfig()


# -----------------------------------------------------------------------------
# Canonical inputs, parsed
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalGap:
    """One entry of `state.global_gaps`, as this module needs it."""

    global_gap_id: int
    order_index: int
    """Position in the engine's own gap list, kept so a gap dropped for bad
    timing can still be detected as *missing between* two good ones."""
    master_time: Optional[float]
    valid: bool
    reason: str = ""
    half_span_s_by_camera: Mapping[str, float] = field(default_factory=dict)
    """Half of the gap's visible duration, per camera that observed it."""

    def tolerance_s(self, camera_id: str, cfg: AssociationConfig) -> float:
        """Ambiguity half-band around this gap, for this camera."""
        half = self.half_span_s_by_camera.get(camera_id)
        if half is None and self.half_span_s_by_camera:
            # This camera never observed the gap; the gap's physical duration is
            # a property of the train, so another camera's span is a better
            # estimate than a flat default.
            half = min(self.half_span_s_by_camera.values())
        if half is None or not math.isfinite(half) or half <= 0:
            half = cfg.default_tolerance_s
        return min(cfg.tolerance_cap_s, max(cfg.tolerance_floor_s, float(half)))


@dataclass(frozen=True)
class WagonSpan:
    """One canonical roster entry, in master-clock seconds."""

    global_wagon_id: str
    wagon_index: int
    start_time: float
    end_time: float
    classification: str = ""


def _finite(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def parse_canonical_gaps(
    global_gaps: Sequence[Mapping[str, Any]],
) -> List[CanonicalGap]:
    """Read `state.global_gaps` into `CanonicalGap`s, in engine order.

    Nothing is discarded: a gap with unusable timing is kept with
    `valid=False`, because its *absence* from the boundary list is exactly what
    makes the surrounding interval untrustworthy.
    """
    out: List[CanonicalGap] = []
    for i, g in enumerate(global_gaps or ()):
        if not isinstance(g, Mapping):
            continue
        try:
            gid = int(g.get("global_gap_id"))
        except (TypeError, ValueError):
            gid = i + 1
        t = _finite(g.get("master_time"))
        reason = ""
        if t is None:
            reason = "master_time missing or not a finite number"
        elif t < 0:
            t, reason = None, "master_time is negative"

        spans: Dict[str, float] = {}
        for cam, obs in (g.get("support_observations") or {}).items():
            if not isinstance(obs, Mapping):
                continue
            fps = _finite(obs.get("fps"))
            span = _finite(obs.get("span_frames"))
            if span is None:
                sf, ef = _finite(obs.get("start_frame")), _finite(obs.get("end_frame"))
                span = (ef - sf) if (sf is not None and ef is not None) else None
            if fps and fps > 0 and span is not None and span > 0:
                spans[str(cam)] = (span / 2.0) / fps

        out.append(CanonicalGap(
            global_gap_id=gid, order_index=i, master_time=t,
            valid=t is not None, reason=reason,
            half_span_s_by_camera=spans,
        ))
    return out


def parse_wagon_spans(wagons: Sequence[Any]) -> List[WagonSpan]:
    """Read the canonical roster into master-clock spans.  Read-only."""
    out: List[WagonSpan] = []
    for w in wagons or ():
        gw = str(getattr(w, "global_id", "") or "")
        if not gw:
            continue
        st = _finite(getattr(w, "start_time", None))
        en = _finite(getattr(w, "end_time", None))
        if st is None or en is None or en <= st:
            continue
        out.append(WagonSpan(
            global_wagon_id=gw,
            wagon_index=int(getattr(w, "wagon_index", 0) or 0),
            start_time=st, end_time=en,
            classification=str(getattr(w, "classification", "") or ""),
        ))
    out.sort(key=lambda s: (s.start_time, s.wagon_index))
    return out


# -----------------------------------------------------------------------------
# One detection in, one assignment out
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """A feature detection to be associated.  Feature-agnostic by design."""

    camera_id: str
    local_frame: int
    feature: str = ""
    detection_id: str = ""


@dataclass(frozen=True)
class Assignment:
    """The canonical wagon for one detection, with full provenance."""

    # ---- what was associated -------------------------------------------
    feature: str
    detection_id: str
    camera_id: str

    # ---- the three clocks ----------------------------------------------
    local_frame: Optional[int]
    local_time: Optional[float]
    global_time: Optional[float]
    local_fps: Optional[float]
    camera_time_offset: float
    offset_status: str

    # ---- the canonical anchors -----------------------------------------
    global_wagon_id: Optional[str]
    associated_global_gap_id: Optional[int]
    previous_wagon_id: Optional[str]
    next_wagon_id: Optional[str]

    # ---- the verdict ---------------------------------------------------
    method: str
    status: str
    confidence: float
    reason: str
    boundary_margin_s: Optional[float] = None
    tolerance_s: Optional[float] = None

    @property
    def resolved(self) -> bool:
        return self.status in RESOLVED_STATUSES and bool(self.global_wagon_id)

    def to_dict(self) -> Dict[str, Any]:
        def _r(v, n=4):
            return round(v, n) if isinstance(v, float) else v
        return {
            "schema": "wagon_eye.canonical_association.v1",
            "feature": self.feature,
            "detection_id": self.detection_id,
            "camera_id": self.camera_id,
            "local_frame": self.local_frame,
            "local_time": _r(self.local_time),
            "global_time": _r(self.global_time),
            "local_fps": _r(self.local_fps),
            "camera_time_offset": _r(self.camera_time_offset),
            "offset_status": self.offset_status,
            "global_wagon_id": self.global_wagon_id,
            "associated_global_gap_id": self.associated_global_gap_id,
            "previous_wagon_id": self.previous_wagon_id,
            "next_wagon_id": self.next_wagon_id,
            "method": self.method,
            "status": self.status,
            "confidence": _r(self.confidence),
            "reason": self.reason,
            "boundary_margin_s": _r(self.boundary_margin_s),
            "tolerance_s": _r(self.tolerance_s),
        }

    def render(self) -> str:
        gw = self.global_wagon_id or "—"
        gap = (f"GAP_{self.associated_global_gap_id}"
               if self.associated_global_gap_id is not None else "no-gap")
        t = f"{self.global_time:.3f}" if self.global_time is not None else "?"
        return (f"[DAMAGE-ASSOC] {self.camera_id} f={self.local_frame} "
                f"t_global={t}s {self.method} {gap} -> {gw} "
                f"[{self.status}] {self.reason}")


# -----------------------------------------------------------------------------
# The timeline
# -----------------------------------------------------------------------------

class CanonicalTimeline:
    """The canonical gap-delimited wagon intervals, built once per train.

    Construction is the only place the roster and the gap list are read; after
    that `assign()` is a pure lookup.  Nothing here mutates either input.
    """

    def __init__(
        self,
        *,
        gaps: Sequence[CanonicalGap],
        wagons: Sequence[WagonSpan],
        per_camera_fps: Mapping[str, float],
        camera_offsets: Mapping[str, float],
        offset_statuses: Optional[Mapping[str, str]] = None,
        cfg: AssociationConfig = DEFAULT_CONFIG,
    ) -> None:
        self.cfg = cfg
        self.gaps: Tuple[CanonicalGap, ...] = tuple(gaps)
        self.wagons: Tuple[WagonSpan, ...] = tuple(wagons)
        self.per_camera_fps = dict(per_camera_fps or {})
        self.camera_offsets = dict(camera_offsets or {})
        self.offset_statuses = dict(offset_statuses or {})

        #: Valid gaps only, ascending in master time -- the actual boundaries.
        self.boundaries: Tuple[CanonicalGap, ...] = tuple(
            sorted((g for g in self.gaps if g.valid),
                   key=lambda g: (g.master_time, g.order_index))
        )
        self._boundary_times: List[float] = [float(g.master_time)
                                             for g in self.boundaries]
        #: Gaps the engine emitted but whose timing is unusable.
        self.invalid_gaps: Tuple[CanonicalGap, ...] = tuple(
            g for g in self.gaps if not g.valid)

        #: slot i is bounded by boundaries[i-1] on the left and boundaries[i]
        #: on the right; slot 0 and slot len(boundaries) are the open ends.
        self._slot_owner: List[Optional[str]] = self._map_slots_to_wagons()
        self._span_by_id: Dict[str, WagonSpan] = {
            w.global_wagon_id: w for w in self.wagons}

    # ---- construction helpers -----------------------------------------

    def _slot_bounds(self, slot: int) -> Tuple[float, float]:
        lo = (self._boundary_times[slot - 1] if slot > 0 else -math.inf)
        hi = (self._boundary_times[slot] if slot < len(self._boundary_times)
              else math.inf)
        return lo, hi

    def _map_slots_to_wagons(self) -> List[Optional[str]]:
        """Give each gap-delimited slot the roster wagon that overlaps it most.

        Overlap, not equality: the roster's wagon spans were themselves cut at
        these gaps (`global_alignment.build_global_wagons`), so in a healthy
        train the correspondence is exact and the overlap test simply confirms
        it.  When a gap was dropped for bad timing the slot is wider than one
        wagon, and `_gap_missing_between` -- not this mapping -- is what refuses
        to attribute a detection inside it.
        """
        owners: List[Optional[str]] = []
        for slot in range(len(self._boundary_times) + 1):
            lo, hi = self._slot_bounds(slot)
            best_id, best_overlap = None, 0.0
            for w in self.wagons:
                ov = min(w.end_time, hi) - max(w.start_time, lo)
                if ov > best_overlap:
                    best_id, best_overlap = w.global_wagon_id, ov
            owners.append(best_id if best_overlap > self.cfg.min_overlap_s
                          else None)
        return owners

    @classmethod
    def build(
        cls,
        *,
        state: Any,
        global_gaps: Sequence[Mapping[str, Any]],
        per_camera_fps: Mapping[str, float],
        cfg: AssociationConfig = DEFAULT_CONFIG,
    ) -> "CanonicalTimeline":
        """Build from the canonical state and the engine's gap list.

        `global_gaps` is passed in rather than read off `state` on purpose: the
        v4 `GlobalTrainState` that reaches the inspection stages keeps only
        `global_gap_count`, and the full list lives on the counting engine's own
        state object (or its `global_train_state.json`).  Taking it as an
        argument is what lets the identical resolver run in both pipeline modes.
        """
        offsets: Dict[str, float] = {}
        statuses: Dict[str, str] = {}
        raw = getattr(state, "camera_offsets", None) or {}
        for cam, meta in raw.items():
            if not isinstance(meta, Mapping):
                continue
            st = str(meta.get("status") or "")
            statuses[str(cam)] = st or "UNKNOWN"
            if st in OFFSET_RESOLVED_STATUSES:
                d = _finite(meta.get("delta"))
                if d is not None:
                    offsets[str(cam)] = d
        return cls(
            gaps=parse_canonical_gaps(global_gaps),
            wagons=parse_wagon_spans(getattr(state, "wagons", None) or ()),
            per_camera_fps=per_camera_fps,
            camera_offsets=offsets,
            offset_statuses=statuses,
            cfg=cfg,
        )

    # ---- normalization -------------------------------------------------

    def offset_for(self, camera_id: str) -> Tuple[float, str]:
        """`(delta, status)`.  An unresolved camera gets 0.0, flagged as such --
        the historical shared-`t=0` assumption, never a guessed shift."""
        st = self.offset_statuses.get(camera_id, "UNKNOWN")
        if camera_id in self.camera_offsets:
            return self.camera_offsets[camera_id], st
        return 0.0, (st if st not in OFFSET_RESOLVED_STATUSES else "UNKNOWN")

    def normalize(self, camera_id: str, local_frame: int
                  ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """`(local_time, global_time, fps)` for one camera-local frame."""
        fps = _finite(self.per_camera_fps.get(camera_id))
        if fps is None or fps <= 0:
            return None, None, None
        t_local = float(local_frame) / fps
        delta, _ = self.offset_for(camera_id)
        return t_local, t_local + delta, fps

    # ---- the interval question -----------------------------------------

    def _gap_missing_between(self, left: Optional[CanonicalGap],
                             right: Optional[CanonicalGap]) -> Optional[CanonicalGap]:
        """A timing-less gap sitting between two boundaries, in engine order.

        If one exists, the slot those boundaries delimit is not one wagon wide,
        so ownership inside it is undecidable -- the case the spec calls out as
        "the relevant gap has no valid timing".
        """
        lo = left.order_index if left is not None else -1
        hi = right.order_index if right is not None else len(self.gaps)
        for g in self.invalid_gaps:
            if lo < g.order_index < hi:
                return g
        return None

    def wagon_at(self, global_time: float) -> Optional[str]:
        """The canonical wagon owning a normalized time, ignoring tolerance."""
        slot = bisect.bisect_right(self._boundary_times, float(global_time))
        return self._slot_owner[slot]

    def assign(self, detection: Detection) -> Assignment:
        """Associate one detection with the canonical wagon that owns it."""
        cam = str(detection.camera_id or "")
        delta, off_status = self.offset_for(cam)

        def _fail(status: str, reason: str, **kw) -> Assignment:
            base = dict(
                feature=detection.feature, detection_id=detection.detection_id,
                camera_id=cam, local_frame=detection.local_frame,
                local_time=None, global_time=None, local_fps=None,
                camera_time_offset=delta, offset_status=off_status,
                global_wagon_id=None, associated_global_gap_id=None,
                previous_wagon_id=None, next_wagon_id=None,
                method=METHOD_UNASSIGNED, status=status, confidence=0.0,
                reason=reason,
            )
            base.update(kw)
            return Assignment(**base)

        if detection.local_frame is None or int(detection.local_frame) < 0:
            return _fail(STATUS_UNRESOLVED_NO_FRAME,
                         "detection carries no usable local frame index")

        t_local, t_global, fps = self.normalize(cam, int(detection.local_frame))
        if t_global is None:
            return _fail(STATUS_UNRESOLVED_NO_CAMERA_FPS,
                         f"no usable fps for {cam}; a local frame cannot be "
                         f"normalized without one")

        if not self.boundaries:
            return _fail(STATUS_UNRESOLVED_NO_CANONICAL_GAPS,
                         "the canonical gap sequence carries no usable timing",
                         local_time=t_local, global_time=t_global, local_fps=fps)

        slot = bisect.bisect_right(self._boundary_times, t_global)
        left = self.boundaries[slot - 1] if slot > 0 else None
        right = self.boundaries[slot] if slot < len(self.boundaries) else None

        missing = self._gap_missing_between(left, right)
        if missing is not None:
            return _fail(
                STATUS_UNRESOLVED_NO_GAP_TIMING,
                f"GAP_{missing.global_gap_id} has no usable master_time "
                f"({missing.reason or 'unspecified'}), so this interval may "
                f"span more than one wagon",
                local_time=t_local, global_time=t_global, local_fps=fps,
                previous_wagon_id=self._slot_owner[slot],
            )

        # The nearer bracketing gap is the one this detection is measured
        # against; which side of it we are on names the method and the wagon.
        d_left = (abs(t_global - left.master_time) if left else math.inf)
        d_right = (abs(right.master_time - t_global) if right else math.inf)
        if d_right <= d_left:
            gap, method, margin = right, METHOD_BEFORE_GAP, d_right
        else:
            gap, method, margin = left, METHOD_AFTER_GAP, d_left

        # previous/next are the wagons bracketing THE ASSOCIATED GAP, so a
        # BEFORE_GAP assignment always equals previous_wagon_id and an
        # AFTER_GAP assignment always equals next_wagon_id.
        gap_slot = self.boundaries.index(gap)
        prev_id = self._slot_owner[gap_slot]
        next_id = self._slot_owner[gap_slot + 1]
        owner = prev_id if method == METHOD_BEFORE_GAP else next_id

        tol = gap.tolerance_s(cam, self.cfg)
        if off_status not in OFFSET_RESOLVED_STATUSES:
            tol *= self.cfg.assumed_offset_tolerance_factor

        common = dict(
            feature=detection.feature, detection_id=detection.detection_id,
            camera_id=cam, local_frame=int(detection.local_frame),
            local_time=t_local, global_time=t_global, local_fps=fps,
            camera_time_offset=delta, offset_status=off_status,
            associated_global_gap_id=gap.global_gap_id,
            previous_wagon_id=prev_id, next_wagon_id=next_id,
            boundary_margin_s=margin, tolerance_s=tol,
        )

        if margin <= tol:
            return Assignment(
                **common, global_wagon_id=None, method=method,
                status=STATUS_BOUNDARY_AMBIGUOUS, confidence=0.0,
                reason=(f"{margin:.3f}s from GAP_{gap.global_gap_id} is within "
                        f"the {tol:.3f}s tolerance; ownership between "
                        f"{prev_id or '—'} and {next_id or '—'} cannot be "
                        f"called from time alone"),
            )

        if owner is None:
            return Assignment(
                **common, global_wagon_id=None, method=method,
                status=STATUS_UNRESOLVED_OUTSIDE_WAGON_REGION, confidence=0.0,
                reason=(f"the interval {method.lower().replace('_', ' ')} "
                        f"GAP_{gap.global_gap_id} contains no canonical wagon "
                        f"(leading engine or trailing brake van territory)"),
            )

        # The first and last slots are open-ended -- nothing bounds them but the
        # single gap at their inner edge -- so on the bare interval rule a
        # detection minutes past the brake van would still be "in the last
        # wagon's interval".  Ownership is therefore also required to fall
        # inside the owning wagon's OWN canonical span, widened by the same
        # tolerance band so rounding at an edge cannot reject a genuine
        # detection.  Interior slots are covered by their wagon by construction,
        # so this only ever bites outside the train.
        span = self._span_by_id.get(owner)
        if span is not None and not (span.start_time - tol <= t_global
                                     <= span.end_time + tol):
            return Assignment(
                **common, global_wagon_id=None, method=method,
                status=STATUS_UNRESOLVED_OUTSIDE_WAGON_REGION, confidence=0.0,
                reason=(f"normalized time {t_global:.3f}s falls outside "
                        f"{owner}'s canonical span "
                        f"{span.start_time:.3f}-{span.end_time:.3f}s "
                        f"(+/-{tol:.3f}s); it is outside the counted train"),
            )

        # Confidence is purely about distance from the boundary: 0 at the edge
        # of the tolerance band, 1.0 once two band-widths clear of it.  How
        # trustworthy the CLOCK was is a separate axis, reported as
        # `offset_status` / the ASSUMED_OFFSET status rather than folded in.
        conf = max(0.0, min(1.0, margin / (2.0 * tol))) if tol > 0 else 1.0
        assumed = off_status not in OFFSET_RESOLVED_STATUSES
        return Assignment(
            **common, global_wagon_id=owner, method=method,
            status=(STATUS_RESOLVED_ASSUMED_OFFSET if assumed
                    else STATUS_RESOLVED),
            confidence=round(conf, 4),
            reason=(f"{margin:.3f}s {method.lower().replace('_', ' ')} "
                    f"GAP_{gap.global_gap_id}"
                    + (f"; {cam} clock offset unresolved, normalized on the "
                       f"shared-t=0 assumption" if assumed else "")),
        )

    def assign_all(self, detections: Sequence[Detection]) -> List[Assignment]:
        return [self.assign(d) for d in detections or ()]

    # ---- audit ---------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "canonical_wagons": len(self.wagons),
            "canonical_gaps": len(self.gaps),
            "usable_gap_boundaries": len(self.boundaries),
            "gaps_without_timing": [g.global_gap_id for g in self.invalid_gaps],
            "wagon_intervals": [
                {"slot": i, "global_wagon_id": owner,
                 "start_time": (None if math.isinf(self._slot_bounds(i)[0])
                                else round(self._slot_bounds(i)[0], 4)),
                 "end_time": (None if math.isinf(self._slot_bounds(i)[1])
                              else round(self._slot_bounds(i)[1], 4))}
                for i, owner in enumerate(self._slot_owner)
            ],
            "camera_time_offsets": {c: round(v, 4)
                                    for c, v in sorted(self.camera_offsets.items())},
            "camera_offset_statuses": dict(sorted(self.offset_statuses.items())),
            "per_camera_fps": {c: round(float(v), 4)
                               for c, v in sorted(self.per_camera_fps.items())
                               if _finite(v)},
        }
