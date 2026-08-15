"""Fragment reassembly: rebuild one physical gap from the pieces a tracker split it into.

    GapTracker completed fragments
        -> fragment stitching / reassembly      <-- this module
        -> reconstructed physical gap tracks
        -> existing gap validation, unchanged

WHY THIS LAYER EXISTS
---------------------
A gap crossing the frame is one physical object, but the tracker can emit it as
several short tracks. When the detector misses a frame the object's next centre
lands beyond the association gate, the track is closed, and a new track id opens
where it reappears. Nothing is wrong with the detections or the tracker -- the
gap is plainly visible in every frame -- but the evidence for one object arrives
split across several ids.

Validation then judges each piece on its own. Each piece is shorter than the
minimum track duration, so each is rejected as insufficient temporal evidence,
and the gap they collectively prove is never counted. Measured on a real train:
three consecutive wagon boundaries were lost this way, leaving a single 184-frame
"wagon" where the median is 54 frames.

The fix belongs BEFORE validation, not inside it. Lowering the duration floor
would admit noise and would still mis-handle the second case below. Reassembling
the object first means every existing gate then applies to the whole physical
gap, exactly as it was designed to.

TWO CASES, ONE RULE
-------------------
Measured fragments come in two kinds, and the same compatibility rule resolves
both without special-casing:

  1. Fragments whose neighbours are also fragments. Together they constitute a
     gap that nothing else counted -> they reassemble into one new gap.

  2. Fragments whose compatible neighbour is a track that already passes
     validation on its own. The fragment is the leading edge of a gap that is
     ALREADY counted -> it is absorbed into that track, which stays one gap.

Case 2 is why this cannot be solved by relaxing a threshold: recovering those
fragments as separate gaps would double-count.

THE ASYMMETRY THAT KEEPS THIS SAFE
----------------------------------
Merging is the action that requires evidence; refusing to merge is the safe
default that preserves today's behaviour. So every criterion below is a reason to
merge, and failing any one of them leaves the fragments exactly as the tracker
emitted them -- separate, and subject to the same validation as before.

This layer never accepts a gap. It only decides which observations belong to the
same object. A reassembled track still faces every existing gate (static motion,
direction, coverage, detection holes, confidence, speed, duplicates, minimum
separation) and can still be rejected by any of them.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from global_train_state import GapEvent
from gap_validation import compute_motion_features

__all__ = [
    "FragmentStitchConfig", "DEFAULT_FRAGMENT_STITCH", "StitchChain",
    "StitchResult", "reassemble_fragments",
]


# =============================================================================
# CONFIG -- camera-independent by construction
# =============================================================================

@dataclass
class FragmentStitchConfig:
    """Stitching criteria. Every value is seconds, a frame-width fraction, a
    dimensionless ratio, or a count -- never pixels, frames or a speed. Absolute
    quantities are resolved per camera at runtime from that camera's own width
    and fps, so nothing here is tied to one geometry, train or speed."""

    enabled: bool = True

    # ---- temporal adjacency -------------------------------------------------
    max_seam_seconds: float = 0.27
    """Largest temporal hole across which two fragments may be joined.

    Every genuine seam measured on real data was a single frame (0.067 s at
    15 fps). 0.27 s allows for a few consecutively missed detections while
    staying far below the spacing between distinct gaps: the closest observed
    spacing between real consecutive boundaries was 39 frames (2.6 s), a ~10x
    margin, so this window cannot merge two different wagon gaps."""

    # ---- seam plausibility --------------------------------------------------
    seam_speed_tolerance: float = 3.5
    """How far the seam jump may exceed what the local reference speed predicts
    over the seam interval.

    Provenance (measured, one camera, 66 candidates, 643 within-track steps):
    genuine seams spanned 106.9-129.6 px against a reference per-frame step of
    43.7 px, i.e. 2.45x-2.97x. Steps occurring INSIDE tracks that already pass
    validation reach 2.05x at p99 and 4.20x at maximum. 3.5x therefore covers
    every observed genuine seam with margin while still admitting nothing more
    extreme than the jitter already tolerated within a single accepted track."""

    max_seam_frac: float = 0.30
    """Hard cap on seam displacement as a fraction of frame width, independent
    of the speed prediction. Stops a long seam interval from authorising a jump
    across the frame. Observed genuine seams peaked at 0.135 of frame width, so
    this leaves ~2.2x margin."""

    min_seam_floor_frac: float = 0.005
    """Floor under the predicted seam displacement, as a fraction of frame width.

    Without it a near-zero reference speed (a stopped train) would predict a
    zero-length seam and make stitching impossible exactly when fragments sit on
    top of each other. With it, two fragments of a stalled gap can still be
    rejoined."""

    # ---- fragment eligibility ----------------------------------------------
    min_hits_to_stitch: int = 2
    """A fragment needs at least two detections to have a measurable direction.

    This is also the guarantee that no raw detection can bypass tracking: a
    one-frame detection is never eligible to stitch, so it can never contribute
    to a reassembled gap, and on its own it still fails validation."""

    min_fragments_for_direction: int = 3
    """Fragments needed before a dominant direction is trusted. Below this the
    camera's flow is unknown and nothing is stitched -- the conservative default."""

    reference_window: int = 5
    """Temporal neighbours used for the LOCAL reference advance at a seam.

    Local rather than global because the train accelerates, decelerates, stops
    and resumes; a global median would mispredict every seam recorded while the
    speed differed from it."""

    reference_min_seconds: float = 0.27
    """Minimum track duration to help measure the reference advance rate.

    A FRAGMENT CANNOT MEASURE THE TRAIN'S SPEED, and this is the subtlety that
    makes the whole layer work. Measured on real data, tracks of exactly 3 hits
    advance 29.7-41.7 px/frame (median 34.2) while every track long enough to
    pass validation on its own advances at a median of 47.7 px/frame. Two effects
    compound: a short track's displacement spans one fewer interval than its
    duration suggests, and the detector's centre advances in a stick-then-jump
    pattern whose jumps a 3-frame window never contains.

    Estimating a seam from the fragments beside it therefore under-predicts by
    ~2x and refuses exactly the seams it should accept -- observed directly: a
    reference of 341.8 px/s where the true local speed was ~750 px/s.

    This deliberately equals the validator's `min_track_seconds`, because the
    population to trust for a speed measurement is precisely the one whose
    duration validation already accepts."""

    min_reference_tracks: int = 3
    """Reliable tracks needed before any seam is predicted. Below this the
    train's speed is unknown, so nothing is stitched -- preserving exactly the
    behaviour that exists without this layer."""


DEFAULT_FRAGMENT_STITCH = FragmentStitchConfig()


@dataclass
class ResolvedStitchThresholds:
    """`FragmentStitchConfig` resolved into one camera's pixels and frames."""
    frame_width: int
    fps: float
    max_seam_frames: int
    max_seam_px: float
    min_seam_floor_px: float
    reference_min_hits: int

    def to_dict(self) -> Dict[str, Any]:
        return {"frame_width": self.frame_width, "fps": round(self.fps, 4),
                "max_seam_frames": self.max_seam_frames,
                "max_seam_px": round(self.max_seam_px, 3),
                "min_seam_floor_px": round(self.min_seam_floor_px, 3),
                "reference_min_hits": self.reference_min_hits}


def resolve_stitch_thresholds(cfg: FragmentStitchConfig, frame_width: int,
                              fps: float) -> ResolvedStitchThresholds:
    w = int(frame_width) if frame_width and frame_width > 0 else 0
    f = float(fps) if fps and fps > 0 else 0.0
    return ResolvedStitchThresholds(
        frame_width=w, fps=f,
        max_seam_frames=max(1, int(round(cfg.max_seam_seconds * f))) if f else 1,
        max_seam_px=cfg.max_seam_frac * w,
        min_seam_floor_px=cfg.min_seam_floor_frac * w,
        reference_min_hits=(max(2, int(round(cfg.reference_min_seconds * f)))
                            if f else 2))


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class StitchChain:
    """One reconstructed physical gap and the fragments it was rebuilt from."""
    member_track_ids: List[int]
    merged: GapEvent
    seams: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_reassembled(self) -> bool:
        return len(self.member_track_ids) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {"track_id": self.merged.track_id,
                "member_track_ids": list(self.member_track_ids),
                "start_frame": self.merged.start_frame,
                "end_frame": self.merged.end_frame,
                "hit_count": self.merged.hit_count,
                "confidence": round(self.merged.confidence, 4),
                "seams": list(self.seams)}


@dataclass
class StitchResult:
    """What reassembly produced for one camera."""
    camera_id: str
    events: List[GapEvent] = field(default_factory=list)
    """The candidate set to validate: reassembled tracks plus untouched singletons."""

    chains: List[StitchChain] = field(default_factory=list)
    input_count: int = 0
    dominant_direction: int = 0
    reference_advance: Optional[float] = None
    resolved_thresholds: Dict[str, Any] = field(default_factory=dict)
    rejected_seams: List[Dict[str, Any]] = field(default_factory=list)
    """Pairs considered and refused, with the criterion that refused them. Kept
    so a missed reassembly can be diagnosed without re-running the tracker."""

    @property
    def reassembled_count(self) -> int:
        return sum(1 for c in self.chains if c.is_reassembled)

    @property
    def absorbed_fragment_count(self) -> int:
        return sum(len(c.member_track_ids) - 1
                   for c in self.chains if c.is_reassembled)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "input_candidates": self.input_count,
            "output_candidates": len(self.events),
            "reassembled_gaps": self.reassembled_count,
            "fragments_absorbed": self.absorbed_fragment_count,
            "dominant_direction": self.dominant_direction,
            "reference_advance_px_per_frame": (round(self.reference_advance, 2)
                                              if self.reference_advance else None),
            "resolved_thresholds": dict(self.resolved_thresholds),
            "chains": [c.to_dict() for c in self.chains if c.is_reassembled],
            "rejected_seams": list(self.rejected_seams),
        }


# =============================================================================
# INTERNALS
# =============================================================================

@dataclass
class _Frag:
    """A candidate plus the measurements stitching decides on."""
    event: GapEvent
    order: int
    start_frame: int
    end_frame: int
    x_start: float
    x_end: float
    hits: int
    direction: int
    speed: float
    advance_px_per_frame: float
    """|displacement| divided by the number of frame INTERVALS the track spans.

    The per-interval form is what a seam prediction needs, and it is independent
    of how track duration is defined."""

    @property
    def center_frame(self) -> float:
        return (self.start_frame + self.end_frame) / 2.0


def _describe(gaps: Sequence[GapEvent]) -> List[_Frag]:
    """Measure each candidate. Order is by (start_frame, track_id) so the whole
    pass is deterministic regardless of the caller's ordering."""
    frags: List[_Frag] = []
    ordered = sorted(enumerate(gaps),
                     key=lambda p: (p[1].start_frame, p[1].track_id))
    for order, (_, g) in enumerate(ordered):
        traj = [float(x) for x in (g.center_x_trajectory or [])]
        if len(traj) < 1:
            continue
        f = compute_motion_features(g)
        if f is None:
            # One hit, or no fps: no direction can be measured, so it can never
            # be stitched. It stays in the output untouched and validation
            # rejects it as it does today.
            frags.append(_Frag(event=g, order=order, start_frame=g.start_frame,
                               end_frame=g.end_frame, x_start=traj[0],
                               x_end=traj[-1], hits=g.hit_count, direction=0,
                               speed=0.0, advance_px_per_frame=0.0))
            continue
        intervals = max(1, len(traj) - 1)
        frags.append(_Frag(event=g, order=order, start_frame=f.frame_start,
                           end_frame=f.frame_end, x_start=f.center_start,
                           x_end=f.center_end, hits=f.hits,
                           direction=f.direction,
                           speed=f.velocity_px_per_sec,
                           advance_px_per_frame=f.abs_displacement_px / intervals))
    return frags


def _dominant_direction(frags: Sequence[_Frag], cfg: FragmentStitchConfig) -> int:
    """The camera's flow, from the candidates themselves.

    Weighted by hits so long, well-observed tracks outvote brief ones. Returns 0
    when there is not enough evidence, which disables stitching entirely."""
    votes = [(f.direction, f.hits) for f in frags if f.direction != 0]
    if len(votes) < cfg.min_fragments_for_direction:
        return 0
    plus = sum(h for d, h in votes if d > 0)
    minus = sum(h for d, h in votes if d < 0)
    if plus == minus:
        return 0
    return 1 if plus > minus else -1


def _local_reference_advance(frags: Sequence[_Frag], at_frame: float,
                             cfg: FragmentStitchConfig,
                             thr: ResolvedStitchThresholds) -> Optional[float]:
    """Median advance in PIXELS PER FRAME INTERVAL near `at_frame`.

    Per frame interval, not per second, because that is exactly the quantity a
    seam prediction needs -- it removes any dependence on how a track's duration
    is defined, which is one of the two biases that made a fragment-derived
    reference under-predict.

    Measured only from tracks of at least `reference_min_hits`, which is the
    other half of the fix: short fragments systematically understate the advance
    rate (see `reference_min_seconds`), so including them would bias the very
    prediction used to decide whether a fragment should be rejoined.

    Local, not global: the train accelerates, decelerates, stops and resumes, so
    the seam is predicted from the motion recorded around it."""
    pool = [(abs(f.center_frame - at_frame), f.advance_px_per_frame)
            for f in frags
            if f.hits >= thr.reference_min_hits and f.advance_px_per_frame > 0]
    if len(pool) < cfg.min_reference_tracks:
        return None
    pool.sort(key=lambda p: p[0])
    window = [a for _, a in pool[:max(1, cfg.reference_window)]]
    return statistics.median(window)


def _seam_verdict(a: _Frag, b: _Frag, frags: Sequence[_Frag],
                  direction: int, cfg: FragmentStitchConfig,
                  thr: ResolvedStitchThresholds) -> Tuple[bool, str, Dict[str, Any]]:
    """Do A (earlier) and B (later) belong to the same physical gap?

    Returns (ok, reason_if_not, measurements)."""
    seam_frames = b.start_frame - a.end_frame
    seam_disp = b.x_start - a.x_end
    info: Dict[str, Any] = {
        "from_track_id": a.event.track_id, "to_track_id": b.event.track_id,
        "seam_frames": seam_frames, "seam_displacement_px": round(seam_disp, 2),
    }

    # Both pieces must be trackable objects, not lone detections.
    if a.hits < cfg.min_hits_to_stitch or b.hits < cfg.min_hits_to_stitch:
        return False, (f"a fragment has fewer than {cfg.min_hits_to_stitch} hits, "
                       f"so it has no measurable direction and can never be "
                       f"stitched"), info

    # Strictly sequential. Fragments that overlap in time are either duplicate
    # tracks or two simultaneous gaps; the existing duplicate rule owns that
    # case, and merging them here would fabricate motion.
    if seam_frames < 1:
        return False, "fragments overlap in time rather than following each other", info
    if seam_frames > thr.max_seam_frames:
        return False, (f"temporal hole {seam_frames} frames exceeds "
                       f"{thr.max_seam_frames} ({cfg.max_seam_seconds}s)"), info

    # Each piece must move with the camera's flow...
    if a.direction != direction or b.direction != direction:
        return False, "a fragment does not move in the camera's dominant direction", info

    # ...and B must be AHEAD of A along that flow. A fragment behind its
    # predecessor is a different object, not a continuation.
    if seam_disp == 0 or (seam_disp > 0) != (direction > 0):
        return False, ("the later fragment is not ahead of the earlier one along "
                       "the direction of travel, so they are different objects"), info

    # The jump must match what the LOCAL speed predicts over the seam interval.
    ref = _local_reference_advance(
        frags, (a.end_frame + b.start_frame) / 2.0, cfg, thr)
    info["local_reference_advance_px_per_frame"] = round(ref, 2) if ref else None
    if ref is None:
        return False, (f"fewer than {cfg.min_reference_tracks} tracks long enough "
                       f"to measure the advance rate, so no seam can be predicted"), info
    predicted = max(ref * seam_frames, thr.min_seam_floor_px)
    allowed = predicted * cfg.seam_speed_tolerance
    info["predicted_seam_px"] = round(predicted, 2)
    info["allowed_seam_px"] = round(allowed, 2)
    info["seam_ratio"] = round(abs(seam_disp) / predicted, 3) if predicted else None
    if abs(seam_disp) > allowed:
        return False, (f"seam jump {abs(seam_disp):.1f}px exceeds "
                       f"{cfg.seam_speed_tolerance}x the {predicted:.1f}px the local "
                       f"advance rate predicts over {seam_frames} frame(s)"), info
    if abs(seam_disp) > thr.max_seam_px:
        return False, (f"seam jump {abs(seam_disp):.1f}px exceeds the "
                       f"{cfg.max_seam_frac:.2f} frame-width cap "
                       f"({thr.max_seam_px:.1f}px)"), info
    return True, "", info


def _merge(members: Sequence[_Frag]) -> GapEvent:
    """Build one GapEvent from an ordered chain of fragments.

    Observations are unioned by frame and re-sorted, so the merged track carries
    exactly the evidence the tracker recorded -- nothing synthesised. Confidence
    is the hits-weighted mean of the members, so a long confident piece is not
    outvoted by a brief uncertain one."""
    by_frame: Dict[int, Tuple[float, Optional[List[float]]]] = {}
    for m in members:
        g = m.event
        traj = [float(x) for x in (g.center_x_trajectory or [])]
        frames = list(g.hit_frames or [])
        if not frames:
            # No per-hit frame record: fall back to the track's own span so the
            # observations still land in the right order.
            frames = list(range(g.start_frame, g.start_frame + len(traj)))
        bboxes = list(g.bbox_history or [])
        for i, fr in enumerate(frames[:len(traj)]):
            if fr in by_frame:
                continue                      # first writer wins: deterministic
            by_frame[fr] = (traj[i],
                            list(bboxes[i]) if i < len(bboxes) else None)

    frames_sorted = sorted(by_frame)
    traj = [by_frame[f][0] for f in frames_sorted]
    bboxes = [by_frame[f][1] for f in frames_sorted if by_frame[f][1] is not None]

    start = min(m.start_frame for m in members)
    end = max(m.end_frame for m in members)
    span = max(1, end - start + 1)
    total_hits = sum(m.hits for m in members) or len(frames_sorted)
    conf = (sum(m.event.confidence * m.hits for m in members) / total_hits
            if total_hits else members[0].event.confidence)

    first = members[0].event
    return GapEvent(
        track_id=min(m.event.track_id for m in members),
        camera_id=first.camera_id,
        start_frame=start, end_frame=end,
        confidence=conf,
        hit_count=len(frames_sorted),
        center_x_trajectory=traj,
        fps=first.fps,
        temporal_consistency_score=min(1.0, len(frames_sorted) / span),
        hit_frames=frames_sorted,
        bbox_history=bboxes,
        class_label=first.class_label,
    )


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def reassemble_fragments(
    gaps: Sequence[GapEvent],
    camera_id: str,
    cfg: FragmentStitchConfig = DEFAULT_FRAGMENT_STITCH,
    frame_width: int = 0,
    fps: float = 0.0,
    verbose: bool = True,
) -> StitchResult:
    """Rebuild physical gaps from tracker fragments, before validation runs.

    Pure function of its inputs: it holds no module state, mutates no input
    event, and returns a fresh candidate list. Two trains processed in the same
    process therefore cannot influence each other.
    """
    result = StitchResult(camera_id=camera_id, input_count=len(gaps))

    if not fps:
        fps = next((g.fps for g in gaps if g.fps), 0.0)
    thr = resolve_stitch_thresholds(cfg, frame_width, fps)
    result.resolved_thresholds = thr.to_dict()

    if not cfg.enabled or len(gaps) < 2 or not thr.fps:
        result.events = list(gaps)
        if verbose and not cfg.enabled:
            print(f"  [STITCH/{camera_id}] disabled -- "
                  f"{len(gaps)} candidate(s) passed through unchanged")
        return result

    frags = _describe(gaps)
    direction = _dominant_direction(frags, cfg)
    result.dominant_direction = direction
    result.reference_advance = _local_reference_advance(
        frags, statistics.median([f.center_frame for f in frags]), cfg, thr
    ) if frags else None

    if direction == 0:
        result.events = list(gaps)
        if verbose:
            print(f"  [STITCH/{camera_id}] dominant direction undetermined "
                  f"({len(frags)} candidate(s)) -- nothing stitched")
        return result

    # ---- link each fragment to its successor -------------------------------
    #
    # Single forward pass in temporal order. `successor[i] = j` means fragment j
    # continues fragment i. A fragment is claimed at most once as a successor,
    # so chains are simple paths and the result cannot depend on iteration
    # order.
    n = len(frags)
    successor: Dict[int, int] = {}
    claimed: set = set()
    seam_info: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for i in range(n):
        a = frags[i]
        best: Optional[int] = None
        for j in range(i + 1, n):
            b = frags[j]
            if b.start_frame - a.end_frame > thr.max_seam_frames:
                break                           # ordered by start: no later one fits
            # Overlapping pairs fall through to _seam_verdict rather than being
            # skipped here, so that refusal is RECORDED. Silently dropping them
            # would leave a missed reassembly with no explanation in the
            # diagnostics, which is the one thing this record exists to prevent.
            if j in claimed:
                continue
            ok, why, info = _seam_verdict(a, b, frags, direction, cfg, thr)
            if ok:
                best = j
                seam_info[(i, j)] = info
                break                           # nearest compatible wins
            info = dict(info)
            info["refused_because"] = why
            result.rejected_seams.append(info)
        if best is not None:
            successor[i] = best
            claimed.add(best)

    # ---- walk the chains ---------------------------------------------------
    chain_of: List[Optional[List[int]]] = [None] * n
    for i in range(n):
        if i in claimed or chain_of[i] is not None:
            continue                            # not a chain head
        chain: List[int] = [i]
        cur = i
        while cur in successor:
            cur = successor[cur]
            chain.append(cur)
        for k in chain:
            chain_of[k] = chain

    seen: set = set()
    events: List[GapEvent] = []
    for i in range(n):
        chain = chain_of[i]
        if chain is None or id(chain) in seen:
            continue
        seen.add(id(chain))
        members = [frags[k] for k in chain]
        if len(members) == 1:
            merged = members[0].event            # untouched: same object identity
        else:
            merged = _merge(members)
        sc = StitchChain(
            member_track_ids=[m.event.track_id for m in members],
            merged=merged,
            seams=[seam_info[(chain[t], chain[t + 1])]
                   for t in range(len(chain) - 1)
                   if (chain[t], chain[t + 1]) in seam_info])
        result.chains.append(sc)
        events.append(merged)

    events.sort(key=lambda g: (g.start_frame, g.track_id))
    result.events = events

    if verbose:
        if result.reassembled_count:
            print(f"  [STITCH/{camera_id}] {result.input_count} candidate(s) -> "
                  f"{len(result.events)}: reassembled {result.reassembled_count} "
                  f"physical gap(s) from {result.absorbed_fragment_count + result.reassembled_count} "
                  f"fragment(s)")
            for c in result.chains:
                if not c.is_reassembled:
                    continue
                print(f"      tracks {c.member_track_ids} -> "
                      f"frames {c.merged.start_frame}-{c.merged.end_frame} "
                      f"({c.merged.hit_count} hits, conf {c.merged.confidence:.2f})")
        else:
            print(f"  [STITCH/{camera_id}] {result.input_count} candidate(s), "
                  f"no fragments to reassemble")
    return result
