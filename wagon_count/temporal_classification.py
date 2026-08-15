"""
temporal_classification.py  --  track/segment-level temporal classification
===========================================================================

THE PROBLEM
-----------
The classifier is trusted per observation, so one bad prediction can move the
train-structure boundary:

    ... WAGON WAGON  BRAKE_VAN  WAGON WAGON ...
                     ^^^^^^^^^ one 0.6-confidence sample

If that burst lands inside a short segment's 5-sample majority vote it flips the
whole segment's label, and if the flipped segment sits at either end of the
wagon region it shifts FIRST_VALID_WAGON / LAST_VALID_WAGON -- changing the
count for no physical reason.

THE FIX -- TWO LAYERS, BOTH TEMPORAL
------------------------------------
Layer 1, WITHIN a segment: confidence-weighted voting over the sampled frames,
replacing plain majority + alphabetical tiebreak. A 0.6-confidence outlier no
longer outvotes 1.0-confidence agreement.

Layer 2, ACROSS segments: a hysteresis state machine. The stable class only
changes when the challenger accumulates enough confidence-weighted evidence,
measured in SECONDS of train, not in observation count.

    per-frame samples
        |  layer 1: confidence-weighted vote
    per-segment raw label
        |  layer 2: hysteresis + confidence-weighted evidence
    STABLE class intervals
        |
    train structure -> FIRST_VALID_WAGON .. LAST_VALID_WAGON

WHY PERSISTENCE IS MEASURED IN TIME, NOT IN SEGMENT COUNT
--------------------------------------------------------
This matters more than it looks. On this project's real data the genuine brake
van behind the locomotive is ONE segment (frames 182-239, 3.87 s, confidence
0.998), corroborated independently by a second model on a second camera
(RIGHT_UP_TOP: 4.00 s, confidence 1.000). A rule of "require 3 consecutive
segments" would delete it, move FIRST_VALID_WAGON earlier and inflate the count
-- i.e. it would suppress a real vehicle.

Meanwhile the measured noise bursts are single frame samples of 0.33 s at
confidence 0.605-0.645, an order of magnitude shorter and far less confident.

So:
    primary   : accumulated evidence duration >= min_stable_region_s (1.0 s)
    secondary : a RUN of sub-threshold observations may also qualify once it
                reaches switch_persistence (3) consecutive same-class
                observations -- this is where a "3 consecutive" rule is
                meaningful, and it cannot delete a single long region.

SYMMETRY
--------
No class is privileged. WAGON->BRAKE_VAN, BRAKE_VAN->WAGON, ENGINE->WAGON and
WAGON->ENGINE all go through the identical test. The module contains no rule of
the form "brake vans are at the end" and no rule that keeps a class sticky
beyond its evidence.

WHAT THIS MODULE DOES NOT TOUCH
-------------------------------
Nothing about detection, gap tracking, gap validation, the master gap sequence,
the fixed-master invariant, or model weights. `tracker_engine.py` stays
byte-identical: the within-segment vote is provided by a subclass.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from global_train_state import SegmentClass

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TemporalClassificationConfig:
    """Temporal smoothing parameters, all derived from measured behaviour.

    MEASUREMENT (real videos, 848x480 @ 15 fps, every 5th frame classified):

      RIGHT_UP / side_classification.pt
        UNKNOWN    6 samples  2.00 s  conf med 1.000
        ENGINE    23 samples  7.67 s  conf med 1.000
        BRAKE_VAN 11 samples  3.67 s  conf med 1.000   <- genuine
        ENGINE     1 sample   0.33 s  conf     0.605   <- NOISE
        WAGON    259 samples 86.33 s  conf med 1.000

      RIGHT_UP_TOP / top_classification.pt
        UNKNOWN   15 samples  5.00 s  conf med 1.000
        ENGINE    24 samples  8.00 s  conf med 1.000
        BRAKE_VAN 12 samples  4.00 s  conf med 1.000   <- genuine
        WAGON    189 samples 63.00 s  conf med 1.000
        BRAKE_VAN  1 sample   0.33 s  conf     0.645   <- NOISE
        WAGON     59 samples 19.67 s  conf med 1.000

    Genuine regions: >= 2.00 s at confidence ~1.000.
    Noise bursts   : 0.33 s at confidence 0.605-0.645.
    The two populations are separated by ~6x in duration and ~0.35 in
    confidence, so the defaults below sit between them with margin on each side.
    """

    enabled: bool = True
    """False = pass raw labels straight through (previous behaviour)."""

    # ---- layer 2: hysteresis --------------------------------------------
    min_stable_region_s: float = 1.0
    """Accumulated evidence (seconds) a challenger class needs before the stable
    class changes. Measured: shortest genuine region 2.00 s, longest noise burst
    0.33 s -> 1.0 s leaves a 2x margin below real and a 3x margin above noise."""

    switch_persistence: int = 3
    """Consecutive same-class observations that also qualify a switch, for runs
    of observations each shorter than `min_stable_region_s`. Measured: noise
    appears as 1 observation, so 3 is a 3x margin. This is a SECONDARY route --
    a single observation longer than min_stable_region_s switches on its own, so
    this rule can never delete a long genuine region."""

    min_confidence_to_challenge: float = 0.50
    """An observation below this confidence never counts as evidence for a
    switch. Measured noise sat at 0.605-0.645 and genuine regions at ~1.000, so
    this is deliberately permissive: duration does the discriminating, not a
    confidence cutoff. Raising it toward 0.7 would also suppress the measured
    noise, but tuning confidence alone is exactly the wrong fix -- the problem is
    temporal, so the temporal test must carry the decision."""

    min_switch_confidence: float = 0.75
    """Mean confidence the challenging run must reach to take over.

    This is what makes a WEAK challenger lose even if it repeats:

        WAGON 0.99, WAGON 1.00, BRAKE_VAN 0.61, BRAKE_VAN 0.63, WAGON 0.99
            -> mean challenger confidence 0.62 < 0.75  -> stays WAGON

        WAGON 0.99, BRAKE_VAN 0.99, BRAKE_VAN 1.00, BRAKE_VAN 1.00, BRAKE_VAN 1.00
            -> mean 1.00 >= 0.75 and 4 >= switch_persistence -> becomes BRAKE_VAN

    Measured basis: the two noise bursts sit at 0.605 and 0.645, while every
    genuine region's segment confidence is >= 0.90. 0.75 sits between them with
    margin on both sides.

    NOTE this is deliberately NOT the primary mechanism -- persistence and
    duration decide the ordinary case. It exists only to stop a low-confidence
    run from switching on repetition alone. An earlier attempt compared the
    challenger's confidence-seconds against the incumbent's over an equal
    OBSERVATION window, which was abandoned as brittle: it rejected the genuine
    ENGINE(7.67 s) -> BRAKE_VAN(3.67 s) transition simply because the engine
    segment was longer, and comparing confidence densities instead failed on
    0.998 vs 1.000. Both scores are still recorded per transition for
    diagnostics; neither gates the decision."""

    # ---- layer 1: within-segment voting ---------------------------------
    use_confidence_weighted_vote: bool = True
    """Weight each sampled frame by its confidence instead of counting votes.
    Replaces plain majority + alphabetical tiebreak, under which three
    0.6-confidence samples beat two 1.0-confidence samples."""

    def describe(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_TEMPORAL_CONFIG = TemporalClassificationConfig()


# =============================================================================
# Data holders
# =============================================================================

@dataclass
class ClassSample:
    """One classified frame."""
    frame: int
    time: float
    raw_label: str
    semantic: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"frame": self.frame, "time": round(self.time, 4),
                "raw_label": self.raw_label, "semantic": self.semantic,
                "confidence": round(self.confidence, 4)}


@dataclass
class SegmentObservation:
    """One segment, its sampled frames, and its raw (pre-hysteresis) label."""
    segment_index: int
    start_frame: int
    end_frame: int
    fps: float
    raw_label: str = SegmentClass.UNKNOWN
    raw_confidence: float = 0.0
    samples: List[ClassSample] = field(default_factory=list)
    weighted_scores: Dict[str, float] = field(default_factory=dict)
    stable_label: str = SegmentClass.UNKNOWN
    held_by_hysteresis: bool = False

    # ---- provenance / continuity (used when available) ------------------
    track_id: Optional[int] = None
    """Identity of the underlying track, when the caller has one. Lets the
    tracker distinguish 'the same object was momentarily misread' from 'a new
    object entered the region'."""

    track_continuous_with_previous: bool = True
    """False when this observation belongs to a different track than the one
    before it -- which makes a class change physically plausible rather than
    noise, so the tracker requires less temporal evidence for it."""

    center_x: Optional[float] = None
    """Image-plane position, when available, for motion continuity."""

    # ---- tracker state at the moment this observation was consumed ------
    previous_stable_label: Optional[str] = None
    candidate_label: Optional[str] = None
    consecutive_candidate_count: int = 0
    consecutive_candidate_duration_s: float = 0.0
    stable_run_duration_s: float = 0.0

    @property
    def start_time(self) -> float:
        return self.start_frame / self.fps if self.fps > 0 else 0.0

    @property
    def end_time(self) -> float:
        return (self.end_frame + 1) / self.fps if self.fps > 0 else 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        d = {
            "segment_index": self.segment_index,
            "start_frame": self.start_frame, "end_frame": self.end_frame,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "duration_s": round(self.duration_s, 4),
            "raw_label": self.raw_label,
            "raw_confidence": round(self.raw_confidence, 4),
            "stable_label": self.stable_label,
            "changed_by_smoothing": self.raw_label != self.stable_label,
            "held_by_hysteresis": self.held_by_hysteresis,
            "n_samples": len(self.samples),
            "weighted_scores": {k: round(v, 4)
                                for k, v in self.weighted_scores.items()},
        }
        if include_samples:
            d["samples"] = [s.to_dict() for s in self.samples]
        return d


@dataclass
class ClassTransition:
    """One considered class change -- accepted or rejected, always recorded."""
    from_class: str
    to_class: str
    segment_index: int
    frame: int
    time: float
    accepted: bool
    reason: str
    supporting_segments: int = 0
    supporting_duration_s: float = 0.0
    challenger_score: float = 0.0
    incumbent_score: float = 0.0
    mean_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_class": self.from_class, "to_class": self.to_class,
            "segment_index": self.segment_index, "frame": self.frame,
            "time": round(self.time, 4), "accepted": self.accepted,
            "reason": self.reason,
            "supporting_segments": self.supporting_segments,
            "supporting_duration_s": round(self.supporting_duration_s, 4),
            "challenger_score": round(self.challenger_score, 4),
            "incumbent_score": round(self.incumbent_score, 4),
            "mean_confidence": round(self.mean_confidence, 4),
        }


@dataclass
class StableInterval:
    """A maximal run of segments sharing one stable class."""
    classification: str
    first_segment_index: int
    last_segment_index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    n_segments: int
    mean_confidence: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification,
            "first_segment_index": self.first_segment_index,
            "last_segment_index": self.last_segment_index,
            "start_frame": self.start_frame, "end_frame": self.end_frame,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "duration_s": round(self.duration_s, 4),
            "n_segments": self.n_segments,
            "mean_confidence": round(self.mean_confidence, 4),
        }


@dataclass
class TemporalClassificationResult:
    camera_id: str
    observations: List[SegmentObservation] = field(default_factory=list)
    transitions: List[ClassTransition] = field(default_factory=list)
    stable_intervals: List[StableInterval] = field(default_factory=list)
    config_used: Dict[str, Any] = field(default_factory=dict)

    @property
    def stable_labels(self) -> List[str]:
        return [o.stable_label for o in self.observations]

    @property
    def raw_labels(self) -> List[str]:
        return [o.raw_label for o in self.observations]

    @property
    def n_changed(self) -> int:
        return sum(1 for o in self.observations if o.raw_label != o.stable_label)

    def class_counts(self, stable: bool = True) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for o in self.observations:
            lb = o.stable_label if stable else o.raw_label
            out[lb] = out.get(lb, 0) + 1
        return out

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        acc = [t for t in self.transitions if t.accepted]
        rej = [t for t in self.transitions if not t.accepted]
        return {
            "camera_id": self.camera_id,
            "n_segments": len(self.observations),
            "raw_class_counts": self.class_counts(stable=False),
            "stable_class_counts": self.class_counts(stable=True),
            "segments_relabelled_by_smoothing": self.n_changed,
            "n_transitions_accepted": len(acc),
            "n_transitions_rejected": len(rej),
            "transitions": [t.to_dict() for t in self.transitions],
            "stable_intervals": [s.to_dict() for s in self.stable_intervals],
            "observations": [o.to_dict(include_samples) for o in self.observations],
            "config_used": dict(self.config_used),
        }


# =============================================================================
# LAYER 1 -- confidence-weighted vote WITHIN one segment
# =============================================================================

def aggregate_samples(
    samples: Sequence[ClassSample],
    cfg: TemporalClassificationConfig = DEFAULT_TEMPORAL_CONFIG,
) -> Tuple[str, float, Dict[str, float]]:
    """Collapse a segment's sampled frames into one label + confidence.

    FORMULA (confidence-weighted vote):

        score(c) = SUM over samples s with semantic(s) == c  of  confidence(s)

        label    = argmax_c score(c)
        conf     = mean( confidence(s) for s where semantic(s) == label )

    Ties break toward the class with the higher mean confidence, then
    alphabetically, so the result is deterministic.

    With `use_confidence_weighted_vote=False` every sample contributes 1.0
    instead of its confidence, reproducing a plain majority vote.

    Why this matters: measured noise samples sit at 0.605-0.645 while genuine
    ones sit at ~1.000, so under a plain majority three noisy samples outvote
    two confident ones. Weighting by confidence makes 3 x 0.62 = 1.86 lose to
    2 x 1.00 = 2.00.
    """
    if not samples:
        return SegmentClass.UNKNOWN, 0.0, {}

    scores: Dict[str, float] = {}
    for s in samples:
        w = s.confidence if cfg.use_confidence_weighted_vote else 1.0
        scores[s.semantic] = scores.get(s.semantic, 0.0) + w

    def _mean_conf(cls: str) -> float:
        cs = [s.confidence for s in samples if s.semantic == cls]
        return statistics.mean(cs) if cs else 0.0

    best = max(scores.keys(), key=lambda c: (scores[c], _mean_conf(c), -ord(c[0])))
    return best, _mean_conf(best), scores


# =============================================================================
# LAYER 2 -- hysteresis across segments
# =============================================================================

def _evidence_score(obs: Sequence[SegmentObservation], cls: str) -> float:
    """Confidence-weighted evidence, in confidence-seconds.

        score(c) = SUM over observations o with raw_label(o) == c
                       of  confidence(o) * duration_s(o)

    Duration is the weight because an observation's evidential value is how long
    the train was actually seen that way. This is what lets a single 3.87 s
    brake-van segment outweigh a 0.33 s burst without any class-specific rule.
    """
    return sum(o.raw_confidence * o.duration_s
               for o in obs if o.raw_label == cls)


def _window_score(obs: Sequence[SegmentObservation], cls: str, n: int) -> float:
    """Confidence-weighted score of the LAST `n` observations, for class `cls`.

    Used to give the incumbent a fair, equally-sized window to defend itself in.
    """
    if n <= 0:
        return 0.0
    return _evidence_score(obs[-n:], cls)


class TemporalClassificationTracker:
    """Stateful temporal classifier: raw class observations -> stable timeline.

    A first-class component sitting between the classifier and
    ``train_structure``. It maintains, and exposes for diagnostics:

        current stable class          previous stable class
        candidate (challenging) class consecutive candidate evidence
        duration of the current stable run
        the full transition history, accepted AND rejected

    Decision mechanism, applied identically in every direction:

        A challenge (a run of consecutive observations disagreeing with the
        stable class) is accepted only when ALL hold

          (1) every challenging observation >= min_confidence_to_challenge
          (2) the run agrees on a single successor class
          (3) PERSISTENCE: accumulated duration >= min_stable_region_s
                        OR run length >= switch_persistence
                        OR the run starts a NEW track (continuity broken)
          (4) CONFIDENCE: mean challenger confidence >= min_switch_confidence,
                          so a weak run cannot take over by repeating

        score(c) = SUM over observations o of class c  of  confidence(o) x duration_s(o)

    (3) makes a genuine multi-second vehicle switch on its own, so a
    single-segment brake van is never deleted. (4) makes weak-but-repeated noise
    lose to confident history. Track discontinuity is treated as physical
    evidence that the object really did change.
    """

    def __init__(self, camera_id: str = "",
                 cfg: TemporalClassificationConfig = DEFAULT_TEMPORAL_CONFIG):
        self.camera_id = camera_id
        self.cfg = cfg
        self.stable: Optional[str] = None
        self.previous_stable: Optional[str] = None
        self.pending: List[SegmentObservation] = []
        self.consumed: List[SegmentObservation] = []
        self.transitions: List[ClassTransition] = []

    # ------------------------------------------------------------------
    @property
    def stable_run_duration_s(self) -> float:
        d = 0.0
        for o in reversed(self.consumed):
            if o.stable_label != self.stable:
                break
            d += o.duration_s
        return d

    def _annotate(self, o: SegmentObservation) -> None:
        o.previous_stable_label = self.previous_stable
        o.candidate_label = (self.pending[0].raw_label if self.pending else None)
        o.consecutive_candidate_count = len(self.pending)
        o.consecutive_candidate_duration_s = sum(p.duration_s for p in self.pending)
        o.stable_run_duration_s = self.stable_run_duration_s

    def _flush_pending_as(self, cls: str) -> None:
        for p in self.pending:
            p.stable_label = cls
            p.held_by_hysteresis = (cls != p.raw_label)
            self.consumed.append(p)
        self.pending.clear()

    def _evaluate_challenge(self) -> Tuple[bool, str, float, float]:
        """Decide the open challenge. Returns (accept, reason, ch_score, in_score)."""
        cfg = self.cfg
        pending = self.pending
        labels = {p.raw_label for p in pending}
        dur = sum(p.duration_s for p in pending)

        if len(labels) != 1:
            return (False,
                    f"held: the {len(pending)} challenging observation(s) do not "
                    f"agree on a successor class ({sorted(labels)})", 0.0, 0.0)

        challenger = next(iter(labels))
        ch_score = _evidence_score(pending, challenger)
        in_score = _window_score(self.consumed, self.stable, len(pending))

        weak = [p for p in pending
                if p.raw_confidence < cfg.min_confidence_to_challenge]
        new_track = any(not p.track_continuous_with_previous for p in pending)
        long_enough = dur >= cfg.min_stable_region_s
        persistent = len(pending) >= cfg.switch_persistence
        mean_conf = statistics.mean([p.raw_confidence for p in pending])

        if weak:
            return (False,
                    f"rejected: {len(weak)} observation(s) below "
                    f"min_confidence_to_challenge="
                    f"{cfg.min_confidence_to_challenge}", ch_score, in_score)
        if mean_conf < cfg.min_switch_confidence:
            return (False,
                    f"rejected on confidence: challenger {challenger} mean "
                    f"confidence {mean_conf:.3f} < min_switch_confidence="
                    f"{cfg.min_switch_confidence} over {len(pending)} "
                    f"observation(s) / {dur:.2f}s -- a weak run does not take "
                    f"over however often it repeats", ch_score, in_score)
        if not (long_enough or persistent or new_track):
            return (False,
                    f"held by hysteresis: {len(pending)} observation(s) / "
                    f"{dur:.2f}s is under both min_stable_region_s="
                    f"{cfg.min_stable_region_s}s and switch_persistence="
                    f"{cfg.switch_persistence}, and the track is continuous",
                    ch_score, in_score)

        why = ("a new track began" if new_track and not (long_enough or persistent)
               else "duration" if long_enough else "run length")
        return (True,
                f"accepted on {why}: {len(pending)} observation(s) / {dur:.2f}s, "
                f"evidence {ch_score:.2f} vs incumbent {in_score:.2f} "
                f"confidence-seconds", ch_score, in_score)

    def _record(self, accepted: bool, reason: str,
                ch_score: float, in_score: float) -> None:
        first = self.pending[0]
        self.transitions.append(ClassTransition(
            from_class=self.stable or SegmentClass.UNKNOWN,
            to_class=first.raw_label, segment_index=first.segment_index,
            frame=first.start_frame, time=first.start_time, accepted=accepted,
            reason=reason, supporting_segments=len(self.pending),
            supporting_duration_s=sum(p.duration_s for p in self.pending),
            challenger_score=ch_score, incumbent_score=in_score,
            mean_confidence=statistics.mean([p.raw_confidence
                                             for p in self.pending]),
        ))

    # ------------------------------------------------------------------
    def update(self, o: SegmentObservation) -> str:
        """Feed one observation; return the stable class in force after it."""
        self._annotate(o)

        if self.stable is None:                 # opening state
            self.stable = o.raw_label
            o.stable_label = self.stable
            self.consumed.append(o)
            return self.stable

        if o.raw_label == self.stable:
            if self.pending:                    # the challenge was noise
                # Carry the reason the challenge actually failed, so the
                # diagnostics say WHY, not merely that it was abandoned.
                why = getattr(self, "_last_reject_reason", "")
                self._record(False,
                             f"challenge abandoned after {len(self.pending)} "
                             f"observation(s) / "
                             f"{sum(p.duration_s for p in self.pending):.2f}s: "
                             f"the stable class resumed, so the run was noise"
                             + (f" [{why}]" if why else ""),
                             _evidence_score(self.pending,
                                             self.pending[0].raw_label),
                             _window_score(self.consumed, self.stable,
                                           len(self.pending)))
                self._flush_pending_as(self.stable)
            o.stable_label = self.stable
            self.consumed.append(o)
            return self.stable

        self.pending.append(o)
        accept, reason, ch, inc = self._evaluate_challenge()
        if accept:
            challenger = self.pending[0].raw_label
            self._record(True, reason, ch, inc)
            self._flush_pending_as(challenger)
            self.previous_stable, self.stable = self.stable, challenger
            self._last_reject_reason = ""
        else:
            self._last_reject_reason = reason
            o.stable_label = self.stable
            o.held_by_hysteresis = True
        return self.stable

    def finish(self) -> None:
        """Resolve a challenge still open at the end of the sequence."""
        if not self.pending:
            return
        accept, reason, ch, inc = self._evaluate_challenge()
        challenger = self.pending[0].raw_label
        self._record(accept, f"end of sequence: {reason}", ch, inc)
        if accept:
            self._flush_pending_as(challenger)
            self.previous_stable, self.stable = self.stable, challenger
        else:
            self._flush_pending_as(self.stable or challenger)

    def state(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "current_stable_class": self.stable,
            "previous_stable_class": self.previous_stable,
            "candidate_class": (self.pending[0].raw_label if self.pending else None),
            "consecutive_candidate_count": len(self.pending),
            "consecutive_candidate_duration_s": round(
                sum(p.duration_s for p in self.pending), 4),
            "stable_run_duration_s": round(self.stable_run_duration_s, 4),
            "n_transitions": len(self.transitions),
        }


def smooth_class_sequence(
    observations: List[SegmentObservation],
    camera_id: str = "",
    cfg: TemporalClassificationConfig = DEFAULT_TEMPORAL_CONFIG,
    verbose: bool = True,
) -> TemporalClassificationResult:
    """Apply hysteresis to a segment-label sequence.

    The state machine holds a `stable` class. A run of consecutive observations
    disagreeing with it is a *challenge*; the stable class changes only when the
    challenge is supported:

        (1) every challenging observation is >= min_confidence_to_challenge
        (2) the challenging run agrees on one class
        (3) EITHER accumulated duration >= min_stable_region_s
            OR     run length          >= switch_persistence
        (4) mean challenger confidence >= min_switch_confidence

    While a challenge is unsupported the observations keep the stable label and
    are marked `held_by_hysteresis` -- the raw label is never discarded, so
    nothing is hidden.

    Both accepted and rejected transitions are recorded with their evidence.
    """
    result = TemporalClassificationResult(camera_id=camera_id,
                                          observations=observations,
                                          config_used=cfg.describe())
    if not observations:
        return result

    if not cfg.enabled:
        for o in observations:
            o.stable_label = o.raw_label
        result.stable_intervals = build_stable_intervals(observations)
        return result

    tracker = TemporalClassificationTracker(camera_id=camera_id, cfg=cfg)
    for o in observations:
        tracker.update(o)
    tracker.finish()
    result.transitions = tracker.transitions

    result.stable_intervals = build_stable_intervals(observations)

    if verbose:
        acc = sum(1 for t in result.transitions if t.accepted)
        rej = len(result.transitions) - acc
        print(f"  [TEMPORAL/{camera_id}] {len(observations)} segment(s)  "
              f"relabelled={result.n_changed}  "
              f"transitions accepted={acc} rejected={rej}")
        print(f"      raw   : {result.class_counts(stable=False)}")
        print(f"      stable: {result.class_counts(stable=True)}")
        for t in result.transitions:
            mark = "ACCEPT" if t.accepted else "REJECT"
            print(f"      [{mark}] {t.from_class} -> {t.to_class} "
                  f"@ seg {t.segment_index} f{t.frame} t={t.time:.2f}s")
            print(f"               {t.reason}")

    return result


def build_stable_intervals(
    observations: Sequence[SegmentObservation],
) -> List[StableInterval]:
    """Collapse the smoothed labels into maximal same-class intervals."""
    out: List[StableInterval] = []
    for o in observations:
        if out and out[-1].classification == o.stable_label:
            iv = out[-1]
            iv.last_segment_index = o.segment_index
            iv.end_frame = o.end_frame
            iv.end_time = o.end_time
            iv.n_segments += 1
        else:
            out.append(StableInterval(
                classification=o.stable_label,
                first_segment_index=o.segment_index,
                last_segment_index=o.segment_index,
                start_frame=o.start_frame, end_frame=o.end_frame,
                start_time=o.start_time, end_time=o.end_time,
                n_segments=1, mean_confidence=o.raw_confidence))
    # mean confidence per interval, from the contributing observations
    for iv in out:
        cs = [o.raw_confidence for o in observations
              if iv.first_segment_index <= o.segment_index <= iv.last_segment_index]
        iv.mean_confidence = statistics.mean(cs) if cs else 0.0
    return out


# =============================================================================
# Bridge to the existing pipeline
# =============================================================================

def observations_from_classifications(
    classifications: Sequence[Any],
    fps: float,
    sample_history: Optional[Dict[int, List[ClassSample]]] = None,
) -> List[SegmentObservation]:
    """Wrap `_MasterClassification` records as SegmentObservations.

    `sample_history` optionally supplies the per-frame samples behind each
    segment (segment_index -> samples), enabling layer 1. Without it the
    segment's own label/confidence is used as a single observation, so layer 2
    still applies.
    """
    out: List[SegmentObservation] = []
    for c in classifications:
        obs = SegmentObservation(
            segment_index=c.segment_index, start_frame=c.start_frame,
            end_frame=c.end_frame, fps=fps,
            raw_label=c.label, raw_confidence=c.confidence,
        )
        if sample_history and c.segment_index in sample_history:
            obs.samples = list(sample_history[c.segment_index])
        out.append(obs)
    return out


def apply_temporal_classification(
    classifications: Sequence[Any],
    fps: float,
    camera_id: str = "",
    cfg: TemporalClassificationConfig = DEFAULT_TEMPORAL_CONFIG,
    sample_history: Optional[Dict[int, List[ClassSample]]] = None,
    verbose: bool = True,
) -> Tuple[List[Any], TemporalClassificationResult]:
    """Smooth a camera's segment labels.

    Returns `(smoothed_classifications, result)`. The returned classification
    records are copies carrying the STABLE label, so downstream train-structure
    code needs no change: it simply receives temporally consistent labels.
    """
    import dataclasses

    obs = observations_from_classifications(classifications, fps, sample_history)

    # Layer 1: re-vote within each segment when samples are available.
    if cfg.enabled:
        for o in obs:
            if o.samples:
                label, conf, scores = aggregate_samples(o.samples, cfg)
                o.raw_label, o.raw_confidence = label, conf
                o.weighted_scores = scores

    # Layer 2: hysteresis across segments.
    result = smooth_class_sequence(obs, camera_id=camera_id, cfg=cfg,
                                   verbose=verbose)

    smoothed = [dataclasses.replace(c, label=o.stable_label)
                for c, o in zip(classifications, obs)]
    return smoothed, result
