"""
train_structure.py  --  wagon-only train structure (ENGINE / WAGON / BRAKE_VAN)
==============================================================================

A train looks like

    ENGINE ENGINE ENGINE  WAGON WAGON ... WAGON  BRAKE_VAN

Only the middle region is counted. This module answers three questions:

    1. Which classifier does each camera use?          (camera -> model mapping)
    2. Where does the wagon region start and end?      (get_master_wagon_window)
    3. Which segments are wagons, and what are the     (TrainStructure)
       leading / trailing non-wagon objects?

THE COUNTING RULE
-----------------
    ENGINE is not a wagon.  BRAKE_VAN is not a wagon.
    Neither ever receives a GW id, and neither extends the wagon timeline.

    global wagon timeline = first WAGON .. last WAGON

Everything before the first WAGON is the leading non-wagon region; everything
after the last WAGON is the trailing non-wagon region. Both are preserved as
metadata (for the PDF, the processed videos and diagnostics) -- they are never
deleted, and frames are never re-ordered or re-timed. Only their eligibility to
receive a GW id changes.

HOW IT REUSES EXISTING CODE
---------------------------
`global_alignment.build_global_wagons` is called UNCHANGED to build the segment
list from the validated master gaps: it already applies the `b <= prev`
boundary-collapse rule, the `N gaps -> N+1` segmentation, the classification
inheritance and the leading/trailing gap provenance. This module then *selects*
the wagon-region subset of that output and renumbers it `GW_1..GW_N`. No segment
mathematics is reimplemented.

Consequence for transitions, which is exactly what is wanted:
the ENGINE->WAGON boundary and the WAGON->BRAKE_VAN boundary are not wagon
boundaries, because they bound segments that are outside the wagon region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from global_train_state import (
    CAMERA_LEFT_UP, CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP, CAMERA_RIGHT_UP_TOP,
    GlobalWagon, SegmentClass,
)

# =============================================================================
# Camera -> classifier mapping
# =============================================================================

SIDE_CLASSIFICATION_MODEL = "side_classification.pt"
TOP_CLASSIFICATION_MODEL = "top_classification.pt"
#: LEFT_UP_TOP's OWN classifier, trained on its own overhead view.
#: RIGHT_UP_TOP keeps `top_classification.pt` -- the two top cameras no longer
#: share a classifier, so this mapping is now the only thing that decides which
#: weights each one loads and every reader must go through it.
LEFT_UP_TOP_CLASSIFICATION_MODEL = "ltop.pt"

#: Which classification model each camera uses.
#:
#:   RIGHT_UP      -> side_classification.pt   (master authority, UNCHANGED)
#:   LEFT_UP       -> side_classification.pt   (a side view, same geometry)
#:   RIGHT_UP_TOP  -> top_classification.pt    (UNCHANGED)
#:   LEFT_UP_TOP   -> ltop.pt                  (its own, trained for its view)
#:
#: LEFT_UP keeps the side model because it is a side view with the same geometry
#: as the master; the top models are trained on the overhead view. Note that
#: before top-camera classification existed NO support camera was classified at
#: all, so this mapping only ever adds capability.
#:
#: LEFT_UP_TOP moved off the shared top model once a classifier was trained for
#: its own view. Replacing "the top model" would have moved RIGHT_UP_TOP too,
#: which is wrong: this is a per-camera change, and a classifier applied to the
#: wrong camera still returns confident ENGINE / WAGON / BRAKE_VAN labels -- just
#: the wrong ones, which is far worse than a visible failure.
CAMERA_CLASSIFICATION_MODEL: Dict[str, str] = {
    CAMERA_RIGHT_UP: SIDE_CLASSIFICATION_MODEL,
    CAMERA_LEFT_UP: SIDE_CLASSIFICATION_MODEL,
    CAMERA_RIGHT_UP_TOP: TOP_CLASSIFICATION_MODEL,
    CAMERA_LEFT_UP_TOP: LEFT_UP_TOP_CLASSIFICATION_MODEL,
}


def classification_model_for(camera_id: str) -> str:
    """The classifier `camera_id` must use. Raises on an unknown camera.

    No default: handing a camera another camera's classifier produces confident
    labels from weights nobody chose, and those labels decide which segments are
    excluded from wagon synchronization. A KeyError is the cheap failure.
    """
    try:
        return CAMERA_CLASSIFICATION_MODEL[camera_id]
    except KeyError:
        raise KeyError(
            f"no classification model configured for camera {camera_id!r}; "
            f"known cameras: {sorted(CAMERA_CLASSIFICATION_MODEL)}") from None

TOP_CAMERAS_USING_TOP_MODEL = (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP)


# =============================================================================
# Semantic label mapping, built from the model's ACTUAL class names
# =============================================================================

#: Substrings that identify each semantic class. Matching is done on the real
#: strings returned by `model.names`; class INDICES are never assumed.
_ENGINE_TOKENS = ("engine", "loco", "locomotive", "locono", "engine_head")
_BRAKEVAN_TOKENS = ("brakevan", "brake_van", "brake-van", "guard_van",
                    "guardvan", "tail", "wagon_tail")
_WAGON_TOKENS = ("wagon", "coach", "bogie", "container", "boxn")
_BACKGROUND_TOKENS = ("track", "tracks", "empty_track", "empty", "background",
                      "rail", "rails", "none", "other", "unknown", "nothing")


@dataclass
class LabelMapping:
    """Mapping from one model's real class names to SegmentClass values."""
    model_path: str
    names: Dict[int, str] = field(default_factory=dict)
    mapping: Dict[str, str] = field(default_factory=dict)
    unmapped: List[str] = field(default_factory=list)

    @property
    def task_ok(self) -> bool:
        return bool(self.names)

    def semantic_for(self, raw_label: str) -> str:
        return self.mapping.get((raw_label or "").strip().lower(),
                                SegmentClass.UNKNOWN)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "class_count": len(self.names),
            "names": {int(k): v for k, v in self.names.items()},
            "mapping": dict(self.mapping),
            "unmapped_classes": list(self.unmapped),
        }


def build_label_mapping(model_names: Dict[int, str], model_path: str = "") -> LabelMapping:
    """Build a semantic mapping from a model's real `model.names`.

    Class indices are never assumed. An unrecognised class is mapped to
    ``SegmentClass.UNKNOWN`` and recorded in ``unmapped`` so it can be reported
    -- it is NEVER silently treated as a WAGON, because that would inflate the
    count with whatever the model happens to emit.
    """
    lm = LabelMapping(model_path=model_path,
                      names={int(k): str(v) for k, v in (model_names or {}).items()})
    for raw in lm.names.values():
        key = raw.strip().lower()
        if any(t in key for t in _BRAKEVAN_TOKENS):
            lm.mapping[key] = SegmentClass.BRAKE_VAN
        elif any(t in key for t in _ENGINE_TOKENS):
            lm.mapping[key] = SegmentClass.ENGINE
        elif any(t in key for t in _WAGON_TOKENS):
            lm.mapping[key] = SegmentClass.WAGON
        elif any(t in key for t in _BACKGROUND_TOKENS):
            lm.mapping[key] = SegmentClass.UNKNOWN
        else:
            lm.mapping[key] = SegmentClass.UNKNOWN
            lm.unmapped.append(raw)
    return lm


# Order matters: 'brakevan' contains no 'wagon' substring, but 'wagon_tail'
# contains both 'wagon' and 'tail'. BRAKE_VAN is therefore tested first so a
# tail-of-train label is never mistaken for an ordinary wagon.


# =============================================================================
# The wagon window
# =============================================================================

NON_WAGON_CLASSES = (SegmentClass.ENGINE, SegmentClass.BRAKE_VAN)


@dataclass
class NonWagonObject:
    """One segment outside the wagon region (engine, brake van, or unknown)."""
    classification: str
    classification_confidence: float
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    position: str
    """"leading" | "trailing" | "interior" | "leading_merged".

    `leading_merged` is the end-anchored walk's outcome: a segment the classifier
    labelled ENGINE that the canonical evidence says also holds the first wagon.
    It is COUNTED, unlike leading/trailing, and it is not interior either -- it
    is the region's first unit."""
    segment_index: int           # index in the full pre-selection segment list

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification,
            "classification_confidence": round(self.classification_confidence, 4),
            "start_frame": self.start_frame, "end_frame": self.end_frame,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "position": self.position, "segment_index": self.segment_index,
        }


@dataclass
class WagonWindow:
    """The counted region of the train: first WAGON .. last WAGON."""
    found: bool = False
    reason: str = ""

    first_wagon_segment_index: Optional[int] = None
    last_wagon_segment_index: Optional[int] = None
    wagon_start_frame: Optional[int] = None
    wagon_end_frame: Optional[int] = None
    wagon_start_time: Optional[float] = None
    wagon_end_time: Optional[float] = None

    #: The wagons themselves, renumbered GW_1..GW_N.
    wagon_units: List[GlobalWagon] = field(default_factory=list)

    leading_non_wagon_objects: List[NonWagonObject] = field(default_factory=list)
    """Segments before the first WAGON. Outside the window -> NO GW id."""

    trailing_non_wagon_objects: List[NonWagonObject] = field(default_factory=list)
    """Segments after the last WAGON. Outside the window -> NO GW id."""

    interior_non_wagon_objects: List[NonWagonObject] = field(default_factory=list)
    """ENGINE / BRAKE_VAN labels found INSIDE the wagon window.

    These are recorded as classification ANOMALIES and are STILL COUNTED. The
    RIGHT_UP master gap sequence is authoritative: every segment between the
    first and last wagon is bounded by validated master gaps, so an interior
    engine/brake-van label is a classification error, not grounds to delete a
    master wagon or renumber GW ids.

    Excluding them (the original behaviour) let a single misclassification
    silently remove a wagon from the authoritative count -- classification
    controlling an individual wagon, which it must never do. Classification
    decides only where the window starts and ends."""

    reverse_extended_objects: List[NonWagonObject] = field(default_factory=list)
    """Segments the END-ANCHORED backward walk pulled back into the region.

    Deliberately NOT in `interior_non_wagon_objects`: these sit at the region's
    leading edge, not inside it, and calling them interior would misreport where
    the ambiguity is. They ARE counted (they hold the first wagon), so the
    `wagons + leading + trailing == total_segments` invariant still balances."""

    reverse_anchor: Optional["ReverseAnchor"] = None
    """The backward walk's audit trail, forward boundary and disagreement
    included. None when the reverse derivation did not run."""

    total_segments: int = 0

    @property
    def master_wagon_count(self) -> int:
        return len(self.wagon_units)

    def summary(self) -> Dict[str, Any]:
        def _cls_counts(objs: Sequence[NonWagonObject]) -> Dict[str, int]:
            out: Dict[str, int] = {}
            for o in objs:
                out[o.classification] = out.get(o.classification, 0) + 1
            return out

        return {
            "found": self.found,
            "reason": self.reason,
            "master_wagon_count": self.master_wagon_count,
            "total_segments": self.total_segments,
            "first_wagon_segment_index": self.first_wagon_segment_index,
            "last_wagon_segment_index": self.last_wagon_segment_index,
            "wagon_start_frame": self.wagon_start_frame,
            "wagon_end_frame": self.wagon_end_frame,
            "wagon_start_time": (round(self.wagon_start_time, 4)
                                 if self.wagon_start_time is not None else None),
            "wagon_end_time": (round(self.wagon_end_time, 4)
                               if self.wagon_end_time is not None else None),
            "first_wagon": (self.wagon_units[0].global_id if self.wagon_units else None),
            "last_wagon": (self.wagon_units[-1].global_id if self.wagon_units else None),
            "leading_non_wagon_count": len(self.leading_non_wagon_objects),
            "trailing_non_wagon_count": len(self.trailing_non_wagon_objects),
            "interior_non_wagon_count": len(self.interior_non_wagon_objects),
            "interior_classification_anomalies": len(self.interior_non_wagon_objects),
            "interior_anomalies_are_still_counted": True,
            "leading_non_wagon_classes": _cls_counts(self.leading_non_wagon_objects),
            "trailing_non_wagon_classes": _cls_counts(self.trailing_non_wagon_objects),
            "interior_non_wagon_classes": _cls_counts(self.interior_non_wagon_objects),
            "leading_non_wagon_objects": [o.to_dict()
                                          for o in self.leading_non_wagon_objects],
            "trailing_non_wagon_objects": [o.to_dict()
                                           for o in self.trailing_non_wagon_objects],
            "interior_non_wagon_objects": [o.to_dict()
                                           for o in self.interior_non_wagon_objects],
            "reverse_extended_count": len(self.reverse_extended_objects),
            "reverse_extended_objects": [o.to_dict()
                                         for o in self.reverse_extended_objects],
            "reverse_anchor": (self.reverse_anchor.to_dict()
                               if self.reverse_anchor is not None else None),
            "boundaries_agree": (self.reverse_anchor.boundaries_agree
                                 if self.reverse_anchor is not None else None),
        }


# =============================================================================
# Reverse / end-anchored boundary derivation
# =============================================================================
#
# WHY THE END IS THE ANCHOR
# -------------------------
# Validated on real footage: the trailing WAGON -> BRAKE_VAN/ENGINE transition is
# reliably detected, while the leading ENGINE -> first-WAGON transition is not.
# The physical gap between the locomotive and the first wagon is the one most
# often missed, and when it is missed `build_global_wagons` emits ENGINE and the
# first wagon as ONE segment. That merged segment inherits the ENGINE label, the
# forward rule ("the window starts at the first segment labelled WAGON") starts
# the region at the SECOND wagon, and the first wagon is silently lost into the
# leading non-wagon region.
#
# So the end -- the boundary that can be trusted -- becomes the anchor, and the
# region is walked backwards from it.
#
# WHAT THE REVERSE WALK MAY AND MAY NOT DO
# ----------------------------------------
# It is a boundary/validation mechanism, not a second counting algorithm. It only
# ever RETAINS segments that `build_global_wagons` already produced from the
# validated RIGHT_UP master gaps. It never splits a segment, never invents a
# boundary, never renumbers a gap and never consults a support camera. The most
# it can do is move one already-existing master segment from the leading
# non-wagon list into the counted region.
#
# WHY DURATION ALONE CANNOT DECIDE
# --------------------------------
# The tempting test -- "this leading segment is much longer than a wagon, so it
# must be engine + wagon" -- does not work: a locomotive on its own is ALSO much
# longer than a wagon. Length cannot separate "long engine" from "engine plus
# wagon". What separates them is evidence of a BOUNDARY inside the segment, and
# that evidence already exists: gap validation keeps every candidate it rejected,
# with the frames it spanned (`GapValidationResult.rejected`, persisted as
# `gap_validation.json`). A rejected candidate sitting inside the segment is the
# missed coupling.
#
# The existing WAGON_ACTIVE recovery pass cannot find this one, which is exactly
# why the wagon is lost: it only re-admits soft-failed gaps INSIDE the wagon
# window, and the engine->first-wagon coupling is by definition at that window's
# leading edge, outside it.


@dataclass(frozen=True)
class RejectedGapSpan:
    """One gap candidate that validation rejected, with the frames it spanned.

    Read-only evidence. Nothing here re-admits a gap, changes the master
    sequence or renumbers anything -- the reverse walk only asks "was a boundary
    seen and discarded inside this segment?".
    """
    frame_start: int
    frame_end: int
    reason: str = ""
    soft: bool = False
    track_id: Optional[int] = None

    @property
    def center_frame(self) -> int:
        return (int(self.frame_start) + int(self.frame_end)) // 2

    def to_dict(self) -> Dict[str, Any]:
        return {"frame_start": self.frame_start, "frame_end": self.frame_end,
                "center_frame": self.center_frame, "reason": self.reason,
                "soft": self.soft, "track_id": self.track_id}


def rejected_gap_spans_from_validation(result: Any) -> List[RejectedGapSpan]:
    """Adapter for an in-memory `gap_validation.GapValidationResult`."""
    out: List[RejectedGapSpan] = []
    for r in (getattr(result, "rejected", None) or ()):
        f = getattr(r, "features", None)
        fs = getattr(f, "frame_start", None)
        fe = getattr(f, "frame_end", None)
        if fs is None or fe is None:
            continue
        out.append(RejectedGapSpan(
            frame_start=int(fs), frame_end=int(fe),
            reason=str(getattr(r, "reason", "") or ""),
            soft=bool(getattr(r, "is_soft", False)),
            track_id=getattr(f, "track_id", None)))
    return out


def rejected_gap_spans_from_json(doc: Any) -> List[RejectedGapSpan]:
    """Adapter for a persisted `gap_validation.json`.

    Sequential mode reads the same evidence back off disk that batch mode holds
    in memory, so both pipelines feed the reverse walk identical input.
    """
    out: List[RejectedGapSpan] = []
    if not isinstance(doc, dict):
        return out
    for r in (doc.get("rejections") or ()):
        if not isinstance(r, dict):
            continue
        f = r.get("features") or {}
        fs, fe = f.get("frame_start"), f.get("frame_end")
        if fs is None or fe is None:
            continue
        try:
            out.append(RejectedGapSpan(
                frame_start=int(fs), frame_end=int(fe),
                reason=str(r.get("reason") or ""),
                soft=bool(r.get("soft", False)),
                track_id=f.get("track_id")))
        except (TypeError, ValueError):
            continue
    return out


@dataclass(frozen=True)
class ReverseAnchorConfig:
    """Tunables for the backward walk.

    Every threshold is a RATIO against this train's own median wagon, never a
    frame count and never a number of seconds: the same train runs at different
    speeds on different days, so a fixed offset would be right once.
    """

    merge_duration_ratio: float = 1.4
    """A leading segment must be at least this many median wagons long before it
    can be considered to hold the engine AND a wagon. Below it there is no room
    for two vehicles, whatever the rejected candidate was."""

    min_trailing_wagon_ratio: float = 0.5
    max_trailing_wagon_ratio: float = 1.5
    """The stretch from the rejected boundary to the end of the segment has to
    look like ONE wagon: at least half a median wagon and at most one and a half.

    The lower bound is the obvious one -- there must be room for a wagon behind
    the coupling. The upper bound is the one that does the real work, and it was
    added because without it a bare locomotive was being turned into a wagon: a
    long engine is several median wagons wide, so a soft rejection almost
    anywhere inside it leaves "at least half a wagon" behind it and passes. A
    wagon is one wagon long; a remainder of two or three wagons is not one
    wagon, and one boundary cannot explain it either way."""

    boundary_margin_ratio: float = 0.15
    """A rejected candidate this close (as a fraction of a median wagon) to
    either end of the segment is a re-detection of a boundary already there, not
    a missed interior one."""


DEFAULT_REVERSE_CONFIG = ReverseAnchorConfig()


REVERSE_REASON_NO_SEGMENTS = "NO_SEGMENTS"
REVERSE_REASON_NO_WAGON_LABEL = "NO_SEGMENT_LABELLED_WAGON"
REVERSE_REASON_AGREES = "FORWARD_AND_REVERSE_AGREE"
REVERSE_REASON_EXTENDED = "REVERSE_RETAINED_MERGED_LEADING_WAGON"
REVERSE_REASON_NO_EVIDENCE = "NO_REJECTED_BOUNDARY_EVIDENCE"


@dataclass
class ReverseAnchor:
    """The audit trail of the backward walk, forward boundary included.

    Both boundaries are always reported. When they disagree the disagreement is
    recorded rather than quietly resolved -- a region that moved is exactly what
    a reviewer needs to see.
    """

    found: bool = False
    reason: str = ""

    # ---- the anchor: the reliable trailing boundary ----
    end_frame: Optional[int] = None
    last_wagon_segment_index: Optional[int] = None
    last_wagon: Optional[str] = None

    # ---- the derived leading boundary ----
    start_frame: Optional[int] = None
    first_wagon_segment_index: Optional[int] = None
    first_wagon: Optional[str] = None

    wagons_retained: int = 0
    leading_non_wagon: int = 0
    trailing_non_wagon: int = 0

    # ---- forward vs reverse ----
    forward_first_wagon_segment_index: Optional[int] = None
    forward_start_frame: Optional[int] = None
    boundaries_agree: bool = True
    disagreement: str = ""

    #: Segments the backward walk pulled back into the region, with why.
    extended_segments: List[Dict[str, Any]] = field(default_factory=list)
    #: Candidates considered and refused, with why. Both halves are recorded.
    rejected_extensions: List[Dict[str, Any]] = field(default_factory=list)
    median_wagon_frames: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "reason": self.reason,
            "end_frame": self.end_frame,
            "last_wagon_segment_index": self.last_wagon_segment_index,
            "last_wagon": self.last_wagon,
            "start_frame": self.start_frame,
            "first_wagon_segment_index": self.first_wagon_segment_index,
            "first_wagon": self.first_wagon,
            "wagons_retained": self.wagons_retained,
            "leading_non_wagon": self.leading_non_wagon,
            "trailing_non_wagon": self.trailing_non_wagon,
            "forward_first_wagon_segment_index":
                self.forward_first_wagon_segment_index,
            "forward_start_frame": self.forward_start_frame,
            "boundaries_agree": self.boundaries_agree,
            "disagreement": self.disagreement,
            "extended_segments": list(self.extended_segments),
            "rejected_extensions": list(self.rejected_extensions),
            "median_wagon_frames": (round(self.median_wagon_frames, 2)
                                    if self.median_wagon_frames is not None
                                    else None),
        }

    def render(self) -> str:
        return (
            f"[ACTIVE-REGION] REVERSE-ANCHOR  "
            f"end_frame={self.end_frame}  last_wagon={self.last_wagon}  "
            f"start_frame={self.start_frame}  first_wagon={self.first_wagon}  "
            f"wagons_retained={self.wagons_retained}  "
            f"leading_non_wagon={self.leading_non_wagon}  "
            f"trailing_non_wagon={self.trailing_non_wagon}  "
            f"reason={self.reason}"
        )


def _segment_frames(seg: GlobalWagon) -> int:
    """Length of an INCLUSIVE `[start_frame_master, end_frame_master]` span."""
    return int(seg.end_frame_master) - int(seg.start_frame_master) + 1


def _median(values: Sequence[float]) -> Optional[float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _merge_evidence(
    seg: GlobalWagon,
    *,
    median_wagon_frames: Optional[float],
    rejected: Sequence[RejectedGapSpan],
    cfg: ReverseAnchorConfig,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Does this leading non-wagon segment also contain the first wagon?

    Returns `(retain, reason, detail)`. Every refusal carries its reason, so a
    wagon that stays out can be explained without re-running anything.
    """
    detail: Dict[str, Any] = {
        "segment_start_frame": int(seg.start_frame_master),
        "segment_end_frame": int(seg.end_frame_master),
        "segment_frames": _segment_frames(seg),
        "classification": seg.classification,
        "classification_confidence": round(seg.classification_confidence, 4),
    }

    if median_wagon_frames is None or median_wagon_frames <= 0:
        return False, "NO_WAGON_SCALE", detail

    span = _segment_frames(seg)
    need = cfg.merge_duration_ratio * median_wagon_frames
    detail["median_wagon_frames"] = round(median_wagon_frames, 2)
    detail["required_frames"] = round(need, 2)
    if span < need:
        return False, "TOO_SHORT_TO_HOLD_ENGINE_AND_WAGON", detail

    margin = cfg.boundary_margin_ratio * median_wagon_frames
    min_tail = cfg.min_trailing_wagon_ratio * median_wagon_frames
    max_tail = cfg.max_trailing_wagon_ratio * median_wagon_frames
    detail["boundary_margin_frames"] = round(margin, 2)
    detail["min_trailing_wagon_frames"] = round(min_tail, 2)
    detail["max_trailing_wagon_frames"] = round(max_tail, 2)

    best: Optional[RejectedGapSpan] = None
    n_hard_skipped = 0
    for r in rejected:
        if not r.soft:
            # A HARD rejection is not weak evidence of a boundary, it is a
            # measurement that this is NOT one: no trajectory, static artefact,
            # travelling against the train, an already-accepted duplicate.
            # `gap_validation` calls these "the false-positive defences" and
            # never relaxes them, and `recover_wagon_active_candidates` re-admits
            # only SOFT failures inside the wagon region. The same rule has to
            # hold at the leading edge, or this becomes a way to smuggle back
            # exactly the detections validation exists to throw out -- a long
            # locomotive produces plenty of them.
            n_hard_skipped += 1
            continue
        c = r.center_frame
        if not (int(seg.start_frame_master) + margin <= c
                <= int(seg.end_frame_master) - margin):
            continue                       # at or outside an existing boundary
        tail = int(seg.end_frame_master) - c
        if tail < min_tail or tail > max_tail:
            continue                       # the remainder is not one wagon
        # The LAST such boundary is the coupling closest to the first wagon.
        if best is None or c > best.center_frame:
            best = r
    detail["hard_rejections_ignored"] = n_hard_skipped
    if best is None:
        return False, REVERSE_REASON_NO_EVIDENCE, detail

    detail["rejected_boundary"] = best.to_dict()
    detail["trailing_wagon_frames"] = int(seg.end_frame_master) - best.center_frame
    return True, REVERSE_REASON_EXTENDED, detail


def _as_non_wagon(w: GlobalWagon, idx: int, position: str) -> NonWagonObject:
    return NonWagonObject(
        classification=w.classification,
        classification_confidence=w.classification_confidence,
        start_frame=w.start_frame_master, end_frame=w.end_frame_master,
        start_time=w.start_time, end_time=w.end_time,
        position=position, segment_index=idx)


def get_master_wagon_window(
    segments: Sequence[GlobalWagon],
    *,
    rejected_gap_spans: Optional[Sequence[RejectedGapSpan]] = None,
    reverse_cfg: ReverseAnchorConfig = DEFAULT_REVERSE_CONFIG,
    verbose: bool = True,
) -> WagonWindow:
    """Select the counted wagon region from the master's full segment list.

    Parameters
    ----------
    segments :
        The complete segment list as produced by
        ``global_alignment.build_global_wagons`` from the VALIDATED master gaps.
        Each carries its inherited classification.

    Returns
    -------
    WagonWindow
        ``wagon_units`` are renumbered ``GW_1..GW_N`` and are the ONLY objects
        that receive a global id. Engines and brake vans are preserved in the
        leading / trailing / interior non-wagon lists.

    Rules
    -----
    * The window runs from the FIRST segment classified WAGON to the LAST
      segment classified WAGON, inclusive.
    * Inside the window, a segment classified ENGINE or BRAKE_VAN is EXCLUDED
      from the count (the hard rule: they never receive a GW id) and recorded as
      an interior non-wagon object.
    * Inside the window, a segment classified UNKNOWN is COUNTED. It sits
      between two identified wagons, so it is physically a vehicle the
      classifier could not label; excluding it would silently undercount. It is
      still reported, so the ambiguity stays visible.
    * If no segment is classified WAGON, the window is empty and the wagon count
      is 0. Nothing is invented.
    """
    win = WagonWindow(total_segments=len(segments))

    if not segments:
        win.reason = "no master segments were produced"
        if verbose:
            print("  [WAGONWIN] no segments -> wagon count 0")
        return win

    wagon_idx = [i for i, s in enumerate(segments)
                 if s.classification == SegmentClass.WAGON]

    if not wagon_idx:
        counts: Dict[str, int] = {}
        for s in segments:
            counts[s.classification] = counts.get(s.classification, 0) + 1
        win.reason = (f"no segment was classified WAGON (labels seen: {counts}); "
                      f"wagon count is 0 -- nothing is invented")
        for i, s in enumerate(segments):
            win.leading_non_wagon_objects.append(_as_non_wagon(s, i, "leading"))
        if verbose:
            print(f"  [WAGONWIN] {win.reason}")
        return win

    # ---- END-ANCHORED derivation -------------------------------------
    #
    # The trailing WAGON -> non-wagon transition is the reliable boundary, so it
    # anchors the region; the leading edge is then derived by walking BACKWARDS
    # from it. `lw` is that anchor and is unchanged from the forward rule -- the
    # end was never the problem. `fw` is what the backward walk may move.
    lw = wagon_idx[-1]
    fw_forward = wagon_idx[0]

    anchor = ReverseAnchor(
        found=True,
        end_frame=int(segments[lw].end_frame_master),
        last_wagon_segment_index=lw,
        forward_first_wagon_segment_index=fw_forward,
        forward_start_frame=int(segments[fw_forward].start_frame_master),
    )

    # This train's own wagon scale, measured from the segments the forward rule
    # already accepts. Taken BEFORE any extension, so a candidate can never
    # widen the yardstick it is about to be measured against.
    anchor.median_wagon_frames = _median(
        [_segment_frames(segments[i]) for i in wagon_idx])

    fw = fw_forward
    spans = list(rejected_gap_spans or ())
    i = fw_forward - 1
    while i >= 0:
        seg = segments[i]
        if seg.classification not in NON_WAGON_CLASSES:
            # Only ENGINE / BRAKE_VAN sit between the video start and the first
            # wagon. Anything else here is not a boundary question, so the walk
            # stops rather than guessing.
            anchor.rejected_extensions.append(
                {"segment_index": i, "retained": False,
                 "reason": "NOT_A_NON_WAGON_LABEL",
                 "classification": seg.classification})
            break
        keep, why, detail = _merge_evidence(
            seg, median_wagon_frames=anchor.median_wagon_frames,
            rejected=spans, cfg=reverse_cfg)
        detail["segment_index"] = i
        detail["retained"] = keep
        detail["reason"] = why
        if not keep:
            anchor.rejected_extensions.append(detail)
            break
        anchor.extended_segments.append(detail)
        fw = i
        # A segment holding ENGINE + the first wagon IS the leading boundary:
        # everything before it is pure locomotive. Stopping here is what keeps
        # the walk from eating its way into the engine.
        break

    win.found = True
    win.first_wagon_segment_index = fw
    win.last_wagon_segment_index = lw

    anchor.first_wagon_segment_index = fw
    anchor.start_frame = int(segments[fw].start_frame_master)
    anchor.boundaries_agree = (fw == fw_forward)
    if anchor.boundaries_agree:
        anchor.reason = REVERSE_REASON_AGREES
    else:
        anchor.reason = REVERSE_REASON_EXTENDED
        anchor.disagreement = (
            f"forward boundary starts at segment {fw_forward} "
            f"(frame {anchor.forward_start_frame}); the end-anchored walk starts "
            f"at segment {fw} (frame {anchor.start_frame}), retaining "
            f"{fw_forward - fw} segment(s) the forward rule placed in the "
            f"leading non-wagon region")
    win.reverse_anchor = anchor

    extended_indices = {int(d["segment_index"]) for d in anchor.extended_segments}

    for i, s in enumerate(segments):
        if i < fw:
            win.leading_non_wagon_objects.append(_as_non_wagon(s, i, "leading"))
        elif i > lw:
            win.trailing_non_wagon_objects.append(_as_non_wagon(s, i, "trailing"))
        elif i in extended_indices:
            # Retained by the backward walk. Counted, and recorded separately
            # from the interior anomalies so the audit says where it came from.
            win.reverse_extended_objects.append(_as_non_wagon(s, i, "leading_merged"))
            win.wagon_units.append(s)
        else:
            # INSIDE the window. Every segment here is bounded by validated
            # RIGHT_UP master gaps, which are authoritative, so it counts as a
            # wagon regardless of its label. An ENGINE / BRAKE_VAN label inside
            # the window is recorded as a classification anomaly -- it must not
            # delete a master wagon or renumber GW ids.
            if s.classification in NON_WAGON_CLASSES:
                win.interior_non_wagon_objects.append(
                    _as_non_wagon(s, i, "interior"))
            win.wagon_units.append(s)

    # Renumber the survivors GW_1..GW_N, preserving the existing naming scheme.
    for new_index, w in enumerate(win.wagon_units, start=1):
        w.global_id = f"GW_{new_index}"
        w.wagon_index = new_index

    if win.wagon_units:
        # Read straight off the retained units, never recomputed: the segments
        # own the inclusive/exclusive convention (`end_frame_master` inclusive,
        # `end_time == (end_frame + 1) / fps`) and re-deriving it here is exactly
        # how an off-by-one gets in.
        win.wagon_start_frame = win.wagon_units[0].start_frame_master
        win.wagon_end_frame = win.wagon_units[-1].end_frame_master
        win.wagon_start_time = win.wagon_units[0].start_time
        win.wagon_end_time = win.wagon_units[-1].end_time
        if win.reverse_anchor is not None:
            a = win.reverse_anchor
            a.wagons_retained = len(win.wagon_units)
            a.leading_non_wagon = len(win.leading_non_wagon_objects)
            a.trailing_non_wagon = len(win.trailing_non_wagon_objects)
            a.first_wagon = win.wagon_units[0].global_id
            a.last_wagon = win.wagon_units[-1].global_id
            # The anchor reports the SAME frames the window does.
            a.start_frame = win.wagon_start_frame
            a.end_frame = win.wagon_end_frame
    else:
        win.found = False
        win.reason = ("every segment in the wagon region was ENGINE or BRAKE_VAN; "
                      "wagon count is 0")

    if verbose:
        lead = ", ".join(f"{o.classification}" for o in win.leading_non_wagon_objects) or "none"
        trail = ", ".join(f"{o.classification}" for o in win.trailing_non_wagon_objects) or "none"
        if win.reverse_anchor is not None:
            a = win.reverse_anchor
            print(f"  {a.render()}")
            if not a.boundaries_agree:
                print(f"      FORWARD vs REVERSE DISAGREE: {a.disagreement}")
                for d in a.extended_segments:
                    rb = d.get("rejected_boundary") or {}
                    print(f"      retained segment {d['segment_index']} "
                          f"({d['classification']}, {d['segment_frames']} frames, "
                          f"median wagon {d.get('median_wagon_frames')}): "
                          f"rejected boundary at frame {rb.get('center_frame')} "
                          f"[{rb.get('reason')}] leaves "
                          f"{d.get('trailing_wagon_frames')} frames of wagon")
            else:
                print(f"      forward and reverse boundaries agree "
                      f"(segment {a.forward_first_wagon_segment_index}, "
                      f"frame {a.forward_start_frame})")
            for d in a.rejected_extensions:
                # "no candidates at all" and "candidates existed but validation
                # hard-rejected them" are different findings and the log has to
                # tell them apart -- the second one is the case worth looking at.
                hard = d.get("hard_rejections_ignored") or 0
                extra = (f"  ({hard} hard-rejected candidate(s) ignored: "
                         f"validation measured them as not-a-gap)" if hard else "")
                print(f"      NOT retained: segment {d.get('segment_index')} "
                      f"({d.get('classification')}) -- {d.get('reason')}{extra}")
        print(f"  [WAGONWIN] segments={win.total_segments}  "
              f"wagon region = segment {fw}..{lw}  ->  "
              f"{win.master_wagon_count} wagon(s) GW_1..GW_{win.master_wagon_count}")
        print(f"      leading non-wagon : {lead}")
        print(f"      trailing non-wagon: {trail}")
        if win.interior_non_wagon_objects:
            inner = ", ".join(f"{o.classification}@seg{o.segment_index}"
                              for o in win.interior_non_wagon_objects)
            print(f"      interior non-wagon (excluded from the count): {inner}")
        if win.wagon_start_time is not None:
            print(f"      wagon window: frames {win.wagon_start_frame}-"
                  f"{win.wagon_end_frame}  "
                  f"t={win.wagon_start_time:.2f}-{win.wagon_end_time:.2f}s")

    return win


# =============================================================================
# Support-camera local wagon region
# =============================================================================

def _attach_sample_recorder(clf, mapping: "LabelMapping") -> None:
    """Make a classifier record the per-frame samples behind each segment.

    Needed so the temporal layer can re-vote within a segment with confidence
    weighting. Implemented by wrapping two methods on the instance -- the
    existing sampling logic in ``tracker_engine.MasterClassifier`` is reused
    verbatim, not duplicated, so the two cannot drift apart.

    After ``classify_segments`` the classifier carries
    ``sample_history: {segment_index: [ClassSample, ...]}``.
    """
    from temporal_classification import ClassSample

    clf.sample_history = {}
    clf._frame_buffer = []
    clf._segment_counter = 0
    _orig_frame = clf.classify_frame
    _orig_one = clf._classify_one
    _orig_segments = clf.classify_segments

    def classify_frame(frame):
        raw, conf = _orig_frame(frame)
        clf._frame_buffer.append((raw, float(conf)))
        return raw, conf

    def _classify_one(cap, start_frame, end_frame):
        # _classify_one is invoked exactly once per segment, in segment order,
        # so the samples buffered during this call belong to this segment.
        mark = len(clf._frame_buffer)
        label, conf = _orig_one(cap, start_frame, end_frame)
        idx = clf._segment_counter
        clf._segment_counter += 1
        clf.sample_history[idx] = [
            ClassSample(frame=-1, time=0.0, raw_label=raw,
                        semantic=mapping.semantic_for(raw), confidence=conf)
            for raw, conf in clf._frame_buffer[mark:]
        ]
        return label, conf

    def classify_segments(video_path, segments):
        clf.sample_history = {}
        clf._frame_buffer = []
        clf._segment_counter = 0
        return _orig_segments(video_path, segments)

    clf.classify_frame = classify_frame
    clf._classify_one = _classify_one
    clf.classify_segments = classify_segments


def load_segment_classifier(model_path: str, num_samples: int = 5,
                            verbose: bool = True):
    """Load a classification model and pair it with a mapping of its REAL names.

    Returns ``(classifier, LabelMapping)``. The classifier reuses the existing
    ``tracker_engine.MasterClassifier`` sampling / majority-vote machinery
    unchanged; only the raw-label -> SegmentClass mapping is replaced, so that an
    unrecognised class becomes UNKNOWN and is reported instead of being silently
    counted as a WAGON.

    ``tracker_engine.py`` itself is not modified: this is a subclass.
    """
    from tracker_engine import MasterClassifier

    class _MappedClassifier(MasterClassifier):
        """MasterClassifier with an explicit, model-derived label mapping."""

        def __init__(self, path: str, mapping: LabelMapping, **kw):
            super().__init__(path, **kw)
            self._mapping = mapping

        # Shadows the base staticmethod; the base calls self._label_to_class(...)
        def _label_to_class(self, label: str) -> str:   # type: ignore[override]
            return self._mapping.semantic_for(label)

    probe = MasterClassifier(model_path, num_samples=num_samples, verbose=False)
    mapping = build_label_mapping(probe.class_names, model_path)
    clf = _MappedClassifier(model_path, mapping, num_samples=num_samples,
                            verbose=verbose)
    _attach_sample_recorder(clf, mapping)
    if verbose:
        print(f"  [CLASSIFIER] {model_path}")
        print(f"      task={getattr(clf.model, 'task', '?')}  "
              f"classes={len(mapping.names)}  names={list(mapping.names.values())}")
        print(f"      semantic mapping: "
              f"{ {k: v for k, v in mapping.mapping.items()} }")
        if mapping.unmapped:
            print(f"      ** UNEXPECTED CLASS NAMES (mapped to UNKNOWN, never to "
                  f"WAGON): {mapping.unmapped} **")
    return clf, mapping


@dataclass
class LocalWagonRegion:
    """A support camera's own wagon region, in that camera's LOCAL time."""
    camera_id: str
    classifier_model: str = ""
    found: bool = False
    reason: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    class_counts: Dict[str, int] = field(default_factory=dict)
    segment_labels: List[str] = field(default_factory=list)
    unmapped_classes: List[str] = field(default_factory=list)

    def contains_time(self, t_local: float) -> bool:
        """Is a local instant inside the wagon region?

        When the region is unknown, returns True: a missing classification must
        not silently discard support evidence. Falling back to 'accept' only
        affects evidence association, never the count.
        """
        if not self.found or self.start_time is None or self.end_time is None:
            return True
        return self.start_time <= t_local <= self.end_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "classifier_model": self.classifier_model,
            "found": self.found, "reason": self.reason,
            "start_frame": self.start_frame, "end_frame": self.end_frame,
            "start_time": (round(self.start_time, 4)
                           if self.start_time is not None else None),
            "end_time": (round(self.end_time, 4)
                         if self.end_time is not None else None),
            "class_counts": dict(self.class_counts),
            "segment_labels": list(self.segment_labels),
            "unmapped_classes": list(self.unmapped_classes),
        }


def build_local_wagon_region(
    camera_id: str,
    segments: Sequence[Tuple[int, int]],
    labels: Sequence[str],
    fps: float,
    classifier_model: str = "",
    unmapped_classes: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> LocalWagonRegion:
    """Determine a support camera's local wagon region from its own labels.

    Used to keep engine / brake-van observations out of wagon synchronization.
    It cannot influence the count -- support cameras are evidence only.
    """
    reg = LocalWagonRegion(camera_id=camera_id, classifier_model=classifier_model,
                           unmapped_classes=list(unmapped_classes or []))
    reg.segment_labels = list(labels)
    for lb in labels:
        reg.class_counts[lb] = reg.class_counts.get(lb, 0) + 1

    idx = [i for i, lb in enumerate(labels) if lb == SegmentClass.WAGON]
    if not idx or fps <= 0:
        reg.reason = ("no WAGON segment identified on this camera; "
                      "support evidence is not restricted by region")
        if verbose:
            print(f"  [LOCALWIN/{camera_id}] {reg.reason}")
        return reg

    fw, lw = idx[0], idx[-1]
    reg.found = True
    reg.start_frame = segments[fw][0]
    reg.end_frame = segments[lw][1]
    reg.start_time = reg.start_frame / fps
    reg.end_time = (reg.end_frame + 1) / fps
    if verbose:
        print(f"  [LOCALWIN/{camera_id}] wagon region = segment {fw}..{lw}  "
              f"frames {reg.start_frame}-{reg.end_frame}  "
              f"t={reg.start_time:.2f}-{reg.end_time:.2f}s  "
              f"labels={reg.class_counts}")
    return reg
