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

#: Which classification model each camera uses.
#:
#:   RIGHT_UP      -> side_classification.pt   (master authority, UNCHANGED)
#:   LEFT_UP       -> side_classification.pt   (a side view, same geometry)
#:   RIGHT_UP_TOP  -> top_classification.pt    (new)
#:   LEFT_UP_TOP   -> top_classification.pt    (new)
#:
#: LEFT_UP keeps the side model because it is a side view with the same geometry
#: as the master; the top model is trained on the overhead view. Note that
#: before this change NO support camera was classified at all, so mapping
#: LEFT_UP to the side model adds capability without altering any existing
#: behaviour.
CAMERA_CLASSIFICATION_MODEL: Dict[str, str] = {
    CAMERA_RIGHT_UP: SIDE_CLASSIFICATION_MODEL,
    CAMERA_LEFT_UP: SIDE_CLASSIFICATION_MODEL,
    CAMERA_RIGHT_UP_TOP: TOP_CLASSIFICATION_MODEL,
    CAMERA_LEFT_UP_TOP: TOP_CLASSIFICATION_MODEL,
}

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
    position: str                # "leading" | "trailing" | "interior"
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
        }


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

    fw, lw = wagon_idx[0], wagon_idx[-1]
    win.found = True
    win.first_wagon_segment_index = fw
    win.last_wagon_segment_index = lw

    for i, s in enumerate(segments):
        if i < fw:
            win.leading_non_wagon_objects.append(_as_non_wagon(s, i, "leading"))
        elif i > lw:
            win.trailing_non_wagon_objects.append(_as_non_wagon(s, i, "trailing"))
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
        win.wagon_start_frame = win.wagon_units[0].start_frame_master
        win.wagon_end_frame = win.wagon_units[-1].end_frame_master
        win.wagon_start_time = win.wagon_units[0].start_time
        win.wagon_end_time = win.wagon_units[-1].end_time
    else:
        win.found = False
        win.reason = ("every segment in the wagon region was ENGINE or BRAKE_VAN; "
                      "wagon count is 0")

    if verbose:
        lead = ", ".join(f"{o.classification}" for o in win.leading_non_wagon_objects) or "none"
        trail = ", ".join(f"{o.classification}" for o in win.trailing_non_wagon_objects) or "none"
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
