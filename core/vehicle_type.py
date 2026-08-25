"""Resolve each canonical Global Wagon's VEHICLE TYPE from the side cameras.

Global Wagon IDENTITY and vehicle TYPE are separate questions, and this module
owns only the second. Identity -- which objects exist, and their order -- is
decided by the RIGHT_UP master reconstruction and is not touched here. This
resolves what each already-established object IS, and records why.

Why a separate layer at all
---------------------------
Type already comes from a side camera. `global_alignment` assigns it at exactly
one place, from `initial_classifications`, which `run_global_count` builds from
the MASTER alone:

    initial_classifications = _classify_master_pre_fusion(...)
    initial_classifications, _ = tcls.apply_temporal_classification(
        initial_classifications, master.fps, camera_id=CAMERA_RIGHT_UP, ...)

and the fixed-master invariant already forbids the rest:

    "total_wagons == the WAGON units of the master's wagon window. ENGINE and
     BRAKE_VAN are preserved as metadata but never receive a GW id and never
     extend the wagon timeline. Support cameras contribute association +
     evidence + diagnostics only."

So a top camera cannot set a type or create a wagon today. What was missing is
narrower: LEFT_UP played no part at all, there was no fallback when RIGHT_UP had
nothing to say, and no record of WHY a wagon carries the type it carries.

The rule
--------
RIGHT_UP is primary and always wins. LEFT_UP corroborates or dissents -- both
recorded, neither decisive -- and becomes the source only where RIGHT_UP has no
classification for that wagon at all.

That is deliberately the conservative choice. `total_wagons` counts WAGON units,
so a rule that let LEFT_UP flip a wagon to ENGINE would change the wagon count
from a support camera's opinion. Under this rule the resolved type equals the
existing type in every case where RIGHT_UP has an opinion, which is the normal
case: the timeline and the count CANNOT move. A test asserts exactly that.

Top-camera predictions are accepted as input and recorded as ignored, so an
audit can show what they said and that it carried no weight. They are never
consulted for the decision.

Frames of reference
-------------------
A wagon's start/end are MASTER times. A side camera's classifications are in
THAT camera's local frames. Mapping between them uses the offset the pipeline
already computed -- `local_time = global_time - delta` -- the same arithmetic
the materializer uses to bucket support evidence. This module does not derive
offsets, does not read video, and runs no model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("core.vehicle_type")

_TAG = "[TYPE-RESOLUTION]"

#: Type authority, in order. RIGHT_UP first: it is the master, and the only
#: camera whose classification may decide. LEFT_UP is the fallback ONLY.
SIDE_AUTHORITY: Tuple[str, ...] = (C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP)

#: Recorded, reported, and never consulted for the decision.
NON_AUTHORITATIVE_CAMERAS: Tuple[str, ...] = tuple(C.TOP_CAMERAS)

PRIMARY = "PRIMARY"
FALLBACK = "FALLBACK"
UNRESOLVED = "UNRESOLVED"

#: Why a top camera's opinion was not used. One string, so a log line and a
#: report field cannot drift apart.
TOP_IGNORED_REASON = "TOP_CAMERA_NON_AUTHORITATIVE"


@dataclass
class TypeDecision:
    """One canonical wagon's resolved type, and the whole basis for it."""

    global_id: str
    resolved_type: str = C.CLASS_WAGON
    source_camera: str = ""
    source_track_id: Optional[Any] = None
    confidence: float = 0.0
    decision: str = UNRESOLVED
    corroborated_by: List[str] = field(default_factory=list)
    dissent: Dict[str, Any] = field(default_factory=dict)
    top_predictions: Dict[str, Any] = field(default_factory=dict)
    previous_type: str = ""
    changed: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_id":        self.global_id,
            "resolved_type":    self.resolved_type,
            "source_camera":    self.source_camera,
            "source_track_id":  self.source_track_id,
            "confidence":       round(float(self.confidence or 0.0), 4),
            "decision":         self.decision,
            "corroborated_by":  list(self.corroborated_by),
            "dissent":          dict(self.dissent),
            "top_predictions":  dict(self.top_predictions),
            "top_ignored_reason": (TOP_IGNORED_REASON
                                   if self.top_predictions else ""),
            "previous_type":    self.previous_type,
            "changed":          bool(self.changed),
            "reason":           self.reason,
        }

    def render(self) -> str:
        bits = [f"{_TAG} {self.global_id} type={self.resolved_type}"]
        if self.source_camera:
            bits.append(f"source={self.source_camera}")
        bits.append(f"confidence={float(self.confidence or 0.0):.3f}")
        bits.append(f"decision={self.decision}")
        return "  ".join(bits)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def classification_for_window(
    records: Sequence[Any], start_local: float, end_local: float,
    fps: float,
) -> Optional[Tuple[str, float, Any, float]]:
    """`(label, confidence, track_id, overlap_seconds)` for a local window.

    The record with the greatest TEMPORAL overlap wins, ties broken by
    confidence. Overlap rather than the window's centre frame: a wagon whose
    centre happens to land in a one-sample burst would otherwise take that
    burst's label over a record covering nearly the whole wagon. Duration is the
    evidence; a single frame is not.
    """
    if fps <= 0 or not records:
        return None
    best = None
    for r in records:
        sf = getattr(r, "start_frame", None)
        ef = getattr(r, "end_frame", None)
        if sf is None or ef is None:
            continue
        ov = _overlap(start_local, end_local, float(sf) / fps,
                      (float(ef) + 1.0) / fps)
        if ov <= 0:
            continue
        conf = float(getattr(r, "confidence", 0.0) or 0.0)
        key = (ov, conf)
        if best is None or key > best[0]:
            best = (key, str(getattr(r, "label", C.CLASS_WAGON) or C.CLASS_WAGON),
                    conf, getattr(r, "segment_index", None), ov)
    if best is None:
        return None
    _key, label, conf, track_id, ov = best
    return label, conf, track_id, ov


def _local_window(wagon: Any, delta: float) -> Tuple[float, float]:
    """A wagon's window in one camera's local clock.

    `local_time = global_time - delta`, the same arithmetic the materializer
    uses to bucket support evidence. Nothing new is derived here.
    """
    return (float(getattr(wagon, "start_time", 0.0) or 0.0) - float(delta or 0.0),
            float(getattr(wagon, "end_time", 0.0) or 0.0) - float(delta or 0.0))


def resolve_wagon(
    wagon: Any,
    *,
    side_classifications: Dict[str, Sequence[Any]],
    camera_fps: Dict[str, float],
    camera_offsets: Optional[Dict[str, float]] = None,
    top_classifications: Optional[Dict[str, Sequence[Any]]] = None,
) -> TypeDecision:
    """Resolve one canonical wagon's type. Never raises."""
    offsets = camera_offsets or {}
    tops = top_classifications or {}
    gw_id = str(getattr(wagon, "global_id", "") or "")
    previous = str(getattr(wagon, "classification", "") or "")
    d = TypeDecision(global_id=gw_id, resolved_type=previous or C.CLASS_WAGON,
                     previous_type=previous)

    # What each SIDE camera says, in its own clock.
    found: Dict[str, Tuple[str, float, Any, float]] = {}
    for cam in SIDE_AUTHORITY:
        recs = side_classifications.get(cam) or []
        fps = float(camera_fps.get(cam) or 0.0)
        s, e = _local_window(wagon, offsets.get(cam, 0.0))
        got = classification_for_window(recs, s, e, fps)
        if got is not None:
            found[cam] = got

    # What the TOP cameras say -- recorded, never consulted.
    for cam in NON_AUTHORITATIVE_CAMERAS:
        recs = tops.get(cam) or []
        fps = float(camera_fps.get(cam) or 0.0)
        s, e = _local_window(wagon, offsets.get(cam, 0.0))
        got = classification_for_window(recs, s, e, fps)
        if got is not None:
            d.top_predictions[cam] = {"type": got[0],
                                      "confidence": round(float(got[1]), 4),
                                      "ignored_reason": TOP_IGNORED_REASON}

    master = found.get(C.CAMERA_RIGHT_UP)
    support = found.get(C.CAMERA_LEFT_UP)

    if master is not None:
        label, conf, track_id, _ov = master
        d.resolved_type, d.confidence = label, conf
        d.source_camera, d.source_track_id = C.CAMERA_RIGHT_UP, track_id
        d.decision = PRIMARY
        d.reason = "RIGHT_UP is the type authority"
        if support is not None:
            if support[0] == label:
                d.corroborated_by.append(C.CAMERA_LEFT_UP)
            else:
                # Recorded, not applied. LEFT_UP flipping a type would change
                # total_wagons from a support camera's opinion.
                d.dissent[C.CAMERA_LEFT_UP] = {
                    "type": support[0],
                    "confidence": round(float(support[1]), 4),
                    "applied": False,
                }
    elif support is not None:
        label, conf, track_id, _ov = support
        d.resolved_type, d.confidence = label, conf
        d.source_camera, d.source_track_id = C.CAMERA_LEFT_UP, track_id
        d.decision = FALLBACK
        d.reason = ("RIGHT_UP has no classification for this wagon; "
                    "LEFT_UP used as the explicit fallback")
    else:
        d.decision = UNRESOLVED
        d.reason = ("no side-camera classification covers this wagon; "
                    "the existing type is kept")

    d.changed = bool(previous and d.resolved_type != previous)
    return d


def resolve_train(
    state: Any,
    *,
    side_classifications: Dict[str, Sequence[Any]],
    camera_fps: Dict[str, float],
    camera_offsets: Optional[Dict[str, float]] = None,
    top_classifications: Optional[Dict[str, Sequence[Any]]] = None,
    apply: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Resolve every canonical wagon's type.  ONE function, both pipelines.

    Iterates `state.wagons` -- the canonical timeline -- so it can neither add
    nor drop a wagon: it only annotates the objects the master reconstruction
    already created. `apply=False` reports without writing.
    """
    decisions: List[TypeDecision] = []
    for w in (getattr(state, "wagons", None) or []):
        try:
            d = resolve_wagon(
                w, side_classifications=side_classifications,
                camera_fps=camera_fps, camera_offsets=camera_offsets,
                top_classifications=top_classifications)
        except Exception as e:                                   # noqa: BLE001
            d = TypeDecision(global_id=str(getattr(w, "global_id", "?")),
                             resolved_type=str(getattr(w, "classification", "")
                                               or C.CLASS_WAGON),
                             decision=UNRESOLVED,
                             reason=f"resolver error: {type(e).__name__}: {e}")
        decisions.append(d)
        if verbose:
            log.info("%s", d.render())
            if d.corroborated_by:
                log.info("%s %s side_support=%s agrees", _TAG, d.global_id,
                         ",".join(d.corroborated_by))
            for cam, info in d.dissent.items():
                log.info("%s %s side_dissent=%s type=%s applied=False", _TAG,
                         d.global_id, cam, info.get("type"))
            for cam, info in d.top_predictions.items():
                log.info("%s %s top_prediction=%s ignored_for_type_authority "
                         "reason=%s", _TAG, d.global_id, info.get("type"),
                         info.get("ignored_reason"))

    # `GlobalWagon` is a FROZEN dataclass -- deliberately: the roster is
    # immutable and `assert_roster_unchanged` guards it. So the type is applied
    # by rebuilding the list with `dataclasses.replace`, which copies every
    # identity field (global_id, wagon_index, frames, times, gaps, supporting
    # cameras) and changes only the two type fields. Identity is preserved by
    # construction rather than by care, and an in-place setattr -- which would
    # have raised FrozenInstanceError and been swallowed -- cannot silently do
    # nothing.
    if apply and any(d.decision in (PRIMARY, FALLBACK) for d in decisions):
        import dataclasses
        by_id = {d.global_id: d for d in decisions}
        rebuilt = []
        for w in (getattr(state, "wagons", None) or []):
            d = by_id.get(str(getattr(w, "global_id", "") or ""))
            if d is None or d.decision == UNRESOLVED:
                rebuilt.append(w)
                continue
            try:
                rebuilt.append(dataclasses.replace(
                    w, classification=d.resolved_type,
                    classification_confidence=float(d.confidence)))
            except Exception as e:                               # noqa: BLE001
                log.warning("%s %s could not apply type: %s: %s", _TAG,
                            d.global_id, type(e).__name__, e)
                rebuilt.append(w)
        try:
            state.wagons = rebuilt
        except Exception as e:                                   # noqa: BLE001
            log.warning("%s could not write the resolved types back: %s: %s",
                        _TAG, type(e).__name__, e)

    changed = [d.global_id for d in decisions if d.changed]
    summary = {
        "wagons":        len(decisions),
        "primary":       sum(1 for d in decisions if d.decision == PRIMARY),
        "fallback":      sum(1 for d in decisions if d.decision == FALLBACK),
        "unresolved":    sum(1 for d in decisions if d.decision == UNRESOLVED),
        "corroborated":  sum(1 for d in decisions if d.corroborated_by),
        "dissenting":    sum(1 for d in decisions if d.dissent),
        "changed":       changed,
        "decisions":     [d.to_dict() for d in decisions],
    }
    if verbose:
        log.info("%s %d wagon(s): primary=%d fallback=%d unresolved=%d "
                 "corroborated=%d dissent=%d changed=%d", _TAG,
                 summary["wagons"], summary["primary"], summary["fallback"],
                 summary["unresolved"], summary["corroborated"],
                 summary["dissenting"], len(changed))
    return summary
