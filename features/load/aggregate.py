"""Load aggregation, separated from load inference.

`_aggregate_camera` does two jobs in one loop: it classifies frames pulled from
a materialized wagon cache, and it votes those classifications into a per-camera
LOADED / EMPTY verdict. Only the first needs frames. The second is arithmetic.

This module is the second job alone, reproducing the legacy vote exactly:

    is_loaded = (loaded_count / frames_used) > 0.35

Two details of that rule are easy to get wrong and are preserved deliberately:

* The denominator is `frames_used` -- EVERY frame the classifier looked at --
  not the number of frames that voted. A frame whose label canonicalises to
  neither LOADED nor EMPTY still dilutes the loaded ratio. Switching to
  `n_loaded + n_empty` would make wagons load-positive that the old pipeline
  called empty.
* Confidence is the mean over the WINNING side only, not over all frames.

Nothing here opens a video, loads a model, runs inference, or reads a wagon
cache. `tests/test_load_aggregation_equivalence.py` proves the verdict, the
counts, the confidence and the chosen best frames are identical to the
pre-refactor implementation on a non-empty corpus.

Snapshots: `BestFrameTracker.update()` refuses a `None` frame, so a pure
aggregator with no images cannot populate one. The best frame INDEX per side is
therefore always computed, and the image is attached only when a `frames` map is
supplied -- deferred snapshot resolution, the same shape used for Damage. The
tie-break matches the tracker: strictly-greater wins, so the FIRST frame at the
maximum confidence is kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core import constants as C

#: Legacy threshold, RIGHT_UP_TOP/damage_processor.py:1047. Imported rather
#: than re-declared would be circular; it is asserted equal in the test.
LOADED_RATIO_THRESHOLD = 0.35


def canonical_load(raw: str) -> str:
    """Identical to `features.load.processor._canonical_load`."""
    return C.LOAD_LABEL_TO_STATE.get((raw or "").strip().lower(), C.NO_DATA)


@dataclass
class LoadClassification:
    """One classifier result on one frame. What Phase 1 hands over."""
    frame_idx: int
    class_name: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"frame_idx": int(self.frame_idx),
                "class_name": str(self.class_name),
                "confidence": float(self.confidence)}


@dataclass
class LoadAggregate:
    """Exactly the tuple `_aggregate_camera` returns, named."""
    load_status: str = C.NO_DATA
    confidence: float = 0.0
    frames_used: int = 0
    loaded_count: int = 0
    empty_count: int = 0
    best_loaded: Any = None          # BestFrameTracker
    best_empty: Any = None           # BestFrameTracker
    best_loaded_idx: int = -1
    best_empty_idx: int = -1

    def as_tuple(self) -> Tuple[str, float, int, int, int, Any, Any]:
        """The legacy return shape, positionally identical."""
        return (self.load_status, self.confidence, self.frames_used,
                self.loaded_count, self.empty_count,
                self.best_loaded, self.best_empty)

    def per_camera_row(self) -> Dict[str, Any]:
        """The `per_camera[cam]` dict the processor writes, byte-identical."""
        used = self.frames_used
        return {
            "load_status":  self.load_status,
            "confidence":   round(float(self.confidence), 4),
            "frames_used":  used,
            "loaded_count": self.loaded_count,
            "empty_count":  self.empty_count,
            "loaded_ratio": round(self.loaded_count / used, 4) if used else 0.0,
        }


def classifications_from_observations(observations: Iterable[Any]
                                      ) -> List[LoadClassification]:
    """`core.timeline_evidence.Observation` -> aggregator input."""
    out: List[LoadClassification] = []
    for o in observations or []:
        if getattr(o, "kind", None) != "load":
            continue
        if o.local_frame is None:
            continue
        out.append(LoadClassification(
            frame_idx=int(o.local_frame), class_name=str(o.label or ""),
            confidence=float(o.confidence or 0.0)))
    return out


def aggregate_load_from_classifications(
    classifications: Sequence[LoadClassification],
    *,
    camera_id: str,
    frames: Optional[Dict[int, Any]] = None,
) -> LoadAggregate:
    """Per-camera load verdict from already-collected classifications.

    The body of `_aggregate_camera` with the frame reader and the model call
    removed; every branch, threshold and tie-break is unchanged.

    `classifications` must contain one entry per frame the classifier scored,
    in the order it scored them -- including frames whose label is neither
    LOADED nor EMPTY, because those count toward `frames_used`.
    """
    from features._evidence import BestFrameTracker

    loaded_confs: List[float] = []
    empty_confs: List[float] = []
    used = 0
    best_loaded = BestFrameTracker()
    best_empty = BestFrameTracker()
    best_loaded_idx, best_loaded_score = -1, -1.0
    best_empty_idx, best_empty_score = -1, -1.0
    imgs = frames or {}

    for rec in classifications:
        fi = int(rec.frame_idx)
        cls = str(rec.class_name)
        conf = float(rec.confidence)
        cls_canon = canonical_load(cls)
        used += 1
        if cls_canon == C.LOAD_LOADED:
            loaded_confs.append(conf)
            if conf > best_loaded_score:
                best_loaded_score, best_loaded_idx = conf, fi
            best_loaded.update(score=conf, frame=imgs.get(fi), frame_idx=fi,
                               camera_id=camera_id, class_name=cls,
                               confidence=conf)
        elif cls_canon == C.LOAD_EMPTY:
            empty_confs.append(conf)
            if conf > best_empty_score:
                best_empty_score, best_empty_idx = conf, fi
            best_empty.update(score=conf, frame=imgs.get(fi), frame_idx=fi,
                              camera_id=camera_id, class_name=cls,
                              confidence=conf)

    agg = LoadAggregate(best_loaded=best_loaded, best_empty=best_empty,
                        best_loaded_idx=best_loaded_idx,
                        best_empty_idx=best_empty_idx)

    if used == 0:
        return agg

    n_loaded = len(loaded_confs)
    n_empty = len(empty_confs)
    total = max(1, used)
    loaded_ratio = n_loaded / total

    agg.frames_used = used
    agg.loaded_count = n_loaded
    agg.empty_count = n_empty

    if loaded_ratio > LOADED_RATIO_THRESHOLD and n_loaded > 0:
        agg.load_status = C.LOAD_LOADED
        agg.confidence = float(sum(loaded_confs) / n_loaded)
        return agg
    if n_empty > 0:
        agg.load_status = C.LOAD_EMPTY
        agg.confidence = float(sum(empty_confs) / n_empty)
        return agg
    agg.load_status = C.NO_DATA
    agg.confidence = 0.0
    return agg


def aggregate_load_from_observations(
    observations: Iterable[Any],
    *,
    camera_id: str,
    frames: Optional[Dict[int, Any]] = None,
) -> LoadAggregate:
    """The same, taking `Observation` records straight from Phase 1."""
    return aggregate_load_from_classifications(
        classifications_from_observations(observations),
        camera_id=camera_id, frames=frames)
