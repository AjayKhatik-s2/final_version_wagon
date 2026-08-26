"""Door aggregation, separated from door inference.

The sampled Door path scores frames from a materialized wagon cache AND builds
the per-camera decisions, evidence buckets and overlay trajectory in one loop.
Only the scoring needs frames. Everything after it is arithmetic over
detections, and welding the two together is what forces Door inference to wait
for a wagon cache to exist.

This module is the second half alone. It returns the SAME 6-tuple
`_run_sampled_one_camera` returns, so `run()` is unchanged for both modes.

Where the seam falls, and why there:

* `detection_quality(frame, bbox)` genuinely reads pixels, so it is the last
  thing that happens at collection time. Its scalar result travels with the
  detection. Everything downstream of it -- `snapshot_score`, `expand_bbox`,
  the class-to-state mapping, the aggregator, the bucket ranking -- is pure and
  lives here.
* A frame whose inference RAISED is counted in `used` but is never declared to
  the aggregator (the old loop `continue`s past `add_frame`). A frame that was
  scored and yielded nothing IS declared as empty. `DoorFrameRecord.errored`
  keeps the two apart because that is what the old code does and because `used`
  is reported -- NOT because it changes the decisions. Measured: declaring the
  empty frames does not move the accepted set at any hit-to-frame ratio, so
  `EvidenceAggregator.add_frame(fi, [])` is behaviourally inert here. The same
  holds for Damage. Preserved for fidelity, not for effect.
* The `_canonical()` bucket-key quirk is replicated deliberately, exactly as
  the old path notes: the aggregator groups on `DOOR_LABEL_TO_STATE.get(raw,
  _canonical(raw))` while the evidence bucket keys on `_canonical(raw)` alone.
  Fixing that here would conflate a refactor with a behaviour change.
* The overlay `trajectory` keys on `len(trajectory) + 1`, which never collides
  with an existing key, so every detection becomes its own single-frame track.
  That is what the old code does and what the renderer already consumes.

Snapshots are deferred exactly as in Load: `BestFrameTracker.update()` refuses a
`None` frame, so the winning frame INDEX and its metadata are always computed
and the image is attached only when a `frames` map is supplied.

Nothing here opens a video, loads a model, runs inference, or reads a wagon
cache. `tests/test_door_aggregation_equivalence.py` proves the decisions, the
evidence buckets and the overlay are identical to the pre-refactor
implementation on a non-empty corpus.

Scope: this extracts the SAMPLED path. The legacy tracker path
(`_run_tracker_one_camera`) drives DoorTracker and DoorIdentityMerger, which
consume frames internally to build snapshots and cannot be made pure without
changing those modules -- out of scope here, and untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core import constants as C
from core.frame_quality import (
    snapshot_score, expand_bbox, _DOOR_BBOX_EXPAND_FRAC,
)


def canonical_door(state_value: str) -> str:
    """Identical to `features.door.processor._canonical`."""
    from features.door.processor import _canonical
    return _canonical(state_value)


@dataclass
class DoorDetection:
    """One kept YOLO box on one frame, plus the one pixel-derived scalar.

    `crop_quality` is `detection_quality(frame, bbox)`, computed at collection
    because it is the only value in the whole path that needs the image.
    """
    frame_idx: int
    raw_class: str
    confidence: float
    bbox: Sequence[float]
    crop_quality: float

    def to_dict(self) -> Dict[str, Any]:
        return {"frame_idx": int(self.frame_idx),
                "raw_class": str(self.raw_class),
                "confidence": float(self.confidence),
                "bbox": [float(v) for v in self.bbox],
                "crop_quality": float(self.crop_quality)}


@dataclass
class DoorFrameRecord:
    """One iterated frame. Empty detections and a failed score are different."""
    frame_idx: int
    detections: List[DoorDetection] = field(default_factory=list)
    errored: bool = False


def detections_from_observations(observations: Iterable[Any]
                                 ) -> List[DoorDetection]:
    """`core.timeline_evidence.Observation` -> aggregator input."""
    out: List[DoorDetection] = []
    for o in observations or []:
        if getattr(o, "kind", None) != "door":
            continue
        if o.local_frame is None or o.bbox is None:
            continue
        payload = getattr(o, "payload", None) or {}
        out.append(DoorDetection(
            frame_idx=int(o.local_frame), raw_class=str(o.label or ""),
            confidence=float(o.confidence or 0.0), bbox=list(o.bbox),
            crop_quality=float(payload.get("crop_quality", 0.0))))
    return out


def frame_records_from_detections(detections: Sequence[DoorDetection],
                                  scored_frames: Sequence[int]
                                  ) -> List[DoorFrameRecord]:
    """Group flat detections back into per-frame records, in scoring order."""
    by_frame: Dict[int, List[DoorDetection]] = {}
    for d in detections:
        by_frame.setdefault(int(d.frame_idx), []).append(d)
    return [DoorFrameRecord(int(fi), by_frame.get(int(fi), []))
            for fi in scored_frames]


def aggregate_door_from_frames(
    records: Sequence[DoorFrameRecord],
    *,
    camera_id: str,
    frame_width: int,
    frame_height: int,
    stride: int,
    frames: Optional[Dict[int, Any]] = None,
) -> Tuple[List[Dict[str, Any]], int, int, int,
           Dict[str, Any], Dict[str, Any]]:
    """The 6-tuple `_run_sampled_one_camera` returns, from collected detections.

    `(decisions, used, frame_w, frame_h, cands, overlay)` -- same aggregator,
    same acceptance rule, same bucket keying, same overlay shape.
    """
    from features._evidence import BestFrameTracker
    from features.evidence_aggregator import EvidenceAggregator, Observation

    used = 0
    cands: Dict[str, BestFrameTracker] = {}
    best_meta: Dict[str, Dict[str, Any]] = {}
    trajectory: Dict[int, Dict[str, Any]] = {}
    imgs = frames or {}

    if not records or frame_width == 0:
        return [], len(records), frame_width, frame_height, cands, {
            "tracks": [], "events": []}

    agg = EvidenceAggregator(frame_width=frame_width,
                             frame_height=frame_height, stride=stride)

    for rec in records:
        used += 1
        if rec.errored:
            # The old loop `continue`s before add_frame: a frame whose
            # inference raised is NOT declared to the aggregator at all.
            continue
        fi = int(rec.frame_idx)
        if not rec.detections:
            agg.add_frame(fi, [])
            continue

        observations: List[Observation] = []
        for det in rec.detections:
            raw = str(det.raw_class).lower()
            canon_state = C.DOOR_LABEL_TO_STATE.get(raw, canonical_door(raw))
            bl = [float(v) for v in det.bbox]
            crop_q = float(det.crop_quality)
            sc = snapshot_score(bl, float(det.confidence), crop_q,
                                frame_width, frame_height)
            observations.append(Observation(
                frame_idx=fi, state=canon_state,
                confidence=float(det.confidence),
                bbox=(bl[0], bl[1], bl[2], bl[3]), score=float(sc),
            ))

            # Evidence buckets -- keyed exactly as the legacy path keys them.
            bucket_key = canonical_door(raw)
            bbox_store = expand_bbox(bl, _DOOR_BBOX_EXPAND_FRAC,
                                     frame_width, frame_height)
            bucket = cands.setdefault(bucket_key, BestFrameTracker())
            prev = best_meta.get(bucket_key, {}).get("score", -1.0)
            if sc > prev:
                best_meta[bucket_key] = {
                    "score": float(sc), "frame_idx": fi, "bbox": bbox_store,
                    "state": bucket_key, "confidence": float(det.confidence),
                    "raw_class": raw, "quality": crop_q,
                }
                bucket.update(score=sc, frame=imgs.get(fi), bbox=bbox_store,
                              frame_idx=fi, state=bucket_key,
                              confidence=float(det.confidence), raw_class=raw,
                              quality=crop_q)

            entry = trajectory.setdefault(len(trajectory) + 1, {
                "camera_id": camera_id, "track_id": len(trajectory) + 1,
                "frames": [],
            })
            entry["frames"].append({
                "frame_idx": fi, "bbox": bbox_store,
                "state_raw": raw, "last_class": raw,
                "confidence": float(det.confidence), "velocity": [0.0, 0.0],
            })

        agg.add_frame(fi, observations)

    result = agg.finalize()
    decisions: List[Dict[str, Any]] = []
    for g in result["accepted"]:
        best = g.get("best")
        decisions.append({
            "camera_id":   camera_id,
            "track_id":    int(g["candidate_id"]),
            "state":       str(g["state"]),
            "confidence":  float(g["confidence"]),
            "first_frame": int(g["first_frame"]),
            "last_frame":  int(g["last_frame"]),
            "total_hits":  int(g["frame_support"]),
            "mean_center_x": float(best.center[0]) if best else 0.0,
        })

    overlay = {"tracks": list(trajectory.values()), "events": []}
    return decisions, used, frame_width, frame_height, cands, overlay


def best_frame_indices(records: Sequence[DoorFrameRecord], *,
                       frame_width: int, frame_height: int
                       ) -> Dict[str, int]:
    """Which frame each evidence bucket would keep, without any image.

    Deferred snapshot resolution: the caller can decode exactly these frames
    later instead of holding every candidate image in memory during collection.
    """
    best: Dict[str, Tuple[float, int]] = {}
    for rec in records:
        if rec.errored:
            continue
        for det in rec.detections:
            raw = str(det.raw_class).lower()
            sc = snapshot_score([float(v) for v in det.bbox],
                                float(det.confidence),
                                float(det.crop_quality),
                                frame_width, frame_height)
            key = canonical_door(raw)
            if sc > best.get(key, (-1.0, -1))[0]:
                best[key] = (float(sc), int(rec.frame_idx))
    return {k: v[1] for k, v in best.items()}
