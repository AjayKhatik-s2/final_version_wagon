"""Damage feature processor (v4, train-state-native, ALL legacy intelligence
ported).

Per wagon, for each top camera (RIGHT_UP_TOP / LEFT_UP_TOP):

    1. Iterate cached JPEGs.
    2. Run YOLO `damage.pt`.  Apply legacy detection filters:
         - confidence floor (0.55)
         - bbox area filter: skip < 0.5% or > 40% of frame (FP guards)
         - skip negative classes (`no_damage`, `outer_wall_damage` -- the
           outer-wall variant is a side concern, not a top concern)
    3. Apply edge-zone suppression on top-camera detections:
         - X-axis: skip if bbox center < 12% of width or > 88%
         - Y-axis: skip if bbox center < 10% top or > 85% bottom
         - bypass: keep at conf >= 0.70 even at edges (real damage tail)
    4. Feed surviving detections into the mature DamageTracker (Kalman +
       Hungarian, distance-only cost @ 200 px, min_hits=2, max_age=30,
       confidence-weighted majority vote per track, gap-frame-aware
       snapshot selection).
    5. Cross-track dedup on the final track set (same-class + spatially
       close tracks collapse to one).
    6. If load_status of this wagon is LOADED, drop floor_damage tracks
       (can't see the floor when loaded).
    7. Verdict per top camera:
         - DAMAGE if >=1 confirmed track survives
         - OK     if frames seen and no damage track survives
         - NO_DATA if no frames at all
    8. Cross-camera fusion: ANY top camera reporting DAMAGE wins.

Output JSON shape (per wagon):
    {
        "global_id":  "GW_7",
        "feature":    "damage",
        "status":     "OK" | "NO_FRAMES" | "FAILED" | "NO_DATA",
        "top_damage": "DAMAGE" | "OK" | "NO_DATA",
        "top_damage_details": [{class_name, confidence, bbox, frame_idx,
                                camera_id, track_id}, ...],
        "per_camera": {RIGHT_UP_TOP: {damage_status, frames, tracks},
                       LEFT_UP_TOP:  {...}},
        "supporting_cameras": [...],
        "frame_count": ...,
    }
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core import constants as C
from core.global_state_loader import GlobalTrainState

from features._common import (
    load_yolo, iter_wagon_frames, list_wagon_frames,
    write_per_wagon_json, empty_payload, FeatureTimer,
)

# Mature intelligence ported from legacy
from features.inference_lib.damage_tracker import (
    DamageTracker, DamageTrackerConfig, yolo_to_damage_detections,
)
from features._evidence import (
    wagon_evidence_dir, save_jpeg, safe_crop,
    write_metadata, draw_annotated_bbox,
)
# EXPERIMENTAL sampled path only; the legacy tracker path never touches this.
from features.evidence_aggregator import EvidenceAggregator, Observation


FEATURE_NAME = "damage"


# -----------------------------------------------------------------------------
# Legacy filter parameters (kept verbatim from RIGHT_UP_TOP/damage_processor.py)
# -----------------------------------------------------------------------------

_AREA_MIN_RATIO       = 0.005    # 0.5% of frame
_AREA_MAX_RATIO       = 0.40     # 40% of frame
_EDGE_X_MIN_RATIO     = 0.12     # 12% from left
_EDGE_X_MAX_RATIO     = 0.88     # 88% from left
_EDGE_Y_MIN_RATIO     = 0.10     # 10% from top
_EDGE_Y_MAX_RATIO     = 0.85     # 85% from top (floor view)
_EDGE_BYPASS_CONF     = 0.70     # bypass edge filter if conf >= 0.70

# Classes to skip on TOP cameras (the legacy SKIP_CLASSES set)
_SKIP_CLASSES_TOP = {
    "no_damage", "wagon", "engine", "tail", "brake_van", "guard_van",
    "background",
    "outer_wall_damage",        # side concern, not a top one
}


# -----------------------------------------------------------------------------
# Per-frame filtering pipeline
# -----------------------------------------------------------------------------

def _filter_detections_for_top(
    boxes: np.ndarray,
    confs: np.ndarray,
    cls_ids: np.ndarray,
    class_names: Dict[int, str],
    frame_w: int,
    frame_h: int,
    confidence_floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply ALL legacy top-camera damage filters in one pass.

    Returns the surviving (boxes, confs, cls_ids).
    """
    keep_mask: List[bool] = []
    frame_area = max(1.0, float(frame_w) * float(frame_h))

    for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
        cls_name = str(class_names.get(int(cls_id), "unknown")).strip().lower()
        if cls_name in _SKIP_CLASSES_TOP:
            keep_mask.append(False)
            continue
        if float(conf) < confidence_floor:
            keep_mask.append(False)
            continue

        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        area_ratio = (x2 - x1) * (y2 - y1) / frame_area
        if area_ratio < _AREA_MIN_RATIO or area_ratio > _AREA_MAX_RATIO:
            keep_mask.append(False)
            continue

        # Edge-zone suppression (with bypass for high-confidence hits)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cx_r = cx / max(1.0, frame_w)
        cy_r = cy / max(1.0, frame_h)
        at_edge = (
            cx_r < _EDGE_X_MIN_RATIO or cx_r > _EDGE_X_MAX_RATIO
            or cy_r < _EDGE_Y_MIN_RATIO or cy_r > _EDGE_Y_MAX_RATIO
        )
        if at_edge and float(conf) < _EDGE_BYPASS_CONF:
            keep_mask.append(False)
            continue
        keep_mask.append(True)

    if not keep_mask:
        empty = np.zeros((0,), dtype=boxes.dtype)
        return np.zeros((0, 4), dtype=boxes.dtype), empty, empty.astype(int)
    mask = np.asarray(keep_mask, dtype=bool)
    return boxes[mask], confs[mask], cls_ids[mask]


# -----------------------------------------------------------------------------
# Per-camera tracker run
# -----------------------------------------------------------------------------

def _run_tracker_one_camera(
    yolo_model,
    tracker_config: DamageTrackerConfig,
    cache_root: str,
    gw_id: str,
    camera_id: str,
    confidence_floor: float,
) -> Tuple[List[Dict[str, Any]], int, int, int, List[Dict[str, Any]]]:
    """Returns (track_decisions, n_frames_used, frame_w, frame_h, frame_dets).

    ``frame_dets`` is the per-frame list of RAW filtered detections (one entry
    per surviving box per frame).  The legacy damage ``_tracked.mp4`` drew the
    raw per-frame YOLO survivors -- NOT the Kalman-smoothed track box -- so this
    is exactly what the Stage-4b overlay replays for top cameras (faithful,
    including legacy's deliberately un-smoothed damage-box behaviour).
    """
    paths = list_wagon_frames(cache_root, gw_id, camera_id, trim_stable=True)
    if not paths:
        return [], 0, 0, 0, []

    tracker = DamageTracker(config=tracker_config)
    tracker.reset()

    frame_w, frame_h = 0, 0
    used = 0
    frame_dets: List[Dict[str, Any]] = []
    for fi, frame in iter_wagon_frames(cache_root, gw_id, camera_id, trim_stable=True):
        if frame_w == 0:
            frame_h, frame_w = frame.shape[:2]
        used += 1

        try:
            results = yolo_model(frame, verbose=False)[0]
        except Exception:
            continue
        if results.boxes is None or len(results.boxes) == 0:
            tracker.update([], frame=frame,
                           frame_width=frame_w, frame_height=frame_h)
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss  = results.boxes.cls.cpu().numpy().astype(int)
        names = getattr(yolo_model, "names", {}) or {}

        boxes, confs, clss = _filter_detections_for_top(
            boxes, confs, clss, names, frame_w, frame_h, confidence_floor,
        )
        if len(boxes) == 0:
            tracker.update([], frame=frame,
                           frame_width=frame_w, frame_height=frame_h)
            continue

        # Record the raw per-frame survivors for the overlay (legacy drew
        # these directly, un-smoothed).
        for bb, cf, ci in zip(boxes, confs, clss):
            frame_dets.append({
                "camera_id":  camera_id,
                "frame_idx":  int(fi),
                "bbox":       [float(bb[0]), float(bb[1]),
                               float(bb[2]), float(bb[3])],
                "class_name": str(names.get(int(ci), "")).lower(),
                "confidence": float(cf),
            })

        detections = yolo_to_damage_detections(
            boxes=boxes, confidences=confs, class_ids=clss,
            class_names=names,
        )
        tracker.update(detections=detections, frame=frame,
                       frame_width=frame_w, frame_height=frame_h)

    final = tracker.get_final_damage_states()
    out: List[Dict[str, Any]] = []
    for tid, info in final.items():
        bbox_raw = info.get("best_snapshot_bbox")
        bbox_list = (bbox_raw.tolist() if bbox_raw is not None
                     and hasattr(bbox_raw, "tolist") else None)
        out.append({
            "camera_id":   camera_id,
            "track_id":    int(tid),
            "class_name":  str(info.get("class_name", "")).lower(),
            "confidence":  float(info.get("confidence", 0.0) or 0.0),
            "best_confidence": float(info.get("best_confidence", 0.0) or 0.0),
            "total_hits":  int(info.get("total_hits", 0)),
            "first_frame": int(info.get("first_frame", 0)),
            "last_frame":  int(info.get("last_frame", 0)),
            "best_frame_idx": int(info.get("best_frame_idx", 0) or 0),
            "bbox":        bbox_list,
            # in-memory snapshot ndarray (stripped before JSON serialization)
            "_snapshot":   info.get("best_snapshot"),
        })
    return out, used, frame_w, frame_h, frame_dets


# -----------------------------------------------------------------------------
# EXPERIMENTAL sampled path (inference_mode="sampled")
# -----------------------------------------------------------------------------

def _run_sampled_one_camera(
    yolo_model,
    tracker_config: DamageTrackerConfig,
    cache_root: str,
    gw_id: str,
    camera_id: str,
    confidence_floor: float,
    sample_stride: int = 2,
) -> Tuple[List[Dict[str, Any]], int, int, int, List[Dict[str, Any]]]:
    """Sampled-frame Damage inference -- EXPERIMENTAL, not the default path.

    Returns the SAME 5-tuple as `_run_tracker_one_camera`, so everything after
    it in `run()` -- cross-track dedup, the loaded-wagon floor-damage filter,
    evidence persistence and the JSON contract -- is byte-for-byte unchanged.

    Unchanged from legacy: the model, `_filter_detections_for_top` (confidence
    floor, area bounds, edge-zone suppression with the high-confidence bypass),
    and the class vocabulary.  Only the frame sample rate and the grouping
    mechanism differ: EvidenceAggregator requires a support FRACTION across
    sampled frames rather than the tracker's absolute hit count, so a lone
    high-confidence false positive cannot mark a wagon damaged.
    """
    paths = list_wagon_frames(cache_root, gw_id, camera_id, trim_stable=True)
    if not paths:
        return [], 0, 0, 0, []

    stride = max(1, int(sample_stride))
    frame_w, frame_h = 0, 0
    used = 0
    frame_dets: List[Dict[str, Any]] = []
    agg: Optional[EvidenceAggregator] = None
    snapshots: Dict[int, Any] = {}

    for fi, frame in iter_wagon_frames(cache_root, gw_id, camera_id,
                                       every_nth=stride, trim_stable=True):
        if frame_w == 0:
            frame_h, frame_w = frame.shape[:2]
            agg = EvidenceAggregator(frame_width=frame_w, frame_height=frame_h,
                                     stride=stride)
        used += 1

        try:
            results = yolo_model(frame, verbose=False)[0]
        except Exception:
            continue
        if results.boxes is None or len(results.boxes) == 0:
            agg.add_frame(fi, [])
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)
        names = getattr(yolo_model, "names", {}) or {}

        # IDENTICAL legacy filtering -- not relaxed for sampling.
        boxes, confs, clss = _filter_detections_for_top(
            boxes, confs, clss, names, frame_w, frame_h, confidence_floor,
        )
        if len(boxes) == 0:
            agg.add_frame(fi, [])
            continue

        observations: List[Observation] = []
        for bb, cf, ci in zip(boxes, confs, clss):
            cls_name = str(names.get(int(ci), "")).lower()
            bl = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
            frame_dets.append({
                "camera_id": camera_id, "frame_idx": int(fi), "bbox": bl,
                "class_name": cls_name, "confidence": float(cf),
            })
            observations.append(Observation(
                frame_idx=int(fi), state=cls_name, confidence=float(cf),
                bbox=(bl[0], bl[1], bl[2], bl[3]), score=float(cf),
            ))
        snapshots[int(fi)] = frame
        agg.add_frame(fi, observations)

    if agg is None:
        return [], used, frame_w, frame_h, frame_dets

    out: List[Dict[str, Any]] = []
    for g in agg.finalize()["accepted"]:
        best = g.get("best")
        if best is None:
            continue
        out.append({
            "camera_id":   camera_id,
            "track_id":    int(g["candidate_id"]),
            "class_name":  str(g["state"]).lower(),
            "confidence":  float(g["confidence"]),
            "best_confidence": float(best.confidence),
            "total_hits":  int(g["frame_support"]),
            "first_frame": int(g["first_frame"]),
            "last_frame":  int(g["last_frame"]),
            "best_frame_idx": int(best.frame_idx),
            "bbox":        list(best.bbox),
            "_snapshot":   snapshots.get(int(best.frame_idx)),
        })
    return out, used, frame_w, frame_h, frame_dets


# -----------------------------------------------------------------------------
# Cross-track dedup (same class + spatially close tracks collapse)
# -----------------------------------------------------------------------------

def _dedup_cross_tracks(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove near-duplicate tracks within ONE camera.

    Rule (legacy): same class + IoU>=0.2 OR center distance < 100 px.
    Survivor = highest confidence (then more hits).
    """
    if len(tracks) < 2:
        return tracks
    survivors: List[Dict[str, Any]] = []
    for tr in sorted(tracks, key=lambda t: (-t["confidence"], -t["total_hits"])):
        bbox = tr.get("bbox")
        keep = True
        if bbox and len(bbox) == 4:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            for s in survivors:
                if s["class_name"] != tr["class_name"]:
                    continue
                sb = s.get("bbox")
                if not sb or len(sb) != 4:
                    continue
                scx = (sb[0] + sb[2]) / 2.0
                scy = (sb[1] + sb[3]) / 2.0
                if (cx - scx) ** 2 + (cy - scy) ** 2 < 100.0 ** 2:
                    keep = False
                    break
                # IoU
                ax1, ay1, ax2, ay2 = bbox
                bx1, by1, bx2, by2 = sb
                ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    a_area = (ax2 - ax1) * (ay2 - ay1)
                    b_area = (bx2 - bx1) * (by2 - by1)
                    union = max(1.0, a_area + b_area - inter)
                    if inter / union >= 0.20:
                        keep = False
                        break
        if keep:
            survivors.append(tr)
    return survivors


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def run(
    *,
    state: GlobalTrainState,
    cache_root: str,
    feature_models_dir: str,
    output_dir: str,
    evidence_root: Optional[str] = None,
    confidence: float = C.CONF_DAMAGE,
    verbose: bool = True,
    every_nth: int = 1,
    max_frames: int = 0,
    min_persistent_frames: int = 2,
    inference_mode: str = "legacy",
    sample_stride: int = 2,
) -> Dict[str, str]:
    """`inference_mode`:

        "legacy"  (DEFAULT) every frame + DamageTracker.  Known-good path.
        "sampled" EXPERIMENTAL: every `sample_stride`-th frame +
                  EvidenceAggregator.  The orchestrator does not pass this
                  argument, so production behaviour is unchanged by
                  construction.

    Both modes share the model, `_filter_detections_for_top`, cross-track
    dedup, the loaded-wagon floor-damage suppression, evidence persistence and
    the JSON contract.
    """
    del every_nth, max_frames, min_persistent_frames  # legacy tracker owns persistence
    mode = str(inference_mode or "legacy").strip().lower()
    if mode not in ("legacy", "sampled"):
        raise ValueError(
            f"damage inference_mode must be 'legacy' or 'sampled', got {mode!r}")

    model_path = os.path.join(feature_models_dir, C.MODEL_DAMAGE)
    yolo_model = load_yolo(model_path)

    feature_out = os.path.join(output_dir, FEATURE_NAME)
    os.makedirs(feature_out, exist_ok=True)
    timer = FeatureTimer("damage")
    summary: Dict[str, str] = {}

    tracker_cfg = DamageTrackerConfig(
        confidence_threshold=confidence,
        max_age=30, n_init=2, min_hits_for_decision=2,
        max_center_distance=200.0, iou_weight=0.0, distance_weight=1.0,
    )

    # Stage-2 fusion needs to know which wagons are LOADED to drop
    # floor_damage tracks; we read those from sibling `load` JSONs.
    load_dir = os.path.join(output_dir, "load")

    if yolo_model is None and verbose:
        print(f"[FEAT/damage] WARNING: {model_path} missing -- NO_DATA for all wagons.")

    if verbose:
        print(f"[FEAT/damage] running on {len(state.wagons)} wagons "
              f"(legacy DamageTracker + edge-zone + loaded-wagon filter)")

    for gw in state.wagons:
        gw_id = gw.global_id
        t0 = time.time()
        try:
            if yolo_model is None:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.NO_DATA,
                    top_damage=C.NO_DATA, top_damage_details=[],
                    per_camera={}, supporting_cameras=[],
                    error="damage.pt not present",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.NO_DATA
                continue

            # Engine / brake-van wagons rarely carry meaningful damage signal
            # (no floor / loaded body to inspect).  Skip cleanly.
            if gw.classification in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN):
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_OK,
                    top_damage=C.NO_DATA, top_damage_details=[],
                    per_camera={}, supporting_cameras=[],
                    skipped_reason=f"classification={gw.classification}",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_OK
                continue

            per_camera: Dict[str, Dict[str, Any]] = {}
            all_evidence: List[Dict[str, Any]] = []
            all_frame_dets: List[Dict[str, Any]] = []   # raw per-frame overlay
            supporting: List[str] = []
            any_damage = False
            total_frames = 0

            for cam in C.TOP_CAMERAS:
                if mode == "sampled":
                    tracks, used, _, _, fdets = _run_sampled_one_camera(
                        yolo_model, tracker_cfg, cache_root, gw_id, cam,
                        confidence_floor=confidence,
                        sample_stride=sample_stride,
                    )
                else:
                    tracks, used, _, _, fdets = _run_tracker_one_camera(
                        yolo_model, tracker_cfg, cache_root, gw_id, cam,
                        confidence_floor=confidence,
                    )
                all_frame_dets.extend(fdets)
                tracks = _dedup_cross_tracks(tracks)
                per_camera[cam] = {
                    "damage_status":  (C.DAMAGE_PRESENT if tracks else
                                       (C.DAMAGE_OK if used > 0 else C.NO_DATA)),
                    "frames_used":    used,
                    "track_count":    len(tracks),
                    "tracks":         tracks,
                }
                if used > 0:
                    supporting.append(cam)
                    total_frames += used
                if tracks:
                    any_damage = True
                    all_evidence.extend(tracks)

            # Loaded-wagon filter:  if our companion `load` feature already
            # marked this wagon LOADED, drop floor_damage tracks
            # (can't see the floor under a load).
            load_path = os.path.join(load_dir, f"{gw_id}.json")
            if os.path.exists(load_path) and any_damage:
                try:
                    import json
                    with open(load_path, "r", encoding="utf-8") as f:
                        load_payload = json.load(f)
                    if (load_payload.get("status") == C.STATUS_OK
                            and load_payload.get("load_status") == C.LOAD_LOADED):
                        before = len(all_evidence)
                        all_evidence = [
                            e for e in all_evidence
                            if e["class_name"] != "floor_damage"
                        ]
                        # Keep the overlay consistent with the verdict: a wagon
                        # we declare not-floor-damaged shouldn't flash floor
                        # boxes in its video.
                        all_frame_dets = [
                            d for d in all_frame_dets
                            if d["class_name"] != "floor_damage"
                        ]
                        if before != len(all_evidence) and verbose:
                            print(f"  [damage/{gw_id}] loaded wagon -> "
                                  f"dropped {before - len(all_evidence)} "
                                  f"floor_damage track(s)")
                        any_damage = bool(all_evidence)
                except Exception:
                    pass  # fail open

            # Persist raw per-frame damage detections for the Stage-4b overlay
            # (legacy drew these directly, un-smoothed, per frame).
            if evidence_root and all_frame_dets:
                ov_dir = wagon_evidence_dir(evidence_root, gw_id, FEATURE_NAME)
                write_metadata(
                    os.path.join(ov_dir, "overlay.json"),
                    {"global_id": gw_id, "feature": FEATURE_NAME,
                     "detections": all_frame_dets},
                )

            if not supporting:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
                    top_damage=C.NO_DATA, top_damage_details=[],
                    per_camera=per_camera, supporting_cameras=[],
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_NO_FRAMES
                continue

            top_damage = C.DAMAGE_PRESENT if any_damage else C.DAMAGE_OK

            # Persist evidence:  per-track best snapshot + bbox crop into
            # evidence/<gw>/damage/.  We then strip the in-memory `_snapshot`
            # ndarrays before serialising any track dict to JSON.
            evidence_paths: Dict[str, str] = {}
            if evidence_root and all_evidence:
                ev_dir = wagon_evidence_dir(evidence_root, gw_id, FEATURE_NAME)
                track_meta: List[Dict[str, Any]] = []
                for i, tr in enumerate(all_evidence, start=1):
                    snap = tr.get("_snapshot")
                    if snap is None:
                        continue
                    annotated = draw_annotated_bbox(
                        snap, tr.get("bbox"),
                        label=f"{tr['class_name']} {tr['best_confidence']:.2f}",
                        color=(0, 0, 255),
                    )
                    full_p = os.path.join(ev_dir, f"track_{i}.jpg")
                    crop_p = os.path.join(ev_dir, f"track_{i}_crop.jpg")
                    save_jpeg(full_p, annotated)
                    crop_img = safe_crop(snap, tr.get("bbox"), pad=10)
                    if crop_img is not None:
                        save_jpeg(crop_p, crop_img)
                    evidence_paths[f"track_{i}"] = full_p
                    if crop_img is not None:
                        evidence_paths[f"track_{i}_crop"] = crop_p
                    track_meta.append({
                        "track_idx":       i,
                        "camera_id":       tr["camera_id"],
                        "track_id":        tr["track_id"],
                        "class_name":      tr["class_name"],
                        "confidence":      tr["confidence"],
                        "best_confidence": tr["best_confidence"],
                        "best_frame_idx":  tr["best_frame_idx"],
                        "bbox":            tr.get("bbox"),
                    })
                write_metadata(os.path.join(ev_dir, "metadata.json"), {
                    "global_id":  gw_id,
                    "feature":    FEATURE_NAME,
                    "top_damage": top_damage,
                    "tracks":     track_meta,
                })

            # JSON shouldn't carry ndarray snapshots.
            serializable_evidence = [
                {k: v for k, v in tr.items() if k != "_snapshot"}
                for tr in all_evidence
            ]
            for cam_info in per_camera.values():
                cam_info["tracks"] = [
                    {k: v for k, v in tr.items() if k != "_snapshot"}
                    for tr in cam_info.get("tracks", [])
                ]

            payload: Dict[str, Any] = {
                "global_id": gw_id,
                "feature":   FEATURE_NAME,
                "status":    C.STATUS_OK,
                "top_damage":         top_damage,
                "top_damage_details": serializable_evidence,
                "per_camera":         per_camera,
                "supporting_cameras": supporting,
                "frame_count":        total_frames,
                "evidence":           evidence_paths,
            }
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_OK
            if verbose:
                print(f"  [damage/{gw_id}] {top_damage}  tracks={len(all_evidence)}  "
                      f"frames={total_frames}  cams={','.join(supporting)}")
        except Exception as e:
            payload = empty_payload(
                gw_id, FEATURE_NAME, C.STATUS_FAILED,
                top_damage=C.NO_DATA,
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(limit=2),
            )
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_FAILED
            if verbose:
                print(f"  [damage/{gw_id}] FAILED: {e}")
        finally:
            timer.stamp(gw_id, t0)

    if verbose:
        n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
        print(f"[FEAT/damage] done in {timer.total():.1f}s  ok={n_ok}/{len(summary)}")
    return summary
