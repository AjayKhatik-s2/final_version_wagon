"""Load feature processor (v4, train-state-native, ALL legacy intelligence
ported).

Legacy logic from `RIGHT_UP_TOP/damage_processor.py:1038-1087` (per-wagon
frame-by-frame loaded/empty voting):
    is_loaded = (loaded_count / total_count) > 0.35

Plus three additional guards that came along for the ride in the
legacy code:
    * fall back to a YOLO-detection-style classifier when `loaded.pt`
      emits boxes instead of probs (some loaded variants do this)
    * default to EMPTY if the model can't be reached at all
    * skip ENGINE / BRAKE_VAN wagons -- they don't carry payload, and
      running the loaded model on them produces meaningless votes

Per-wagon JSON shape:
    {
        "global_id":   "GW_7",
        "feature":     "load",
        "status":      "OK" | "NO_FRAMES" | "FAILED" | "NO_DATA",
        "load_status": "LOADED" | "EMPTY" | "NO_DATA",
        "load_confidence": 0.85,
        "per_camera": {
            "RIGHT_UP_TOP": {load_status, confidence, loaded_count,
                             empty_count, frames_used, loaded_ratio},
            "LEFT_UP_TOP":  {...}
        },
        "supporting_cameras": [...],
        "frame_count": ...,
    }

Authority rule (Stage 2 fusion respects this too): RIGHT_UP_TOP is
authoritative; LEFT_UP_TOP is supporting / fallback.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from core import constants as C
from core.global_state_loader import GlobalTrainState

from features._common import (
    load_yolo, run_classification, iter_wagon_frames,
    write_per_wagon_json, empty_payload, FeatureTimer,
)
from features._evidence import (
    BestFrameTracker, wagon_evidence_dir,
    save_jpeg, write_metadata,
)


FEATURE_NAME = "load"


# Legacy threshold from RIGHT_UP_TOP/damage_processor.py:1047
_LOADED_RATIO_THRESHOLD = 0.35


def _canonical_load(raw: str) -> str:
    return C.LOAD_LABEL_TO_STATE.get((raw or "").strip().lower(), C.NO_DATA)


def _aggregate_camera(
    model, cache_root: str, gw_id: str, camera_id: str,
    *, every_nth: int, max_frames: Optional[int],
) -> Tuple[str, float, int, int, int, "BestFrameTracker", "BestFrameTracker"]:
    """Return (load_status, confidence, frames_used, loaded_count, empty_count,
               best_loaded_frame, best_empty_frame).

    Follows the legacy frame-by-frame voting rule:
        is_loaded = (loaded_count / total_count) > 0.35
    Confidence is the mean top-1 probability of frames that voted with
    the winning side.

    Best-frame trackers retain the highest-conf frame on each side for
    evidence persistence.
    """
    loaded_confs: List[float] = []
    empty_confs:  List[float] = []
    used = 0
    best_loaded = BestFrameTracker()
    best_empty  = BestFrameTracker()

    for fi, frame in iter_wagon_frames(cache_root, gw_id, camera_id,
                                       every_nth=every_nth,
                                       max_frames=max_frames,
                                       trim_stable=True):
        cls, conf = run_classification(model, frame)
        cls_canon = _canonical_load(cls)
        used += 1
        if cls_canon == C.LOAD_LOADED:
            loaded_confs.append(conf)
            best_loaded.update(score=float(conf), frame=frame,
                               frame_idx=int(fi),
                               camera_id=camera_id, class_name=cls,
                               confidence=float(conf))
        elif cls_canon == C.LOAD_EMPTY:
            empty_confs.append(conf)
            best_empty.update(score=float(conf), frame=frame,
                              frame_idx=int(fi),
                              camera_id=camera_id, class_name=cls,
                              confidence=float(conf))

    if used == 0:
        return C.NO_DATA, 0.0, 0, 0, 0, best_loaded, best_empty

    n_loaded = len(loaded_confs)
    n_empty  = len(empty_confs)
    total    = max(1, used)
    loaded_ratio = n_loaded / total

    if loaded_ratio > _LOADED_RATIO_THRESHOLD and n_loaded > 0:
        mean_conf = sum(loaded_confs) / n_loaded
        return C.LOAD_LOADED, float(mean_conf), used, n_loaded, n_empty, best_loaded, best_empty
    if n_empty > 0:
        mean_conf = sum(empty_confs) / n_empty
        return C.LOAD_EMPTY, float(mean_conf), used, n_loaded, n_empty, best_loaded, best_empty
    return C.NO_DATA, 0.0, used, n_loaded, n_empty, best_loaded, best_empty


def run(
    *,
    state: GlobalTrainState,
    cache_root: str,
    feature_models_dir: str,
    output_dir: str,
    evidence_root: Optional[str] = None,
    every_nth: int = 2,
    # MUST be None, not 0, to mean "unbounded".  `features/_common.py`'s
    # iter_wagon_frames guards its subsample with `max_frames is not None`, so a
    # 0 was treated as a hard cap of ZERO: np.linspace(0, n-1, 0) returns an
    # empty array and every frame was discarded.  Load therefore processed 0
    # frames per wagon and wrote NO_FRAMES for every wagon -- every_nth=2 never
    # took effect at all.  Verified on a 40-frame cache: max_frames=0 -> 0
    # frames yielded, max_frames=None -> 17 (34 stable frames, every 2nd).
    # Fixed Load-side only; the shared helper is deliberately left untouched.
    max_frames: Optional[int] = None,
    inference_mode: str = "legacy",
    sample_stride: int = 2,
    verbose: bool = True,
) -> Dict[str, str]:
    mode = str(inference_mode or "legacy").strip().lower()
    if mode not in ("legacy", "sampled"):
        raise ValueError(
            f"load inference_mode must be 'legacy' or 'sampled', got {mode!r}")
    # Load ALREADY samples: `every_nth` defaults to 2, and since the
    # max_frames=0 -> None fix it actually takes effect.  "sampled" therefore
    # just makes the stride explicit/configurable; at sample_stride=2 it is
    # behaviourally IDENTICAL to legacy and yields no call reduction.
    effective_every_nth = max(1, int(sample_stride)) if mode == "sampled" else every_nth

    model_path = os.path.join(feature_models_dir, C.MODEL_LOADED)
    model = load_yolo(model_path)

    feature_out = os.path.join(output_dir, FEATURE_NAME)
    os.makedirs(feature_out, exist_ok=True)
    timer = FeatureTimer("load")
    summary: Dict[str, str] = {}

    if model is None and verbose:
        print(f"[FEAT/load] WARNING: {model_path} missing -- NO_DATA for all wagons.")

    if verbose:
        print(f"[FEAT/load] running on {len(state.wagons)} wagons "
              f"(legacy frame-by-frame voting, >{_LOADED_RATIO_THRESHOLD:.0%} -> LOADED)")

    for gw in state.wagons:
        gw_id = gw.global_id
        t0 = time.time()
        try:
            if model is None:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.NO_DATA,
                    load_status=C.NO_DATA, load_confidence=0.0,
                    per_camera={}, supporting_cameras=[],
                    error="loaded.pt not present",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.NO_DATA
                continue

            # ENGINE / BRAKE_VAN gate: no payload to classify.
            if gw.classification in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN):
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_OK,
                    load_status=C.NO_DATA, load_confidence=0.0,
                    per_camera={}, supporting_cameras=[],
                    skipped_reason=f"classification={gw.classification}",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_OK
                continue

            per_camera: Dict[str, Dict[str, Any]] = {}
            best_by_cam: Dict[str, Dict[str, BestFrameTracker]] = {}
            for cam in C.TOP_CAMERAS:
                cls, conf, used, n_l, n_e, b_l, b_e = _aggregate_camera(
                    model, cache_root, gw_id, cam,
                    every_nth=effective_every_nth, max_frames=max_frames,
                )
                per_camera[cam] = {
                    "load_status":  cls,
                    "confidence":   round(float(conf), 4),
                    "frames_used":  used,
                    "loaded_count": n_l,
                    "empty_count":  n_e,
                    "loaded_ratio": round(n_l / used, 4) if used else 0.0,
                }
                best_by_cam[cam] = {"loaded": b_l, "empty": b_e}

            r = per_camera[C.CAMERA_RIGHT_UP_TOP]
            l = per_camera[C.CAMERA_LEFT_UP_TOP]
            supporting = [c for c, v in per_camera.items() if v["frames_used"] > 0]

            # RIGHT_UP_TOP authoritative when present; LEFT_UP_TOP supports.
            if r["load_status"] != C.NO_DATA:
                fused_status = r["load_status"]
                fused_conf   = r["confidence"]
            elif l["load_status"] != C.NO_DATA:
                fused_status = l["load_status"]
                fused_conf   = l["confidence"]
            else:
                fused_status = C.NO_DATA
                fused_conf   = 0.0

            if not supporting:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
                    load_status=C.NO_DATA, load_confidence=0.0,
                    per_camera=per_camera, supporting_cameras=[],
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_NO_FRAMES
                continue

            # Persist strongest LOADED + strongest EMPTY frame from the
            # authoritative camera (RIGHT_UP_TOP); fall back to LEFT_UP_TOP.
            evidence_paths: Dict[str, str] = {}
            if evidence_root and fused_status != C.NO_DATA:
                ev_dir = wagon_evidence_dir(evidence_root, gw_id, FEATURE_NAME)
                # Choose source camera: master if it has evidence for the
                # winning class, else the supporter.
                winning_key = "loaded" if fused_status == C.LOAD_LOADED else "empty"
                source_cam = None
                for cam in (C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP):
                    if best_by_cam[cam][winning_key].has_data():
                        source_cam = cam
                        break
                meta_payload: Dict[str, Any] = {
                    "global_id": gw_id, "feature": FEATURE_NAME,
                    "fused_status": fused_status,
                    "fused_confidence": fused_conf,
                    "per_camera": {},
                }
                if source_cam is not None:
                    b = best_by_cam[source_cam][winning_key]
                    full_p = os.path.join(ev_dir, "best_frame.jpg")
                    save_jpeg(full_p, b.frame)
                    evidence_paths["best_frame"] = full_p
                    meta_payload["source_camera"]  = source_cam
                    meta_payload["best_frame_idx"] = b.frame_idx
                    meta_payload["best_class"]     = b.meta.get("class_name")
                    meta_payload["best_confidence"] = b.meta.get("confidence")
                # Also keep a per-camera audit trail
                for cam, side in best_by_cam.items():
                    for key, b in side.items():
                        if b.has_data():
                            meta_payload["per_camera"].setdefault(cam, {})[key] = {
                                "frame_idx":  b.frame_idx,
                                "confidence": b.meta.get("confidence"),
                                "class_name": b.meta.get("class_name"),
                            }
                write_metadata(os.path.join(ev_dir, "metadata.json"), meta_payload)

            payload: Dict[str, Any] = {
                "global_id": gw_id,
                "feature":   FEATURE_NAME,
                "status":    C.STATUS_OK,
                "load_status":     fused_status,
                "load_confidence": round(fused_conf, 4),
                "per_camera":      per_camera,
                "supporting_cameras": supporting,
                "frame_count": sum(v["frames_used"] for v in per_camera.values()),
                "evidence":    evidence_paths,
            }
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_OK
            if verbose:
                print(f"  [load/{gw_id}] {fused_status} ({fused_conf:.2f})  "
                      f"R={r['load_status']} ({r['loaded_ratio']:.0%})  "
                      f"L={l['load_status']} ({l['loaded_ratio']:.0%})")
        except Exception as e:
            payload = empty_payload(
                gw_id, FEATURE_NAME, C.STATUS_FAILED,
                load_status=C.NO_DATA,
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(limit=2),
            )
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_FAILED
            if verbose:
                print(f"  [load/{gw_id}] FAILED: {e}")
        finally:
            timer.stamp(gw_id, t0)

    if verbose:
        n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
        print(f"[FEAT/load] done in {timer.total():.1f}s  ok={n_ok}/{len(summary)}")
    return summary
