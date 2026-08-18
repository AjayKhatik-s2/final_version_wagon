"""OCR feature processor (v4, train-state-native).

Two interchangeable OCR engines, selected by ``WAGONEYE_OCR_ENGINE``:

``rekognition`` (DEFAULT) -- AWS Rekognition ``DetectText``, the V4
Train-Inspection-Engine's engine, reproduced exactly:

    1. YOLO (`wagon_number_update.pt`, falling back to `wagon_id_counting.pt`)
       detects wagon-number plates on RIGHT_UP frames (OCR authority).
    2. Detections are grouped into temporal BANDS (one band == one physical
       plate crossing the frame), `gap_tolerance=8` frames.
    3. Per band, THREE frames are chosen and their raw crops stacked on ONE
       white vertical sheet (20px gutters, crops copied verbatim):
           loaded rake -> primary End-7/-5/-3, fallback Start+3/+5/+7
           empty rake  -> primary Start+3/+5/+7, fallback End-7/-5/-3
       The rake's load state comes from this run's own `load` feature output,
       which is why Stage 3 runs Load before OCR.
    4. The sheet goes through the V4 `Preprocessor` (3x cubic upscale, min
       width 800) and then a single Rekognition ``DetectText`` call.  Because
       the crops are stacked vertically each reads as its own ``LINE``, so the
       reader picks the single best VALID 11-digit reading rather than
       concatenating three copies of the number.
    5. The first sheet yielding a valid 11-digit number wins; otherwise the
       highest-confidence attempt is still reported, flagged invalid.

    Rekognition calls are bounded per wagon (`max_calls`, default 4) so a rake
    of 58 wagons cannot fan out into an unbounded API bill.

``easyocr`` -- the legacy local pipeline (kept as a no-network fallback, and
unchanged):
    1. YOLO `wagon_id_counting.pt` detects wagon-number bbox regions on
       RIGHT_UP frames (master / OCR authority).
    2. Each crop is fed through the legacy `WagonNumberOCR`:
           padding 10 -> 3x cubic upscale -> NLMeans denoise (h=8) ->
           CLAHE (clipLimit=3.5, tile 8x8) -> unsharp masking ->
           easyocr (allowlist='0123456789') -> digit extraction ->
           wagon-type confusion-map correction (first 2 digits in 10-39)
           -> WagonNumberValidator (length=11, structure check).
    3. Surviving candidates per frame are added to the legacy
       `WagonNumberAggregator` which performs:
           exact-string grouping with digit-level voting at each position
           min 2 frames + min OCR conf 0.3
    4. Best aggregated number is picked by (observations, mean conf).

Output JSON shape (both engines write the same keys; the V4-parity fields are
what `delivery.inspection_json` carries into the per-camera dashboard JSON):
    {
        "global_id":  "GW_7",
        "feature":    "ocr",
        "status":     "OK" | "NO_FRAMES" | "FAILED" | "NO_DATA",
        "engine":     "rekognition" | "easyocr",
        "wagon_identifier":  "32145678901",
        "wagon_identifier_confidence": 0.83,
        # V4-parity fields:
        "raw_number":        "32145678901",
        "display_number":    "32145678901" | "-",
        "is_valid_11_digit": true,
        "fallback_triggered": false,
        "best_frame":  123,
        "best_bbox":   [x1, y1, x2, y2],
        "candidates":  [...],
        "supporting_cameras": ["RIGHT_UP"],
        "frame_count": ...,
    }
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from core import constants as C
from core.global_state_loader import GlobalTrainState

from features._common import (
    load_yolo, run_detection, iter_wagon_frames, crop_bbox,
    write_per_wagon_json, empty_payload, FeatureTimer,
)

# Mature intelligence ported from legacy
from features.inference_lib.wagon_number_ocr import WagonNumberOCR, WagonNumber
from features.inference_lib.wagon_number_aggregator import (
    WagonNumberAggregator, AggregatorConfig,
)
from features._evidence import (
    BestFrameTracker, wagon_evidence_dir,
    save_jpeg, safe_crop, write_metadata, draw_annotated_bbox,
)
# Rekognition engine (V4 parity).  Imported lazily inside the engine path so a
# checkout without boto3 can still run the easyocr engine.


FEATURE_NAME = "ocr"

ENGINE_REKOGNITION = "rekognition"
ENGINE_EASYOCR = "easyocr"


def resolve_engine(explicit: Optional[str] = None) -> str:
    """Which OCR engine this run uses.

    ``WAGONEYE_OCR_ENGINE=rekognition`` (default) | ``easyocr``.  An unknown
    value falls back to Rekognition rather than raising, so a typo degrades to
    the V4-parity default instead of taking a batch down.
    """
    raw = (explicit or os.getenv("WAGONEYE_OCR_ENGINE")
           or ENGINE_REKOGNITION).strip().lower()
    if raw in (ENGINE_REKOGNITION, "aws", "detect_text"):
        return ENGINE_REKOGNITION
    if raw in (ENGINE_EASYOCR, "easy", "local"):
        return ENGINE_EASYOCR
    print(f"[FEAT/ocr] unknown WAGONEYE_OCR_ENGINE={raw!r} -- "
          f"using {ENGINE_REKOGNITION}")
    return ENGINE_REKOGNITION


def _detector_path(feature_models_dir: str) -> str:
    """The plate detector, preferring V4's `wagon_number_update.pt`.

    `C.feature_model_path` falls back to the older `wagon_id_counting.pt` when
    the canonical file is absent, so an existing checkout keeps working.
    """
    return C.feature_model_path(feature_models_dir, C.MODEL_WAGON_NUMBER)


def _wagon_is_loaded(states_root: str, gw_id: str) -> bool:
    """Load state for this wagon, from this run's own `load` feature output.

    V4 orders the three-frame triplet by rake load state, because cargo occludes
    the plate from one end of the band depending on travel direction.  Stage 3
    runs Load to completion before OCR, so this read is deterministic.  A
    missing/failed load result means EMPTY -- the same default V4 uses.
    """
    p = os.path.join(states_root, "load", f"{gw_id}.json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return False
    return (payload.get("status") == C.STATUS_OK
            and payload.get("load_status") == C.LOAD_LOADED)


# -----------------------------------------------------------------------------
# Per-process singleton (easyocr Reader is heavy; load once)
# -----------------------------------------------------------------------------

_OCR_SINGLETON: Optional[WagonNumberOCR] = None


def _get_ocr() -> Optional[WagonNumberOCR]:
    global _OCR_SINGLETON
    if _OCR_SINGLETON is not None:
        return _OCR_SINGLETON
    try:
        _OCR_SINGLETON = WagonNumberOCR(
            use_gpu=True,
            min_confidence=0.30,        # legacy default for cross-frame aggregation
            resize_factor=3.0,
        )
        if getattr(_OCR_SINGLETON, "reader", None) is None:
            _OCR_SINGLETON = None
    except Exception as e:
        print(f"[FEAT/ocr] WagonNumberOCR init failed: {e}")
        _OCR_SINGLETON = None
    return _OCR_SINGLETON


# -----------------------------------------------------------------------------
# Per-wagon driver
# -----------------------------------------------------------------------------

def _process_one_wagon(
    yolo_model,
    ocr: WagonNumberOCR,
    cache_root: str,
    gw_id: str,
    det_confidence: float,
) -> Dict[str, Any]:
    """Iterate cached RIGHT_UP frames, run YOLO + OCR, aggregate."""
    aggregator = WagonNumberAggregator(AggregatorConfig(
        min_frame_count=2,
        min_confidence=0.3,
        require_validation=True,
    ))

    used = 0
    raw_candidates: List[Dict[str, Any]] = []
    best = BestFrameTracker()    # remembers the highest-conf OCR snapshot

    for fi, frame in iter_wagon_frames(cache_root, gw_id, C.CAMERA_RIGHT_UP, trim_stable=True):
        used += 1

        # Stage A: YOLO detection -- locate wagon-number bbox regions.
        # fp32: this used to request half=True, which on the CPU-only torch
        # build is emulated fp16 -- measured on door_state.pt at 167x slower
        # with ZERO detections surviving threshold. OCR is off by default so it
        # never bit here, but it would have the moment OCR was enabled.
        try:
            results = yolo_model(frame, verbose=False)[0]
        except Exception:
            continue
        if results.boxes is None or len(results.boxes) == 0:
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        # Stage B: per-detection OCR pipeline (preprocess + easyocr +
        # validate + reconstruct)
        for bbox, yolo_conf in zip(boxes, confs):
            if float(yolo_conf) < det_confidence:
                continue
            bbox_list = [float(b) for b in bbox]
            crop = crop_bbox(frame, bbox_list, pad=10)
            if crop is None or crop.size == 0:
                continue
            try:
                wagon_num = ocr.reconstruct_wagon_number(
                    crop, float(yolo_conf), debug=False,
                )
            except Exception:
                continue
            if wagon_num is None:
                continue
            aggregator.add_wagon_number(wagon_num, frame_idx=fi)

            full = getattr(wagon_num, "full_number", None)
            if full:
                ocr_conf = float(getattr(wagon_num, "ocr_confidence", 0.0))
                raw_candidates.append({
                    "frame_idx":       int(fi),
                    "full_number":     str(full),
                    "ocr_confidence":  ocr_conf,
                    "yolo_confidence": float(getattr(wagon_num, "yolo_confidence", 0.0)),
                    "bbox":            bbox_list,
                })
                # Track best snapshot:  prefer full-length (11-digit) numbers
                # and within that bucket, highest OCR confidence.
                is_full = int(len(str(full)) == C.WAGON_NUMBER_LENGTH)
                score = is_full * 10.0 + ocr_conf
                best.update(
                    score=score, frame=frame, bbox=bbox_list,
                    frame_idx=fi,
                    full_number=str(full),
                    ocr_confidence=ocr_conf,
                    yolo_confidence=float(yolo_conf),
                    is_full_length=bool(is_full),
                )

    # Stage C: pick the dominant aggregated wagon number
    aggregated = aggregator.get_aggregated_numbers()
    return {
        "frame_count": used,
        "aggregated":  aggregated,
        "raw":         raw_candidates,
        "best":        best,
    }


# -----------------------------------------------------------------------------
# Rekognition engine -- per-wagon driver (V4 parity)
# -----------------------------------------------------------------------------

def _rekognition_client():
    """Shared Rekognition client, or None when unavailable.

    `RekognitionUnavailable` (no boto3 / no credentials / bad region) is not an
    error worth failing a batch over: the caller records NO_DATA for OCR and
    every other feature still reports.
    """
    try:
        from core.rekognition import get_client
        return get_client()
    except Exception as e:  # noqa: BLE001 - degrade, never crash the stage
        print(f"[FEAT/ocr] Rekognition unavailable ({e}) -- OCR will be NO_DATA. "
              f"Set WAGONEYE_OCR_ENGINE=easyocr to use the local engine.")
        return None


def _collect_plate_detections(
    yolo_model, cache_root: str, gw_id: str, det_confidence: float,
) -> tuple:
    """One YOLO pass over this wagon's RIGHT_UP frames.

    Returns ``(detections, frames_seen, frame_cache)`` where each detection is
    ``{frame, confidence, bbox}`` -- exactly the shape
    `group_detections_into_bands` expects.  Frames are cached by index so the
    sheet builder can re-read only the three frames a triplet needs, without a
    second decode pass over the whole wagon.
    """
    detections = []
    frames_seen = 0
    frame_cache = {}
    for fi, frame in iter_wagon_frames(cache_root, gw_id, C.CAMERA_RIGHT_UP,
                                       trim_stable=True):
        frames_seen += 1
        try:
            results = yolo_model(frame, verbose=False)[0]
        except Exception:
            continue
        if results.boxes is None or len(results.boxes) == 0:
            continue
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        kept = False
        for bbox, conf in zip(boxes, confs):
            if float(conf) < det_confidence:
                continue
            detections.append({
                "frame": int(fi),
                "confidence": float(conf),
                "bbox": [float(b) for b in bbox],
            })
            kept = True
        if kept:
            frame_cache[int(fi)] = frame
    return detections, frames_seen, frame_cache


def _process_one_wagon_rekognition(
    yolo_model,
    client,
    cache_root: str,
    gw_id: str,
    det_confidence: float,
    is_loaded: bool,
) -> Dict[str, Any]:
    """Detect plates, band them, and read the winning band via Rekognition."""
    from features.inference_lib.rekognition_wagon_number import (
        RekognitionWagonNumberOCR, group_detections_into_bands,
    )

    detections, frames_seen, frame_cache = _collect_plate_detections(
        yolo_model, cache_root, gw_id, det_confidence)
    if not detections:
        return {"frame_count": frames_seen, "result": None, "detections": []}

    bands = group_detections_into_bands(detections, gap_tolerance=8)
    reader = RekognitionWagonNumberOCR(client)
    result = reader.read_wagon_number(
        bands=bands,
        frame_loader=lambda fi: frame_cache.get(int(fi)),
        is_loaded=is_loaded,
        kind="wagon",
    )
    return {
        "frame_count": frames_seen,
        "result": result,
        "detections": detections,
        "band_count": len(bands),
        "frame_cache": frame_cache,
    }


def _persist_rekognition_evidence(
    *, evidence_root: Optional[str], gw_id: str, outcome: Dict[str, Any],
) -> Dict[str, str]:
    """Write the Rekognition engine's evidence into this package's flat layout.

    Filenames are deliberately the SAME as the easyocr engine's
    (`best_frame.jpg` / `number_crop.jpg`) so the reports and the dashboard
    adapter resolve evidence identically whichever engine ran.  The assembled
    sheet -- the exact image sent to Rekognition -- is additionally saved as
    `ocr_sheet.jpg`, which is what `inspection_json` prefers for
    `ocr_frame_s3_url`.
    """
    paths: Dict[str, str] = {}
    result = outcome.get("result") or {}
    if not evidence_root or not result:
        return paths

    ev_dir = wagon_evidence_dir(evidence_root, gw_id, FEATURE_NAME)
    sheet = result.get("_sheet")
    if sheet is not None:
        sheet_p = os.path.join(ev_dir, "ocr_sheet.jpg")
        if save_jpeg(sheet_p, sheet):
            paths["ocr_sheet"] = sheet_p

    frame_idx = result.get("best_frame")
    bbox = result.get("best_bbox")
    frame = (outcome.get("frame_cache") or {}).get(
        int(frame_idx)) if frame_idx is not None else None
    if frame is not None and bbox:
        number = result.get("display_number") or "?"
        annotated = draw_annotated_bbox(
            frame, bbox,
            label=f"OCR {number} {float(result.get('confidence') or 0.0):.2f}",
            color=(0, 255, 0),
        )
        full_p = os.path.join(ev_dir, "best_frame.jpg")
        if save_jpeg(full_p, annotated):
            paths["best_frame"] = full_p
        crop_img = safe_crop(frame, bbox, pad=4)
        if crop_img is not None:
            crop_p = os.path.join(ev_dir, "number_crop.jpg")
            if save_jpeg(crop_p, crop_img):
                paths["number_crop"] = crop_p

    write_metadata(os.path.join(ev_dir, "metadata.json"), {
        "global_id":         gw_id,
        "feature":           FEATURE_NAME,
        "camera_id":         C.CAMERA_RIGHT_UP,
        "engine":            ENGINE_REKOGNITION,
        "frame_idx":         frame_idx,
        "bbox":              bbox,
        "full_number":       result.get("raw_number"),
        "display_number":    result.get("display_number"),
        "is_valid_11_digit": result.get("is_valid_11_digit"),
        "ocr_confidence":    result.get("ocr_confidence"),
        "fallback_triggered": result.get("fallback_triggered"),
        "band_id":           result.get("band_id"),
        "band_count":        result.get("band_count"),
        "rekognition_calls": result.get("rekognition_calls"),
    })
    return paths


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
    det_confidence: float = C.CONF_OCR_BOX,
    wagon_number_length: int = C.WAGON_NUMBER_LENGTH,
    every_nth: int = 1,
    max_frames: int = 0,
    engine: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """Run OCR on every wagon.

    `engine` selects the reader (default from WAGONEYE_OCR_ENGINE):
    ``rekognition`` for V4 parity, ``easyocr`` for the local pipeline.
    """
    del every_nth, max_frames, wagon_number_length  # engines use own thresholds

    engine = resolve_engine(engine)
    model_path = _detector_path(feature_models_dir)
    yolo_model = load_yolo(model_path)

    # Only construct the reader the selected engine actually needs: the easyocr
    # Reader is a heavy local model load, and the Rekognition client needs
    # credentials -- neither should be touched by a run using the other engine.
    ocr = None
    rekog_client = None
    if engine == ENGINE_REKOGNITION:
        rekog_client = _rekognition_client()
        reader_ready = rekog_client is not None
    else:
        ocr = _get_ocr()
        reader_ready = ocr is not None

    feature_out = os.path.join(output_dir, FEATURE_NAME)
    os.makedirs(feature_out, exist_ok=True)
    timer = FeatureTimer("ocr")
    summary: Dict[str, str] = {}

    if yolo_model is None and verbose:
        print(f"[FEAT/ocr] WARNING: {model_path} missing -- NO_DATA for all wagons.")
    if not reader_ready and verbose:
        print(f"[FEAT/ocr] WARNING: {engine} reader unavailable -- "
              f"NO_DATA for all wagons.")

    if verbose:
        detail = ("AWS Rekognition DetectText, 3-frame sheet per band"
                  if engine == ENGINE_REKOGNITION
                  else "legacy WagonNumberOCR + WagonNumberAggregator")
        print(f"[FEAT/ocr] running on {len(state.wagons)} wagons "
              f"(engine={engine}: {detail}, RIGHT_UP only)")
        print(f"[FEAT/ocr] detector: {os.path.basename(model_path)}")

    for gw in state.wagons:
        gw_id = gw.global_id
        t0 = time.time()
        try:
            if yolo_model is None or not reader_ready:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.NO_DATA,
                    wagon_identifier=C.NO_DATA,
                    wagon_identifier_confidence=0.0,
                    candidates=[], supporting_cameras=[],
                    engine=engine,
                    error="detector or OCR engine unavailable",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.NO_DATA
                continue

            # ENGINE / BRAKE_VAN wagons rarely carry the standard 11-digit
            # wagon number; running OCR on them produces noise.  Skip but
            # still record the wagon entry.
            if gw.classification in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN):
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_OK,
                    wagon_identifier=C.NO_DATA,
                    wagon_identifier_confidence=0.0,
                    candidates=[],
                    supporting_cameras=[C.CAMERA_RIGHT_UP],
                    engine=engine,
                    skipped_reason=f"classification={gw.classification}",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_OK
                continue

            if engine == ENGINE_REKOGNITION:
                outcome = _process_one_wagon_rekognition(
                    yolo_model, rekog_client, cache_root, gw_id, det_confidence,
                    _wagon_is_loaded(output_dir, gw_id),
                )
                used = outcome["frame_count"]
                result = outcome.get("result")

                if used == 0:
                    payload = empty_payload(
                        gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
                        wagon_identifier=C.NO_DATA,
                        wagon_identifier_confidence=0.0,
                        candidates=[], supporting_cameras=[], engine=engine,
                    )
                    write_per_wagon_json(feature_out, gw_id, payload)
                    summary[gw_id] = C.STATUS_NO_FRAMES
                    continue

                evidence_paths = _persist_rekognition_evidence(
                    evidence_root=evidence_root, gw_id=gw_id, outcome=outcome)

                # `_sheet` is an in-memory image; it must never reach the JSON.
                serializable = {k: v for k, v in (result or {}).items()
                                if k != "_sheet"}
                payload = {
                    "global_id":   gw_id,
                    "feature":     FEATURE_NAME,
                    "status":      C.STATUS_OK,
                    "engine":      engine,
                    "wagon_identifier": (serializable.get("wagon_identifier")
                                         or C.NO_DATA),
                    "wagon_identifier_confidence":
                        round(float(serializable.get(
                            "wagon_identifier_confidence") or 0.0), 4),
                    "candidates":  [],
                    "supporting_cameras": [C.CAMERA_RIGHT_UP],
                    "frame_count": used,
                    "detection_count": len(outcome.get("detections") or []),
                    "evidence":    evidence_paths,
                }
                payload.update(serializable)
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_OK
                if verbose:
                    print(f"  [ocr/{gw_id}] {payload.get('display_number', '-')} "
                          f"(valid={bool(payload.get('is_valid_11_digit'))}, "
                          f"conf={float(payload.get('confidence') or 0.0):.2f}, "
                          f"bands={outcome.get('band_count', 0)}, "
                          f"calls={payload.get('rekognition_calls')}, "
                          f"frames={used})")
                continue

            outcome = _process_one_wagon(
                yolo_model, ocr, cache_root, gw_id, det_confidence,
            )
            used = outcome["frame_count"]
            aggregated = outcome["aggregated"]

            if used == 0:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
                    wagon_identifier=C.NO_DATA,
                    wagon_identifier_confidence=0.0,
                    candidates=[],
                    supporting_cameras=[],
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_NO_FRAMES
                continue

            # Build serialized candidate list from the aggregator's output
            candidates_out: List[Dict[str, Any]] = []
            for agg in aggregated:
                candidates_out.append({
                    "full_number":     str(getattr(agg, "wagon_number", "")),
                    "observations":    int(getattr(agg, "frame_count", 0)),
                    "mean_conf":       float(getattr(agg, "avg_confidence", 0.0)),
                    "yolo_conf":       float(getattr(agg, "avg_yolo_confidence",
                                              getattr(agg, "yolo_confidence", 0.0))),
                    "is_full_length":  len(str(getattr(agg, "wagon_number", "")))
                                       == C.WAGON_NUMBER_LENGTH,
                })

            # Aggregator already enforces min_frame_count + min_confidence.
            # The "best" candidate is the one with the highest combined
            # (observations, mean_conf) score.
            candidates_out.sort(
                key=lambda c: (
                    -int(c["is_full_length"]),
                    -c["observations"],
                    -c["mean_conf"],
                    c["full_number"],
                )
            )

            if candidates_out and candidates_out[0]["is_full_length"]:
                top = candidates_out[0]
                ident = top["full_number"]
                conf  = top["mean_conf"]
            else:
                ident = C.NO_DATA
                conf  = 0.0

            # Persist best-frame evidence:  full annotated frame +
            # tight crop of the wagon-number plate.
            evidence_paths: Dict[str, str] = {}
            best_obj = outcome.get("best")
            if evidence_root and best_obj is not None and best_obj.has_data():
                ev_dir = wagon_evidence_dir(evidence_root, gw_id, FEATURE_NAME)
                annotated = draw_annotated_bbox(
                    best_obj.frame, best_obj.bbox,
                    label=f"OCR {best_obj.meta.get('full_number','?')} "
                          f"{best_obj.meta.get('ocr_confidence',0.0):.2f}",
                    color=(0, 255, 0),
                )
                full_p = os.path.join(ev_dir, "best_frame.jpg")
                crop_p = os.path.join(ev_dir, "number_crop.jpg")
                save_jpeg(full_p, annotated)
                crop_img = safe_crop(best_obj.frame, best_obj.bbox, pad=4)
                if crop_img is not None:
                    save_jpeg(crop_p, crop_img)
                evidence_paths["best_frame"] = full_p
                if crop_img is not None:
                    evidence_paths["number_crop"] = crop_p
                write_metadata(os.path.join(ev_dir, "metadata.json"), {
                    "global_id":       gw_id,
                    "feature":         FEATURE_NAME,
                    "camera_id":       C.CAMERA_RIGHT_UP,
                    "frame_idx":       best_obj.frame_idx,
                    "bbox":            best_obj.bbox,
                    "full_number":     best_obj.meta.get("full_number"),
                    "ocr_confidence":  best_obj.meta.get("ocr_confidence"),
                    "yolo_confidence": best_obj.meta.get("yolo_confidence"),
                    "is_full_length":  best_obj.meta.get("is_full_length"),
                    "aggregated_winner": ident,
                    "aggregated_confidence": conf,
                })

            # The V4-parity fields are emitted by BOTH engines so the
            # per-camera inspection JSON reads identically whichever ran.
            _is_valid = bool(ident != C.NO_DATA
                             and len(str(ident)) == C.WAGON_NUMBER_LENGTH)
            _best_meta = (best_obj.meta if (best_obj is not None
                                            and best_obj.has_data()) else {})
            payload: Dict[str, Any] = {
                "global_id":   gw_id,
                "feature":     FEATURE_NAME,
                "status":      C.STATUS_OK,
                "engine":      engine,
                "wagon_identifier":            ident,
                "wagon_identifier_confidence": round(float(conf), 4),
                "raw_number":        (str(_best_meta.get("full_number") or "")
                                      if not _is_valid else str(ident)),
                "display_number":    str(ident) if _is_valid else "-",
                "is_valid_11_digit": _is_valid,
                "confidence":        round(float(conf), 4),
                "ocr_confidence":    round(float(conf), 4),
                "best_frame":        (best_obj.frame_idx
                                      if best_obj is not None else None),
                "best_bbox":         (best_obj.bbox
                                      if best_obj is not None else None),
                # easyocr has no sheet-escalation step, so a valid read is never
                # the product of a fallback attempt.
                "fallback_triggered": bool(not _is_valid),
                "candidates":  candidates_out[:8],
                "raw_candidates_first_8":      outcome["raw"][:8],
                "supporting_cameras": [C.CAMERA_RIGHT_UP],
                "frame_count": used,
                "evidence":    evidence_paths,
            }
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_OK
            if verbose:
                print(f"  [ocr/{gw_id}] {ident} (conf={conf:.2f}, "
                      f"candidates={len(candidates_out)}, frames={used})")
        except Exception as e:
            payload = empty_payload(
                gw_id, FEATURE_NAME, C.STATUS_FAILED,
                wagon_identifier=C.NO_DATA,
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(limit=2),
            )
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_FAILED
            if verbose:
                print(f"  [ocr/{gw_id}] FAILED: {e}")
        finally:
            timer.stamp(gw_id, t0)

    if verbose:
        n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
        print(f"[FEAT/ocr] done in {timer.total():.1f}s  ok={n_ok}/{len(summary)}")
    return summary
