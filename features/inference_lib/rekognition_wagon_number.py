"""Rekognition-backed wagon-number reading, scoped to ONE Global Wagon.

This is the global_train adaptation of the V4 Train-Inspection-Engine's
`inspection/ocr/wagon_number.py`.  The OCR *algorithm* is unchanged; only the
unit of work differs:

    V4 engine        : per SEGMENT of a camera pass, frames on disk in a segment dir
    global_train     : per GLOBAL WAGON (GW_n), frames in wagon_cache/<GW_n>/right_up/

So this module takes the YOLO detections already produced for one wagon, groups
them into bands, and for each band builds up to two three-frame sheets (primary +
fallback) which it sends to Rekognition until a valid 11-digit number is read.

Why sheets instead of per-frame OCR
-----------------------------------
EasyOCR was cheap enough to run on every frame of a wagon and majority-vote the
result.  A Rekognition ``DetectText`` call is a network round-trip and is billed
per image, so voting over ~100 frames per wagon is neither fast nor affordable.
Instead three well-chosen crops are stacked on one sheet, each reading as its own
``LINE``, and the reader picks the best VALID 11-digit line -- one call, three
chances.  Which three frames depends on rake load state, because cargo occludes
the plate from one end of the band depending on travel direction:

    loaded rake -> primary End-7/-5/-3, fallback Start+3/+5/+7
    empty rake  -> primary Start+3/+5/+7, fallback End-7/-5/-3

Call budget is capped per wagon (`core.rekognition.MAX_CALLS_PER_WAGON`) so a
noisy wagon with many bands can never blow up the bill.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core import constants as C
from core.logging_setup import get_logger

from .ocr_preprocessor import Preprocessor
from .rekognition_reader import read_digits
from .three_frame_sheet import build_vertical_sheet, select_frame_positions

log = get_logger("features.ocr.rekognition")


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def is_valid_wagon_number(number: str) -> bool:
    """Wagon numbers are exactly 11 digits (V4 rule, verbatim)."""
    return bool(number) and number.isdigit() and len(number) == C.WAGON_NUMBER_LENGTH


#: Locomotive numbers are 5 digits (V4 `is_valid_loco_number`).
LOCO_NUMBER_LENGTH = 5


def is_valid_loco_number(number: str) -> bool:
    """Loco numbers are exactly 5 digits (V4 rule, verbatim)."""
    return bool(number) and number.isdigit() and len(number) == LOCO_NUMBER_LENGTH


# ---------------------------------------------------------------------------
# Crop helper (V4 WagonImageEnhancer)
# ---------------------------------------------------------------------------

#: Fraction of the bbox width/height added as padding on each side before OCR.
PADDING_FRACTION = 0.25


def crop_plate(frame: np.ndarray, bbox: Sequence[float],
               padding_fraction: float = PADDING_FRACTION) -> Optional[np.ndarray]:
    """Crop ``frame`` to ``bbox`` expanded by ``padding_fraction`` on each side.

    Mirrors V4's ``WagonImageEnhancer.crop`` (which padded by 25% of the box
    dimensions), clamped to the frame.  Returns ``None`` for a degenerate box.
    """
    if frame is None or bbox is None or len(bbox) != 4:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    pad_x = (x2 - x1) * padding_fraction
    pad_y = (y2 - y1) * padding_fraction
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame[cy1:cy2, cx1:cx2].copy()


# ---------------------------------------------------------------------------
# Band grouping
# ---------------------------------------------------------------------------

def group_detections_into_bands(
    detections: Sequence[Dict[str, Any]],
    gap_tolerance: int = 8,
) -> List[Dict[str, Any]]:
    """Group per-frame plate detections into temporally contiguous bands.

    ``detections`` is a sequence of ``{"frame": int, "confidence": float,
    "bbox": [x1,y1,x2,y2]}``.  A new band starts when the frame gap to the
    previous detection exceeds ``gap_tolerance`` -- i.e. the plate left view and a
    different plate came in.

    Each returned band carries both orderings the sheet selector needs:
      * ``frames_by_time``  -- ascending frame number (drives Start/End offsets)
      * ``top_frames``      -- descending confidence (drives the short-band
                               single-crop fallback and the representative frame)

    Ported from V4's ``WagonNumberDetector.detect_on_segment`` banding, with the
    per-frame YOLO loop hoisted out to the caller (global_train already batches
    detection through `features._common`).
    """
    raw = [d for d in detections if d.get("bbox") is not None]
    if not raw:
        return []
    raw = sorted(raw, key=lambda d: int(d["frame"]))

    groups: List[List[Dict[str, Any]]] = [[raw[0]]]
    for det in raw[1:]:
        if int(det["frame"]) - int(groups[-1][-1]["frame"]) <= gap_tolerance + 1:
            groups[-1].append(det)
        else:
            groups.append([det])

    bands: List[Dict[str, Any]] = []
    for band_id, group in enumerate(groups, start=1):
        # One entry per frame, keeping that frame's highest-confidence box.
        by_frame: Dict[int, Dict[str, Any]] = {}
        for det in group:
            fn = int(det["frame"])
            prev = by_frame.get(fn)
            if prev is None or float(det["confidence"]) > float(prev["confidence"]):
                by_frame[fn] = {
                    "frame": fn,
                    "confidence": float(det["confidence"]),
                    "bbox": [float(v) for v in det["bbox"]],
                }
        frames_by_conf = sorted(by_frame.values(),
                                key=lambda f: (-f["confidence"], f["frame"]))
        frames_by_time = sorted(by_frame.values(), key=lambda f: f["frame"])
        best = frames_by_conf[0]
        bands.append({
            "band_id": band_id,
            "best_frame": best["frame"],
            "best_confidence": best["confidence"],
            "best_bbox": best["bbox"],
            "top_frames": frames_by_conf,
            "frames_by_time": frames_by_time,
            "frame_count": len(by_frame),
            "avg_confidence": float(np.mean([f["confidence"]
                                             for f in by_frame.values()])),
        })
    return bands


# ---------------------------------------------------------------------------
# OCR orchestrator
# ---------------------------------------------------------------------------

#: A band needs at least this many detected frames before the three-frame sheet
#: offsets are meaningful; shorter bands fall back to a single best-frame crop.
MIN_FRAMES_FOR_SHEET_SELECTION = 3


class RekognitionWagonNumberOCR:
    """Reads the 11-digit wagon number for one wagon via Rekognition DetectText.

    Parameters
    ----------
    client:
        A `core.rekognition.RekognitionClient` (or any object with
        ``detect_text(bytes) -> list``).
    max_calls:
        Hard ceiling on DetectText calls per wagon.
    gap_tolerance:
        Frame gap that separates two plate bands.
    """

    def __init__(self, client, *, max_calls: int = 4, gap_tolerance: int = 8,
                 logger=None) -> None:
        self.client = client
        self.max_calls = max(1, int(max_calls))
        self.gap_tolerance = int(gap_tolerance)
        self.log = logger or log

    # -- public ------------------------------------------------------------

    def read_loco_number(
        self,
        *,
        bands: List[Dict[str, Any]],
        frame_loader: Callable[[int], Optional[np.ndarray]],
    ) -> Dict[str, Any]:
        """Read the 5-digit LOCO number from ``bands``.

        Same sheet mechanism as the wagon path, but the V4 loco rule selects
        ``Middle-2 / Middle / Middle+2`` (a loco's plate is centred on its face,
        so there is no loaded/empty occlusion to steer around) and validates 5
        digits instead of 11.
        """
        return self.read_wagon_number(bands=bands, frame_loader=frame_loader,
                                      is_loaded=False, kind="loco")

    def read_wagon_number(
        self,
        *,
        bands: List[Dict[str, Any]],
        frame_loader: Callable[[int], Optional[np.ndarray]],
        is_loaded: bool,
        kind: str = "wagon",
    ) -> Dict[str, Any]:
        """Read the wagon number from ``bands``.

        ``frame_loader`` maps a frame index to its BGR image (the processor backs
        this with the wagon_cache JPEGs, so only the 3-6 selected frames are ever
        decoded).  ``is_loaded`` picks the primary triplet order.

        ``kind`` selects the plate contract: ``"wagon"`` (11 digits, loaded/empty
        triplet order) or ``"loco"`` (5 digits, middle triplet).

        Returns a V4-shaped result dict; ``wagon_identifier`` is the canonical
        global_train field and is ``NO_DATA`` unless a valid number was read.
        """
        is_loco = (kind == "loco")
        validator = is_valid_loco_number if is_loco else is_valid_wagon_number
        best_raw, best_conf = "", 0.0
        best_sheet: Optional[np.ndarray] = None
        best_info: Optional[Dict[str, Any]] = None
        primary_band = (max(bands, key=lambda b: b["best_confidence"])
                        if bands else {})

        calls = 0
        attempt_index = 0
        for band in bands:
            for triplet in self._candidate_triplets(band, is_loaded, kind=kind):
                if calls >= self.max_calls:
                    self.log.debug(
                        "[REKOGNITION] call budget %d reached -- stopping",
                        self.max_calls)
                    break
                sheet, rep_info = self._build_sheet(triplet, frame_loader)
                if sheet is None:
                    attempt_index += 1
                    continue
                # Preprocessor runs on the ASSEMBLED sheet, not per crop.
                img = Preprocessor.primary(sheet)
                calls += 1
                # Vertical sheet -> the crops read as separate LINEs; pick the
                # single best VALID reading instead of concatenating.
                num, conf = read_digits(self.client, img, validator=validator)
                if num and conf >= best_conf:
                    best_raw, best_conf = num, conf
                    best_sheet, best_info = img, rep_info
                if validator(num):
                    return self._result(
                        num, conf, img, rep_info, band.get("band_id", 0),
                        is_valid=True, fallback_triggered=attempt_index > 0,
                        calls=calls, bands=bands, kind=kind)
                attempt_index += 1
            if calls >= self.max_calls:
                break

        if best_info is None:
            best_info = {"frame": primary_band.get("best_frame"),
                         "bbox": primary_band.get("best_bbox"),
                         "confidence": primary_band.get("best_confidence", 0.0)}
        return self._result(
            best_raw, best_conf, best_sheet, best_info,
            primary_band.get("band_id", 0), is_valid=False,
            fallback_triggered=True, calls=calls, bands=bands, kind=kind)

    # -- internals ---------------------------------------------------------

    def _candidate_triplets(self, band: Dict[str, Any], is_loaded: bool,
                            *, kind: str = "wagon") -> List[List[Dict[str, Any]]]:
        """Ordered three-frame triplets to sheet for one band.

        Normally two triplets -- the primary set then the fallback set -- deduped
        so a short band that clamps offsets onto the same frame doesn't repeat
        crops.  Bands with too few detected frames yield a single one-crop triplet
        of the highest-confidence frame.
        """
        frames = band.get("frames_by_time") or []
        n = len(frames)
        if n == 0:
            return []
        if n < MIN_FRAMES_FOR_SHEET_SELECTION:
            return [[{
                "frame": band["best_frame"],
                "confidence": band["best_confidence"],
                "bbox": band["best_bbox"],
            }]]
        # A loco's plate sits centred on its face, so V4 reads Middle-2/Middle/
        # Middle+2 rather than steering around cargo occlusion.
        category = ("loco" if kind == "loco"
                    else ("loaded" if is_loaded else "empty"))
        triplets: List[List[Dict[str, Any]]] = []
        seen_signatures = set()
        for use_fallback in (False, True):
            positions = select_frame_positions(n, category, use_fallback)
            seen: set = set()
            triplet = [frames[p] for p in positions
                       if not (p in seen or seen.add(p))]
            if not triplet:
                continue
            sig = tuple(f["frame"] for f in triplet)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            triplets.append(triplet)
        return triplets

    def _build_sheet(self, triplet: List[Dict[str, Any]],
                     frame_loader: Callable[[int], Optional[np.ndarray]]):
        """Crop each frame in ``triplet`` and lay the raw crops on one sheet.

        Returns ``(sheet, representative_frame_info)`` where the representative is
        the highest-confidence frame in the triplet (reported as the winning
        frame/bbox).  Returns ``(None, None)`` when no crop could be extracted.
        """
        raw_crops: List[np.ndarray] = []
        used: List[Dict[str, Any]] = []
        for fi in triplet:
            frame = frame_loader(int(fi["frame"]))
            if frame is None:
                continue
            crop = crop_plate(frame, fi["bbox"])
            if crop is None or crop.size == 0:
                continue
            raw_crops.append(crop)
            used.append(fi)
        if not raw_crops:
            return None, None
        sheet = build_vertical_sheet(raw_crops)
        rep = max(used, key=lambda f: f.get("confidence", 0.0))
        return sheet, rep

    @staticmethod
    def _result(number: str, conf: float, sheet, frame_info, band_id: int, *,
                is_valid: bool, fallback_triggered: bool, calls: int,
                bands: List[Dict[str, Any]],
                kind: str = "wagon") -> Dict[str, Any]:
        frame_info = frame_info or {}
        result = {
            # canonical global_train fields
            "wagon_identifier": number if is_valid else C.NO_DATA,
            "wagon_identifier_confidence": round(float(conf), 4),
            # V4-parity fields (carried into the per-camera inspection JSON)
            "raw_number": number,
            "display_number": number if is_valid else "-",
            "confidence": round(float(conf), 4),
            "ocr_confidence": round(float(conf), 4),
            "band_id": band_id,
            "best_frame": frame_info.get("frame"),
            "best_bbox": frame_info.get("bbox"),
            "fallback_triggered": bool(fallback_triggered),
            # observability
            "engine": "rekognition",
            "kind": kind,
            "rekognition_calls": calls,
            "band_count": len(bands),
            "_sheet": sheet,          # in-memory only; stripped before JSON write
        }
        # V4 names the validity flag after the digit count it enforces.
        if kind == "loco":
            result["is_valid_5_digit"] = bool(is_valid)
        else:
            result["is_valid_11_digit"] = bool(is_valid)
        return result
