"""Three-frame OCR sheet builder.

Ported from the V4 Train-Inspection-Engine `inspection/ocr/three_frame_sheet.py`.

Instead of OCR-ing a single frame per wagon, three frames are selected, a crop is
extracted from each, and the three crops are stacked on ONE white sheet
(vertically by default, so each plate lands on its own horizontal band and
Rekognition reads each as a separate ``LINE``).  That single sheet is then fed
through :class:`ocr_preprocessor.Preprocessor` + the Rekognition reader -- one
API call reads three chances at the plate.

Selection rules (positions index the band's frame list, ordered by time; ``Start``
is the first frame, ``End`` the last, ``Middle`` == ``len // 2``):

* **Loco**      -> ``Middle-2, Middle, Middle+2``
* **Empty train**
    * primary   -> ``Start+3, Start+5, Start+7``
    * fallback  -> ``End-7, End-5, End-3``
* **Loaded train**
    * primary   -> ``End-7, End-5, End-3``
    * fallback  -> ``Start+3, Start+5, Start+7``

Cargo occludes the plate from one end of the band depending on travel direction,
which is why the loaded/empty orders are mirrored.  Callers decide when to
escalate to the fallback (i.e. when the primary sheet yields no valid 11-digit
number), exactly as in V4.

Design notes
------------
* Crops are placed **exactly as extracted** -- never resized, padded per-crop, or
  blended.  The only compositing is: white canvas + 20 px gaps between crops.
* The preprocessor runs on the ASSEMBLED sheet, never on the individual crops.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .ocr_preprocessor import Preprocessor

# Spacing / background are fixed by the V4 spec: 20 px white gutters.
SHEET_SPACING_PX = 20
SHEET_BG_COLOR = (255, 255, 255)  # BGR white

# Offset triplets, expressed as (anchor, offset) pairs.  ``start`` offsets index
# from the first frame (frames[k]); ``end`` offsets index from the last frame
# (frames[n-1-k]); ``middle`` offsets from frames[n//2].
_START_TRIPLET = [("start", 3), ("start", 5), ("start", 7)]
_END_TRIPLET = [("end", 7), ("end", 5), ("end", 3)]
_MIDDLE_TRIPLET = [("middle", -2), ("middle", 0), ("middle", 2)]

# (primary, fallback) offset sets per category.
_RULES = {
    "loco": (_MIDDLE_TRIPLET, _MIDDLE_TRIPLET),
    "empty": (_START_TRIPLET, _END_TRIPLET),
    "loaded": (_END_TRIPLET, _START_TRIPLET),
}


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def _resolve_offset(anchor: str, offset: int, n: int) -> int:
    """Turn an (anchor, offset) pair into a concrete 0-based position, clamped
    into ``[0, n-1]`` so short bands never index out of range."""
    if anchor == "start":
        pos = offset
    elif anchor == "end":
        pos = (n - 1) - offset
    else:  # middle
        pos = (n // 2) + offset
    return max(0, min(n - 1, pos))


def select_frame_positions(n_frames: int, category: str,
                           use_fallback: bool = False) -> list:
    """Return the three 0-based frame positions to use for ``category``.

    Positions are clamped to the available range; if the band is shorter than the
    requested offsets the triplet may contain duplicates (still three entries) so
    the sheet builder always has something to place.  Callers dedupe.
    """
    if n_frames <= 0:
        return []
    if category not in _RULES:
        raise ValueError(
            f"Unknown category {category!r}; expected one of {sorted(_RULES)}")
    primary, fallback = _RULES[category]
    triplet = fallback if use_fallback else primary
    return [_resolve_offset(anchor, off, n_frames) for anchor, off in triplet]


def select_frames(frames: Sequence, category: str,
                  use_fallback: bool = False) -> list:
    """Map :func:`select_frame_positions` onto an ordered ``frames`` sequence.

    ``frames`` may be frame indices, paths, images, or dicts -- anything
    indexable.  Returns the three selected items in sheet order.
    """
    positions = select_frame_positions(len(frames), category, use_fallback)
    return [frames[p] for p in positions]


# ---------------------------------------------------------------------------
# Sheet assembly
# ---------------------------------------------------------------------------

def _prepare_crops(crops: Sequence[np.ndarray]) -> list:
    """Validate + normalise crops to 3-channel BGR (grayscale/BGRA promoted)."""
    valid = [c for c in crops if c is not None and getattr(c, "size", 0) > 0]
    if not valid:
        raise ValueError("sheet builder requires at least one crop")

    def _to_bgr(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    return [_to_bgr(c) for c in valid]


def build_vertical_sheet(
    crops: Sequence[np.ndarray],
    spacing: int = SHEET_SPACING_PX,
    bg_color=SHEET_BG_COLOR,
) -> np.ndarray:
    """Stack ``crops`` top-to-bottom on one white canvas with ``spacing`` px gaps.

    Every crop is copied in **verbatim** -- no resizing, no per-crop padding, no
    blending.  Crops of differing widths are left-aligned; the canvas width is the
    widest crop.  Grayscale crops are promoted to 3-channel.

    Vertical stacking keeps each plate on its own horizontal band, so Rekognition
    reads each crop as a separate LINE instead of running three side-by-side
    numbers together.

    Raises ``ValueError`` if ``crops`` is empty or all ``None``.
    """
    prepared = _prepare_crops(crops)
    total_width = max(c.shape[1] for c in prepared)
    total_height = sum(c.shape[0] for c in prepared) + spacing * (len(prepared) - 1)

    sheet = np.full((total_height, total_width, 3), bg_color, dtype=np.uint8)

    y = 0
    for crop in prepared:
        h, w = crop.shape[:2]
        sheet[y:y + h, 0:w] = crop  # left-aligned, pixel-exact copy
        y += h + spacing
    return sheet


def build_horizontal_sheet(
    crops: Sequence[np.ndarray],
    spacing: int = SHEET_SPACING_PX,
    bg_color=SHEET_BG_COLOR,
) -> np.ndarray:
    """Lay ``crops`` left-to-right on one white canvas with ``spacing`` px gaps.

    Same verbatim-copy guarantees as :func:`build_vertical_sheet`, laid out
    horizontally: crops are top-aligned and the canvas height is the tallest.
    """
    prepared = _prepare_crops(crops)
    total_height = max(c.shape[0] for c in prepared)
    total_width = sum(c.shape[1] for c in prepared) + spacing * (len(prepared) - 1)

    sheet = np.full((total_height, total_width, 3), bg_color, dtype=np.uint8)

    x = 0
    for crop in prepared:
        h, w = crop.shape[:2]
        sheet[0:h, x:x + w] = crop  # top-aligned, pixel-exact copy
        x += w + spacing
    return sheet


def build_sheet(
    crops: Sequence[np.ndarray],
    orientation: str = "vertical",
    spacing: int = SHEET_SPACING_PX,
    bg_color=SHEET_BG_COLOR,
) -> np.ndarray:
    """Build a sheet in the given ``orientation`` (default ``"vertical"``)."""
    if orientation == "vertical":
        return build_vertical_sheet(crops, spacing, bg_color)
    if orientation == "horizontal":
        return build_horizontal_sheet(crops, spacing, bg_color)
    raise ValueError(
        f"orientation must be 'vertical' or 'horizontal', got {orientation!r}")


# ---------------------------------------------------------------------------
# High-level assembly + preprocessing
# ---------------------------------------------------------------------------

def _default_frame_loader(frame_ref) -> Optional[np.ndarray]:
    """Load a frame reference (path or already-decoded ndarray) into BGR."""
    if isinstance(frame_ref, np.ndarray):
        return frame_ref
    if isinstance(frame_ref, (str, os.PathLike)):
        return cv2.imread(os.fspath(frame_ref))
    raise TypeError(f"Cannot load frame reference of type {type(frame_ref)!r}")


def build_sheet_from_frames(
    frames: Sequence,
    category: str,
    use_fallback: bool = False,
    crop_fn: Optional[Callable] = None,
    loader: Callable = _default_frame_loader,
    spacing: int = SHEET_SPACING_PX,
    orientation: str = "vertical",
) -> Optional[np.ndarray]:
    """Select three frames from ``frames`` and build one sheet.

    Returns the assembled sheet (BGR), or ``None`` if no usable crop could be
    produced (e.g. every selected frame failed to load).
    """
    selected = select_frames(frames, category, use_fallback)
    crops = []
    for ref in selected:
        img = loader(ref)
        if img is None:
            continue
        crop = crop_fn(img) if crop_fn is not None else img
        if crop is None or crop.size == 0:
            continue
        crops.append(crop)
    if not crops:
        return None
    return build_sheet(crops, orientation=orientation, spacing=spacing)


def preprocess_sheet(sheet: np.ndarray) -> np.ndarray:
    """Run the assembled sheet through the primary OCR preprocessing."""
    return Preprocessor.primary(sheet)
