"""Shared AWS Rekognition digit-plate reading.

Ported from the V4 Train-Inspection-Engine
`inspection/ocr/rekognition_reader.py`.

Replaces EasyOCR's ``readtext(image, allowlist="0123456789")``.  Rekognition has
no digit allowlist, so every ``LINE`` detection is filtered down to its digit
characters after the call.

Two things shape how the ``LINE`` detections are turned into a number:

* **Multi-row single plates** -- a plate is sometimes stacked across two rows
  (e.g. "221423" over "25759").  Rekognition segments those as separate ``LINE``
  detections, so rows are re-ordered top-to-bottom by their bounding box before
  concatenating -- the job EasyOCR's center-Y clustering used to do.
* **Three-frame sheets** -- when the image is a vertical sheet of three plate
  crops (see :mod:`three_frame_sheet`), Rekognition returns one (or two) LINEs
  *per crop*.  Blindly concatenating **all** of them would fuse three copies of
  the same number into a 22/33-digit run that fails validation.  So when a
  ``validator`` is supplied, :func:`read_digits` instead **picks the single best
  valid-length reading** -- the highest-confidence individual LINE that passes the
  validator, or, failing that, the highest-confidence run of vertically-
  consecutive LINEs that passes it (covering two-row plates on the sheet).  Only
  if nothing validates does it fall back to the legacy concatenate-everything
  behaviour, so the caller still gets a best-effort string to report.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

_DIGITS_RE = re.compile(r"\D")

#: JPEG quality used when encoding the sheet for the DetectText request.
ENCODE_QUALITY = 95


def digits_only(text: str) -> str:
    """Strip every non-digit character."""
    return _DIGITS_RE.sub("", text or "")


def _pick_best_valid(lines: List[Tuple[float, str, float]],
                     validator: Callable[[str], bool]):
    """Return the best ``(digits, confidence_0_1)`` that passes ``validator``.

    ``lines`` is ``(top, digits, confidence_0_100)`` sorted top-to-bottom.
    Preference order:
      1. the highest-confidence **single** LINE that validates;
      2. otherwise the highest-confidence run of **consecutive** LINEs (smallest
         run first) whose concatenation validates -- this reconstructs a plate
         split across two rows without gluing separate crops together.
    Returns ``None`` when nothing validates.
    """
    # 1) single LINE candidates
    best = None
    for _, digits, conf in lines:
        if validator(digits) and (best is None or conf > best[1]):
            best = (digits, conf)
    if best is not None:
        return best[0], best[1] / 100.0

    # 2) consecutive-run candidates (two-row plates); prefer smallest run,
    #    then highest average confidence.
    n = len(lines)
    for size in range(2, n + 1):
        best = None
        for i in range(0, n - size + 1):
            window = lines[i:i + size]
            text = "".join(d for _, d, _ in window)
            if not validator(text):
                continue
            avg = sum(c for _, _, c in window) / size
            if best is None or avg > best[1]:
                best = (text, avg)
        if best is not None:
            return best[0], best[1] / 100.0
    return None


def read_digits(
    client,
    image: np.ndarray,
    validator: Optional[Callable[[str], bool]] = None,
) -> Tuple[str, float]:
    """Run DetectText on a BGR image, return ``(digits_text, confidence_0_1)``.

    When ``validator`` is given (e.g. ``is_valid_wagon_number``), the single best
    valid-length reading is returned instead of concatenating every LINE -- see
    the module docstring.  Without a validator the legacy behaviour (all LINEs
    joined top-to-bottom) is used.

    Never raises: an encode failure, a throttle, or any botocore error degrades to
    ``("", 0.0)`` so one bad wagon can never abort the Stage-3 run.
    """
    if client is None or image is None or getattr(image, "size", 0) == 0:
        return "", 0.0

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY),
                                           ENCODE_QUALITY])
    if not ok:
        return "", 0.0

    try:
        detections = client.detect_text(buf.tobytes())
    except Exception:
        return "", 0.0

    lines: List[Tuple[float, str, float]] = []
    for det in detections:
        if det.get("Type") != "LINE":
            continue
        digits = digits_only(det.get("DetectedText", ""))
        if not digits:
            continue
        try:
            top = det["Geometry"]["BoundingBox"]["Top"]
        except (KeyError, TypeError):
            top = 0.0
        lines.append((float(top), digits, float(det.get("Confidence", 0.0))))

    if not lines:
        return "", 0.0

    lines.sort(key=lambda item: item[0])  # top-to-bottom

    if validator is not None:
        picked = _pick_best_valid(lines, validator)
        if picked is not None:
            return picked

    # No validator, or nothing validated -- legacy best-effort concatenation.
    text = "".join(digits for _, digits, _ in lines)
    avg_conf = sum(c for _, _, c in lines) / len(lines) / 100.0
    return text, avg_conf
