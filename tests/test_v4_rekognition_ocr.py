"""Rekognition wagon-number OCR: V4's algorithm, driven by a fake client.

The V4 Train-Inspection-Engine reads wagon numbers with AWS Rekognition
``DetectText``, not EasyOCR, and the *how* matters as much as the *what*:

* One plate crossing the frame is ONE band of detections, and the three frames
  sheeted from that band are chosen by the rake's LOAD state -- cargo occludes
  the plate from one end of the band depending on travel direction.  Getting the
  triplet order wrong silently lowers the read rate rather than failing.
* The three crops are stacked VERTICALLY so Rekognition returns one ``LINE`` per
  crop.  The reader must then pick the single best VALID 11-digit line; naively
  concatenating all three produces a 33-digit run that fails validation, which
  is the specific bug the sheet design exists to avoid.
* Two-row plates (one physical plate printed across two lines) must still
  reassemble, so consecutive-line runs are tried when no single line validates.

None of these tests touch AWS: a fake client returns canned ``TextDetections``,
so the assertions are about this pipeline's logic, not Rekognition's accuracy.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                  # noqa: E402

from core import constants as C                                     # noqa: E402
from features.inference_lib import rekognition_reader as RR          # noqa: E402
from features.inference_lib import three_frame_sheet as TFS          # noqa: E402
from features.inference_lib.ocr_preprocessor import Preprocessor     # noqa: E402
from features.inference_lib import rekognition_wagon_number as RWN   # noqa: E402
from features.ocr import processor as OCR                            # noqa: E402


def _line(text, top, conf=99.0):
    return {"Type": "LINE", "DetectedText": text,
            "Confidence": conf,
            "Geometry": {"BoundingBox": {"Top": top, "Left": 0.0,
                                         "Width": 1.0, "Height": 0.1}}}


class _FakeRekognition:
    """Returns a scripted DetectText response; records every call."""

    def __init__(self, detections):
        self._detections = detections
        self.calls = 0

    def detect_text(self, image_bytes):
        self.calls += 1
        d = self._detections
        return d(self.calls) if callable(d) else d


def _frame(h=80, w=240):
    return np.full((h, w, 3), 200, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Digit reading
# ---------------------------------------------------------------------------

class TestReadDigits(unittest.TestCase):

    def test_single_valid_line_wins(self):
        client = _FakeRekognition([
            _line("12345678901", 0.10, 98.0),
            _line("999", 0.50, 99.9),
        ])
        text, conf = RR.read_digits(client, _frame(),
                                    validator=RWN.is_valid_wagon_number)
        self.assertEqual(text, "12345678901")
        self.assertAlmostEqual(conf, 0.98, places=2)

    def test_three_crop_sheet_is_not_concatenated(self):
        """The whole point of the sheet + validator: pick one, never glue three."""
        n = "12345678901"
        client = _FakeRekognition([_line(n, 0.1), _line(n, 0.4), _line(n, 0.7)])
        text, _ = RR.read_digits(client, _frame(),
                                 validator=RWN.is_valid_wagon_number)
        self.assertEqual(text, n)
        self.assertNotEqual(len(text), 33)

    def test_two_row_plate_reassembles_from_consecutive_lines(self):
        client = _FakeRekognition([_line("221423", 0.10), _line("25759", 0.20)])
        text, _ = RR.read_digits(client, _frame(),
                                 validator=RWN.is_valid_wagon_number)
        self.assertEqual(text, "22142325759")
        self.assertTrue(RWN.is_valid_wagon_number(text))

    def test_rows_are_ordered_top_to_bottom_not_by_arrival(self):
        client = _FakeRekognition([_line("25759", 0.20), _line("221423", 0.10)])
        text, _ = RR.read_digits(client, _frame(),
                                 validator=RWN.is_valid_wagon_number)
        self.assertEqual(text, "22142325759")

    def test_non_digits_are_stripped(self):
        client = _FakeRekognition([_line("WR 1234-5678-901", 0.1)])
        text, _ = RR.read_digits(client, _frame(),
                                 validator=RWN.is_valid_wagon_number)
        self.assertEqual(text, "12345678901")

    def test_without_a_validator_lines_are_concatenated(self):
        client = _FakeRekognition([_line("111", 0.1), _line("222", 0.2)])
        text, _ = RR.read_digits(client, _frame())
        self.assertEqual(text, "111222")

    def test_no_detections_is_empty_not_an_error(self):
        text, conf = RR.read_digits(_FakeRekognition([]), _frame())
        self.assertEqual(text, "")
        self.assertEqual(conf, 0.0)

    def test_word_type_detections_are_ignored(self):
        client = _FakeRekognition([
            {"Type": "WORD", "DetectedText": "99999999999", "Confidence": 99.0,
             "Geometry": {"BoundingBox": {"Top": 0.1}}},
            _line("12345678901", 0.2),
        ])
        text, _ = RR.read_digits(client, _frame(),
                                 validator=RWN.is_valid_wagon_number)
        self.assertEqual(text, "12345678901")

    def test_client_failure_degrades_to_empty(self):
        class _Boom:
            def detect_text(self, _b):
                raise RuntimeError("throttled")
        text, conf = RR.read_digits(_Boom(), _frame())
        self.assertEqual((text, conf), ("", 0.0))


# ---------------------------------------------------------------------------
# Frame selection + sheet assembly (V4's rules, verbatim)
# ---------------------------------------------------------------------------

class TestFrameSelection(unittest.TestCase):

    def test_loaded_primary_reads_from_the_END_of_the_band(self):
        pos = TFS.select_frame_positions(20, "loaded", use_fallback=False)
        self.assertEqual(pos, [12, 14, 16])   # End-7, End-5, End-3

    def test_loaded_fallback_reads_from_the_START(self):
        pos = TFS.select_frame_positions(20, "loaded", use_fallback=True)
        self.assertEqual(pos, [3, 5, 7])

    def test_empty_is_the_mirror_of_loaded(self):
        self.assertEqual(TFS.select_frame_positions(20, "empty", False), [3, 5, 7])
        self.assertEqual(TFS.select_frame_positions(20, "empty", True),
                         [12, 14, 16])

    def test_short_band_clamps_instead_of_indexing_out_of_range(self):
        pos = TFS.select_frame_positions(3, "loaded", False)
        self.assertEqual(len(pos), 3)
        self.assertTrue(all(0 <= p <= 2 for p in pos))

    def test_vertical_sheet_stacks_crops_verbatim(self):
        a = np.full((10, 30, 3), 1, dtype=np.uint8)
        b = np.full((20, 50, 3), 2, dtype=np.uint8)
        sheet = TFS.build_vertical_sheet([a, b], spacing=20)
        self.assertEqual(sheet.shape, (10 + 20 + 20, 50, 3))
        # crops are copied pixel-exact, left-aligned
        self.assertTrue((sheet[0:10, 0:30] == 1).all())
        self.assertTrue((sheet[30:50, 0:50] == 2).all())

    def test_empty_crop_list_is_rejected(self):
        with self.assertRaises(ValueError):
            TFS.build_vertical_sheet([])

    def test_preprocessor_upscales_and_meets_min_width(self):
        out = Preprocessor.primary(np.full((10, 20, 3), 128, dtype=np.uint8))
        self.assertGreaterEqual(out.shape[1], Preprocessor.OCR_TARGET_WIDTH)


# ---------------------------------------------------------------------------
# Banding + the per-wagon reader
# ---------------------------------------------------------------------------

class TestBandingAndReader(unittest.TestCase):

    @staticmethod
    def _dets(frames):
        return [{"frame": f, "confidence": 0.8, "bbox": [10, 10, 60, 30]}
                for f in frames]

    def test_contiguous_detections_form_one_band(self):
        bands = RWN.group_detections_into_bands(
            self._dets(range(100, 112)), gap_tolerance=8)
        self.assertEqual(len(bands), 1)
        self.assertEqual(bands[0]["frame_count"], 12)

    def test_a_wide_gap_splits_bands(self):
        bands = RWN.group_detections_into_bands(
            self._dets(list(range(100, 106)) + list(range(300, 306))),
            gap_tolerance=8)
        self.assertEqual(len(bands), 2)

    def test_no_detections_no_bands(self):
        self.assertEqual(RWN.group_detections_into_bands([]), [])

    def test_reader_returns_a_valid_number_and_v4_fields(self):
        client = _FakeRekognition([_line("12345678901", 0.1)])
        bands = RWN.group_detections_into_bands(self._dets(range(100, 112)))
        out = RWN.RekognitionWagonNumberOCR(client).read_wagon_number(
            bands=bands, frame_loader=lambda _fi: _frame(),
            is_loaded=True, kind="wagon")
        self.assertTrue(out["is_valid_11_digit"])
        self.assertEqual(out["display_number"], "12345678901")
        self.assertEqual(out["wagon_identifier"], "12345678901")
        self.assertEqual(out["engine"], "rekognition")
        for key in ("raw_number", "confidence", "ocr_confidence", "band_id",
                    "best_frame", "best_bbox", "fallback_triggered"):
            self.assertIn(key, out)

    def test_first_valid_sheet_wins_without_extra_calls(self):
        client = _FakeRekognition([_line("12345678901", 0.1)])
        bands = RWN.group_detections_into_bands(self._dets(range(100, 112)))
        RWN.RekognitionWagonNumberOCR(client).read_wagon_number(
            bands=bands, frame_loader=lambda _fi: _frame(), is_loaded=True)
        self.assertEqual(client.calls, 1, "escalated after a valid read")

    def test_fallback_triplet_is_tried_when_the_primary_fails(self):
        def script(n):
            return [_line("123", 0.1)] if n == 1 else [_line("12345678901", 0.1)]
        client = _FakeRekognition(script)
        bands = RWN.group_detections_into_bands(self._dets(range(100, 112)))
        out = RWN.RekognitionWagonNumberOCR(client).read_wagon_number(
            bands=bands, frame_loader=lambda _fi: _frame(), is_loaded=True)
        self.assertTrue(out["is_valid_11_digit"])
        self.assertTrue(out["fallback_triggered"])
        self.assertEqual(client.calls, 2)

    def test_invalid_read_is_reported_not_invented(self):
        client = _FakeRekognition([_line("1234", 0.1)])
        bands = RWN.group_detections_into_bands(self._dets(range(100, 112)))
        out = RWN.RekognitionWagonNumberOCR(client).read_wagon_number(
            bands=bands, frame_loader=lambda _fi: _frame(), is_loaded=False)
        self.assertFalse(out["is_valid_11_digit"])
        self.assertEqual(out["display_number"], "-")
        self.assertEqual(out["wagon_identifier"], C.NO_DATA)
        self.assertEqual(out["raw_number"], "1234")

    def test_call_budget_is_bounded_per_wagon(self):
        """A 58-wagon rake must not fan out into unbounded API spend."""
        client = _FakeRekognition([_line("1", 0.1)])
        bands = RWN.group_detections_into_bands(
            self._dets(list(range(100, 112)) + list(range(300, 312))
                       + list(range(500, 512)) + list(range(700, 712))))
        RWN.RekognitionWagonNumberOCR(client, max_calls=3).read_wagon_number(
            bands=bands, frame_loader=lambda _fi: _frame(), is_loaded=True)
        self.assertLessEqual(client.calls, 3)

    def test_unloadable_frames_do_not_crash(self):
        client = _FakeRekognition([_line("12345678901", 0.1)])
        bands = RWN.group_detections_into_bands(self._dets(range(100, 112)))
        out = RWN.RekognitionWagonNumberOCR(client).read_wagon_number(
            bands=bands, frame_loader=lambda _fi: None, is_loaded=True)
        self.assertFalse(out["is_valid_11_digit"])


# ---------------------------------------------------------------------------
# Engine selection in the processor
# ---------------------------------------------------------------------------

class TestEngineSelection(unittest.TestCase):

    def setUp(self):
        self._saved = os.environ.pop("WAGONEYE_OCR_ENGINE", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("WAGONEYE_OCR_ENGINE", None)
        else:
            os.environ["WAGONEYE_OCR_ENGINE"] = self._saved

    def test_rekognition_is_the_default(self):
        self.assertEqual(OCR.resolve_engine(), OCR.ENGINE_REKOGNITION)

    def test_env_selects_easyocr(self):
        os.environ["WAGONEYE_OCR_ENGINE"] = "easyocr"
        self.assertEqual(OCR.resolve_engine(), OCR.ENGINE_EASYOCR)

    def test_explicit_argument_beats_env(self):
        os.environ["WAGONEYE_OCR_ENGINE"] = "easyocr"
        self.assertEqual(OCR.resolve_engine("rekognition"),
                         OCR.ENGINE_REKOGNITION)

    def test_unknown_value_degrades_to_the_default(self):
        os.environ["WAGONEYE_OCR_ENGINE"] = "tesseract"
        self.assertEqual(OCR.resolve_engine(), OCR.ENGINE_REKOGNITION)

    def test_aliases_resolve(self):
        for alias in ("aws", "detect_text", "REKOGNITION"):
            self.assertEqual(OCR.resolve_engine(alias), OCR.ENGINE_REKOGNITION)
        for alias in ("local", "EasyOCR"):
            self.assertEqual(OCR.resolve_engine(alias), OCR.ENGINE_EASYOCR)


# ---------------------------------------------------------------------------
# Detector + load-state resolution
# ---------------------------------------------------------------------------

class TestDetectorAndLoadState(unittest.TestCase):

    def test_v4s_plate_detector_is_preferred(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, C.MODEL_WAGON_NUMBER), "wb").close()
            open(os.path.join(d, C.MODEL_WAGON_ID_COUNTING), "wb").close()
            self.assertEqual(os.path.basename(OCR._detector_path(d)),
                             C.MODEL_WAGON_NUMBER)

    def test_legacy_detector_name_still_satisfies_the_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, C.MODEL_WAGON_ID_COUNTING), "wb").close()
            self.assertEqual(os.path.basename(OCR._detector_path(d)),
                             C.MODEL_WAGON_ID_COUNTING)

    def test_missing_detector_reports_the_canonical_name(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(os.path.basename(OCR._detector_path(d)),
                             C.MODEL_WAGON_NUMBER)

    def test_load_state_comes_from_this_runs_own_load_output(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "load"))
            with open(os.path.join(d, "load", "GW_1.json"), "w") as f:
                json.dump({"status": C.STATUS_OK,
                           "load_status": C.LOAD_LOADED}, f)
            with open(os.path.join(d, "load", "GW_2.json"), "w") as f:
                json.dump({"status": C.STATUS_OK,
                           "load_status": C.LOAD_EMPTY}, f)
            self.assertTrue(OCR._wagon_is_loaded(d, "GW_1"))
            self.assertFalse(OCR._wagon_is_loaded(d, "GW_2"))

    def test_missing_load_result_defaults_to_empty(self):
        """V4's default too -- never guess LOADED, which reorders the triplet."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(OCR._wagon_is_loaded(d, "GW_9"))

    def test_failed_load_result_defaults_to_empty(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "load"))
            with open(os.path.join(d, "load", "GW_1.json"), "w") as f:
                json.dump({"status": C.STATUS_FAILED,
                           "load_status": C.LOAD_LOADED}, f)
            self.assertFalse(OCR._wagon_is_loaded(d, "GW_1"))


if __name__ == "__main__":
    unittest.main()
