"""Load aggregation, extracted, must produce an identical verdict.

`_aggregate_camera` classifies frames from a materialized wagon cache AND votes
them into a LOADED / EMPTY verdict in one loop. Only the classification needs
frames. The extraction is proven the way Damage and GapTracker were: run the OLD
function over a real wagon-cache fixture, run the same classifier over the same
frames to collect its outputs, feed those to the new pure aggregator, and
require an identical result. A non-vacuity guard refuses an empty comparison.

The second class tests the rule that is easiest to break silently: the loaded
ratio is measured against every frame the classifier LOOKED at, not against the
frames that voted.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

import cv2
import numpy as np

from core import constants as C
from features.load.aggregate import (
    LOADED_RATIO_THRESHOLD, LoadClassification,
    aggregate_load_from_classifications, aggregate_load_from_observations,
    canonical_load, classifications_from_observations,
)

CAM = C.CAMERA_RIGHT_UP_TOP
GW = "GW_1"
W, H = 320, 240
EVERY_NTH = 2


class _P:
    def __init__(self, top1, conf):
        self.top1, self.top1conf = top1, conf


class _Res:
    def __init__(self, probs):
        self.probs, self.boxes = probs, None


class StubLoadModel:
    """Three brightness bands -> loaded / an unvotable label / empty.

    Keyed on brightness, not an index painted into a pixel: the cache is JPEG
    and lossy, so a single index pixel does not survive the round trip while the
    mean of a flat field does. The middle band emits a label that canonicalises
    to NO_DATA, which is what makes the denominator observable.
    """
    names = {0: "loaded", 1: "locono", 2: "empty"}

    def __init__(self):
        self.calls = 0

    def __call__(self, frame, verbose=False):
        self.calls += 1
        m = float(frame.mean())
        if m > 170:
            return [_Res(_P(0, 0.91))]
        if m > 80:
            return [_Res(_P(1, 0.55))]
        return [_Res(_P(2, 0.77))]


def _wagon_cache(root, n_frames=60):
    """A real wagon cache in the layout `iter_wagon_frames` expects."""
    d = os.path.join(root, GW, C.CAMERA_FOLDER[CAM])
    os.makedirs(d, exist_ok=True)
    for i in range(n_frames):
        val = 210 if i % 3 == 0 else (120 if i % 3 == 1 else 30)
        cv2.imwrite(os.path.join(d, "frame_%06d.jpg" % i),
                    np.full((H, W, 3), val, dtype=np.uint8))
    return root


class TestLoadAggregationEquivalence(unittest.TestCase):
    """Old fused implementation versus extracted aggregation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = _wagon_cache(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _old(self):
        from features.load.processor import _aggregate_camera
        model = StubLoadModel()
        out = _aggregate_camera(model, self.cache, GW, CAM,
                                every_nth=EVERY_NTH, max_frames=None)
        return out, model

    def _collect(self, frames_wanted=False):
        """Phase 1: the same classifier over the same frames, nothing more."""
        from features._common import iter_wagon_frames, run_classification
        model = StubLoadModel()
        recs, imgs = [], {}
        for fi, frame in iter_wagon_frames(self.cache, GW, CAM,
                                           every_nth=EVERY_NTH,
                                           max_frames=None, trim_stable=True):
            cls, conf = run_classification(model, frame)
            recs.append(LoadClassification(int(fi), cls, float(conf)))
            if frames_wanted:
                imgs[int(fi)] = frame
        return recs, imgs, model

    def test_the_corpus_is_not_empty_and_exercises_every_branch(self):
        (cls, conf, used, n_l, n_e, _bl, _be), _m = self._old()
        self.assertGreater(used, 0, "no frame was classified")
        self.assertGreater(n_l, 0, "no LOADED vote")
        self.assertGreater(n_e, 0, "no EMPTY vote")
        self.assertGreater(used, n_l + n_e,
                           "no unvotable frame: the denominator is untested")
        self.assertNotEqual(cls, C.NO_DATA)

    def test_the_same_frames_are_seen(self):
        (_c, _cf, used, _l, _e, _bl, _be), _m = self._old()
        recs, _i, _m2 = self._collect()
        self.assertEqual(len(recs), used,
                         "collection saw a different frame set than the old "
                         "path; the comparison would not be like-for-like")

    def test_verdict_counts_and_confidence_are_identical(self):
        (cls, conf, used, n_l, n_e, _bl, _be), _m = self._old()
        recs, _i, _m2 = self._collect()
        new = aggregate_load_from_classifications(recs, camera_id=CAM)
        self.assertEqual(new.load_status, cls)
        self.assertAlmostEqual(new.confidence, conf, places=9)
        self.assertEqual(new.frames_used, used)
        self.assertEqual(new.loaded_count, n_l)
        self.assertEqual(new.empty_count, n_e)

    def test_the_legacy_tuple_shape_is_reproduced_positionally(self):
        old, _m = self._old()
        recs, _i, _m2 = self._collect()
        new = aggregate_load_from_classifications(recs, camera_id=CAM).as_tuple()
        self.assertEqual(old[:5], new[:5])

    def test_the_per_camera_row_matches_what_the_processor_writes(self):
        (cls, conf, used, n_l, n_e, _bl, _be), _m = self._old()
        old_row = {
            "load_status":  cls,
            "confidence":   round(float(conf), 4),
            "frames_used":  used,
            "loaded_count": n_l,
            "empty_count":  n_e,
            "loaded_ratio": round(n_l / used, 4) if used else 0.0,
        }
        recs, _i, _m2 = self._collect()
        new = aggregate_load_from_classifications(recs, camera_id=CAM)
        self.assertEqual(old_row, new.per_camera_row())

    def test_the_same_best_frames_are_chosen(self):
        (_c, _cf, _u, _l, _e, b_l, b_e), _m = self._old()
        recs, imgs, _m2 = self._collect(frames_wanted=True)
        new = aggregate_load_from_classifications(recs, camera_id=CAM,
                                                  frames=imgs)
        self.assertEqual(new.best_loaded.frame_idx, b_l.frame_idx)
        self.assertEqual(new.best_empty.frame_idx, b_e.frame_idx)
        self.assertAlmostEqual(new.best_loaded.score, b_l.score, places=9)
        self.assertAlmostEqual(new.best_empty.score, b_e.score, places=9)
        self.assertTrue(new.best_loaded.has_data())
        self.assertTrue(new.best_empty.has_data())

    def test_the_best_index_is_known_even_with_no_images(self):
        """Deferred snapshots: the index is decided without the frame."""
        (_c, _cf, _u, _l, _e, b_l, b_e), _m = self._old()
        recs, _i, _m2 = self._collect()
        new = aggregate_load_from_classifications(recs, camera_id=CAM)
        self.assertEqual(new.best_loaded_idx, b_l.frame_idx)
        self.assertEqual(new.best_empty_idx, b_e.frame_idx)
        self.assertFalse(new.best_loaded.has_data())

    def test_observations_route_to_the_same_result(self):
        from core.timeline_evidence import Observation
        recs, _i, _m2 = self._collect()
        direct = aggregate_load_from_classifications(recs, camera_id=CAM)
        obs = [Observation(camera_id=CAM, kind="load",
                           t_start=r.frame_idx / 15.0, confidence=r.confidence,
                           local_frame=r.frame_idx, label=r.class_name)
               for r in recs]
        self.assertEqual(len(classifications_from_observations(obs)), len(recs))
        routed = aggregate_load_from_observations(obs, camera_id=CAM)
        self.assertEqual(routed.as_tuple()[:5], direct.as_tuple()[:5])


class TestTheDenominatorIsFramesUsed(unittest.TestCase):
    """The ratio is over frames LOOKED AT, not frames that voted.

    A discriminating case, stated as a fixture rather than hoped for: 10 loaded,
    5 empty and 14 unvotable frames. Against frames_used (29) the ratio is
    0.345, at or below the 0.35 threshold, so the wagon is EMPTY. Against the
    voting frames only (15) it would be 0.667 and the wagon would be LOADED.
    Getting this wrong flips real wagons from empty to loaded.
    """

    def _mix(self, n_loaded, n_empty, n_other):
        recs, i = [], 0
        for _ in range(n_loaded):
            recs.append(LoadClassification(i, "loaded", 0.9))
            i += 1
        for _ in range(n_empty):
            recs.append(LoadClassification(i, "empty", 0.8))
            i += 1
        for _ in range(n_other):
            recs.append(LoadClassification(i, "locono", 0.5))
            i += 1
        return recs

    def test_unvotable_frames_dilute_the_loaded_ratio(self):
        recs = self._mix(10, 5, 14)
        agg = aggregate_load_from_classifications(recs, camera_id=CAM)
        self.assertEqual(agg.frames_used, 29)
        self.assertEqual(agg.load_status, C.LOAD_EMPTY)
        self.assertGreater(10 / 15, LOADED_RATIO_THRESHOLD,
                           "the fixture no longer discriminates")

    def test_removing_them_flips_the_verdict(self):
        agg = aggregate_load_from_classifications(self._mix(10, 5, 0),
                                                  camera_id=CAM)
        self.assertEqual(agg.load_status, C.LOAD_LOADED)

    def test_the_threshold_is_strictly_greater_than(self):
        """35 of 100 is NOT loaded; 36 is."""
        self.assertEqual(
            aggregate_load_from_classifications(self._mix(35, 65, 0),
                                                camera_id=CAM).load_status,
            C.LOAD_EMPTY)
        self.assertEqual(
            aggregate_load_from_classifications(self._mix(36, 64, 0),
                                                camera_id=CAM).load_status,
            C.LOAD_LOADED)

    def test_confidence_is_the_winning_side_mean_only(self):
        agg = aggregate_load_from_classifications(self._mix(10, 5, 0),
                                                  camera_id=CAM)
        self.assertAlmostEqual(agg.confidence, 0.9, places=9)

    def test_no_frames_is_no_data(self):
        agg = aggregate_load_from_classifications([], camera_id=CAM)
        self.assertEqual(agg.load_status, C.NO_DATA)
        self.assertEqual(agg.frames_used, 0)
        self.assertEqual(agg.per_camera_row()["loaded_ratio"], 0.0)

    def test_only_unvotable_frames_is_no_data(self):
        agg = aggregate_load_from_classifications(self._mix(0, 0, 7),
                                                  camera_id=CAM)
        self.assertEqual(agg.load_status, C.NO_DATA)
        self.assertEqual(agg.frames_used, 7)

    def test_the_threshold_still_matches_the_processor(self):
        from features.load import processor
        self.assertEqual(processor._LOADED_RATIO_THRESHOLD,
                         LOADED_RATIO_THRESHOLD)

    def test_canonicalisation_matches_the_processor(self):
        from features.load import processor
        for raw in ("loaded", "LOADED", " full ", "empty", "unload", "locono",
                    "", "engine_head"):
            with self.subTest(raw=raw):
                self.assertEqual(canonical_load(raw),
                                 processor._canonical_load(raw))


class TestNoInferenceInLoadAggregation(unittest.TestCase):
    """Structural: the aggregator cannot decode or infer, by construction."""

    def test_no_video_decode_and_no_model_call(self):
        import ast
        import inspect
        from features.load import aggregate
        src = inspect.getsource(aggregate)
        called = {ast.unparse(n.func) for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.Call)}
        for banned in ("VideoCapture", "load_yolo", "YOLO", "imread",
                       "iter_wagon_frames", "list_wagon_frames", "predict",
                       "run_classification"):
            with self.subTest(token=banned):
                self.assertFalse([c for c in called if banned in c],
                                 "%s appears in the aggregator" % banned)

    def test_it_takes_no_cache_root_and_no_gw_id(self):
        import inspect
        p = inspect.signature(aggregate_load_from_classifications).parameters
        for banned in ("cache_root", "gw_id", "model", "every_nth",
                       "max_frames"):
            self.assertNotIn(banned, p)

    def test_the_model_is_not_invoked_during_aggregation(self):
        with tempfile.TemporaryDirectory() as root:
            cache = _wagon_cache(root)
            from features._common import iter_wagon_frames, run_classification
            model = StubLoadModel()
            recs = []
            for fi, frame in iter_wagon_frames(cache, GW, CAM,
                                               every_nth=EVERY_NTH,
                                               max_frames=None,
                                               trim_stable=True):
                cls, conf = run_classification(model, frame)
                recs.append(LoadClassification(int(fi), cls, float(conf)))
            after_inference = model.calls
            self.assertGreater(after_inference, 0)
            aggregate_load_from_classifications(recs, camera_id=CAM)
            self.assertEqual(model.calls, after_inference,
                             "the model ran again during aggregation")


if __name__ == "__main__":
    unittest.main()
