"""Damage aggregation, extracted, must produce identical per-wagon records.

`_run_sampled_one_camera` scores frames AND votes the detections into damage
tracks in one loop. Only the scoring needs the video; the voting is arithmetic.
Welding them together is what forces feature inference to wait for a
materialized wagon cache.

The extraction is proven the way the GapTracker one was: run the OLD function
over a real wagon-cache fixture, take the detections it produced, feed those to
the new aggregator, and require identical records. A non-vacuity guard refuses
to let an empty comparison pass -- that guard is the only reason the GAP
equivalence number meant anything, and the same applies here.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

import cv2
import numpy as np

from core import constants as C
from features.damage.aggregate import (
    DamageDetection, aggregate_damage_from_detections,
    aggregate_damage_from_observations, detections_from_observations,
)

CAM = C.CAMERA_RIGHT_UP_TOP
GW = "GW_1"
W, H = 320, 240
STRIDE = 3


class _Boxes:
    def __init__(self, a, c, k):
        self.xyxy, self.conf, self.cls = _A(a), _A(c), _A(k)

    def __len__(self):
        return len(self.xyxy.numpy())


class _A:
    def __init__(self, a):
        self._a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Res:
    def __init__(self, b):
        self.boxes = b


class StubDamageModel:
    """Fires a centred box on BRIGHT frames.

    Keyed on overall brightness rather than an index painted into a pixel:
    the cache is written as JPEG, which is lossy, so a single index pixel does
    not survive the round trip. Brightness does, so old and new see exactly the
    same detections and the comparison is about aggregation.
    """
    names = {0: "inner_wall_damage"}

    def __init__(self, conf=0.82):
        self.conf = conf
        self.calls = 0

    def __call__(self, frame, verbose=False):
        self.calls += 1
        if float(frame.mean()) < 120:
            return [_Res(_Boxes(np.zeros((0, 4)), np.zeros(0),
                                np.zeros(0, int)))]
        return [_Res(_Boxes(np.array([[W * 0.3, H * 0.3, W * 0.7, H * 0.7]]),
                            np.array([self.conf]), np.array([0], dtype=int)))]


def _wagon_cache(root, hit_frames, n_frames=40):
    """A real wagon cache, the layout `iter_wagon_frames` expects.

    Frames in `hit_frames` are bright so the stub fires on them. Note the
    reader trims the noisy 5% at each end (`trim_stable=True`), so hits are
    chosen well inside the span.
    """
    d = os.path.join(root, GW, C.CAMERA_FOLDER[CAM])
    os.makedirs(d, exist_ok=True)
    hits = set(hit_frames)
    for i in range(n_frames):
        val = 210 if i in hits else 30
        cv2.imwrite(os.path.join(d, f"frame_{i:06d}.jpg"),
                    np.full((H, W, 3), val, dtype=np.uint8))
    return root


def _strip(records):
    """Comparable form: the snapshot is an ndarray, compared separately."""
    out = []
    for r in sorted(records, key=lambda x: (x["track_id"], x["first_frame"])):
        d = {k: v for k, v in r.items() if k != "_snapshot"}
        d["bbox"] = [round(float(v), 6) for v in (d["bbox"] or [])]
        d["confidence"] = round(float(d["confidence"]), 9)
        d["best_confidence"] = round(float(d["best_confidence"]), 9)
        out.append(d)
    return out


class TestDamageAggregationEquivalence(unittest.TestCase):
    """Old fused implementation versus extracted aggregation."""

    #: Well inside the trimmed span, and on the stride.
    HITS = [12, 15, 18, 21, 24, 27]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = _wagon_cache(self.tmp.name, self.HITS)

    def tearDown(self):
        self.tmp.cleanup()

    def _scored_frames(self):
        """Exactly the frames the production reader iterated, in order."""
        from features._common import list_wagon_frames
        paths = list_wagon_frames(self.cache, GW, CAM, trim_stable=True)
        idx = [int(os.path.basename(p)[6:12]) for p in paths]
        return idx[::STRIDE]

    def _old(self):
        """The production sampled path, unchanged, reading real frames."""
        from features.damage.processor import (
            DamageTrackerConfig, _run_sampled_one_camera,
        )
        model = StubDamageModel()
        records, used, fw, fh, frame_dets = _run_sampled_one_camera(
            model, DamageTrackerConfig(), self.cache, GW, CAM,
            confidence_floor=C.CONF_DAMAGE, sample_stride=STRIDE)
        return records, used, fw, fh, frame_dets, model

    def test_the_corpus_is_not_empty(self):
        """An empty comparison would prove nothing at all."""
        records, used, _fw, _fh, frame_dets, _m = self._old()
        self.assertGreater(used, 0, "no frame was scored")
        self.assertTrue(frame_dets, "the stub produced no detection")
        self.assertTrue(records, "the old path produced no damage record")

    def test_records_are_identical(self):
        old, used, fw, fh, frame_dets, _m = self._old()
        scored = sorted({d for d in self._scored_frames()})
        new = aggregate_damage_from_detections(
            [DamageDetection(frame_idx=d["frame_idx"],
                             class_name=d["class_name"],
                             confidence=d["confidence"], bbox=d["bbox"])
             for d in frame_dets],
            camera_id=CAM, frame_width=fw, frame_height=fh, stride=STRIDE,
            scored_frames=scored)
        self.assertEqual(_strip(old), _strip(new))
        self.assertTrue(_strip(new))

    def test_every_persisted_field_matches(self):
        old, _u, fw, fh, frame_dets, _m = self._old()
        scored = sorted({d for d in self._scored_frames()})
        new = aggregate_damage_from_detections(
            [DamageDetection(d["frame_idx"], d["class_name"], d["confidence"],
                             d["bbox"]) for d in frame_dets],
            camera_id=CAM, frame_width=fw, frame_height=fh, stride=STRIDE,
            scored_frames=scored)
        for a, b in zip(_strip(old), _strip(new)):
            for key in ("camera_id", "track_id", "class_name", "confidence",
                        "best_confidence", "total_hits", "first_frame",
                        "last_frame", "best_frame_idx", "bbox"):
                self.assertEqual(a[key], b[key], f"{key} diverged")

    def test_declaring_empty_scored_frames_is_faithful_not_decorative(self):
        """What `scored_frames` does, measured rather than assumed.

        Production calls `agg.add_frame(fi, [])` for a scored frame with no
        detection, so the aggregator sees every frame the model looked at. This
        passes that same list. On this corpus it makes no difference to the
        accepted groups -- the grouping follows the frames carrying
        observations -- and the test records that fact rather than asserting a
        behaviour the aggregator does not have.
        """
        scored = self._scored_frames()
        dets = [DamageDetection(f, "inner_wall_damage", 0.9,
                                [10, 10, 50, 50]) for f in scored[:3]]
        with_empties = aggregate_damage_from_detections(
            dets, camera_id=CAM, frame_width=W, frame_height=H, stride=STRIDE,
            scored_frames=scored)
        without = aggregate_damage_from_detections(
            dets, camera_id=CAM, frame_width=W, frame_height=H, stride=STRIDE)
        self.assertTrue(with_empties, "the comparison must not be empty")
        self.assertEqual(_strip(with_empties), _strip(without))

    def test_observations_route_to_the_same_result(self):
        from core.timeline_evidence import KIND_DAMAGE, Observation
        old, _u, fw, fh, frame_dets, _m = self._old()
        obs = [Observation(camera_id=CAM, kind=KIND_DAMAGE,
                           t_start=d["frame_idx"] / 15.0,
                           confidence=d["confidence"], local_frame=d["frame_idx"],
                           bbox=d["bbox"], label=d["class_name"])
               for d in frame_dets]
        new = aggregate_damage_from_observations(
            obs, camera_id=CAM, frame_width=fw, frame_height=fh,
            stride=STRIDE, scored_frames=self._scored_frames())
        self.assertEqual(_strip(old), _strip(new))

    def test_snapshots_are_attached_when_supplied(self):
        old, _u, fw, fh, frame_dets, _m = self._old()
        img = np.zeros((H, W, 3), dtype=np.uint8)
        best = old[0]["best_frame_idx"]
        new = aggregate_damage_from_detections(
            [DamageDetection(d["frame_idx"], d["class_name"], d["confidence"],
                             d["bbox"]) for d in frame_dets],
            camera_id=CAM, frame_width=fw, frame_height=fh, stride=STRIDE,
            scored_frames=self._scored_frames(),
            snapshots={best: img})
        self.assertIsNotNone(new[0]["_snapshot"])

    def test_no_snapshot_still_produces_every_other_field(self):
        old, _u, fw, fh, frame_dets, _m = self._old()
        new = aggregate_damage_from_detections(
            [DamageDetection(d["frame_idx"], d["class_name"], d["confidence"],
                             d["bbox"]) for d in frame_dets],
            camera_id=CAM, frame_width=fw, frame_height=fh, stride=STRIDE,
            scored_frames=self._scored_frames())
        self.assertIsNone(new[0]["_snapshot"])
        self.assertEqual(_strip(old), _strip(new))


class TestNoInferenceInAggregation(unittest.TestCase):
    """Structural: the aggregator cannot decode or infer, by construction."""

    def test_no_video_decode_and_no_model_call(self):
        import ast
        import inspect
        from features.damage import aggregate
        src = inspect.getsource(aggregate)
        called = {ast.unparse(n.func) for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.Call)}
        for banned in ("VideoCapture", "load_yolo", "YOLO", "imread",
                       "iter_wagon_frames", "list_wagon_frames", "predict"):
            with self.subTest(token=banned):
                self.assertFalse([c for c in called if banned in c],
                                 f"{banned} appears in the aggregator")

    def test_it_takes_no_cache_root_and_no_gw_id(self):
        import inspect
        p = inspect.signature(aggregate_damage_from_detections).parameters
        for banned in ("cache_root", "gw_id", "model", "yolo_model", "frame"):
            self.assertNotIn(banned, p)

    def test_the_model_is_not_invoked_during_aggregation(self):
        with tempfile.TemporaryDirectory() as root:
            cache = _wagon_cache(root, [12, 15, 18, 21, 24])
            from features.damage.processor import (
                DamageTrackerConfig, _run_sampled_one_camera,
            )
            model = StubDamageModel()
            _r, _u, fw, fh, frame_dets = _run_sampled_one_camera(
                model, DamageTrackerConfig(), cache, GW, CAM,
                confidence_floor=C.CONF_DAMAGE, sample_stride=STRIDE)
            after_inference = model.calls
            from features._common import list_wagon_frames
            paths = list_wagon_frames(cache, GW, CAM, trim_stable=True)
            scored = [int(os.path.basename(x)[6:12])
                      for x in paths][::STRIDE]
            aggregate_damage_from_detections(
                [DamageDetection(d["frame_idx"], d["class_name"],
                                 d["confidence"], d["bbox"])
                 for d in frame_dets],
                camera_id=CAM, frame_width=fw, frame_height=fh,
                stride=STRIDE, scored_frames=scored)
            self.assertEqual(model.calls, after_inference,
                             "the model ran again during aggregation")

    def test_the_legacy_tracker_path_is_documented_as_out_of_scope(self):
        """It consumes frames inside DamageTracker and is not separable here."""
        import inspect
        from features.damage import aggregate
        self.assertIn("DamageTracker", inspect.getsource(aggregate))


if __name__ == "__main__":
    unittest.main()
