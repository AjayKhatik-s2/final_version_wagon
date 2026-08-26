"""Door aggregation, extracted, must produce identical decisions and evidence.

The sampled Door path scores frames AND builds decisions, evidence buckets and
the overlay trajectory in one loop. Only the scoring needs frames. The
extraction is proven the way Damage, Load and GapTracker were: run the OLD
function over a real wagon-cache fixture, replay the same model over the same
frames to collect its detections, feed those to the new pure aggregator, and
require an identical 6-tuple. Non-vacuity guards refuse an empty comparison on
each of the three outputs -- decisions, buckets and overlay -- because Door is
the one processor where an empty bucket dict would still look like a pass.

Door is the most sensitive of the three extractions: the evidence buckets feed
the report snapshots, and the `_canonical()` bucket-key quirk (buckets key on
`_canonical(raw)` while the aggregator groups on `DOOR_LABEL_TO_STATE`) is
load-bearing for the A/B comparison. `test_the_bucket_key_quirk_is_replicated`
pins it so a later tidy-up cannot silently change which snapshot is published.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

import cv2
import numpy as np

from core import constants as C
from features.door.aggregate import (
    DoorDetection, DoorFrameRecord, aggregate_door_from_frames,
    best_frame_indices, canonical_door, frame_records_from_detections,
)

CAM = C.CAMERA_RIGHT_UP
GW = "GW_1"
W, H = 320, 240
STRIDE = 2
MIN_CONF = 0.68


class _A:
    def __init__(self, a):
        self._a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, a, c, k):
        self.xyxy, self.conf, self.cls = _A(a), _A(c), _A(k)

    def __len__(self):
        return len(self.xyxy.numpy())


class _Res:
    def __init__(self, b):
        self.boxes = b


class StubDoorModel:
    """Two boxes on bright frames, one below the confidence gate on mid frames.

    Keyed on brightness, not an index painted into a pixel: the cache is JPEG
    and lossy, so a single index pixel does not survive the round trip while the
    mean of a flat field does. The mid band exercises the "all boxes filtered
    out" branch, which must still declare an EMPTY frame to the aggregator.
    """
    names = {0: "open_door", 1: "closed_door", 2: "damage"}

    def __init__(self):
        self.calls = 0

    def __call__(self, frame, verbose=False):
        self.calls += 1
        m = float(frame.mean())
        if m > 170:
            return [_Res(_Boxes(
                np.array([[20.0, 40.0, 120.0, 200.0],
                          [180.0, 40.0, 290.0, 200.0]]),
                np.array([0.93, 0.81]), np.array([0, 1], dtype=int)))]
        if m > 80:
            # Present but below the gate -> filtered to nothing.
            return [_Res(_Boxes(np.array([[20.0, 40.0, 120.0, 200.0]]),
                                np.array([0.20]), np.array([2], dtype=int)))]
        return [_Res(_Boxes(np.zeros((0, 4)), np.zeros(0),
                            np.zeros(0, dtype=int)))]


def _wagon_cache(root, n_frames=60):
    """A real wagon cache in the layout `iter_wagon_frames` expects."""
    d = os.path.join(root, GW, C.CAMERA_FOLDER[CAM])
    os.makedirs(d, exist_ok=True)
    for i in range(n_frames):
        val = 210 if i % 3 == 0 else (120 if i % 3 == 1 else 30)
        cv2.imwrite(os.path.join(d, "frame_%06d.jpg" % i),
                    np.full((H, W, 3), val, dtype=np.uint8))
    return root


def _buckets(cands):
    """Comparable form for the evidence buckets: image compared separately."""
    return {k: {"score": round(float(v.score), 9),
                "frame_idx": int(v.frame_idx),
                "bbox": [round(float(x), 6) for x in (v.bbox or [])],
                "meta": {mk: (round(float(mv), 9)
                              if isinstance(mv, float) else mv)
                         for mk, mv in sorted(v.meta.items())}}
            for k, v in sorted(cands.items())}


class TestDoorAggregationEquivalence(unittest.TestCase):
    """Old fused implementation versus extracted aggregation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = _wagon_cache(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _old(self):
        from features.door.processor import _run_sampled_one_camera
        from features.inference_lib.door_tracker import TrackerConfig
        model = StubDoorModel()
        out = _run_sampled_one_camera(model, TrackerConfig(), self.cache, GW,
                                      CAM, sample_stride=STRIDE)
        return out, model

    def _collect(self, frames_wanted=False):
        """Phase 1: the same model over the same frames, plus crop quality.

        `detection_quality` is the only pixel-reading step in the whole path,
        so it happens here and its scalar travels with the detection.
        """
        from core.frame_quality import detection_quality
        from features._common import iter_wagon_frames
        model = StubDoorModel()
        records, imgs = [], {}
        fw = fh = 0
        for fi, frame in iter_wagon_frames(self.cache, GW, CAM,
                                           every_nth=STRIDE,
                                           trim_stable=True):
            if fw == 0:
                fh, fw = frame.shape[:2]
            res = model(frame, verbose=False)[0]
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                keep = confs >= MIN_CONF
                for bbox, conf, cls_id in zip(boxes[keep], confs[keep],
                                              clss[keep]):
                    bl = [float(v) for v in bbox]
                    dets.append(DoorDetection(
                        frame_idx=int(fi),
                        raw_class=str(model.names.get(int(cls_id), "")).lower(),
                        confidence=float(conf), bbox=bl,
                        crop_quality=float(detection_quality(frame, bl))))
            records.append(DoorFrameRecord(int(fi), dets))
            if frames_wanted:
                imgs[int(fi)] = frame
        return records, imgs, fw, fh, model

    def _new(self, frames_wanted=False):
        records, imgs, fw, fh, _m = self._collect(frames_wanted)
        return aggregate_door_from_frames(
            records, camera_id=CAM, frame_width=fw, frame_height=fh,
            stride=STRIDE, frames=imgs if frames_wanted else None), records

    def test_the_corpus_is_not_empty_on_all_three_outputs(self):
        (dec, used, fw, fh, cands, overlay), _m = self._old()
        self.assertGreater(used, 0, "no frame was scored")
        self.assertTrue(dec, "the old path produced no decision")
        self.assertTrue(cands, "the old path produced no evidence bucket")
        self.assertTrue(overlay["tracks"], "the old path produced no overlay")
        self.assertGreater(fw, 0)

    def test_the_same_frames_are_seen(self):
        (_d, used, _fw, _fh, _c, _o), _m = self._old()
        records, _i, _fw2, _fh2, _m2 = self._collect()
        self.assertEqual(len(records), used,
                         "collection saw a different frame set than the old "
                         "path; the comparison would not be like-for-like")

    def test_decisions_are_identical(self):
        (old_dec, _u, _fw, _fh, _c, _o), _m = self._old()
        (new_dec, _u2, _fw2, _fh2, _c2, _o2), _r = self._new()
        self.assertEqual(old_dec, new_dec)
        self.assertTrue(new_dec)

    def test_every_decision_field_matches(self):
        (old_dec, _u, _fw, _fh, _c, _o), _m = self._old()
        (new_dec, _u2, _fw2, _fh2, _c2, _o2), _r = self._new()
        self.assertEqual(len(old_dec), len(new_dec))
        for a, b in zip(old_dec, new_dec):
            for key in ("camera_id", "track_id", "state", "confidence",
                        "first_frame", "last_frame", "total_hits",
                        "mean_center_x"):
                with self.subTest(key=key):
                    self.assertEqual(a[key], b[key], "%s diverged" % key)

    def test_frame_count_and_dimensions_match(self):
        (_d, used, fw, fh, _c, _o), _m = self._old()
        (_d2, used2, fw2, fh2, _c2, _o2), _r = self._new()
        self.assertEqual((used, fw, fh), (used2, fw2, fh2))

    def test_evidence_buckets_are_identical(self):
        (_d, _u, _fw, _fh, old_c, _o), _m = self._old()
        (_d2, _u2, _fw2, _fh2, new_c, _o2), _r = self._new(frames_wanted=True)
        self.assertTrue(_buckets(new_c))
        self.assertEqual(_buckets(old_c), _buckets(new_c))

    def test_bucket_snapshots_carry_an_image_when_frames_are_supplied(self):
        (_d, _u, _fw, _fh, _c, _o), _m = self._old()
        (_d2, _u2, _fw2, _fh2, new_c, _o2), _r = self._new(frames_wanted=True)
        for key, tracker in new_c.items():
            with self.subTest(bucket=key):
                self.assertTrue(tracker.has_data())

    def test_the_overlay_trajectory_is_identical(self):
        (_d, _u, _fw, _fh, _c, old_o), _m = self._old()
        (_d2, _u2, _fw2, _fh2, _c2, new_o), _r = self._new()
        self.assertTrue(new_o["tracks"])
        self.assertEqual(old_o, new_o)

    def test_the_bucket_key_quirk_is_replicated(self):
        """Buckets key on `_canonical(raw)`; the aggregator does not.

        The two keyings genuinely disagree for these labels, which is the whole
        reason the quirk had to be replicated rather than tidied away.
        """
        (_d, _u, _fw, _fh, old_c, _o), _m = self._old()
        (_d2, _u2, _fw2, _fh2, new_c, _o2), _r = self._new()
        self.assertEqual(sorted(old_c), sorted(new_c))
        disagreed = [raw for raw in ("open_door", "closed_door")
                     if C.DOOR_LABEL_TO_STATE.get(raw) != canonical_door(raw)]
        self.assertTrue(disagreed,
                        "the two keyings now agree; this test no longer pins "
                        "the quirk it was written for")

    def test_deferred_best_indices_match_the_buckets(self):
        """The winning frame is decided without holding any image."""
        (_d, _u, fw, fh, old_c, _o), _m = self._old()
        records, _i, fw2, fh2, _m2 = self._collect()
        idx = best_frame_indices(records, frame_width=fw2, frame_height=fh2)
        self.assertTrue(idx)
        self.assertEqual({k: int(v.frame_idx) for k, v in old_c.items()}, idx)

    def test_a_failed_frame_is_counted_but_never_declared(self):
        """`used` counts it; the aggregator never sees it.

        The old loop `continue`s past `add_frame` when inference raises, while a
        frame that scored and found nothing IS declared empty. Both still count
        toward `used`, and that is the part that is externally visible.
        """
        records, _i, fw, fh, _m = self._collect()
        with_error = [DoorFrameRecord(r.frame_idx, r.detections,
                                      errored=not r.detections)
                      for r in records]
        _d, used_err, _a, _b, _c, _o = aggregate_door_from_frames(
            with_error, camera_id=CAM, frame_width=fw, frame_height=fh,
            stride=STRIDE)
        self.assertEqual(used_err, len(records),
                         "a failed frame must still count toward `used`")

    def test_declaring_empty_frames_is_inert_and_that_is_measured(self):
        """Why `errored` is preserved for fidelity, not for effect.

        `add_frame(fi, [])` turns out not to move the accepted set at any
        hit-to-frame ratio: the aggregator measures support over the frames
        that carry observations. The distinction is kept because the old code
        makes it and because `used` is reported, NOT because it changes a
        decision -- and this test pins the measurement so the docstring cannot
        drift away from the behaviour.
        """
        for n_total in (10, 20, 40):
            records = [
                DoorFrameRecord(i * 2, [DoorDetection(
                    i * 2, "open_door", 0.93, [20., 40., 120., 200.], 0.8)])
                if i < 3 else DoorFrameRecord(i * 2, [])
                for i in range(n_total)
            ]
            errored = [DoorFrameRecord(r.frame_idx, r.detections,
                                       errored=not r.detections)
                       for r in records]
            a = aggregate_door_from_frames(records, camera_id=CAM,
                                           frame_width=W, frame_height=H,
                                           stride=STRIDE)
            b = aggregate_door_from_frames(errored, camera_id=CAM,
                                           frame_width=W, frame_height=H,
                                           stride=STRIDE)
            with self.subTest(n_total=n_total):
                self.assertTrue(a[0], "nothing was accepted; vacuous")
                self.assertEqual(a[0], b[0])
                self.assertEqual(a[1], b[1])

    def test_regrouping_flat_detections_reproduces_the_records(self):
        records, _i, fw, fh, _m = self._collect()
        flat = [d for r in records for d in r.detections]
        rebuilt = frame_records_from_detections(flat,
                                                [r.frame_idx for r in records])
        a = aggregate_door_from_frames(records, camera_id=CAM, frame_width=fw,
                                       frame_height=fh, stride=STRIDE)
        b = aggregate_door_from_frames(rebuilt, camera_id=CAM, frame_width=fw,
                                       frame_height=fh, stride=STRIDE)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[5], b[5])


class TestNoInferenceInDoorAggregation(unittest.TestCase):
    """Structural: the aggregator cannot decode or infer, by construction."""

    def test_no_video_decode_and_no_model_call(self):
        import ast
        import inspect
        from features.door import aggregate
        src = inspect.getsource(aggregate)
        called = {ast.unparse(n.func) for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.Call)}
        for banned in ("VideoCapture", "load_yolo", "YOLO", "imread",
                       "iter_wagon_frames", "list_wagon_frames", "predict",
                       "detection_quality", "yolo_to_detections"):
            with self.subTest(token=banned):
                self.assertFalse([c for c in called if banned in c],
                                 "%s appears in the aggregator" % banned)

    def test_it_takes_no_cache_root_and_no_gw_id(self):
        import inspect
        p = inspect.signature(aggregate_door_from_frames).parameters
        for banned in ("cache_root", "gw_id", "model", "yolo_model",
                       "tracker_config", "sample_stride"):
            self.assertNotIn(banned, p)

    def test_the_model_is_not_invoked_during_aggregation(self):
        with tempfile.TemporaryDirectory() as root:
            cache = _wagon_cache(root)
            from core.frame_quality import detection_quality
            from features._common import iter_wagon_frames
            model = StubDoorModel()
            records = []
            fw = fh = 0
            for fi, frame in iter_wagon_frames(cache, GW, CAM,
                                               every_nth=STRIDE,
                                               trim_stable=True):
                if fw == 0:
                    fh, fw = frame.shape[:2]
                res = model(frame, verbose=False)[0]
                dets = []
                if res.boxes is not None and len(res.boxes) > 0:
                    b = res.boxes.xyxy.cpu().numpy()
                    cf = res.boxes.conf.cpu().numpy()
                    k = res.boxes.cls.cpu().numpy().astype(int)
                    keep = cf >= MIN_CONF
                    for bbox, conf, cid in zip(b[keep], cf[keep], k[keep]):
                        bl = [float(v) for v in bbox]
                        dets.append(DoorDetection(
                            int(fi),
                            str(model.names.get(int(cid), "")).lower(),
                            float(conf), bl,
                            float(detection_quality(frame, bl))))
                records.append(DoorFrameRecord(int(fi), dets))
            after_inference = model.calls
            self.assertGreater(after_inference, 0)
            aggregate_door_from_frames(records, camera_id=CAM, frame_width=fw,
                                       frame_height=fh, stride=STRIDE)
            self.assertEqual(model.calls, after_inference,
                             "the model ran again during aggregation")

    def test_the_legacy_tracker_path_is_documented_as_out_of_scope(self):
        """DoorTracker and the identity merger consume frames internally."""
        import inspect
        from features.door import aggregate
        src = inspect.getsource(aggregate)
        self.assertIn("DoorTracker", src)
        self.assertIn("DoorIdentityMerger", src)


if __name__ == "__main__":
    unittest.main()
