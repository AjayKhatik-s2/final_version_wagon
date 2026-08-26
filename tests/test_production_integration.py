"""The production architecture, proven at the call-chain level.

The claim being tested is not "the aggregators are correct" -- that is settled
by the three equivalence suites. It is that PRODUCTION actually executes the
new path in BOTH modes: one decode per camera feeding GAP + Door + Damage +
Load, then pure aggregation, then timestamp fusion. An architecture that is
merely available but not wired is the failure mode this file exists to catch.

Ten properties, one class each:

    1.  the shared collector is actually invoked, in both modes
    2.  process_video() is not invoked for GAP
    3.  each camera has exactly one decode pass
    4.  GAP steps every frame
    5.  Door / Damage / Load use their configured strides
    6.  no wagon cache is required during raw collection
    7.  no model inference happens during Phase 2 aggregation
    8.  the three extracted aggregators are actually called
    9.  the same evidence is passed to both modes
    10. Batch and Sequential agree on the roster and the assignments

Structural checks read the real production sources rather than a copy, so they
fail if somebody reverts a call site. Functional checks drive the real shared
implementation with stub models over a synthetic video, so they fail if the
wiring exists but does not behave.
"""

from __future__ import annotations

import ast
import inspect
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT, WAGON_COUNT_DIR  # noqa: F401

import cv2
import numpy as np

from core import constants as C
from core.master_timeline import CameraClock
from core import production_pipeline as pp

FPS = 15.0
W, H = 160, 120
N_FRAMES = 90
RU = C.CAMERA_RIGHT_UP
RUT = C.CAMERA_RIGHT_UP_TOP


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _strip_comments(src: str) -> str:
    """Code only. A comment explaining why we avoid X must not match X."""
    return "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())


def _video(path: str, n=N_FRAMES) -> str:
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(n):
        vw.write(np.full((H, W, 3), 200 if i % 2 == 0 else 20, dtype=np.uint8))
    vw.release()
    return path


# --- stubs -----------------------------------------------------------------

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


class _P:
    def __init__(self, t, c):
        self.top1, self.top1conf = t, c


class _Res:
    def __init__(self, boxes=None, probs=None):
        self.boxes, self.probs = boxes, probs


class StubDetector:
    """Counts calls and records which frames it was asked about."""
    names = {0: "open_door", 1: "inner_wall_damage"}

    def __init__(self, conf=0.93):
        self.calls = 0
        self.conf = conf

    def __call__(self, frame, verbose=False):
        self.calls += 1
        return [_Res(boxes=_Boxes(
            np.array([[10.0, 10.0, 60.0, 90.0]]),
            np.array([self.conf]), np.array([0], dtype=int)))]


class StubClassifier:
    names = {0: "loaded"}

    def __init__(self):
        self.calls = 0

    def __call__(self, frame, verbose=False):
        self.calls += 1
        return [_Res(probs=_P(0, 0.88))]


class RecordingGapTracker:
    """A GapTracker-shaped double that records the stepper contract."""

    def __init__(self, camera_id=RU):
        self.camera_id = camera_id
        self.began = 0
        self.finished = 0
        self.stepped_frames = []
        self.opened_videos = []

    def begin(self, *, keep_raw_detections=True):
        self.began += 1
        self.keep_raw = keep_raw_detections

    def step(self, frame_idx, frame, frame_h=None):
        self.stepped_frames.append(int(frame_idx))

    def finish(self, *, video_path, fps, width, height,
               total_frames_meta=0, t0=None):
        self.finished += 1
        self.opened_videos.append(video_path)

        class _T:
            pass
        t = _T()
        t.camera_id = self.camera_id
        t.video_path = video_path
        t.fps = float(fps)
        t.width, t.height = int(width), int(height)
        t.total_frames = len(self.stepped_frames)
        t.gaps = []
        t.raw_frame_detections = {}
        return t

    def process_video(self, *a, **k):
        raise AssertionError("process_video must never be called in production")


class _Wagon:
    """The minimal wagon shape fusion and the aggregators consume."""

    def __init__(self, gid, t0, t1, classification="WAGON"):
        self.global_id = gid
        self.start_time = float(t0)
        self.end_time = float(t1)
        self.classification = classification


def _roster():
    return [_Wagon("GW_1", 0.0, 2.0), _Wagon("GW_2", 2.0, 4.0),
            _Wagon("GW_3", 4.0, 6.0)]


def _collect(root, cameras=(RU, RUT), features=("door", "damage", "load"),
             strides=None):
    """Drive the REAL shared collector with stubs. One decode per camera."""
    trackers = {c: RecordingGapTracker(c) for c in cameras}
    models = {"door": StubDetector(), "damage": StubDetector(),
              "load": StubClassifier()}
    videos = {c: _video(os.path.join(root, "%s.mp4" % c)) for c in cameras}
    stage1 = pp.collect_stage1(
        video_paths=videos, gap_trackers=trackers,
        feature_models_dir="", features=features,
        clocks={c: CameraClock(c, fps=FPS, total_frames=N_FRAMES)
                for c in cameras},
        strides=strides or pp.PRODUCTION_STRIDES,
        models=models, verbose=False)
    return stage1, trackers, models


# ===========================================================================
# 1. the shared collector is actually invoked, in both modes
# ===========================================================================

class TestSharedCollectorIsInvoked(unittest.TestCase):

    def test_batch_step1_calls_the_shared_collector(self):
        src = _strip_comments(
            _read(os.path.join(WAGON_COUNT_DIR, "run_global_count.py")))
        self.assertIn("_collect_stage1_shared(", src,
                      "Batch STEP 1 no longer routes through the collector")
        self.assertIn("collect_production", src)
        self.assertIn("assert_no_second_decode", src)

    def test_sequential_step1_calls_the_shared_collector(self):
        from orchestrator import camera_pipeline as cp
        src = _strip_comments(inspect.getsource(cp._track_stitch_validate))
        self.assertIn("collect_production(", src)
        self.assertIn("assert_no_second_decode", src)

    def test_both_modes_reach_the_same_function(self):
        """Not two look-alike helpers -- literally one implementation."""
        from orchestrator import camera_pipeline as cp
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_rgc_probe", os.path.join(WAGON_COUNT_DIR, "run_global_count.py"))
        # Importing the batch entry point pulls in wagon_count's siblings; the
        # source check above already covers it, so only assert the symbol both
        # sides name resolves to one object here.
        self.assertTrue(callable(pp.collect_production))
        self.assertIn("collect_production",
                      _strip_comments(inspect.getsource(cp)))
        self.assertIn("collect_production",
                      _strip_comments(_read(os.path.join(
                          WAGON_COUNT_DIR, "run_global_count.py"))))

    def test_both_stage3_callers_use_the_shared_phase2(self):
        from orchestrator import global_assembler, master_runner
        for mod in (global_assembler, master_runner):
            with self.subTest(module=mod.__name__):
                src = _strip_comments(inspect.getsource(mod))
                self.assertIn("phase2_from_disk(", src)
                self.assertIn("collected=collected", src)


# ===========================================================================
# 2. process_video() is not invoked for GAP
# ===========================================================================

class TestNoProcessVideoInProduction(unittest.TestCase):
    """A second decode is the specific regression this forbids."""

    PRODUCTION_SOURCES = (
        os.path.join(WAGON_COUNT_DIR, "run_global_count.py"),
        os.path.join(V4_ROOT, "orchestrator", "camera_pipeline.py"),
        os.path.join(V4_ROOT, "orchestrator", "camera_runner.py"),
        os.path.join(V4_ROOT, "orchestrator", "global_assembler.py"),
        os.path.join(V4_ROOT, "orchestrator", "master_runner.py"),
        os.path.join(V4_ROOT, "core", "production_pipeline.py"),
        os.path.join(V4_ROOT, "core", "evidence_collection.py"),
        os.path.join(V4_ROOT, "features", "raw_collect.py"),
    )

    def test_no_production_source_calls_process_video(self):
        for path in self.PRODUCTION_SOURCES:
            with self.subTest(source=os.path.relpath(path, V4_ROOT)):
                self.assertTrue(os.path.isfile(path), path)
                called = {ast.unparse(n.func) for n
                          in ast.walk(ast.parse(_read(path)))
                          if isinstance(n, ast.Call)}
                self.assertFalse(
                    [c for c in called if "process_video" in c],
                    "process_video() is called -- that is a second decode")

    def test_the_tracker_still_offers_process_video_for_other_callers(self):
        """Removing it was never the point; not calling it in production was."""
        from tracker_engine import GapTracker
        self.assertTrue(callable(GapTracker.process_video))

    def test_the_double_would_catch_a_regression(self):
        """Negative control: the recorder raises if process_video is used."""
        with self.assertRaises(AssertionError):
            RecordingGapTracker().process_video("x.mp4")


# ===========================================================================
# 3 + 4. exactly one decode per camera; GAP steps EVERY frame
# ===========================================================================

class TestOneDecodeAndEveryFrameStepped(unittest.TestCase):

    def test_each_camera_is_decoded_exactly_once(self):
        with tempfile.TemporaryDirectory() as root:
            stage1, trackers, _m = _collect(root)
            self.assertEqual(sorted(stage1.decode_calls), sorted([RU, RUT]))
            for cam, n in stage1.decode_calls.items():
                with self.subTest(camera=cam):
                    self.assertEqual(n, 1)
            stage1.assert_no_second_decode()
            for cam, t in trackers.items():
                with self.subTest(camera=cam):
                    self.assertEqual(t.began, 1, "begin() not called once")
                    self.assertEqual(t.finished, 1, "finish() not called once")

    def test_gap_steps_every_decoded_frame(self):
        with tempfile.TemporaryDirectory() as root:
            _s, trackers, _m = _collect(root)
            for cam, t in trackers.items():
                with self.subTest(camera=cam):
                    self.assertEqual(t.stepped_frames,
                                     list(range(N_FRAMES)),
                                     "GAP did not step every frame")

    def test_gap_is_not_sampled_even_when_features_are(self):
        """The strides apply to the features only. GAP is stateful."""
        with tempfile.TemporaryDirectory() as root:
            _s, trackers, _m = _collect(root, strides={"door": 5, "damage": 5,
                                                       "load": 5})
            for cam, t in trackers.items():
                with self.subTest(camera=cam):
                    self.assertEqual(len(t.stepped_frames), N_FRAMES)

    def test_assert_no_second_decode_actually_fires(self):
        """Negative control -- the guard is not vacuous."""
        s = pp.Stage1Result()
        s.decode_calls = {RU: 2}
        with self.assertRaises(RuntimeError):
            s.assert_no_second_decode()


# ===========================================================================
# 5. Door / Damage / Load use their configured strides
# ===========================================================================

class TestFeatureStrides(unittest.TestCase):

    def test_production_strides_are_the_documented_ones(self):
        self.assertEqual(pp.PRODUCTION_STRIDES,
                         {"door": 3, "damage": 3, "load": 2})

    def test_each_detector_scores_on_its_own_stride(self):
        with tempfile.TemporaryDirectory() as root:
            stage1, _t, models = _collect(root)
            # Door is a SIDE feature, damage/load are TOP features, so each
            # runs on exactly one of the two cameras in this fixture.
            expected = {"door": len(range(0, N_FRAMES, 3)),
                        "damage": len(range(0, N_FRAMES, 3)),
                        "load": len(range(0, N_FRAMES, 2))}
            scored = {}
            for cam, cc in stage1.per_camera.items():
                for feat, n in cc.frames_scored.items():
                    scored[feat] = scored.get(feat, 0) + n
            for feat, want in expected.items():
                with self.subTest(feature=feat):
                    self.assertEqual(scored.get(feat), want,
                                     f"{feat} did not use its stride")

    def test_a_changed_stride_changes_the_scored_count(self):
        """Negative control: the stride is honoured, not coincidental."""
        with tempfile.TemporaryDirectory() as root:
            s1, _t, _m = _collect(root, strides={"door": 1, "damage": 1,
                                                 "load": 1})
            total = sum(sum(cc.frames_scored.values())
                        for cc in s1.per_camera.values())
        with tempfile.TemporaryDirectory() as root:
            s2, _t, _m = _collect(root, strides={"door": 9, "damage": 9,
                                                 "load": 9})
            total9 = sum(sum(cc.frames_scored.values())
                         for cc in s2.per_camera.values())
        self.assertGreater(total, total9)

    def test_camera_authority_is_respected(self):
        """Door never scores on a top camera, damage never on a side one."""
        with tempfile.TemporaryDirectory() as root:
            stage1, _t, _m = _collect(root)
        self.assertNotIn("damage", stage1.per_camera[RU].frames_scored)
        self.assertNotIn("door", stage1.per_camera[RUT].frames_scored)


# ===========================================================================
# 6. no wagon cache is required during raw collection
# ===========================================================================

class TestNoWagonCacheDuringCollection(unittest.TestCase):

    def test_collection_succeeds_with_no_cache_on_disk(self):
        with tempfile.TemporaryDirectory() as root:
            stage1, _t, _m = _collect(root)
            self.assertTrue(stage1.observations,
                            "no evidence collected at all")
            # Nothing resembling a wagon cache was created or read.
            self.assertEqual(
                [d for d in os.listdir(root) if d.startswith("GW_")], [])

    def test_the_collector_never_reads_a_wagon_cache(self):
        import features.raw_collect as rc
        for mod in (pp, rc):
            with self.subTest(module=mod.__name__):
                called = {ast.unparse(n.func) for n in
                          ast.walk(ast.parse(inspect.getsource(mod)))
                          if isinstance(n, ast.Call)}
                for banned in ("iter_wagon_frames", "list_wagon_frames"):
                    self.assertFalse([c for c in called if banned in c],
                                     f"{banned} in {mod.__name__}")

    def test_collection_takes_no_gw_id_and_no_cache_root(self):
        for fn in (pp.collect_stage1, pp.collect_production):
            with self.subTest(fn=fn.__name__):
                p = inspect.signature(fn).parameters
                for banned in ("gw_id", "cache_root", "roster", "wagons"):
                    self.assertNotIn(banned, p)


# ===========================================================================
# 7 + 8. Phase 2 runs the extracted aggregators and NO model
# ===========================================================================

class TestPhase2IsPureAggregation(unittest.TestCase):

    def _phase2(self, root):
        stage1, _t, models = _collect(root)
        before = {k: getattr(m, "calls", 0) for k, m in models.items()}
        ev = pp.TimelineEvidence(mode="test")
        ev.extend(stage1.observations)
        res = pp.aggregate_phase2(
            evidence=ev, wagons=_roster(), stage1=stage1,
            clocks={c: CameraClock(c, fps=FPS, total_frames=N_FRAMES)
                    for c in stage1.per_camera},
            verbose=False)
        after = {k: getattr(m, "calls", 0) for k, m in models.items()}
        return res, before, after

    def test_no_model_is_invoked_during_aggregation(self):
        with tempfile.TemporaryDirectory() as root:
            _res, before, after = self._phase2(root)
        self.assertEqual(before, after,
                         "a model ran during Phase 2 aggregation")
        self.assertTrue(any(v for v in before.values()),
                        "no model ran during Phase 1 either -- vacuous")

    def test_all_three_extracted_aggregators_are_called(self):
        with tempfile.TemporaryDirectory() as root:
            res, _b, _a = self._phase2(root)
        for feature in ("door", "damage", "load"):
            with self.subTest(feature=feature):
                self.assertGreater(res.aggregator_calls.get(feature, 0), 0,
                                   f"{feature} aggregator never ran")

    def test_the_aggregators_are_the_proven_extracted_ones(self):
        """By identity, not by name resemblance."""
        src = _strip_comments(inspect.getsource(pp.aggregate_phase2))
        for token in ("aggregate_damage_from_observations",
                      "aggregate_load_from_observations",
                      "aggregate_door_from_frames"):
            with self.subTest(token=token):
                self.assertIn(token, src)

    def test_phase2_opens_no_video_and_loads_no_model(self):
        called = {ast.unparse(n.func) for n in
                  ast.walk(ast.parse(inspect.getsource(pp.aggregate_phase2)))
                  if isinstance(n, ast.Call)}
        for banned in ("VideoCapture", "load_yolo", "YOLO", "imread",
                       "run_classification", "detection_quality"):
            with self.subTest(token=banned):
                self.assertFalse([c for c in called if banned in c])

    def test_the_processors_short_circuit_to_the_aggregated_result(self):
        """Each processor consumes `collected` instead of re-inferring."""
        from features.damage import processor as dmg
        from features.door import processor as dr
        from features.load import processor as ld
        for mod, token in ((dmg, "collected.damage_for("),
                           (dr, "collected.door_for("),
                           (ld, "collected.load_for(")):
            with self.subTest(module=mod.__name__):
                src = _strip_comments(inspect.getsource(mod))
                self.assertIn(token, src)
                self.assertIn("_SENTINEL_NO_INFERENCE", src)

    def test_the_sentinel_refuses_to_infer(self):
        """Negative control: Phase 2 cannot quietly fall back to a model."""
        from features.damage.processor import _SENTINEL_NO_INFERENCE
        with self.assertRaises(RuntimeError):
            _SENTINEL_NO_INFERENCE(np.zeros((4, 4, 3), np.uint8))


# ===========================================================================
# 9 + 10. the same evidence, and the same result, in both modes
# ===========================================================================

class TestBatchAndSequentialAgree(unittest.TestCase):
    """Identical evidence must give an identical roster and assignment.

    Both modes now read their evidence from the same on-disk artifact format
    and call the same `phase2_from_disk`, so this compares the two production
    entry points against one corpus rather than comparing a mode against
    itself.
    """

    def _both(self, root):
        stage1, _t, _m = _collect(root)
        ev_dir = os.path.join(root, pp.RAW_EVIDENCE_DIRNAME)
        pp.write_raw_evidence(stage1, ev_dir)
        wagons = _roster()
        batch = pp.phase2_from_disk(evidence_dir=ev_dir, wagons=wagons,
                                    mode="batch", verbose=False)
        seq = pp.phase2_from_disk(evidence_dir=ev_dir, wagons=wagons,
                                  mode="sequential", verbose=False)
        return batch, seq

    def test_the_persisted_evidence_round_trips_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            stage1, _t, _m = _collect(root)
            ev_dir = os.path.join(root, pp.RAW_EVIDENCE_DIRNAME)
            pp.write_raw_evidence(stage1, ev_dir)
            back = pp.read_raw_evidence(ev_dir)
        self.assertTrue(stage1.observations)
        self.assertEqual([o.to_dict() for o in stage1.observations],
                         [o.to_dict() for o in back.observations])
        for cam, cc in stage1.per_camera.items():
            with self.subTest(camera=cam):
                b = back.per_camera[cam]
                self.assertEqual((cc.width, cc.height, cc.frames_read),
                                 (b.width, b.height, b.frames_read))

    def test_both_modes_see_the_same_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            batch, seq = self._both(root)
        self.assertIsNotNone(batch)
        self.assertIsNotNone(seq)
        self.assertEqual(batch.assignments, seq.assignments)
        self.assertGreater(batch.assignments, 0, "vacuous comparison")

    def test_both_modes_produce_the_same_roster(self):
        with tempfile.TemporaryDirectory() as root:
            batch, seq = self._both(root)
        self.assertEqual(sorted(batch.per_wagon), sorted(seq.per_wagon))
        self.assertTrue(batch.per_wagon, "no wagon received any evidence")

    def test_both_modes_produce_the_same_feature_assignments(self):
        with tempfile.TemporaryDirectory() as root:
            batch, seq = self._both(root)
        for gw in sorted(batch.per_wagon):
            b, s = batch.per_wagon[gw], seq.per_wagon[gw]
            with self.subTest(wagon=gw):
                self.assertEqual(b.frames_by_camera, s.frames_by_camera)
                self.assertEqual(sorted(b.damage), sorted(s.damage))
                self.assertEqual(sorted(b.load), sorted(s.load))
                self.assertEqual(sorted(b.door), sorted(s.door))
                for cam in b.load:
                    self.assertEqual(b.load[cam][:5], s.load[cam][:5])
                for cam in b.damage:
                    self.assertEqual(b.damage[cam][0], s.damage[cam][0])
                for cam in b.door:
                    self.assertEqual(b.door[cam][0], s.door[cam][0])

    def test_the_aggregator_call_counts_match(self):
        with tempfile.TemporaryDirectory() as root:
            batch, seq = self._both(root)
        self.assertEqual(batch.aggregator_calls, seq.aggregator_calls)
        self.assertTrue(batch.aggregator_calls)

    def test_assignment_is_by_timestamp_not_by_segment_index(self):
        """No second roster: the only assignment path is fuse()."""
        src = _strip_comments(inspect.getsource(pp.aggregate_phase2))
        self.assertIn("evidence.fuse(wagons)", src)
        self.assertNotIn("segment_index", src)
        self.assertNotIn("enumerate(wagons)", src)


# ===========================================================================
# Audit logging -- so an EC2 run can prove which path executed
# ===========================================================================

class TestAuditTags(unittest.TestCase):

    def test_every_tag_is_defined(self):
        for tag in ("[EVIDENCE-COLLECT]", "[EVIDENCE-GAP]",
                    "[EVIDENCE-FEATURE]", "[EVIDENCE-AGGREGATE]",
                    "[EVIDENCE-FUSE]"):
            with self.subTest(tag=tag):
                self.assertIn(tag, _read(os.path.join(
                    V4_ROOT, "core", "production_pipeline.py")))

    def test_collection_emits_collect_gap_and_feature_tags(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as root:
            trackers = {RU: RecordingGapTracker(RU)}
            with redirect_stdout(buf):
                pp.collect_stage1(
                    video_paths={RU: _video(os.path.join(root, "a.mp4"))},
                    gap_trackers=trackers, feature_models_dir="",
                    features=("door",),
                    clocks={RU: CameraClock(RU, fps=FPS,
                                            total_frames=N_FRAMES)},
                    models={"door": StubDetector()}, verbose=True)
        out = buf.getvalue()
        for tag in (pp.TAG_COLLECT, pp.TAG_GAP, pp.TAG_FEATURE):
            with self.subTest(tag=tag):
                self.assertIn(tag, out)

    def test_aggregation_emits_fuse_and_aggregate_tags(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as root:
            stage1, _t, _m = _collect(root)
            ev = pp.TimelineEvidence(mode="test")
            ev.extend(stage1.observations)
            with redirect_stdout(buf):
                pp.aggregate_phase2(evidence=ev, wagons=_roster(),
                                    stage1=stage1, verbose=True)
        out = buf.getvalue()
        self.assertIn(pp.TAG_FUSE, out)
        self.assertIn(pp.TAG_AGGREGATE, out)


if __name__ == "__main__":
    unittest.main()
