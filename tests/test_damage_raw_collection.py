"""Phase 1 pilot: damage evidence from raw video, before any wagon exists.

The dependency being broken is `iter_wagon_frames(cache_root, gw_id, camera)`.
Every feature reads frames the materializer already bucketed into
`wagon_cache/<GW_n>/`, so a feature cannot run until the roster exists -- and
its wagon assignment ends up encoded in the directory the frame came out of,
where a wrong bucket is indistinguishable from a right one.

The collector walks the ORIGINAL video instead. These tests run it with no
cache directory and no roster in existence at all, and assert the evidence
comes back timestamped and UNASSIGNED.

A stub detector stands in for `damage.pt`, because what is under test is the
frame source, the timestamps and the phase boundary -- not detection quality.
The batch confidence filter is still the real one, imported from the processor.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

import cv2
import numpy as np

from core import constants as C
from core.evidence_collection import (
    build_timeline_evidence, collect_camera_evidence, reproject,
)
from core.global_state_loader import GlobalWagon
from core.master_timeline import CameraClock
from core.timeline_evidence import KIND_DAMAGE, TimelineEvidence
from features.damage.collector import (
    MODEL_PROVENANCE, collect_all_cameras, collect_damage_observations,
)

RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP
FPS = 10.0
W, H = 96, 64


class _Boxes:
    def __init__(self, arr, conf, cls):
        self._a, self._c, self._k = arr, conf, cls

    def __len__(self):
        return len(self._a)

    @property
    def xyxy(self):
        return _T(self._a)

    @property
    def conf(self):
        return _T(self._c)

    @property
    def cls(self):
        return _T(self._k)


class _T:
    def __init__(self, a):
        self._a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class StubDetector:
    """Fires on frames whose mean brightness exceeds a threshold.

    Deterministic and frame-content driven, so a test can place a detection at
    a chosen TIME by painting that frame bright.
    """
    names = {0: "inner_wall_damage"}

    def __init__(self, threshold=100.0, conf=0.85):
        self.threshold, self.conf = threshold, conf
        self.calls = 0

    def __call__(self, frame, verbose=False):
        self.calls += 1
        if float(frame.mean()) < self.threshold:
            return [_Result(_Boxes(np.zeros((0, 4)), np.zeros(0),
                                   np.zeros(0, dtype=int)))]
        # A central box, comfortably inside the top-camera filter's bounds.
        bb = np.array([[W * 0.25, H * 0.25, W * 0.75, H * 0.75]])
        return [_Result(_Boxes(bb, np.array([self.conf]),
                               np.array([0], dtype=int)))]


def _video(path, n_frames, bright_frames=()):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(n_frames):
        val = 200 if i in bright_frames else 20
        vw.write(np.full((H, W, 3), val, dtype=np.uint8))
    vw.release()
    return path


def _roster():
    """Gaps at 10.0s and 20.0s -> three wagons across 0-30s."""
    return [
        GlobalWagon(global_id=f"GW_{i}", wagon_index=i,
                    start_frame_master=int((i - 1) * 10 * FPS),
                    end_frame_master=int(i * 10 * FPS) - 1,
                    start_time=(i - 1) * 10.0, end_time=i * 10.0,
                    classification=C.CLASS_WAGON,
                    classification_confidence=0.9)
        for i in (1, 2, 3)]


class TestNoCacheNoRosterRequired(unittest.TestCase):
    """The dependency this pilot exists to break."""

    def test_collection_runs_with_no_wagon_cache_on_disk(self):
        with tempfile.TemporaryDirectory() as root:
            vid = _video(os.path.join(root, "top.mp4"), 60, bright_frames=[30])
            self.assertFalse(os.path.exists(os.path.join(root, "wagon_cache")))
            r = collect_damage_observations(
                camera_id=RUT, video_path=vid, feature_models_dir="",
                fps=FPS, stride=1, model=StubDetector(), verbose=False)
            self.assertTrue(r.ok, r.skipped)
            self.assertTrue(r.observations)
            self.assertFalse(os.path.exists(os.path.join(root, "wagon_cache")),
                             "collection created a wagon cache")

    def test_the_collector_never_touches_iter_wagon_frames(self):
        import ast
        import inspect
        from features import raw_collect
        from features.damage import collector
        for mod in (collector, raw_collect):
            src = inspect.getsource(mod)
            called = {ast.unparse(n.func) for n in ast.walk(ast.parse(src))
                      if isinstance(n, ast.Call)}
            for banned in ("iter_wagon_frames", "list_wagon_frames"):
                self.assertFalse([c for c in called if banned in c],
                                 f"{banned} still gates collection in "
                                 f"{mod.__name__}")
        called = {ast.unparse(n.func)
                  for n in ast.walk(ast.parse(inspect.getsource(raw_collect)))
                  if isinstance(n, ast.Call)}
        self.assertIn("cv2.VideoCapture", called,
                      "the shared scorer must read the original video")

    def test_it_takes_no_state_and_no_cache_root(self):
        import inspect
        p = inspect.signature(collect_damage_observations).parameters
        self.assertNotIn("state", p)
        self.assertNotIn("cache_root", p)
        self.assertNotIn("gw_id", p)
        self.assertIn("video_path", p)

    def test_the_batch_confidence_filter_is_reused_not_restated(self):
        import inspect
        from features.damage import collector
        from features import raw_collect
        self.assertIn("from features.damage.processor import "
                      "_filter_detections_for_top",
                      inspect.getsource(raw_collect))


class TestTimestampsArePreserved(unittest.TestCase):
    def test_a_detection_carries_the_time_of_its_frame(self):
        with tempfile.TemporaryDirectory() as root:
            vid = _video(os.path.join(root, "t.mp4"), 60, bright_frames=[25])
            r = collect_damage_observations(
                camera_id=RUT, video_path=vid, feature_models_dir="",
                fps=FPS, stride=1, model=StubDetector(), verbose=False)
            o = r.observations[0]
            self.assertEqual(o.local_frame, 25)
            self.assertAlmostEqual(o.t_start, 25 / FPS)       # 2.5s

    def test_the_camera_offset_is_applied(self):
        with tempfile.TemporaryDirectory() as root:
            vid = _video(os.path.join(root, "t.mp4"), 60, bright_frames=[25])
            clock = CameraClock(RUT, fps=FPS, total_frames=60, offset=7.0)
            r = collect_damage_observations(
                camera_id=RUT, video_path=vid, feature_models_dir="",
                clock=clock, stride=1, model=StubDetector(), verbose=False)
            self.assertAlmostEqual(r.observations[0].t_start, 2.5 + 7.0)

    def test_provenance_is_complete(self):
        with tempfile.TemporaryDirectory() as root:
            vid = _video(os.path.join(root, "t.mp4"), 40, bright_frames=[10])
            o = collect_damage_observations(
                camera_id=RUT, video_path=vid, feature_models_dir="",
                fps=FPS, stride=1, model=StubDetector(), verbose=False
            ).observations[0]
            self.assertEqual(o.camera_id, RUT)
            self.assertEqual(o.kind, KIND_DAMAGE)
            self.assertEqual(o.model, MODEL_PROVENANCE)
            self.assertEqual(o.label, "inner_wall_damage")
            self.assertAlmostEqual(o.confidence, 0.85)
            self.assertEqual(len(o.bbox), 4)
            self.assertEqual(o.payload["fps"], FPS)
            self.assertTrue(o.payload["source_video"].endswith("t.mp4"))

    def test_stride_reduces_the_frames_scored(self):
        with tempfile.TemporaryDirectory() as root:
            vid = _video(os.path.join(root, "t.mp4"), 60)
            det = StubDetector()
            r = collect_damage_observations(
                camera_id=RUT, video_path=vid, feature_models_dir="",
                fps=FPS, stride=3, model=det, verbose=False)
            self.assertEqual(r.frames_read, 60)
            self.assertEqual(r.frames_scored, 20)
            self.assertEqual(det.calls, 20)

    def test_a_missing_video_is_reported_not_raised(self):
        r = collect_damage_observations(
            camera_id=RUT, video_path="/nope.mp4", feature_models_dir="",
            fps=FPS, model=StubDetector(), verbose=False)
        self.assertFalse(r.ok)
        self.assertIn("video unavailable", r.skipped)

    def test_reprojection_uses_the_retained_frame_index(self):
        """Collection may precede offset resolution; the frame index survives."""
        with tempfile.TemporaryDirectory() as root:
            vid = _video(os.path.join(root, "t.mp4"), 60, bright_frames=[25])
            r = collect_damage_observations(
                camera_id=RUT, video_path=vid, feature_models_dir="",
                fps=FPS, stride=1, model=StubDetector(), verbose=False)
            self.assertAlmostEqual(r.observations[0].t_start, 2.5)
            later = reproject(r.observations,
                              {RUT: CameraClock(RUT, fps=FPS, total_frames=60,
                                                offset=12.0)})
            self.assertAlmostEqual(later[0].t_start, 14.5)
            self.assertEqual(later[0].local_frame, 25)


class TestEvidenceIsUnassignedUntilFusion(unittest.TestCase):
    def test_collection_produces_no_wagon_id(self):
        with tempfile.TemporaryDirectory() as root:
            vid = _video(os.path.join(root, "t.mp4"), 60, bright_frames=[25])
            r = collect_damage_observations(
                camera_id=RUT, video_path=vid, feature_models_dir="",
                fps=FPS, stride=1, model=StubDetector(), verbose=False)
            for o in r.observations:
                self.assertNotIn("GW_", str(o.to_dict()))

    def test_the_container_holds_evidence_with_no_assignments(self):
        ev = TimelineEvidence(mode="test")
        with tempfile.TemporaryDirectory() as root:
            vid = _video(os.path.join(root, "t.mp4"), 60, bright_frames=[25])
            r = collect_damage_observations(
                camera_id=RUT, video_path=vid, feature_models_dir="",
                fps=FPS, stride=1, model=StubDetector(), verbose=False)
            ev.extend(r.observations)
        self.assertTrue(ev.observations)
        self.assertEqual(ev.assignments, [])

    def test_fusion_assigns_by_timestamp_afterwards(self):
        with tempfile.TemporaryDirectory() as root:
            # bright at frames 50 and 250 -> 5.0s (GW_1) and 25.0s (GW_3)
            vid = _video(os.path.join(root, "t.mp4"), 300,
                         bright_frames=[50, 250])
            r = collect_damage_observations(
                camera_id=RUT, video_path=vid, feature_models_dir="",
                fps=FPS, stride=1, model=StubDetector(), verbose=False)
        ev = TimelineEvidence(mode="test")
        ev.extend(r.observations)
        ev.fuse(_roster())
        self.assertEqual([a.global_id for a in ev.assignments],
                         ["GW_1", "GW_3"])

    def test_an_observation_outside_the_roster_stays_unassigned(self):
        with tempfile.TemporaryDirectory() as root:
            vid = _video(os.path.join(root, "t.mp4"), 400, bright_frames=[350])
            r = collect_damage_observations(
                camera_id=RUT, video_path=vid, feature_models_dir="",
                fps=FPS, stride=1, model=StubDetector(), verbose=False)
        ev = TimelineEvidence(mode="test")
        ev.extend(r.observations)
        ev.fuse(_roster())                        # roster ends at 30.0s
        self.assertIsNone(ev.assignments[0].global_id)
        self.assertIn("outside", ev.assignments[0].detail)


class TestSharedByBothModes(unittest.TestCase):
    def _collect(self, root, mode):
        # The SAME video files for both modes. Naming them per mode leaked the
        # mode into `source_video` provenance and made the artifacts differ for
        # a reason that had nothing to do with the pipeline.
        vids = {RUT: _video(os.path.join(root, "rut.mp4"), 300,
                            bright_frames=[50, 250]),
                LUT: _video(os.path.join(root, "lut.mp4"), 300,
                            bright_frames=[150])}
        col = collect_camera_evidence(
            video_paths=vids, feature_models_dir="",
            clocks={c: CameraClock(c, fps=FPS, total_frames=300)
                    for c in (RUT, LUT)},
            models={"damage": StubDetector()}, verbose=False)
        ev = build_timeline_evidence(collection=col, mode=mode,
                                     canonical_gaps=[10.0, 20.0])
        ev.fuse(_roster())
        return ev

    def test_both_modes_produce_the_same_observations(self):
        with tempfile.TemporaryDirectory() as root:
            b = self._collect(root, "batch")
            s = self._collect(root, "sequential")
        key = lambda e: sorted((o.camera_id, o.local_frame, round(o.t_start, 4))
                               for o in e.observations)
        self.assertEqual(key(b), key(s))
        self.assertTrue(b.observations)

    def test_both_modes_produce_the_same_assignments(self):
        with tempfile.TemporaryDirectory() as root:
            b = self._collect(root, "batch")
            s = self._collect(root, "sequential")
        key = lambda e: sorted((a.observation.camera_id,
                                a.observation.local_frame, a.global_id,
                                a.reason) for a in e.assignments)
        self.assertEqual(key(b), key(s))

    def test_the_artifacts_match_apart_from_the_mode_label(self):
        with tempfile.TemporaryDirectory() as root:
            b = self._collect(root, "batch").to_dict()
            s = self._collect(root, "sequential").to_dict()
        self.assertNotEqual(b.pop("mode"), s.pop("mode"))
        self.assertEqual(b, s)

    def test_neither_orchestrator_carries_its_own_collection_logic(self):
        import inspect
        from orchestrator import global_assembler
        seq = inspect.getsource(global_assembler)
        batch = open(os.path.join(V4_ROOT, "wagon_count",
                                  "run_global_count.py"),
                     encoding="utf-8").read()
        for src in (seq, batch):
            self.assertNotIn("VideoCapture", src)
            self.assertNotIn("_filter_detections_for_top", src)
            self.assertNotIn("collect_damage_observations", src)

    def test_the_migrated_set_is_explicit(self):
        """Door, Damage and Load score raw video; OCR still needs a wagon."""
        from core.evidence_collection import (POST_ROSTER_FEATURES,
                                              RAW_VIDEO_FEATURES)
        self.assertEqual(set(RAW_VIDEO_FEATURES), {"door", "damage", "load"})
        self.assertEqual(tuple(POST_ROSTER_FEATURES), ("ocr",))

    def test_there_is_one_raw_damage_implementation(self):
        """The damage collector is a view of the shared scorer, not a copy."""
        import inspect
        from features.damage import collector
        src = inspect.getsource(collector.collect_damage_observations)
        self.assertIn("raw_collect.collect_camera", src)
        self.assertNotIn("VideoCapture", src)

    def test_one_missing_camera_does_not_stop_the_other(self):
        with tempfile.TemporaryDirectory() as root:
            # stride defaults to 3, so the bright frame must be a multiple
            # of it or the detector never sees it.
            vids = {RUT: _video(os.path.join(root, "a.mp4"), 100,
                                bright_frames=[21]),
                    LUT: "/nonexistent.mp4"}
            r = collect_all_cameras(video_paths=vids, feature_models_dir="",
                                    clocks={RUT: CameraClock(RUT, fps=FPS,
                                                             total_frames=100)},
                                    model=StubDetector(), verbose=False)
            self.assertTrue(r[RUT].observations)
            self.assertFalse(r[LUT].ok)


class TestGapAndDamageAreIndependentProducers(unittest.TestCase):
    def test_collection_needs_no_gap_evidence(self):
        import inspect
        from features.damage import collector
        p = inspect.signature(collector.collect_damage_observations).parameters
        for gap_arg in ("gaps", "gap_times", "tracks", "wagons", "roster"):
            self.assertNotIn(gap_arg, p)

    def test_fusion_is_where_the_two_meet(self):
        """Damage collected first, gaps established later, assignment last."""
        with tempfile.TemporaryDirectory() as root:
            vid = _video(os.path.join(root, "t.mp4"), 300, bright_frames=[150])
            r = collect_damage_observations(
                camera_id=RUT, video_path=vid, feature_models_dir="",
                fps=FPS, stride=1, model=StubDetector(), verbose=False)
        ev = TimelineEvidence(mode="test")
        ev.extend(r.observations)
        self.assertEqual(ev.assignments, [], "assigned before gaps existed")
        ev.canonical_gaps = [10.0, 20.0]          # gaps arrive now
        ev.fuse(_roster())
        self.assertEqual(ev.assignments[0].global_id, "GW_2")   # 15.0s


if __name__ == "__main__":
    unittest.main()


class TestUnifiedMultiDetectorCollection(unittest.TestCase):
    """One decode pass, several detectors, one coordinate system.

    Door, Damage and Load score the SAME frames of the SAME video in a single
    pass, so they share frame indices and timestamps by construction rather
    than by agreement between three separate readers.
    """

    def _side(self, root):
        return _video(os.path.join(root, "side.mp4"), 120,
                      bright_frames=[30, 60, 90])

    def test_all_enabled_detectors_run_in_one_pass(self):
        from features.raw_collect import collect_camera
        with tempfile.TemporaryDirectory() as root:
            r = collect_camera(
                camera_id=RUT, video_path=self._side(root),
                feature_models_dir="", features=("damage", "load"),
                clock=CameraClock(RUT, fps=FPS, total_frames=120),
                strides={"damage": 1, "load": 1},
                models={"damage": StubDetector(), "load": StubDetector()},
                verbose=False)
            self.assertEqual(sorted(r.detectors_run), ["damage", "load"])
            self.assertEqual(r.frames_read, 120, "the video was decoded once")
            self.assertTrue(r.detections["damage"])
            self.assertTrue(r.detections["load"])

    def test_the_detectors_share_frame_indices_and_timestamps(self):
        from features.raw_collect import collect_camera
        with tempfile.TemporaryDirectory() as root:
            r = collect_camera(
                camera_id=RUT, video_path=self._side(root),
                feature_models_dir="", features=("damage", "load"),
                clock=CameraClock(RUT, fps=FPS, total_frames=120),
                strides={"damage": 1, "load": 1},
                models={"damage": StubDetector(), "load": StubDetector()},
                verbose=False)
        frames = {k: sorted({o.local_frame for o in r.observations
                             if o.kind == k})
                  for k in ("damage", "load")}
        # Load emits on EVERY frame it classifies, including frames it calls
        # neither loaded nor empty, because the legacy vote divides by every
        # frame looked at. Damage emits only where it detects. So at equal
        # strides the damage frames are a SUBSET of the load frames rather than
        # equal to them -- which still proves the two detectors are indexing
        # the same decode, and is the strongest claim the shapes allow.
        self.assertTrue(frames["damage"], "damage produced nothing")
        self.assertTrue(frames["load"], "load produced nothing")
        self.assertLessEqual(set(frames["damage"]), set(frames["load"]),
                             "the detectors are not on the same frame grid")
        times = {(o.local_frame, round(o.t_start, 6)) for o in r.observations}
        by_frame = {}
        for f, t in times:
            by_frame.setdefault(f, set()).add(t)
        for f, ts in by_frame.items():
            self.assertEqual(len(ts), 1, f"frame {f} got two different times")

    def test_a_disabled_detector_is_skipped(self):
        from features.raw_collect import collect_camera
        with tempfile.TemporaryDirectory() as root:
            r = collect_camera(
                camera_id=RUT, video_path=self._side(root),
                feature_models_dir="", features=("damage",),
                clock=CameraClock(RUT, fps=FPS, total_frames=120),
                strides={"damage": 1},
                models={"damage": StubDetector(), "load": StubDetector()},
                verbose=False)
            self.assertEqual(r.detectors_run, ["damage"])
            self.assertNotIn("load", r.detections)

    def test_camera_authority_is_respected(self):
        """Damage is never scored on a side camera, nor door on a top one."""
        from features.raw_collect import DETECTOR_CAMERAS, collect_camera
        self.assertEqual(tuple(DETECTOR_CAMERAS["damage"]), tuple(C.TOP_CAMERAS))
        self.assertEqual(tuple(DETECTOR_CAMERAS["door"]), tuple(C.SIDE_CAMERAS))
        with tempfile.TemporaryDirectory() as root:
            r = collect_camera(
                camera_id=C.CAMERA_RIGHT_UP, video_path=self._side(root),
                feature_models_dir="", features=("damage",),
                clock=CameraClock(C.CAMERA_RIGHT_UP, fps=FPS,
                                  total_frames=120),
                models={"damage": StubDetector()}, verbose=False)
            self.assertFalse(r.ok)
            self.assertIn("no enabled detector applies", r.skipped)

    def test_every_observation_carries_full_provenance(self):
        from features.raw_collect import collect_camera
        with tempfile.TemporaryDirectory() as root:
            r = collect_camera(
                camera_id=RUT, video_path=self._side(root),
                feature_models_dir="", features=("damage", "load"),
                clock=CameraClock(RUT, fps=FPS, total_frames=120, offset=2.0),
                strides={"damage": 1, "load": 1},
                models={"damage": StubDetector(), "load": StubDetector()},
                verbose=False)
        self.assertTrue(r.observations)
        for o in r.observations:
            self.assertEqual(o.camera_id, RUT)
            self.assertIsNotNone(o.local_frame)
            self.assertTrue(o.model)
            for key in ("fps", "stride", "source_video", "offset_applied",
                        "detector", "local_time"):
                self.assertIn(key, o.payload, f"{o.kind} lost {key}")
            self.assertAlmostEqual(o.t_start, o.payload["local_time"] + 2.0)

    def test_no_wagon_id_appears_anywhere_in_phase_one(self):
        from features.raw_collect import collect_camera
        with tempfile.TemporaryDirectory() as root:
            r = collect_camera(
                camera_id=RUT, video_path=self._side(root),
                feature_models_dir="", features=("damage", "load"),
                clock=CameraClock(RUT, fps=FPS, total_frames=120),
                strides={"damage": 1, "load": 1},
                models={"damage": StubDetector(), "load": StubDetector()},
                verbose=False)
        for o in r.observations:
            self.assertNotIn("GW_", str(o.to_dict()))

    def test_feature_flags_reach_the_shared_entry_point(self):
        with tempfile.TemporaryDirectory() as root:
            vids = {RUT: self._side(root)}
            col = collect_camera_evidence(
                video_paths=vids, feature_models_dir="",
                clocks={RUT: CameraClock(RUT, fps=FPS, total_frames=120)},
                features=("damage",),
                models={"damage": StubDetector(), "load": StubDetector()},
                damage_stride=1, verbose=False)
            self.assertIn("damage", col.per_feature)
            self.assertNotIn("load", col.per_feature)
