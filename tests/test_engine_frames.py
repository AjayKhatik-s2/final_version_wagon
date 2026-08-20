"""Engine/loco frames are a train-level asset and must never become wagons.

Two things are proven here:

1. the capture itself -- up to five per side camera, ten a train, camera
   identity and original frame index preserved, never padded with duplicates;
2. the isolation -- that running the collector changes NOTHING about wagons.
   That second half is the important one: the requirement is not merely that
   engine frames are stored somewhere, it is that they cannot leak into a local
   or global wagon timeline, a wagon id, or a wagon count.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

import cv2
import numpy as np

from core import constants as C
from core.camera_evidence import LocalSegment, local_segment_id
from features import engine_frames as EF

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP

FPS = 10.0
W, H = 96, 64


def _write_video(path, n_frames):
    """A clip whose frames differ in sharpness, so ranking is not arbitrary.

    Frame k gets a checkerboard whose square size shrinks with k, raising
    Laplacian variance -- later frames are objectively sharper. That gives the
    selector a real signal instead of identical frames.
    """
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for k in range(n_frames):
        img = np.full((H, W, 3), 128, dtype=np.uint8)
        step = max(2, 16 - (k % 14))
        for y in range(0, H, step):
            for x in range(0, W, step):
                if ((x // step) + (y // step)) % 2 == 0:
                    img[y:y + step, x:x + step] = 255
        vw.write(img)
    vw.release()
    return path


def _segments(cam, *, engine_span=(0, 29), n_wagons=2):
    """One ENGINE segment followed by `n_wagons` WAGON segments."""
    segs = []
    if engine_span is not None:
        segs.append(LocalSegment(
            local_id=local_segment_id(cam, 1), index=1,
            start_frame=engine_span[0], end_frame=engine_span[1],
            start_time=engine_span[0] / FPS, end_time=engine_span[1] / FPS,
            label=C.CLASS_ENGINE, confidence=0.88))
    start = (engine_span[1] + 1) if engine_span else 0
    for i in range(n_wagons):
        s = start + i * 20
        segs.append(LocalSegment(
            local_id=local_segment_id(cam, len(segs) + 1), index=len(segs) + 1,
            start_frame=s, end_frame=s + 19,
            start_time=s / FPS, end_time=(s + 19) / FPS,
            label=C.CLASS_WAGON, confidence=0.9))
    return segs


class TestCapture(unittest.TestCase):

    def test_five_frames_from_a_side_camera(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            r = EF.collect(train_id="T1", camera_id=RU, video_path=v,
                           segments=_segments(RU), output_dir=d,
                           fps=FPS, verbose=False)
            self.assertEqual(r.status, "OK")
            self.assertEqual(r.count, 5)
            self.assertEqual(len(r.frames), EF.MAX_FRAMES_PER_CAMERA)

    def test_ten_frames_across_the_two_side_cameras(self):
        with tempfile.TemporaryDirectory() as d:
            total = 0
            for cam in (RU, LU):
                v = _write_video(os.path.join(d, f"{cam}.mp4"), 80)
                total += EF.collect(train_id="T1", camera_id=cam, video_path=v,
                                    segments=_segments(cam),
                                    output_dir=os.path.join(d, cam),
                                    fps=FPS, verbose=False).count
            self.assertEqual(total, 10)

    def test_top_cameras_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "t.mp4"), 80)
            for cam in (RUT, LUT):
                r = EF.collect(train_id="T1", camera_id=cam, video_path=v,
                               segments=_segments(cam), output_dir=d,
                               fps=FPS, verbose=False)
                with self.subTest(camera=cam):
                    self.assertEqual(r.status, "SKIPPED")
                    self.assertEqual(r.count, 0)

    def test_no_engine_segment_yields_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            r = EF.collect(train_id="T1", camera_id=RU, video_path=v,
                           segments=_segments(RU, engine_span=None),
                           output_dir=d, fps=FPS, verbose=False)
            self.assertEqual(r.status, "NO_ENGINE")
            self.assertEqual(r.count, 0)

    def test_fewer_than_five_valid_frames_is_not_padded(self):
        """Three decodable engine frames must yield three files, not five."""
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            r = EF.collect(train_id="T1", camera_id=RU, video_path=v,
                           segments=_segments(RU, engine_span=(0, 2)),
                           output_dir=d, fps=FPS, verbose=False)
            self.assertEqual(r.count, 3)
            self.assertIn("not padded", r.note)

    def test_saved_frames_are_all_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            r = EF.collect(train_id="T1", camera_id=RU, video_path=v,
                           segments=_segments(RU), output_dir=d,
                           fps=FPS, verbose=False)
            idxs = [f.frame_idx for f in r.frames]
            self.assertEqual(len(idxs), len(set(idxs)),
                             "the same frame was saved twice")
            paths = [f.path for f in r.frames]
            self.assertEqual(len(paths), len(set(paths)))

    def test_frames_come_only_from_the_engine_span(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            r = EF.collect(train_id="T1", camera_id=RU, video_path=v,
                           segments=_segments(RU, engine_span=(10, 25)),
                           output_dir=d, fps=FPS, verbose=False)
            for f in r.frames:
                self.assertTrue(10 <= f.frame_idx <= 25,
                                f"frame {f.frame_idx} is outside the engine "
                                f"segment")

    def test_metadata_carries_everything_a_later_ocr_pass_needs(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            r = EF.collect(train_id="TRAIN_42", camera_id=RU, video_path=v,
                           segments=_segments(RU), output_dir=d,
                           fps=FPS, verbose=False)
            for f in r.frames:
                self.assertEqual(f.train_id, "TRAIN_42")
                self.assertEqual(f.camera_id, RU)
                self.assertGreaterEqual(f.frame_idx, 0)
                self.assertAlmostEqual(f.timestamp, f.frame_idx / FPS, places=4)
                self.assertGreater(f.score, 0.0)
                self.assertTrue(f.reason)
                self.assertTrue(os.path.isfile(f.path))
                self.assertEqual(f.segment_label, C.CLASS_ENGINE)

    def test_files_are_named_by_original_frame_index(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            r = EF.collect(train_id="T1", camera_id=RU, video_path=v,
                           segments=_segments(RU), output_dir=d,
                           fps=FPS, verbose=False)
            for f in r.frames:
                self.assertEqual(os.path.basename(f.path),
                                 f"engine_{f.frame_idx:06d}.jpg")

    def test_ranking_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            picks = []
            for i in range(2):
                out = os.path.join(d, f"run{i}")
                r = EF.collect(train_id="T1", camera_id=RU, video_path=v,
                               segments=_segments(RU), output_dir=out,
                               fps=FPS, verbose=False)
                picks.append([f.frame_idx for f in r.frames])
            self.assertEqual(picks[0], picks[1])

    def test_ranks_are_ordered_by_descending_score(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            r = EF.collect(train_id="T1", camera_id=RU, video_path=v,
                           segments=_segments(RU), output_dir=d,
                           fps=FPS, verbose=False)
            scores = [f.score for f in r.frames]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertEqual([f.rank for f in r.frames],
                             list(range(1, len(r.frames) + 1)))

    def test_unreadable_video_is_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            r = EF.collect(train_id="T1", camera_id=RU,
                           video_path=os.path.join(d, "missing.mp4"),
                           segments=_segments(RU), output_dir=d,
                           fps=FPS, verbose=False)
            self.assertEqual(r.status, "UNREADABLE")
            self.assertEqual(r.count, 0)

    def test_brake_van_is_not_treated_as_an_engine(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            segs = _segments(RU, engine_span=None)
            segs[0].label = C.CLASS_BRAKE_VAN
            r = EF.collect(train_id="T1", camera_id=RU, video_path=v,
                           segments=segs, output_dir=d, fps=FPS, verbose=False)
            self.assertEqual(r.status, "NO_ENGINE")


class TestStorageLayout(unittest.TestCase):

    def test_frames_live_under_engine_frames_and_nowhere_else(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            r = EF.collect(train_id="T1", camera_id=RU, video_path=v,
                           segments=_segments(RU), output_dir=d,
                           fps=FPS, verbose=False)
            for f in r.frames:
                rel = os.path.relpath(f.path, d)
                self.assertTrue(rel.startswith(os.path.join("engine_frames",
                                                            RU)),
                                f"{rel} is not under engine_frames/{RU}")

    def test_nothing_is_written_into_wagon_or_evidence_trees(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            EF.collect(train_id="T1", camera_id=RU, video_path=v,
                       segments=_segments(RU), output_dir=d,
                       fps=FPS, verbose=False)
            for forbidden in ("wagon_cache", "camera_cache", "evidence",
                              "wagon_states", "features"):
                self.assertFalse(os.path.exists(os.path.join(d, forbidden)),
                                 f"{forbidden}/ was created")

    def test_metadata_file_merges_both_cameras(self):
        with tempfile.TemporaryDirectory() as d:
            results = []
            for cam in (RU, LU):
                v = _write_video(os.path.join(d, f"{cam}.mp4"), 80)
                results.append(EF.collect(
                    train_id="T1", camera_id=cam, video_path=v,
                    segments=_segments(cam), output_dir=d, fps=FPS,
                    verbose=False))
                EF.write_metadata(d, [results[-1]])   # written one at a time

            doc = EF.load_metadata(d)
            self.assertEqual(doc["schema"], "wagon_eye.engine_frames.v1")
            self.assertEqual(doc["train_id"], "T1")
            self.assertEqual(set(doc["cameras"]), {RU, LU})
            self.assertEqual(doc["total_frames"], 10)
            self.assertEqual(doc["max_frames_per_camera"], 5)

    def test_a_later_camera_does_not_erase_an_earlier_one(self):
        """Cameras seal minutes apart, so the merge has to be additive."""
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            r1 = EF.collect(train_id="T1", camera_id=RU, video_path=v,
                            segments=_segments(RU), output_dir=d, fps=FPS,
                            verbose=False)
            EF.write_metadata(d, [r1])
            r2 = EF.collect(train_id="T1", camera_id=LU, video_path=v,
                            segments=_segments(LU), output_dir=d, fps=FPS,
                            verbose=False)
            EF.write_metadata(d, [r2])
            self.assertIn(RU, EF.load_metadata(d)["cameras"])

    def test_metadata_absent_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(EF.load_metadata(d), {})


class TestEngineFramesAreNotWagons(unittest.TestCase):
    """The invariant that actually matters."""

    def test_no_wagon_id_is_minted(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            r = EF.collect(train_id="T1", camera_id=RU, video_path=v,
                           segments=_segments(RU), output_dir=d,
                           fps=FPS, verbose=False)
            doc = json.dumps(r.to_dict())
            self.assertNotIn("GW_", doc)
            self.assertNotIn("global_id", doc)
            for f in r.frames:
                self.assertNotIn("GW_", os.path.basename(f.path))

    def test_the_segment_list_is_not_mutated(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            segs = _segments(RU)
            before = [s.to_dict() for s in segs]
            EF.collect(train_id="T1", camera_id=RU, video_path=v,
                       segments=segs, output_dir=d, fps=FPS, verbose=False)
            self.assertEqual([s.to_dict() for s in segs], before)

    def test_wagon_count_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            segs = _segments(RU, n_wagons=3)
            wagons_before = [s for s in segs if s.label == C.CLASS_WAGON]
            EF.collect(train_id="T1", camera_id=RU, video_path=v,
                       segments=segs, output_dir=d, fps=FPS, verbose=False)
            wagons_after = [s for s in segs if s.label == C.CLASS_WAGON]
            self.assertEqual(len(wagons_after), len(wagons_before))
            self.assertEqual([s.local_id for s in wagons_after],
                             [s.local_id for s in wagons_before])

    def test_engine_frames_never_appear_in_a_materialized_wagon_cache(self):
        """Run the real materializer AFTER capture; its output must be
        untouched by the engine frames."""
        from materializer import wagon_cache_builder
        with tempfile.TemporaryDirectory() as d:
            v = _write_video(os.path.join(d, "ru.mp4"), 80)
            segs = _segments(RU)
            cache = os.path.join(d, "camera_cache")
            counts = wagon_cache_builder.build_camera_local(
                camera_id=RU, video_path=v, segments=segs,
                cache_root=cache, verbose=False)
            EF.collect(train_id="T1", camera_id=RU, video_path=v,
                       segments=segs, output_dir=d, fps=FPS, verbose=False)
            counts_after = {k: len(os.listdir(os.path.join(
                cache, k, C.CAMERA_FOLDER[RU])))
                for k in os.listdir(cache)
                if os.path.isdir(os.path.join(cache, k))}
            for gw, n in counts.items():
                if n:
                    self.assertEqual(counts_after.get(gw), n,
                                     f"{gw} frame count changed")
            for name in os.listdir(cache):
                self.assertNotIn("engine_frames", name)

    def test_module_touches_no_wagon_machinery(self):
        """Structural: the collector cannot reach a timeline to corrupt it."""
        import ast
        src = open(os.path.join(V4_ROOT, "features", "engine_frames.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    names.update(a.name.split("."))
            elif isinstance(n, ast.ImportFrom):
                names.update((n.module or "").split("."))
                names.update(a.name for a in n.names)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
            elif isinstance(n, ast.Name):
                names.add(n.id)
        for banned in ("wagon_cache_builder", "global_fusion", "as_feature_wagons",
                       "write_segments", "GlobalTrainState", "GlobalWagon",
                       "build_global_wagons", "reconstruction", "GapTracker",
                       "wagon_state_builder", "run_global_count"):
            with self.subTest(token=banned):
                self.assertNotIn(banned, names)

    def test_engine_class_is_read_not_redefined(self):
        """The label comes from the Stage-1 classifier, unchanged."""
        self.assertTrue(EF.is_engine_segment(
            LocalSegment(local_id="x", index=1, start_frame=0, end_frame=1,
                         start_time=0.0, end_time=1.0,
                         label=C.CLASS_ENGINE, confidence=0.5)))
        for other in (C.CLASS_WAGON, C.CLASS_BRAKE_VAN, C.CLASS_UNKNOWN, ""):
            with self.subTest(label=other):
                self.assertFalse(EF.is_engine_segment(
                    LocalSegment(local_id="x", index=1, start_frame=0,
                                 end_frame=1, start_time=0.0, end_time=1.0,
                                 label=other, confidence=0.5)))


class TestCameraRunnerWiring(unittest.TestCase):

    def test_run_camera_collects_after_segments_are_final(self):
        """Capture must sit AFTER write_segments, so it cannot influence them."""
        import inspect
        from orchestrator import camera_runner
        src = inspect.getsource(camera_runner.run_camera)
        # Anchor on the IMPORT, not the bare word: `collect_engine_frames`
        # appears in the signature long before the call site.
        call = src.index("from features import engine_frames")
        self.assertLess(src.index("bundle.write_segments"), call)
        self.assertLess(call,
                        src.index("wagon_cache_builder.build_camera_local"))

    def test_capture_failure_cannot_fail_the_camera(self):
        import inspect
        from orchestrator import camera_runner
        src = inspect.getsource(camera_runner.run_camera)
        block = src[src.index("if collect_engine_frames"):]
        self.assertIn("except Exception", block.split("_t(\"engine_frames\"")[0])

    def test_result_reports_the_count(self):
        from orchestrator.camera_runner import CameraRunResult
        r = CameraRunResult(camera_id=RU, engine_frames=5)
        self.assertEqual(r.to_dict()["engine_frames"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
