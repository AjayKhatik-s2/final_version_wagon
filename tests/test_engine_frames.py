"""Engine frames are TRAIN-level evidence, never wagons.

A locomotive is not a wagon, and the counting engine already says so: master
classification labels it ENGINE and `get_master_wagon_window()` keeps it out of
the roster, recording it under `wagon_window` as a `NonWagonObject` with no
`GW_n`. This module reuses that decision instead of re-deriving it.

The properties worth pinning are mostly negative -- what engine frames must NOT
touch -- because the risk here is contamination: a stray engine segment
appearing as a wagon would change the count, which is the one number the whole
pipeline exists to produce.

Fixtures build real videos with cv2 so the extractor decodes actual frames and
the ranking runs for real; each frame carries a distinct grey level so a test
can name which frame was chosen.
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
from core.global_state_loader import GlobalTrainState, GlobalWagon
from orchestrator import engine_frames as EF

FPS = 10.0
W, H = 64, 48
RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP


def _make_video(path, n_frames, *, sharp_at=()):
    """A video whose frames differ, with chosen indices made SHARP.

    `detection_quality` rewards Laplacian texture, so a noise frame scores far
    above a flat one. That gives the ranking something real to prefer.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        if i in sharp_at:
            frame = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        else:
            frame = np.full((H, W, 3), 128, dtype=np.uint8)
        vw.write(frame)
    vw.release()
    return path


def _state(engine_spans=((0.0, 1.0),), wagon_spans=((1.0, 5.0),)):
    """A state whose wagon_window carries ENGINE segments, as Stage 1 writes it."""
    wagons = tuple(
        GlobalWagon(global_id=f"GW_{i}", wagon_index=i,
                    start_frame_master=int(s * FPS),
                    end_frame_master=int(e * FPS) - 1,
                    start_time=s, end_time=e,
                    classification=C.CLASS_WAGON,
                    classification_confidence=1.0)
        for i, (s, e) in enumerate(wagon_spans, start=1))
    window = {
        "found": True,
        "leading_non_wagon_objects": [
            {"classification": C.CLASS_ENGINE,
             "classification_confidence": 0.93,
             "start_frame": int(s * FPS), "end_frame": int(e * FPS) - 1,
             "start_time": s, "end_time": e,
             "position": "leading", "segment_index": i}
            for i, (s, e) in enumerate(engine_spans)],
        "trailing_non_wagon_objects": [],
        "interior_non_wagon_objects": [],
    }
    st = GlobalTrainState(total_wagons=len(wagons), wagons=wagons,
                          master_camera=C.MASTER_CAMERA, master_fps=FPS,
                          master_total_frames=int(max(e for _s, e in wagon_spans)
                                                  * FPS))
    st.wagon_window = window
    return st


def _run(root, *, engine_spans=((0.0, 1.0),), cameras=(RU, LU),
         n_frames=60, sharp_at=(), offsets=None, max_per_side=5):
    videos, fps = {}, {}
    for cam in cameras:
        videos[cam] = _make_video(os.path.join(root, f"{cam}.mp4"), n_frames,
                                  sharp_at=sharp_at)
        fps[cam] = FPS
    return EF.extract(state=_state(engine_spans=engine_spans),
                      video_paths=videos, per_camera_fps=fps,
                      output_root=root, camera_offsets=offsets or {},
                      max_frames_per_side=max_per_side, verbose=False)


class TestEngineSegmentDiscovery(unittest.TestCase):
    def test_reads_engine_segments_from_the_wagon_window(self):
        segs = EF.engine_segments(_state(engine_spans=((0.0, 1.0), (5.0, 6.0))))
        self.assertEqual(len(segs), 2)
        for s in segs:
            self.assertEqual(s["classification"], C.CLASS_ENGINE)

    def test_ignores_non_engine_non_wagon_objects(self):
        st = _state()
        st.wagon_window["trailing_non_wagon_objects"] = [
            {"classification": C.CLASS_BRAKE_VAN,
             "classification_confidence": 0.9,
             "start_frame": 100, "end_frame": 120,
             "start_time": 10.0, "end_time": 12.0,
             "position": "trailing", "segment_index": 9}]
        self.assertEqual([s["segment_index"] for s in EF.engine_segments(st)],
                         [0], "only ENGINE segments qualify")

    def test_no_window_means_no_extraction(self):
        st = GlobalTrainState(total_wagons=0, wagons=(),
                              master_camera=C.MASTER_CAMERA)
        self.assertEqual(EF.engine_segments(st), [])

    def test_only_side_cameras_are_used(self):
        """Top cameras look at the roof; a loco number is never there."""
        self.assertEqual(tuple(EF.ENGINE_CAMERAS), tuple(C.SIDE_CAMERAS))
        for cam in C.TOP_CAMERAS:
            self.assertNotIn(cam, EF.ENGINE_CAMERAS)


class TestSelection(unittest.TestCase):
    def test_at_most_five_per_side_and_ten_overall(self):
        with tempfile.TemporaryDirectory() as root:
            r = _run(root, engine_spans=((0.0, 3.0),))   # 30 candidate frames
            self.assertEqual(set(r.counts), {RU, LU})
            for cam, n in r.counts.items():
                self.assertLessEqual(n, EF.MAX_FRAMES_PER_SIDE, cam)
            self.assertEqual(r.counts[RU], 5)
            self.assertEqual(r.counts[LU], 5)
            self.assertLessEqual(r.total, 10)
            self.assertEqual(r.total, 10)

    def test_fewer_candidates_stores_fewer_and_records_the_count(self):
        with tempfile.TemporaryDirectory() as root:
            r = _run(root, engine_spans=((0.0, 0.3),))   # 3 candidate frames
            self.assertEqual(r.counts[RU], 3)
            man = EF.read_manifest(r.root, RU)
            self.assertEqual(man["count"], 3)
            self.assertEqual(man["requested"], 5)
            self.assertFalse(man["complete"])
            paths = [f["path"] for f in man["frames"]]
            self.assertEqual(len(set(paths)), 3, "no frame may be duplicated")

    def test_selection_is_deterministic(self):
        with tempfile.TemporaryDirectory() as root:
            a = _run(os.path.join(root, "a"), engine_spans=((0.0, 2.0),),
                     sharp_at=(3, 7, 11, 15, 19))
            b = _run(os.path.join(root, "b"), engine_spans=((0.0, 2.0),),
                     sharp_at=(3, 7, 11, 15, 19))
            for cam in (RU, LU):
                self.assertEqual(
                    [f.source_frame_index for f in a.frames_by_camera[cam]],
                    [f.source_frame_index for f in b.frames_by_camera[cam]],
                    f"{cam}: ranking is not reproducible")

    def test_sharper_frames_rank_above_flat_ones(self):
        """Ranking uses the existing quality scorer, not frame order."""
        with tempfile.TemporaryDirectory() as root:
            sharp = (4, 9, 14, 19, 24)
            r = _run(root, engine_spans=((0.0, 3.0),), sharp_at=sharp)
            chosen = {f.source_frame_index for f in r.frames_by_camera[RU]}
            self.assertEqual(chosen, set(sharp),
                             f"expected the sharp frames, got {sorted(chosen)}")
            scores = [f.score for f in r.frames_by_camera[RU]]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertEqual([f.rank for f in r.frames_by_camera[RU]],
                             [1, 2, 3, 4, 5])


class TestCameraSeparation(unittest.TestCase):
    def test_the_two_sides_are_stored_apart(self):
        with tempfile.TemporaryDirectory() as root:
            r = _run(root)
            for cam in (RU, LU):
                d = os.path.join(r.root, cam)
                self.assertTrue(os.path.isdir(d), f"{cam} has no directory")
                for f in r.frames_by_camera[cam]:
                    self.assertEqual(f.camera_id, cam)
                    self.assertTrue(f.path.startswith(d),
                                    f"{cam} frame stored outside its directory")

    def test_each_side_is_retrievable_on_its_own(self):
        with tempfile.TemporaryDirectory() as root:
            r = _run(root)
            for cam in (RU, LU):
                man = EF.read_manifest(r.root, cam)
                self.assertEqual(man["camera_id"], cam)
                for fr in man["frames"]:
                    self.assertEqual(fr["camera_id"], cam)

    def test_one_missing_side_does_not_stop_the_other(self):
        with tempfile.TemporaryDirectory() as root:
            r = _run(root, cameras=(RU,))       # no LEFT_UP video at all
            self.assertEqual(r.counts[RU], 5)
            self.assertEqual(r.counts[LU], 0)
            self.assertIn(LU, r.skipped)
            self.assertIn("video unavailable", r.skipped[LU])

    def test_absent_top_cameras_are_irrelevant(self):
        """Engine extraction depends only on the side cameras."""
        with tempfile.TemporaryDirectory() as root:
            r = _run(root, cameras=(RU, LU))
            self.assertEqual(r.total, 10)
            for cam in C.TOP_CAMERAS:
                self.assertNotIn(cam, r.frames_by_camera)

    def test_camera_offset_shifts_only_that_camera(self):
        with tempfile.TemporaryDirectory() as root:
            r = _run(root, engine_spans=((1.0, 2.0),),
                     offsets={LU: -0.5})
            ru = {f.source_frame_index for f in r.frames_by_camera[RU]}
            lu = {f.source_frame_index for f in r.frames_by_camera[LU]}
            # LEFT_UP's clock is 0.5s behind, i.e. +5 frames at 10 fps.
            self.assertTrue(min(lu) >= min(ru),
                            f"offset not applied: {sorted(ru)} vs {sorted(lu)}")


class TestMetadata(unittest.TestCase):
    def test_every_field_a_later_ocr_pass_needs(self):
        with tempfile.TemporaryDirectory() as root:
            r = _run(root)
            man = EF.read_manifest(r.root, RU)
            self.assertEqual(man["schema"], EF.SCHEMA)
            for fr in man["frames"]:
                for key in ("camera_id", "rank", "source_frame_index",
                            "timestamp", "master_time", "score", "path",
                            "source_video", "segment_index",
                            "classification", "classification_confidence"):
                    self.assertIn(key, fr)
                self.assertEqual(fr["classification"], C.CLASS_ENGINE)
                self.assertTrue(os.path.isfile(fr["path"]))

    def test_stored_images_are_real_decodable_frames(self):
        with tempfile.TemporaryDirectory() as root:
            r = _run(root)
            for fr in r.frames_by_camera[RU]:
                img = cv2.imread(fr.path)
                self.assertIsNotNone(img, f"{fr.path} is not a readable image")
                self.assertEqual(img.shape[:2], (H, W))


class TestEngineFramesNeverEnterWagonTimelines(unittest.TestCase):
    """The contamination risk, asserted directly."""

    def test_no_wagon_id_is_produced(self):
        with tempfile.TemporaryDirectory() as root:
            r = _run(root)
            for cam, frames in r.frames_by_camera.items():
                for f in frames:
                    blob = json.dumps(f.to_dict())
                    self.assertNotIn("GW_", blob,
                                     f"{cam} engine frame carries a wagon id")

    def test_the_roster_is_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            st = _state(wagon_spans=((1.0, 5.0), (5.0, 9.0)))
            before = [w.global_id for w in st.wagons]
            videos = {c: _make_video(os.path.join(root, f"{c}.mp4"), 60)
                      for c in (RU, LU)}
            EF.extract(state=st, video_paths=videos,
                       per_camera_fps={RU: FPS, LU: FPS},
                       output_root=root, verbose=False)
            self.assertEqual([w.global_id for w in st.wagons], before)
            self.assertEqual(st.total_wagons, 2)

    def test_engine_output_lives_outside_every_wagon_tree(self):
        with tempfile.TemporaryDirectory() as root:
            r = _run(root)
            self.assertEqual(os.path.basename(r.root), EF.DIR_NAME)
            for reserved in ("wagon_cache", "wagon_states", "evidence",
                             "camera_evidence", "global_state"):
                self.assertFalse(
                    os.path.exists(os.path.join(root, reserved)),
                    f"engine extraction created {reserved}/")

    def test_no_feature_evidence_is_written(self):
        with tempfile.TemporaryDirectory() as root:
            r = _run(root)
            for feature in ("door", "damage", "load", "ocr"):
                hits = []
                for dirpath, _dirs, _files in os.walk(root):
                    if os.path.basename(dirpath) == feature:
                        hits.append(dirpath)
                self.assertEqual(hits, [], f"wrote {feature} evidence")

    def test_extractor_runs_no_inference(self):
        import ast
        import inspect
        src = inspect.getsource(EF)
        for banned in ("YOLO", "load_yolo", "ultralytics", "door_proc",
                       "damage_proc", "load_proc", "predict("):
            with self.subTest(token=banned):
                self.assertNotIn(banned, src)
        # and it must not import a feature processor
        names = {a.name for n in ast.walk(ast.parse(src))
                 if isinstance(n, (ast.Import, ast.ImportFrom))
                 for a in n.names}
        modules = {n.module for n in ast.walk(ast.parse(src))
                   if isinstance(n, ast.ImportFrom) and n.module}
        self.assertFalse([x for x in (names | modules) if "features" in x])


class TestBothModesUseOneImplementation(unittest.TestCase):
    def test_batch_calls_the_shared_extractor(self):
        import inspect
        from orchestrator import master_runner
        src = inspect.getsource(master_runner.process_batch)
        self.assertIn("engine_frames.extract(", src)

    def test_sequential_calls_the_shared_extractor(self):
        import inspect
        from orchestrator import global_assembler
        src = inspect.getsource(global_assembler.assemble)
        self.assertIn("engine_frames.extract(", src)

    def test_neither_mode_reimplements_selection(self):
        """One ranking function, so the two modes cannot diverge."""
        import inspect
        from orchestrator import global_assembler, master_runner
        for mod in (master_runner, global_assembler):
            src = inspect.getsource(mod)
            with self.subTest(module=mod.__name__):
                self.assertNotIn("detection_quality", src)
                self.assertNotIn("MAX_FRAMES_PER_SIDE", src)

    def test_extraction_failure_is_non_fatal(self):
        """A missing loco frame must never fail a batch."""
        import inspect
        from orchestrator import global_assembler, master_runner
        for mod, fn in ((master_runner, "process_batch"),
                        (global_assembler, "assemble")):
            src = inspect.getsource(getattr(mod, fn))
            i = src.index("engine_frames.extract(")
            with self.subTest(function=fn):
                self.assertIn("try:", src[:i])
                self.assertIn("except Exception", src[i:i + 900])


if __name__ == "__main__":
    unittest.main()
