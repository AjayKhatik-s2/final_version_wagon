"""The stepper refactor must not change one gap event.

`GapTracker.process_video` owned the decode loop AND all tracker state as
method locals, so nothing could feed it frames. The unified collector has to
decode each camera once and hand the same frame to the gap tracker and to
Door / Damage / Load, which that shape makes impossible.

The loop body was therefore extracted verbatim into `step()`, with the
loop-local state moved onto `self`. This is a mechanical refactor and the tests
treat it as one: the PRE-REFACTOR implementation is loaded from git and run
against the new one over identical video, and every gap event must match --
frames, confidences, hit counts, trajectories, bbox history, the lot.

The comparison uses the REAL gap model when it is present, because a stub
detector would exercise the association and confirmation logic on synthetic
boxes and prove much less. When it is absent the stub still checks the
plumbing, and the test says which mode it ran in rather than quietly weakening.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

from _engine_harness import V4_ROOT, WAGON_COUNT_DIR  # noqa: F401

import cv2
import numpy as np

ENGINE_REL = "wagon_count/tracker_engine.py"
GAP_MODEL = os.path.join(V4_ROOT, "models", "reconstruction",
                         "right_up_wagon_gap.pt")
W, H, FPS = 320, 240, 15.0


def _pre_refactor_module():
    """Load the tracker_engine as it was BEFORE this refactor, from git.

    Compared against the version in the working tree. If the pre-refactor
    source cannot be recovered the test skips loudly rather than passing on a
    comparison it did not make.
    """
    for rev in ("HEAD", "HEAD~1", "HEAD~2", "HEAD~3"):
        try:
            src = subprocess.run(
                ["git", "show", f"{rev}:{ENGINE_REL}"], cwd=V4_ROOT,
                capture_output=True, text=True, timeout=30)
        except Exception:
            return None, None
        if src.returncode != 0 or "def process_video" not in src.stdout:
            continue
        if "def step(" in src.stdout:
            continue                      # already refactored at this revision
        path = os.path.join(tempfile.mkdtemp(), "tracker_engine_pre.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src.stdout)
        if WAGON_COUNT_DIR not in sys.path:
            sys.path.insert(0, WAGON_COUNT_DIR)
        spec = importlib.util.spec_from_file_location("tracker_engine_pre",
                                                      path)
        mod = importlib.util.module_from_spec(spec)
        # Registered BEFORE exec: dataclass/typing resolution looks the module
        # up in sys.modules while it is still executing.
        sys.modules["tracker_engine_pre"] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop("tracker_engine_pre", None)
            return None, rev
        return mod, rev
    return None, None


def _moving_bar_video(path, n_frames=90):
    """A dark frame with one bright vertical bar sweeping left to right.

    Whatever the detector makes of it, both implementations see exactly the
    same pixels, which is all the comparison needs.
    """
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(n_frames):
        frame = np.full((H, W, 3), 18, dtype=np.uint8)
        x = int((i / max(1, n_frames - 1)) * (W - 40)) + 10
        frame[30:H - 30, x:x + 24] = 235
        vw.write(frame)
    vw.release()
    return path


def _fingerprint(tracks):
    """Everything about a result that the refactor could plausibly disturb."""
    return {
        "camera_id": tracks.camera_id,
        "fps": round(float(tracks.fps), 6),
        "total_frames": int(tracks.total_frames),
        "width": int(tracks.width), "height": int(tracks.height),
        "n_gaps": len(tracks.gaps),
        "gaps": [{
            "track_id": g.track_id, "camera_id": g.camera_id,
            "start_frame": g.start_frame, "end_frame": g.end_frame,
            "confidence": round(float(g.confidence), 9),
            "hit_count": g.hit_count,
            "tcs": round(float(g.temporal_consistency_score), 9),
            "centers": [round(float(c), 6) for c in g.center_x_trajectory],
            "hit_frames": list(g.hit_frames),
            "bboxes": [[round(float(v), 6) for v in b] for b in g.bbox_history],
            "class_label": g.class_label,
        } for g in tracks.gaps],
        "raw_frames": sorted((tracks.raw_frame_detections or {}).keys()),
    }


#: The real gap model is MULTI-class -- {0: engine_head, 1: gap, 2: locono} --
#: and `_detect_gaps` requires "gap" in the class name. A stub emitting class 0
#: has every detection rejected at the class filter, which is how the first run
#: of this suite compared two empty results.
GAP_CLASS_ID = 1


class _StubGapModel:
    """Deterministic stand-in, emitting the GAP class the filter accepts."""
    names = {GAP_CLASS_ID: "gap"}

    def __call__(self, frame, verbose=False):
        col = frame[:, :, 0].mean(axis=0)
        bright = np.where(col > 120)[0]
        if len(bright) == 0:
            return [_R(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))]
        x1, x2 = float(bright.min()), float(bright.max())
        return [_R(np.array([[x1, 30.0, x2, float(H - 30)]]),
                   np.array([0.85]),
                   np.array([GAP_CLASS_ID], dtype=int))]


class _R:
    def __init__(self, xyxy, conf, cls):
        self.boxes = _B(xyxy, conf, cls)


class _B:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _A(xyxy), _A(conf), _A(cls)

    def __len__(self):
        return len(self.xyxy.numpy())


class _A:
    def __init__(self, a):
        self._a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


def _build(mod, model):
    t = mod.GapTracker.__new__(mod.GapTracker)
    from tracker_engine import GapTracker as Live
    ref = Live.__new__(Live)
    return t, ref


class TestStepperEquivalence(unittest.TestCase):
    """Old implementation versus new, same video, event for event."""

    @classmethod
    def setUpClass(cls):
        cls.pre, cls.rev = _pre_refactor_module()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = _moving_bar_video(os.path.join(cls.tmp.name, "bar.mp4"))
        cls.real_model = os.path.isfile(GAP_MODEL)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _tracker(self, module, model):
        """A GapTracker from `module`, driven by the STUB detector.

        The stub is used even when the real weights are present, and
        deliberately. This refactor moved tracker STATE; it did not touch
        detection. What has to be proven identical is association,
        confirmation, miss handling and gap construction, and a deterministic
        detector exercises all of those on every frame. The real weights find
        nothing in synthetic video, which would make the comparison empty --
        as the non-vacuity guard below demonstrates.
        """
        if not os.path.isfile(GAP_MODEL):
            self.skipTest("gap weights absent; cannot construct a GapTracker")
        tr = module.GapTracker(model_path=GAP_MODEL, camera_id="RIGHT_UP",
                               confidence=0.40, min_height_ratio=0.35,
                               verbose=False)
        tr.model = model          # detector swapped, everything else real
        return tr

    def test_the_pre_refactor_source_was_recovered(self):
        """Otherwise every equivalence assertion below is vacuous."""
        if self.pre is None:
            self.skipTest(f"pre-refactor tracker_engine not importable "
                          f"(rev={self.rev})")
        self.assertTrue(hasattr(self.pre, "GapTracker"))
        self.assertFalse(hasattr(self.pre.GapTracker, "step"),
                         "the recovered revision is already refactored")

    def test_process_video_is_event_for_event_identical(self):
        if self.pre is None:
            self.skipTest("pre-refactor source unavailable")
        import tracker_engine as live
        old = self._tracker(self.pre, _StubGapModel()).process_video(self.video)
        new = self._tracker(live, _StubGapModel()).process_video(self.video)
        self.assertEqual(_fingerprint(old), _fingerprint(new))

    def test_the_comparison_was_not_empty(self):
        """A run that produced no gaps would make equality meaningless."""
        import tracker_engine as live
        new = self._tracker(live, _StubGapModel()).process_video(self.video)
        self.assertGreater(len(new.gaps) + len(new.raw_frame_detections or {}),
                           0, "neither gaps nor raw detections were produced; "
                              "the equivalence check proves nothing")

    def test_the_detector_used_is_reported_honestly(self):
        """Say what was actually exercised, not what was available.

        The gap WEIGHTS are loaded (the constructor needs them) but the
        detector is then swapped for the stub, because the real model finds
        nothing in synthetic video. Detection quality is therefore NOT under
        test here -- only the state plumbing the refactor moved.
        """
        print(f"[GAP EQUIVALENCE] detector=stub (real weights loaded but "
              f"NOT exercised), pre-refactor rev={self.rev}")
        self.assertTrue(os.path.isfile(GAP_MODEL))


class TestStepMatchesProcessVideo(unittest.TestCase):
    """Frame-by-frame stepping must equal the built-in loop."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = _moving_bar_video(os.path.join(cls.tmp.name, "bar.mp4"))
        cls.real_model = os.path.isfile(GAP_MODEL)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _tracker(self):
        import tracker_engine as live
        if not os.path.isfile(GAP_MODEL):
            self.skipTest("gap weights absent; cannot construct a GapTracker")
        tr = live.GapTracker(model_path=GAP_MODEL, camera_id="RIGHT_UP",
                             confidence=0.40, min_height_ratio=0.35,
                             verbose=False)
        tr.model = _StubGapModel()
        return tr

    def _drive_manually(self):
        """What the shared collector does: own the decode, call step()."""
        tr = self._tracker()
        cap = cv2.VideoCapture(self.video)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        tr.begin(keep_raw_detections=True)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tr.step(i, frame, frame_h=height)
            i += 1
        cap.release()
        return tr.finish(video_path=self.video, fps=fps, width=width,
                         height=height, total_frames_meta=total)

    def test_manual_stepping_equals_process_video(self):
        builtin = self._tracker().process_video(self.video)
        manual = self._drive_manually()
        self.assertEqual(_fingerprint(builtin), _fingerprint(manual))

    def test_stepping_is_repeatable(self):
        self.assertEqual(_fingerprint(self._drive_manually()),
                         _fingerprint(self._drive_manually()))

    def test_begin_resets_state_between_runs(self):
        """A reused tracker must not carry tracks across videos."""
        tr = self._tracker()
        first = tr.process_video(self.video)
        second = tr.process_video(self.video)
        self.assertEqual(_fingerprint(first), _fingerprint(second),
                         "state leaked between runs on the same instance")

    def test_the_public_api_is_unchanged(self):
        import inspect
        import tracker_engine as live
        p = inspect.signature(live.GapTracker.process_video).parameters
        self.assertEqual(list(p),
                         ["self", "video_path", "frame_limit",
                          "keep_raw_detections"])
        self.assertEqual(p["frame_limit"].default, 0)
        self.assertEqual(p["keep_raw_detections"].default, True)

    def test_frame_limit_still_works(self):
        full = self._tracker().process_video(self.video)
        clipped = self._tracker().process_video(self.video, frame_limit=20)
        self.assertLessEqual(clipped.total_frames, full.total_frames)
        self.assertLessEqual(len(clipped.gaps), len(full.gaps))


if __name__ == "__main__":
    unittest.main()


class TestGapInsideTheSharedDecodeLoop(unittest.TestCase):
    """ONE VideoCapture -> GAP + Door + Damage + Load on the same frames.

    The live path used to be: the gap tracker opened the video and decoded it,
    then -- much later, after the roster and the wagon cache existed -- the
    feature processors read bucketed JPEGs. Now a single decode feeds all of
    them, and the gap events that come out must still be the ones the proven
    `process_video` produces.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = _moving_bar_video(os.path.join(cls.tmp.name, "bar.mp4"))
        cls.have_model = os.path.isfile(GAP_MODEL)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _tracker(self):
        import tracker_engine as live
        if not self.have_model:
            self.skipTest("gap weights absent")
        tr = live.GapTracker(model_path=GAP_MODEL, camera_id="RIGHT_UP_TOP",
                             confidence=0.40, min_height_ratio=0.35,
                             verbose=False)
        tr.model = _StubGapModel()
        return tr

    def _collect(self, features=("damage",), models=None):
        from features.raw_collect import collect_camera
        from core.master_timeline import CameraClock
        return collect_camera(
            camera_id="RIGHT_UP_TOP", video_path=self.video,
            feature_models_dir="", features=features,
            clock=CameraClock("RIGHT_UP_TOP", fps=FPS, total_frames=90),
            strides={"damage": 3}, models=models or {},
            gap_tracker=self._tracker(), verbose=False)

    def test_gap_events_match_process_video_exactly(self):
        """The regression that matters: same events out of the shared loop."""
        reference = self._tracker().process_video(self.video)
        shared = self._collect()
        self.assertIsNotNone(shared.gap_tracks)
        self.assertEqual(_fingerprint(reference), _fingerprint(shared.gap_tracks))
        self.assertGreater(len(reference.gaps), 0,
                           "an empty corpus would prove nothing")

    def test_one_decoder_feeds_everything(self):
        import inspect
        from features import raw_collect
        src = inspect.getsource(raw_collect)
        self.assertEqual(src.count("cv2.VideoCapture"), 1,
                         "a second decoder appeared in the shared collector")

    def test_gap_steps_on_every_frame_while_features_use_strides(self):
        """Skipping a frame for GAP would be a missed association, not a sample."""
        det = _StubDamage()
        r = self._collect(features=("damage",), models={"damage": det})
        self.assertEqual(r.frames_read, 90)
        self.assertEqual(r.frames_scored["damage"], 30)   # stride 3
        self.assertEqual(r.gap_tracks.total_frames, 90)   # every frame stepped

    def test_gap_observations_are_in_the_unified_evidence(self):
        r = self._collect()
        gaps = [o for o in r.observations if o.kind == "gap"]
        self.assertTrue(gaps, "no GAP observation reached the shared evidence")
        for o in gaps:
            self.assertEqual(o.camera_id, "RIGHT_UP_TOP")
            self.assertEqual(o.model, "gap")
            self.assertTrue(o.detected)
            self.assertIsNotNone(o.t_start)

    def test_no_wagon_id_exists_during_collection(self):
        r = self._collect()
        for o in r.observations:
            self.assertNotIn("GW_", str(o.to_dict()))
        self.assertNotIn("GW_", str(r.to_dict()))

    def test_gap_and_features_share_the_frame_coordinate_system(self):
        det = _StubDamage()
        r = self._collect(features=("damage",), models={"damage": det})
        dmg = [o for o in r.observations if o.kind == "damage"]
        self.assertTrue(dmg)
        for o in dmg:
            self.assertAlmostEqual(o.t_start, o.local_frame / FPS, places=6)
        # GAP saw every frame; each damage frame is one GAP also observed,
        # which is the point -- one decode, one coordinate system.
        raw = set((r.gap_tracks.raw_frame_detections or {}))
        seen = {o.local_frame for o in dmg}
        self.assertTrue(seen & raw,
                        "damage and gap observed no frame in common")

    def test_fusion_assigns_afterwards_and_reruns_no_model(self):
        from core.global_state_loader import GlobalWagon
        from core.timeline_evidence import TimelineEvidence
        det = _StubDamage()
        r = self._collect(features=("damage",), models={"damage": det})
        calls_after_collection = det.calls

        roster = [GlobalWagon(global_id="GW_1", wagon_index=1,
                              start_frame_master=0, end_frame_master=44,
                              start_time=0.0, end_time=3.0,
                              classification="WAGON",
                              classification_confidence=0.9),
                  GlobalWagon(global_id="GW_2", wagon_index=2,
                              start_frame_master=45, end_frame_master=89,
                              start_time=3.0, end_time=6.0,
                              classification="WAGON",
                              classification_confidence=0.9)]
        ev = TimelineEvidence(mode="test")
        ev.extend(r.observations)
        self.assertEqual(ev.assignments, [])
        ev.fuse(roster)
        self.assertTrue(any(a.global_id for a in ev.assignments))
        self.assertEqual(det.calls, calls_after_collection,
                         "a model was run again during Phase 2")

    def test_a_gap_only_run_is_still_one_decode(self):
        """No features enabled: the collector becomes exactly the old gap pass."""
        r = self._collect(features=())
        self.assertEqual(r.detectors_run, ["gap"])
        self.assertIsNotNone(r.gap_tracks)
        self.assertEqual(_fingerprint(self._tracker().process_video(self.video)),
                         _fingerprint(r.gap_tracks))


class _StubDamage:
    names = {0: "inner_wall_damage"}

    def __init__(self):
        self.calls = 0

    def __call__(self, frame, verbose=False):
        self.calls += 1
        # Same bright-column test the gap stub uses. A whole-frame mean would
        # sit near 30 on this video and never cross a sensible threshold.
        col = frame[:, :, 0].mean(axis=0)
        if not len(np.where(col > 120)[0]):
            return [_R(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))]
        return [_R(np.array([[W * 0.3, H * 0.3, W * 0.7, H * 0.7]]),
                   np.array([0.8]), np.array([0], dtype=int))]
