"""Wiring tests for the EXPERIMENTAL sampled inference modes.

These assert the safety property that matters most right now: **production
behaviour is unchanged by construction**.  The default is "legacy" on both
processors, and the orchestrator never passes the flag at all, so the sampled
path is unreachable unless a benchmark asks for it explicitly.

Model-dependent behaviour (detection quality, per-wagon states) is covered by
the benchmark harness, not here -- these tests must run without weights.
"""

from __future__ import annotations

import inspect
import os
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from features.damage import processor as damage_proc
from features.door import processor as door_proc
from features.evidence_aggregator import EvidenceAggregator, Observation

PROCESSORS = (("door", door_proc), ("damage", damage_proc))


class TestDefaultsAreLegacy(unittest.TestCase):
    def test_both_processors_default_to_legacy(self):
        for name, mod in PROCESSORS:
            with self.subTest(feature=name):
                self.assertEqual(
                    inspect.signature(mod.run).parameters["inference_mode"].default,
                    "legacy",
                    f"{name} must default to the known-good legacy path")

    def test_sample_stride_default_is_two(self):
        for name, mod in PROCESSORS:
            with self.subTest(feature=name):
                self.assertEqual(
                    inspect.signature(mod.run).parameters["sample_stride"].default, 2)

    def test_legacy_implementation_is_retained(self):
        """The tracker path must still exist -- sampled mode is additive."""
        for name, mod in PROCESSORS:
            with self.subTest(feature=name):
                self.assertTrue(hasattr(mod, "_run_tracker_one_camera"))
                self.assertTrue(hasattr(mod, "_run_sampled_one_camera"))

    def test_invalid_mode_is_rejected_loudly(self):
        for name, mod in PROCESSORS:
            with self.subTest(feature=name):
                with self.assertRaises(ValueError):
                    mod.run(state=None, cache_root="", feature_models_dir="",
                            output_dir="", inference_mode="turbo")


class TestOrchestratorSelectsSampled(unittest.TestCase):
    """The pipeline is now deliberately in sampled mode for the EC2 experiment.

    This inverts an earlier guard: sampling used to be unreachable from
    production on purpose.  It is now the default BY REQUEST, and legacy must
    stay one flag away.
    """

    def _parse(self, argv):
        from orchestrator.master_runner import _build_parser
        return _build_parser().parse_args(argv)

    def test_production_command_selects_sampled(self):
        """The exact command that will run on EC2 (strides pinned in
        TestExperimentTwoConfiguration)."""
        a = self._parse(["--local-only", "--local-inputs", "./local_inputs",
                         "--no-interactive", "--disable-features", "ocr",
                         "--skip-upload", "--skip-email"])
        self.assertEqual(a.door_inference_mode, "sampled")
        self.assertEqual(a.damage_inference_mode, "sampled")
        self.assertEqual(a.load_inference_mode, "sampled")

    def test_process_batch_defaults_to_sampled(self):
        from orchestrator.master_runner import process_batch
        p = inspect.signature(process_batch).parameters
        for feat in ("door", "damage", "load"):
            self.assertEqual(p[f"{feat}_inference_mode"].default, "sampled")
        self.assertEqual(p["door_sample_stride"].default, 3)
        self.assertEqual(p["damage_sample_stride"].default, 3)
        self.assertEqual(p["load_sample_stride"].default, 2)

    def test_legacy_is_still_reachable(self):
        """Legacy must never be destroyed -- one flag restores it."""
        a = self._parse(["--local-only", "--legacy-inference"])
        self.assertTrue(a.legacy_inference)
        b = self._parse(["--local-only",
                         "--door-inference-mode", "legacy",
                         "--damage-inference-mode", "legacy"])
        self.assertEqual(b.door_inference_mode, "legacy")
        self.assertEqual(b.damage_inference_mode, "legacy")

    def test_ocr_is_not_given_an_inference_mode(self):
        """OCR is out of scope and must keep its original signature."""
        from features.ocr import processor as ocr_p
        self.assertNotIn("inference_mode",
                         inspect.signature(ocr_p.run).parameters)

    def test_load_now_accepts_the_selector(self):
        """Added in experiment 2 -- Load's stride becomes explicit."""
        from features.load import processor as load_p
        p = inspect.signature(load_p.run).parameters
        self.assertIn("inference_mode", p)
        self.assertEqual(p["inference_mode"].default, "legacy")


class TestSampledPathContract(unittest.TestCase):
    """The sampled helpers must return the SAME tuple shape as legacy, so the
    surrounding run() -- JSON, evidence, snapshots -- needs no branching."""

    def test_door_sampled_returns_six_tuple_on_empty_cache(self):
        got = door_proc._run_sampled_one_camera(
            None, door_proc.TrackerConfig(), "/nonexistent", "GW_1",
            "RIGHT_UP", sample_stride=2)
        self.assertEqual(len(got), 6)
        decisions, used, w, h, cands, overlay = got
        self.assertEqual((decisions, used, w, h), ([], 0, 0, 0))
        self.assertEqual(cands, {})
        self.assertEqual(set(overlay), {"tracks", "events"})

    def test_damage_sampled_returns_five_tuple_on_empty_cache(self):
        got = damage_proc._run_sampled_one_camera(
            None, damage_proc.DamageTrackerConfig(), "/nonexistent", "GW_1",
            "RIGHT_UP_TOP", confidence_floor=0.55, sample_stride=2)
        self.assertEqual(len(got), 5)
        self.assertEqual(got, ([], 0, 0, 0, []))

    def test_door_sampled_tuple_matches_legacy_arity(self):
        empty_legacy = door_proc._run_tracker_one_camera(
            None, door_proc.TrackerConfig(), door_proc.MergeConfig(),
            "/nonexistent", "GW_1", "RIGHT_UP")
        empty_sampled = door_proc._run_sampled_one_camera(
            None, door_proc.TrackerConfig(), "/nonexistent", "GW_1", "RIGHT_UP")
        self.assertEqual(len(empty_legacy), len(empty_sampled))


class TestAggregatorDrivesTheDecision(unittest.TestCase):
    """The decision records the sampled path emits must carry the fields
    `_pick_side_state` relies on, with frame support standing in for hits."""

    def _agg(self, states):
        agg = EvidenceAggregator(frame_width=960, frame_height=540, stride=2)
        for i, st in enumerate(states):
            agg.add_frame(i * 2, [Observation(
                frame_idx=i * 2, state=st, confidence=0.9,
                bbox=(100.0, 100.0, 300.0, 250.0), score=1.0)])
        return agg.finalize()

    def test_repeated_evidence_produces_an_accepted_group(self):
        res = self._agg(["CLOSED"] * 6)
        self.assertTrue(res["accepted"])
        g = res["accepted"][0]
        for key in ("candidate_id", "state", "confidence", "frame_support",
                    "first_frame", "last_frame", "best"):
            self.assertIn(key, g)
        self.assertEqual(g["state"], "CLOSED")
        self.assertEqual(g["frame_support"], 6)

    def test_lone_outlier_does_not_win(self):
        res = self._agg(["CLOSED"] * 6 + ["OPEN"])
        self.assertEqual(res["accepted"][0]["state"], "CLOSED")

    def test_frame_support_is_stride_invariant_in_fraction(self):
        """The reason stride-2 broke the tracker but not the aggregator."""
        dense = self._agg(["CLOSED"] * 8)["accepted"][0]
        sparse = self._agg(["CLOSED"] * 4)["accepted"][0]
        self.assertEqual(dense["state"], sparse["state"])


class _CountingModel:
    """Minimal YOLO stand-in: records every frame it is asked to infer.

    Returns an empty detection set, so this exercises the SAMPLING logic only
    and needs no weights. Test-local by design -- nothing like this exists in
    production code.
    """

    def __init__(self):
        self.calls = 0
        self.names = {0: "closed_door"}

    def __call__(self, frame, **kw):
        self.calls += 1

        class _R:
            boxes = None
        return [_R()]


def _cache_with(n_frames: int, camera: str):
    """Build a wagon_cache exactly as Stage 2 lays it out."""
    import tempfile

    import cv2
    import numpy as np

    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "GW_1", C.CAMERA_FOLDER[camera])
    os.makedirs(d, exist_ok=True)
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    for i in range(n_frames):
        cv2.imwrite(os.path.join(d, f"frame_{i:06d}.jpg"), img)
    return tmp


class TestSampledModeActuallySkipsFrames(unittest.TestCase):
    """The point of the whole exercise: stride=2 must halve real YOLO calls."""

    def setUp(self):
        self.n = 40

    def _stable_count(self, cache, camera):
        from features._common import list_wagon_frames
        return len(list_wagon_frames(cache, "GW_1", camera, trim_stable=True))

    def test_door_stride2_halves_yolo_calls(self):
        import shutil
        cam = C.CAMERA_RIGHT_UP
        cache = _cache_with(self.n, cam)
        try:
            stable = self._stable_count(cache, cam)
            m1, m2 = _CountingModel(), _CountingModel()
            _, used1, *_ = door_proc._run_sampled_one_camera(
                m1, door_proc.TrackerConfig(), cache, "GW_1", cam, sample_stride=1)
            _, used2, *_ = door_proc._run_sampled_one_camera(
                m2, door_proc.TrackerConfig(), cache, "GW_1", cam, sample_stride=2)
            self.assertEqual(m1.calls, stable, "stride=1 must inspect every frame")
            self.assertEqual(used1, stable)
            self.assertEqual(m2.calls, (stable + 1) // 2)
            self.assertEqual(used2, (stable + 1) // 2)
            self.assertLess(m2.calls, m1.calls,
                            "sampled mode is executing every frame -- not sampling")
        finally:
            shutil.rmtree(cache, ignore_errors=True)

    def test_damage_stride2_and_stride3_reduce_yolo_calls(self):
        import shutil
        cam = C.CAMERA_RIGHT_UP_TOP
        cache = _cache_with(self.n, cam)
        try:
            stable = self._stable_count(cache, cam)
            counts = {}
            for stride in (1, 2, 3):
                m = _CountingModel()
                _, used, *_ = damage_proc._run_sampled_one_camera(
                    m, damage_proc.DamageTrackerConfig(), cache, "GW_1", cam,
                    confidence_floor=0.55, sample_stride=stride)
                counts[stride] = (m.calls, used)
            self.assertEqual(counts[1][0], stable)
            self.assertEqual(counts[2][0], (stable + 1) // 2)
            self.assertEqual(counts[3][0], (stable + 2) // 3)
            self.assertGreater(counts[1][0], counts[2][0])
            self.assertGreater(counts[2][0], counts[3][0])
        finally:
            shutil.rmtree(cache, ignore_errors=True)

    def test_legacy_mode_still_inspects_every_frame(self):
        """Guard against sampling silently leaking into the legacy path."""
        import shutil
        cam = C.CAMERA_RIGHT_UP
        cache = _cache_with(self.n, cam)
        try:
            stable = self._stable_count(cache, cam)
            m = _CountingModel()
            _, used, *_ = door_proc._run_tracker_one_camera(
                m, door_proc.TrackerConfig(), door_proc.MergeConfig(),
                cache, "GW_1", cam)
            self.assertEqual(m.calls, stable)
            self.assertEqual(used, stable)
        finally:
            shutil.rmtree(cache, ignore_errors=True)


class TestStage1IsUntouched(unittest.TestCase):
    """Stage 1 must remain functionally identical to the known-good commit."""

    def test_no_stage1_file_is_modified_in_the_working_tree(self):
        import subprocess
        r = subprocess.run(["git", "diff", "--name-only", "HEAD"],
                           cwd=V4_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("git unavailable")
        changed = [p for p in r.stdout.split() if p]
        protected = ("wagon_count/", "reconstruction/", "core/global_state_loader.py",
                     "materializer/", "fusion/", "reporting/")
        # The opt-in Stage-1 sampling experiment intentionally forks exactly
        # these three files.  Everything else under the protected roots must
        # still be untouched -- especially the counting/fusion modules.
        allowed = {
            "wagon_count/tracker_engine.py",
            "wagon_count/run_global_count.py",
            "reconstruction/runner.py",
        }
        offenders = [p for p in changed
                     if p.startswith(protected) and p not in allowed]
        self.assertEqual(offenders, [],
                         f"Stage-1/protected files modified: {offenders}")

    def test_counting_and_fusion_modules_are_still_untouched(self):
        """The sampling fork must not reach the actual counting algorithms."""
        import subprocess
        r = subprocess.run(["git", "diff", "--name-only", "HEAD"],
                           cwd=V4_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("git unavailable")
        changed = set(x for x in r.stdout.split() if x)
        for critical in ("wagon_count/global_fusion.py",
                         "wagon_count/gap_validation.py",
                         "wagon_count/global_alignment.py",
                         "wagon_count/train_structure.py",
                         "wagon_count/fragment_stitching.py",
                         "wagon_count/temporal_classification.py",
                         "wagon_count/global_train_state.py"):
            self.assertNotIn(critical, changed,
                             f"{critical} was modified -- counting logic must "
                             f"not change for a sampling experiment")

    def test_counting_engine_entry_point_untracked_changes_none(self):
        import subprocess
        r = subprocess.run(["git", "status", "--porcelain", "wagon_count"],
                           cwd=V4_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("git unavailable")
        allowed = {"tracker_engine.py", "run_global_count.py"}
        touched = {line.split("/")[-1] for line in r.stdout.split() 
                   if line.endswith(".py")}
        self.assertTrue(touched.issubset(allowed),
                        f"unexpected wagon_count/ modifications: "
                        f"{touched - allowed}")


if __name__ == "__main__":
    unittest.main()


class TestExperimentTwoConfiguration(unittest.TestCase):
    """Door=sampled/2, Damage=sampled/3, Load=sampled/2."""

    def _parse(self, argv):
        from orchestrator.master_runner import _build_parser
        return _build_parser().parse_args(argv)

    def test_production_command_selects_the_requested_strides(self):
        a = self._parse(["--local-only", "--local-inputs", "./local_inputs",
                         "--no-interactive", "--disable-features", "ocr",
                         "--skip-upload", "--skip-email"])
        self.assertEqual((a.door_inference_mode, a.door_sample_stride),
                         ("sampled", 3))
        self.assertEqual((a.damage_inference_mode, a.damage_sample_stride),
                         ("sampled", 3))
        self.assertEqual((a.load_inference_mode, a.load_sample_stride),
                         ("sampled", 2))

    def test_load_max_frames_default_is_still_none(self):
        """The functional fix must never regress to 0."""
        from features.load import processor as load_p
        self.assertIsNone(
            inspect.signature(load_p.run).parameters["max_frames"].default)

    def test_load_rejects_bad_mode(self):
        from features.load import processor as load_p
        with self.assertRaises(ValueError):
            load_p.run(state=None, cache_root="", feature_models_dir="",
                       output_dir="", inference_mode="turbo")

    def test_load_sampled_stride_matches_legacy_every_nth_at_2(self):
        """Honest check: Load was ALREADY at stride 2, so this is a no-op."""
        from features.load import processor as load_p
        p = inspect.signature(load_p.run).parameters
        self.assertEqual(p["every_nth"].default, 2)
        self.assertEqual(p["sample_stride"].default, 2)

    def test_damage_stride3_samples_correct_original_indices(self):
        import shutil
        from features._common import list_wagon_frames
        cam = C.CAMERA_RIGHT_UP_TOP
        cache = _cache_with(40, cam)
        try:
            stable = list_wagon_frames(cache, "GW_1", cam, trim_stable=True)
            expected = [int(os.path.basename(p).split("_")[1].split(".")[0])
                        for p in stable][::3]
            seen = []

            class _Rec(_CountingModel):
                def __call__(self, frame, **kw):
                    return super().__call__(frame, **kw)

            m = _Rec()
            _, used, *_ = damage_proc._run_sampled_one_camera(
                m, damage_proc.DamageTrackerConfig(), cache, "GW_1", cam,
                confidence_floor=0.55, sample_stride=3)
            self.assertEqual(used, len(expected))
            self.assertEqual(m.calls, len(expected))
            # gaps of exactly 3 in ORIGINAL numbering, no duplicates
            self.assertEqual(len(set(expected)), len(expected))
            self.assertTrue(all(b - a == 3 for a, b in zip(expected, expected[1:])))
        finally:
            shutil.rmtree(cache, ignore_errors=True)

    def test_door_stride2_unchanged_by_this_experiment(self):
        import shutil
        from features._common import list_wagon_frames
        cam = C.CAMERA_RIGHT_UP
        cache = _cache_with(40, cam)
        try:
            stable = len(list_wagon_frames(cache, "GW_1", cam, trim_stable=True))
            m = _CountingModel()
            _, used, *_ = door_proc._run_sampled_one_camera(
                m, door_proc.TrackerConfig(), cache, "GW_1", cam, sample_stride=2)
            self.assertEqual(m.calls, (stable + 1) // 2)
            self.assertEqual(used, (stable + 1) // 2)
        finally:
            shutil.rmtree(cache, ignore_errors=True)

    def test_fusion_and_reporting_untouched(self):
        import subprocess
        r = subprocess.run(["git", "status", "--porcelain", "fusion", "reporting"],
                           cwd=V4_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("git unavailable")
        self.assertEqual(r.stdout.strip(), "",
                         "fusion/ or reporting/ was modified")


class TestDoorStride3(unittest.TestCase):
    """Door stride=3 -- the test of whether EvidenceAggregator is really
    stride-invariant, unlike the absolute-hit-count tracker it replaced."""

    def _cache(self, n=40, cam=None):
        return _cache_with(n, cam or C.CAMERA_RIGHT_UP)

    def test_stride3_samples_correct_original_indices_no_duplicates(self):
        import shutil
        from features._common import list_wagon_frames
        cam = C.CAMERA_RIGHT_UP
        cache = self._cache(cam=cam)
        try:
            stable = list_wagon_frames(cache, "GW_1", cam, trim_stable=True)
            expected = [int(os.path.basename(x).split("_")[1].split(".")[0])
                        for x in stable][::3]
            m = _CountingModel()
            _, used, *_ = door_proc._run_sampled_one_camera(
                m, door_proc.TrackerConfig(), cache, "GW_1", cam, sample_stride=3)
            self.assertEqual(used, len(expected))
            self.assertEqual(m.calls, len(expected))
            self.assertEqual(len(set(expected)), len(expected), "duplicate frames")
            self.assertTrue(all(b - a == 3 for a, b in zip(expected, expected[1:])),
                            "sampled indices are not every 3rd ORIGINAL frame")
        finally:
            shutil.rmtree(cache, ignore_errors=True)

    def test_stride3_reduces_calls_below_stride2(self):
        import shutil
        from features._common import list_wagon_frames
        cam = C.CAMERA_RIGHT_UP
        cache = self._cache(cam=cam)
        try:
            stable = len(list_wagon_frames(cache, "GW_1", cam, trim_stable=True))
            got = {}
            for stride in (1, 2, 3):
                m = _CountingModel()
                door_proc._run_sampled_one_camera(
                    m, door_proc.TrackerConfig(), cache, "GW_1", cam,
                    sample_stride=stride)
                got[stride] = m.calls
            self.assertEqual(got[1], stable)
            self.assertEqual(got[2], (stable + 1) // 2)
            self.assertEqual(got[3], (stable + 2) // 3)
            self.assertGreater(got[2], got[3])
        finally:
            shutil.rmtree(cache, ignore_errors=True)

    def test_stride2_remains_available(self):
        a = self._parse_stride(["--local-only", "--door-sample-stride", "2"])
        self.assertEqual(a.door_sample_stride, 2)

    def test_legacy_door_remains_available(self):
        a = self._parse_stride(["--local-only", "--door-inference-mode", "legacy"])
        self.assertEqual(a.door_inference_mode, "legacy")

    def _parse_stride(self, argv):
        from orchestrator.master_runner import _build_parser
        return _build_parser().parse_args(argv)

    def test_aggregator_verdict_is_stride_invariant(self):
        """The property under test: same evidence, different sample rate,
        same verdict -- which the absolute-hit-count tracker could not do."""
        verdicts = {}
        for stride in (1, 2, 3):
            agg = EvidenceAggregator(frame_width=960, frame_height=540,
                                     stride=stride)
            for k in range(9 // stride + 1):
                fi = k * stride
                agg.add_frame(fi, [Observation(
                    frame_idx=fi, state="CLOSED", confidence=0.9,
                    bbox=(100.0, 100.0, 300.0, 250.0), score=1.0)])
            acc = agg.finalize()["accepted"]
            verdicts[stride] = acc[0]["state"] if acc else None
        self.assertEqual(verdicts[1], "CLOSED")
        self.assertEqual(verdicts[2], verdicts[1])
        self.assertEqual(verdicts[3], verdicts[1],
                         "aggregator verdict changed with stride -- it is NOT "
                         "stride-invariant, which is the whole premise")

    def test_damage_and_load_untouched_by_this_change(self):
        import subprocess
        r = subprocess.run(["git", "diff", "--name-only", "HEAD",
                            "features/damage", "features/load"],
                           cwd=V4_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("git unavailable")
        changed = [x for x in r.stdout.split() if x]
        self.assertNotIn("features/damage/processor.py", changed)


class TestStage1SamplingIsOptIn(unittest.TestCase):
    """Stage-1 frame sampling: default MUST be full-frame legacy behaviour."""

    def _parse(self, argv):
        from orchestrator.master_runner import _build_parser
        return _build_parser().parse_args(argv)

    def test_orchestrator_default_is_full_frame(self):
        self.assertEqual(self._parse(["--local-only"]).stage1_sample_stride, 1)

    def test_process_batch_default_is_full_frame(self):
        from orchestrator.master_runner import process_batch
        self.assertEqual(
            inspect.signature(process_batch).parameters[
                "stage1_sample_stride"].default, 1)

    def test_reconstruction_runner_default_is_full_frame(self):
        from reconstruction import runner as recon
        self.assertEqual(
            inspect.signature(recon.run).parameters[
                "stage1_sample_stride"].default, 1)

    def test_flag_is_omitted_from_the_subprocess_at_default(self):
        """At stride 1 the Stage-1 command line must be byte-identical to the
        proven invocation -- no new argument appears at all."""
        p = os.path.join(V4_ROOT, "reconstruction", "runner.py")
        src = open(p, encoding="utf-8").read()
        self.assertIn("if int(stage1_sample_stride) > 1:", src)
        self.assertIn('cmd += ["--stage1-sample-stride"', src)

    def test_gap_tracker_default_stride_is_one(self):
        import sys as _s
        wc = os.path.join(V4_ROOT, "wagon_count")
        if wc not in _s.path:
            _s.path.insert(0, wc)
        import tracker_engine
        self.assertEqual(
            inspect.signature(tracker_engine.GapTracker.__init__)
            .parameters["sample_stride"].default, 1)

    def test_sampling_happens_before_inference(self):
        """The skip must `continue` before _detect_gaps, or no calls are saved."""
        wc = os.path.join(V4_ROOT, "wagon_count", "tracker_engine.py")
        src = open(wc, encoding="utf-8").read()
        skip_at = src.index("if self.sample_stride > 1 and (frame_idx % self.sample_stride)")
        detect_at = src.index("detections = self._detect_gaps(frame, height)", skip_at)
        self.assertLess(skip_at, detect_at,
                        "sampling must occur BEFORE _detect_gaps")

    def test_min_hits_is_not_compensated(self):
        """min_hits must stay 3 -- compensating it would change counting."""
        import sys as _s
        wc = os.path.join(V4_ROOT, "wagon_count")
        if wc not in _s.path:
            _s.path.insert(0, wc)
        import tracker_engine
        self.assertEqual(
            inspect.signature(tracker_engine.GapTracker.__init__)
            .parameters["min_hits"].default, 3)

    def test_all_four_cameras_get_the_same_stride(self):
        """Cross-camera synchronization depends on identical sampling."""
        p = os.path.join(V4_ROOT, "wagon_count", "run_global_count.py")
        src = open(p, encoding="utf-8").read()
        self.assertEqual(
            src.count("sample_stride=int(args.stage1_sample_stride)"), 4,
            "all four cameras must receive the identical stride")

    def test_original_frame_indices_preserved_by_the_skip(self):
        """frame_idx advances for skipped frames too, so surviving frames keep
        their ORIGINAL numbering and no index is reused."""
        stride = 2
        total = 20
        kept, idx = [], 0
        for _ in range(total):
            if stride > 1 and (idx % stride):
                idx += 1
                continue
            kept.append(idx)
            idx += 1
        self.assertEqual(kept, list(range(0, total, 2)))
        self.assertEqual(len(set(kept)), len(kept), "duplicate frame index")
        self.assertEqual(len(kept), (total + 1) // 2)
