"""Stage-3 sampling: identical in sequential and batch, and CLI-controllable.

Production defaults are Door sampled/3, Damage sampled/3, Load sampled/2. What
these tests guard is not the numbers alone but that there is ONE source for them
and that both pipelines reach it.

The defect this suite exists for: the values lived in THREE places -- argparse
defaults, `process_batch`'s signature, and hardcoded literals in
`camera_runner._feature_plan` / `global_assembler._FEATURE_ORDER`. Nothing tied
them together, and `historical_runner` passed `inference_opts` only to the BATCH
branch. So `--door-sample-stride 5 --mode sequential` printed the requested
stride and then sampled at the hardcoded one, and when the literals were changed
on one side the two modes silently disagreed for the same command.

Now `core.constants.STAGE3_*` is the single source, `camera_runner.stage3_extras`
the single builder, and `inference_opts` is threaded into the sequential path.
"""

from __future__ import annotations

import inspect
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import constants as C                                   # noqa: E402
from orchestrator import camera_runner as CR                      # noqa: E402
from orchestrator import historical_runner as HR                  # noqa: E402
from orchestrator import master_runner as MR                      # noqa: E402
from orchestrator.global_assembler import stage3_order            # noqa: E402


class TestTheProductionDefaults(unittest.TestCase):

    def test_the_values_are_door_3_damage_3_load_2(self):
        e = CR.stage3_extras()
        self.assertEqual((e["door"]["inference_mode"],
                          e["door"]["sample_stride"]), ("sampled", 3))
        self.assertEqual((e["damage"]["inference_mode"],
                          e["damage"]["sample_stride"]), ("sampled", 3))
        self.assertEqual((e["load"]["inference_mode"],
                          e["load"]["sample_stride"]), ("sampled", 2))

    def test_load_carries_the_stride_on_every_nth_too(self):
        """Load samples even in legacy mode; `sample_stride` alone leaves it at
        its own default of 2 while door and damage obey."""
        e = CR.stage3_extras()
        self.assertEqual(e["load"]["every_nth"], e["load"]["sample_stride"])

    def test_ocr_takes_no_stride_arguments(self):
        self.assertEqual(CR.stage3_extras()["ocr"], {})

    def test_the_defaults_come_from_one_place(self):
        self.assertEqual(C.STAGE3_DOOR_STRIDE, 3)
        self.assertEqual(C.STAGE3_DAMAGE_STRIDE, 3)
        self.assertEqual(C.STAGE3_LOAD_STRIDE, 2)
        for m in (C.STAGE3_DOOR_MODE, C.STAGE3_DAMAGE_MODE,
                  C.STAGE3_LOAD_MODE):
            self.assertEqual(m, "sampled")


class TestBothModesSampleIdentically(unittest.TestCase):

    def test_the_per_camera_plan_matches_the_assembly_order(self):
        shared = CR.stage3_extras()
        for name, extra in stage3_order():
            with self.subTest(feature=name):
                self.assertEqual(extra, shared[name])

    def test_every_camera_gets_the_same_stride_for_a_given_feature(self):
        per_feature = {}
        for cam in C.ALL_CAMERAS:
            for name, _mod, extra in CR._feature_plan(
                    cam, {"door", "damage", "load"}):
                per_feature.setdefault(name, set()).add(
                    (extra["inference_mode"], extra["sample_stride"]))
        for name, seen in per_feature.items():
            self.assertEqual(len(seen), 1,
                             f"{name} sampled differently per camera: {seen}")

    def test_batch_signature_and_sequential_builder_agree(self):
        p = inspect.signature(MR.process_batch).parameters
        e = CR.stage3_extras()
        for feat in ("door", "damage", "load"):
            self.assertEqual(p[f"{feat}_inference_mode"].default,
                             e[feat]["inference_mode"])
            self.assertEqual(p[f"{feat}_sample_stride"].default,
                             e[feat]["sample_stride"])

    def test_an_override_reaches_both_the_plan_and_the_assembly(self):
        opts = {"door_sample_stride": 5, "damage_sample_stride": 7,
                "load_sample_stride": 4}
        want = {"door": 5, "damage": 7, "load": 4}
        for name, extra in stage3_order(opts):
            if name in want:
                self.assertEqual(extra["sample_stride"], want[name])
        for cam in C.ALL_CAMERAS:
            for name, _m, extra in CR._feature_plan(
                    cam, set(want), inference_opts=opts):
                self.assertEqual(extra["sample_stride"], want[name])


class TestTheCliReachesSequential(unittest.TestCase):
    """The defect: only the batch branch received `inference_opts`."""

    def test_process_batch_sequential_accepts_the_options(self):
        self.assertIn("inference_opts",
                      inspect.signature(HR.process_batch_sequential).parameters)

    def test_run_camera_accepts_the_options(self):
        self.assertIn("inference_opts",
                      inspect.signature(CR.run_camera).parameters)

    def test_assemble_accepts_the_options(self):
        from orchestrator import global_assembler as GA
        self.assertIn("inference_opts",
                      inspect.signature(GA.assemble).parameters)

    def test_the_sequential_branch_actually_passes_them(self):
        """The CALL site, not the def -- splitting on the first occurrence finds
        the definition and passes for the wrong reason."""
        src = open(os.path.join(ROOT, "orchestrator", "historical_runner.py"),
                   encoding="utf-8").read()
        calls = src.split("asm = process_batch_sequential(")
        self.assertGreater(len(calls), 1, "no call site found")
        self.assertIn("inference_opts=inference_opts", calls[1][:900],
                      "the sequential branch drops the CLI settings")

    def test_the_per_camera_call_passes_them_too(self):
        src = open(os.path.join(ROOT, "orchestrator", "historical_runner.py"),
                   encoding="utf-8").read()
        calls = src.split("camera_runner.run_camera(")
        self.assertGreater(len(calls), 1)
        self.assertIn("inference_opts=inference_opts", calls[-1][:600])

    def test_the_assemble_call_passes_them_too(self):
        src = open(os.path.join(ROOT, "orchestrator", "historical_runner.py"),
                   encoding="utf-8").read()
        calls = src.split("global_assembler.assemble(")
        self.assertGreater(len(calls), 1)
        self.assertIn("inference_opts=inference_opts", calls[-1][:900])

    def test_all_six_options_plus_legacy_exist_on_the_cli(self):
        src = open(os.path.join(ROOT, "orchestrator", "master_runner.py"),
                   encoding="utf-8").read()
        for opt in ("--door-inference-mode", "--door-sample-stride",
                    "--damage-inference-mode", "--damage-sample-stride",
                    "--load-inference-mode", "--load-sample-stride",
                    "--legacy-inference"):
            self.assertIn(f'"{opt}"', src, f"{opt} is missing")


class TestLegacyIsStillReachable(unittest.TestCase):

    def test_legacy_restores_every_frame_door_and_damage(self):
        opts = {"door_inference_mode": "legacy",
                "damage_inference_mode": "legacy"}
        e = CR.stage3_extras(opts)
        self.assertEqual(e["door"]["inference_mode"], "legacy")
        self.assertEqual(e["damage"]["inference_mode"], "legacy")

    def test_legacy_inference_is_a_shorthand_for_all_three(self):
        src = open(os.path.join(ROOT, "orchestrator", "master_runner.py"),
                   encoding="utf-8").read()
        block = src.split("args.legacy_inference", 1)[1][:400]
        for feat in ("door", "damage", "load"):
            self.assertIn(feat, block, f"--legacy-inference misses {feat}")

    def test_the_processors_still_accept_both_modes(self):
        from features.door import processor as dp
        from features.damage import processor as dmg
        from features.load import processor as lp
        for mod in (dp, dmg, lp):
            p = inspect.signature(mod.run).parameters
            self.assertIn("inference_mode", p)
            self.assertIn("sample_stride", p)


class TestSamplingArithmetic(unittest.TestCase):
    """What a stride actually does to the frames fed to a processor."""

    @staticmethod
    def _sampled(paths, every_nth):
        """The subsample `features._common.iter_wagon_frames` performs."""
        return paths[::every_nth] if every_nth > 1 else list(paths)

    def test_stride_3_takes_every_third_frame(self):
        frames = list(range(30))
        got = self._sampled(frames, 3)
        self.assertEqual(got, [0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
        self.assertEqual(len(got), 10)

    def test_stride_2_takes_every_second_frame(self):
        frames = list(range(30))
        self.assertEqual(len(self._sampled(frames, 2)), 15)

    def test_no_frame_is_sampled_twice(self):
        for stride in (1, 2, 3, 5):
            got = self._sampled(list(range(97)), stride)
            self.assertEqual(len(got), len(set(got)),
                             f"stride {stride} produced a duplicate frame")

    def test_the_call_reduction_is_what_the_stride_promises(self):
        n = 300
        self.assertAlmostEqual(len(self._sampled(list(range(n)), 3)) / n,
                               1 / 3, places=2)
        self.assertAlmostEqual(len(self._sampled(list(range(n)), 2)) / n,
                               1 / 2, places=2)

    def test_stride_1_is_every_frame(self):
        frames = list(range(50))
        self.assertEqual(self._sampled(frames, 1), frames)

    def test_a_bad_stride_falls_back_rather_than_dividing_by_zero(self):
        for bad in (0, -1, None, "x"):
            e = CR.stage3_extras({"door_sample_stride": bad})
            self.assertGreaterEqual(e["door"]["sample_stride"], 1)


class TestReportingIsPreserved(unittest.TestCase):
    """YOLO-call reporting and timings must survive the change."""

    def test_the_stage3_line_still_reports_mode_and_stride(self):
        src = open(os.path.join(ROOT, "orchestrator", "master_runner.py"),
                   encoding="utf-8").read()
        self.assertIn("Stage-3 inference: door=", src)
        self.assertIn("stride=", src)

    def test_timings_are_still_recorded(self):
        for parts in (("orchestrator", "master_runner.py"),
                      ("orchestrator", "global_assembler.py")):
            src = open(os.path.join(ROOT, *parts), encoding="utf-8").read()
            self.assertIn("timings", src, f"{parts} stopped recording timings")
