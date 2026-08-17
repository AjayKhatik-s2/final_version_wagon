"""Pin `orchestrator/camera_pipeline.py` to `wagon_count/run_global_count.py`.

The single-camera runner is a LITERAL EXTRACTION of the per-camera portion of
run_global_count. These tests fail if either side drifts -- if a threshold
moves, a step is dropped or reordered, or the recovery helper is "improved".

Model-free: they compare source structure, defaults and call ordering. The
detection-level equivalence run against batch_outputs/20260817_064608/ needs
weights and lives in the verification script, not here.
"""

from __future__ import annotations

import inspect
import os
import re
import unittest

from _engine_harness import V4_ROOT, WAGON_COUNT_DIR  # noqa: F401

from orchestrator import camera_pipeline as cp

RGC = os.path.join(WAGON_COUNT_DIR, "run_global_count.py")


def _rgc() -> str:
    with open(RGC, "r", encoding="utf-8") as f:
        return f.read()


def _norm(s: str) -> str:
    """Collapse whitespace so indentation differences don't matter."""
    return re.sub(r"\s+", " ", s).strip()


class TestDeriveWagonWindowIsLiteral(unittest.TestCase):
    """The one helper that had to be re-expressed rather than imported."""

    def _extract_rgc_body(self) -> str:
        src = _rgc()
        start = src.index("def _derive_wagon_window(")
        end = src.index("def _resolved_camera_offsets(", start)
        return src[start:end]

    def test_calls_the_same_functions_with_the_same_arguments(self):
        ours = _norm(inspect.getsource(cp.derive_wagon_window))
        theirs = _norm(self._extract_rgc_body())
        for token in (
            "if not classifications: return None",
            "segments = ga.build_global_wagons(",
            "list(master.gaps),",
            "master_total_frames=master.total_frames, master_fps=master.fps,",
            "initial_classifications=list(classifications),",
            "support_camera_ids=[])",
            "return ts.get_master_wagon_window(segments, verbose=verbose)",
            "except Exception: return None",
        ):
            with self.subTest(token=token):
                self.assertIn(_norm(token), theirs, "reference drifted")
                self.assertIn(_norm(token), ours, "extraction drifted")

    def test_no_extra_logic_was_added(self):
        """No new branch, threshold or heuristic beyond the original."""
        body = inspect.getsource(cp.derive_wagon_window)
        # Remove the docstring block outright -- prose like "returning None"
        # would otherwise be counted as executable statements.
        parts = body.split('"""')
        code = parts[0] + ("".join(parts[2:]) if len(parts) > 2 else "")
        code = "\n".join(l for l in code.splitlines()
                         if not l.strip().startswith("#"))
        self.assertEqual(code.count("if "), 1, "extra branch introduced")
        self.assertEqual(code.count("return "), 3, "extra return path")
        self.assertNotRegex(code, r"\b0\.\d+\b", "numeric threshold introduced")


class TestDefaultsMatchRunGlobalCount(unittest.TestCase):
    """Config defaults must equal run_global_count's argparse defaults."""

    def test_thresholds_identical(self):
        src = _rgc()
        c = cp.DEFAULT_CONFIG
        for flag, value in (
            ("--side-confidence", c.side_confidence),
            ("--top-confidence", c.top_confidence),
            ("--side-min-height-ratio", c.side_min_height_ratio),
            ("--top-min-height-ratio", c.top_min_height_ratio),
        ):
            with self.subTest(flag=flag):
                m = re.search(
                    re.escape(f'"{flag}"') + r",\s*type=float,\s*default=([0-9.]+)",
                    src)
                self.assertIsNotNone(m, f"{flag} not found in reference")
                self.assertEqual(float(m.group(1)), value)

    def test_classification_samples_identical(self):
        m = re.search(r'"--classification-samples",\s*type=int,\s*default=(\d+)',
                      _rgc())
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)),
                         cp.DEFAULT_CONFIG.classification_samples)

    def test_all_correctness_mechanisms_default_on(self):
        c = cp.DEFAULT_CONFIG
        for name in ("fragment_stitching", "gap_validation",
                     "temporal_classification", "wagon_recovery"):
            with self.subTest(mechanism=name):
                self.assertTrue(getattr(c, name),
                                f"{name} must default ON, as production does")


class TestStepOrdering(unittest.TestCase):
    """Steps must appear in run_global_count's order in both paths."""

    def _positions(self, fn, tokens):
        src = inspect.getsource(fn)
        return [src.index(t) for t in tokens]

    def test_shared_prefix_is_track_stitch_validate(self):
        pos = self._positions(cp._track_stitch_validate, [
            "GapTracker(", "tracker.process_video(",
            "fstitch.reassemble_fragments(", "gval.validate_gap_events(",
            "gval.renumber_gap_events(",
        ])
        self.assertEqual(pos, sorted(pos),
                         "STEP 1 -> 1a -> 1b order violated")

    def test_stitching_precedes_validation(self):
        """run_global_count is explicit: reassembly runs BEFORE validation."""
        src = inspect.getsource(cp._track_stitch_validate)
        self.assertLess(src.index("reassemble_fragments("),
                        src.index("validate_gap_events("))

    def test_master_path_order(self):
        pos = self._positions(cp.run_master_camera, [
            "_track_stitch_validate(", "_classify_master(",
            "apply_temporal_classification(", "derive_wagon_window(",
            "recover_wagon_active_candidates(", "renumber_gap_events(",
        ])
        self.assertEqual(pos, sorted(pos), "master step order violated")

    def test_master_reclassifies_after_recovery(self):
        """Production classifies TWICE when recovery fires."""
        src = inspect.getsource(cp.run_master_camera)
        after = src[src.index("if recovery.recovered:"):]
        self.assertIn("_classify_master(", after)
        self.assertIn("apply_temporal_classification(", after)
        self.assertIn("reclassified = True", after)

    def test_support_path_has_no_recovery(self):
        src = inspect.getsource(cp.run_support_camera)
        self.assertNotIn("recover_wagon_active_candidates", src)
        self.assertNotIn("derive_wagon_window", src)

    def test_support_path_builds_a_wagon_region(self):
        src = inspect.getsource(cp.run_support_camera)
        self.assertIn("ts.build_local_wagon_region(", src)

    def test_neither_path_runs_fusion(self):
        for fn in (cp.run_master_camera, cp.run_support_camera):
            src = inspect.getsource(fn)
            with self.subTest(fn=fn.__name__):
                self.assertNotIn("assemble_global_train_state", src)
                self.assertNotIn("global_fusion", src)


class TestNoGlobalIdsLeak(unittest.TestCase):
    def test_segments_are_camera_local_only(self):
        src = inspect.getsource(cp)
        self.assertNotIn('"GW_', src)
        self.assertNotIn("f'GW_", src)
        self.assertIn("local_segment_id(", src)

    def test_segment_ids_and_ordering(self):
        class _T:
            fps, total_frames = 15.0, 300
            gaps: list = []
        segs = cp._segments_to_local("LEFT_UP", _T())
        self.assertTrue(all(s.local_id.startswith("L_LEFT_UP_") for s in segs))
        self.assertEqual([s.index for s in segs],
                         list(range(1, len(segs) + 1)))


class TestProtectedFilesUntouched(unittest.TestCase):
    def test_wagon_count_and_reconstruction_unmodified(self):
        import subprocess
        r = subprocess.run(["git", "status", "--porcelain",
                            "wagon_count", "reconstruction", "fusion",
                            "reporting", "materializer"],
                           cwd=V4_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("git unavailable")
        self.assertEqual(r.stdout.strip(), "",
                         "sequential mode must not modify protected packages")


if __name__ == "__main__":
    unittest.main()
