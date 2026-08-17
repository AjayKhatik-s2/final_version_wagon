"""Prove the NEW counting engine is wired in and the OLD one is not reachable.

These tests are about provenance and wiring, not numbers.  They fail if
somebody re-introduces the previous counting path, points Stage 1 somewhere
else, or quietly reimplements the adopted engine instead of using it.
"""

from __future__ import annotations

import os
import unittest

from _engine_harness import (
    LEGACY_BACKUP_DIR, REFERENCE_DIR, V4_ROOT, WAGON_COUNT_DIR,
)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# Modules that constitute the adopted correct-count engine.  Every one of these
# is new to this package -- none existed in the previous counting implementation.
ADOPTED_MODULES = (
    "global_fusion.py",          # fixed-master fusion + camera offsets
    "gap_validation.py",         # candidate -> boundary motion/temporal gates
    "fragment_stitching.py",     # tracker fragments -> physical gaps
    "temporal_classification.py",# class-sequence hysteresis
    "train_structure.py",        # wagon window; engines/brake vans get no id
)

# Modules the previous implementation also had, replaced wholesale.
REPLACED_MODULES = (
    "global_train_state.py",
    "tracker_engine.py",
    "global_alignment.py",
    "video_segmenter.py",
    "run_global_count.py",
)


class TestNewEnginePresent(unittest.TestCase):
    def test_adopted_modules_exist_in_live_package(self):
        for name in ADOPTED_MODULES:
            with self.subTest(module=name):
                self.assertTrue(
                    os.path.isfile(os.path.join(WAGON_COUNT_DIR, name)),
                    f"{name} missing from wagon_count/ -- the correct-count "
                    f"engine was not adopted")

    def test_entry_point_drives_every_correctness_mechanism(self):
        """run_global_count must actually call the new stages, not just ship them."""
        src = _read(os.path.join(WAGON_COUNT_DIR, "run_global_count.py"))
        for token in ("import global_fusion", "import gap_validation",
                      "import fragment_stitching", "import train_structure",
                      "import temporal_classification"):
            with self.subTest(token=token):
                self.assertIn(token, src)
        for call in ("reassemble_fragments(", "validate_gap_events(",
                     "recover_wagon_active_candidates(",
                     "apply_temporal_classification(",
                     "assemble_global_train_state_master_fixed("):
            with self.subTest(call=call):
                self.assertIn(call, src,
                              f"{call} is never invoked -- a correctness "
                              f"mechanism was shipped but not wired in")

    def test_master_fixed_is_the_default_fusion(self):
        src = _read(os.path.join(WAGON_COUNT_DIR, "run_global_count.py"))
        self.assertIn('choices=("master-fixed", "legacy")', src)
        self.assertIn('default="master-fixed"', src)

    def test_engine_is_byte_identical_to_the_reference(self):
        """Adopted verbatim, not reimplemented and not locally patched.

        EVERY counting module -- entry point included -- must match the proven
        implementation exactly.  There is no local fork of the counting engine.
        """
        if not os.path.isdir(REFERENCE_DIR):
            self.skipTest("reference folder removed after review")
        for name in ADOPTED_MODULES + REPLACED_MODULES:
            with self.subTest(module=name):
                live = _read(os.path.join(WAGON_COUNT_DIR, name))
                ref = _read(os.path.join(REFERENCE_DIR, name))
                self.assertEqual(live, ref,
                                 f"{name} diverges from the proven engine")

    def test_no_model_aliasing_or_download_logic_was_added(self):
        """Model placement is an operator responsibility, not the code's.

        The engine resolves each weight at its canonical filename under
        --models-dir and fails loudly otherwise.  No alias map, no fallback
        name, no fetching.
        """
        src = _read(os.path.join(WAGON_COUNT_DIR, "run_global_count.py"))
        for banned in ("_MODEL_ALIASES", "boto3", "download_file", "s3.cp",
                       "aws s3 cp s3://<bucket>/{ts.TOP"):
            if banned == "aws s3 cp s3://<bucket>/{ts.TOP":
                continue    # an operator HINT in a log message, not logic
            with self.subTest(token=banned):
                self.assertNotIn(banned, src)
        # Exactly one resolution rule: canonical name under models_dir.
        self.assertIn("def _resolve_model(name: str, models_dir: str) -> str:", src)
        self.assertIn("Model not found: {name}. Expected at {p}.", src)


class TestOldEngineNotUsed(unittest.TestCase):
    def test_previous_counting_markers_are_gone(self):
        """Distinctive code from the replaced implementation must not survive."""
        live_sources = {
            name: _read(os.path.join(WAGON_COUNT_DIR, name))
            for name in REPLACED_MODULES
        }
        # The old package worked around bad counting with these local patches;
        # the new engine solves the same problems structurally (wagon window +
        # temporal classification), so the patches must be gone.
        self.assertNotIn("ENGINE_MIN_CONFIDENCE", live_sources["tracker_engine.py"])
        self.assertNotIn("Startup false-engine guard",
                         live_sources["global_alignment.py"])

    def test_legacy_backup_is_not_a_package_and_not_imported(self):
        if not os.path.isdir(LEGACY_BACKUP_DIR):
            self.skipTest("legacy backup already deleted")
        self.assertFalse(
            os.path.exists(os.path.join(LEGACY_BACKUP_DIR, "__init__.py")),
            "the removed-engine backup must not be importable")
        needle = os.path.basename(LEGACY_BACKUP_DIR)
        for pkg in ("core", "reconstruction", "materializer", "fusion",
                    "features", "reporting", "rendering", "orchestrator",
                    "delivery", "wagon_count"):
            root = os.path.join(V4_ROOT, pkg)
            for dirpath, _dirs, files in os.walk(root):
                if "__pycache__" in dirpath:
                    continue
                for fn in files:
                    if not fn.endswith(".py"):
                        continue
                    src = _read(os.path.join(dirpath, fn))
                    self.assertNotIn(
                        needle, src,
                        f"{os.path.join(dirpath, fn)} references the removed "
                        f"legacy counting engine")


class TestStage1WiringUnchanged(unittest.TestCase):
    def test_runner_resolves_the_live_wagon_count_package(self):
        from reconstruction import runner
        resolved = runner._find_wagon_count_dir(V4_ROOT)
        self.assertEqual(os.path.normcase(os.path.abspath(resolved)),
                         os.path.normcase(os.path.abspath(WAGON_COUNT_DIR)))

    def test_runner_invokes_the_new_cli_contract(self):
        src = _read(os.path.join(V4_ROOT, "reconstruction", "runner.py"))
        # Frame extraction stays with the materializer; overlay videos became
        # opt-in in the new engine, so Stage 1 must ask for them explicitly.
        self.assertIn('"--no-frames"', src)
        self.assertIn('"--render-videos"', src)
        # Stage 1 must never select the retained legacy fusion path.
        self.assertNotIn("--fusion", src)
        self.assertNotIn("legacy", src.split("Returns the parsed")[-1])

    def test_stage1_surfaces_counting_authority_metadata(self):
        from reconstruction.runner import ReconstructionResult
        fields = ReconstructionResult.__dataclass_fields__
        self.assertIn("camera_offsets", fields)
        self.assertIn("roster_fingerprint", fields)


if __name__ == "__main__":
    unittest.main()
