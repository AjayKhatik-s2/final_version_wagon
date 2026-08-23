"""LEFT_UP_TOP uses its own classifier, `ltop.pt`; nobody else's changes.

Both top cameras shared `top_classification.pt` until a classifier was trained
for LEFT_UP_TOP's own overhead view. The danger in a change like this is not that
the new model fails to load -- that is loud -- but that the WRONG model is
applied and nothing says so. A classifier run on the wrong camera still returns
confident ENGINE / WAGON / BRAKE_VAN labels, and those labels decide which
segments are excluded from wagon synchronization. So these tests pin the whole
mapping, both modes, and the two substitutions that were actually reachable:

  * replacing "the top model" and moving RIGHT_UP_TOP with it;
  * batch's `want == TOP_CLASSIFICATION_MODEL ? top : side` ternary, which went
    False the moment LEFT_UP_TOP's `want` became `ltop.pt` and would have handed
    an overhead camera the SIDE classifier.

No S3 credentials are needed: model-sync's requirement list and key construction
are asserted directly, and selection is asserted through the one existing
mapping both pipelines read.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "wagon_count"))

from core import constants as C                                   # noqa: E402
from core import config as CFG                                    # noqa: E402
from core import model_sync                                       # noqa: E402
import train_structure as ts                                      # noqa: E402


def _batch_module():
    """`wagon_count/run_global_count.py` loaded without running it."""
    path = os.path.join(ROOT, "wagon_count", "run_global_count.py")
    spec = importlib.util.spec_from_file_location("_rgc_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _src(*parts):
    return open(os.path.join(ROOT, *parts), encoding="utf-8").read()


# ---------------------------------------------------------------------------
# 1. The existing mapping, extended
# ---------------------------------------------------------------------------

class TestClassificationMapping(unittest.TestCase):

    def test_left_up_top_uses_ltop(self):
        self.assertEqual(
            ts.classification_model_for(C.CAMERA_LEFT_UP_TOP), "ltop.pt")

    def test_the_other_three_are_unchanged(self):
        self.assertEqual(ts.classification_model_for(C.CAMERA_RIGHT_UP),
                         "side_classification.pt")
        self.assertEqual(ts.classification_model_for(C.CAMERA_LEFT_UP),
                         "side_classification.pt")
        self.assertEqual(ts.classification_model_for(C.CAMERA_RIGHT_UP_TOP),
                         "top_classification.pt")

    def test_right_up_top_did_not_follow_left_up_top(self):
        """The trap: replacing 'the top model' would move BOTH cameras."""
        self.assertNotEqual(
            ts.classification_model_for(C.CAMERA_RIGHT_UP_TOP),
            ts.classification_model_for(C.CAMERA_LEFT_UP_TOP))
        self.assertEqual(ts.classification_model_for(C.CAMERA_RIGHT_UP_TOP),
                         ts.TOP_CLASSIFICATION_MODEL)

    def test_only_one_camera_moved_off_the_shared_top_model(self):
        sharing = [c for c in C.ALL_CAMERAS
                   if ts.classification_model_for(c) == ts.TOP_CLASSIFICATION_MODEL]
        self.assertEqual(sharing, [C.CAMERA_RIGHT_UP_TOP])

    def test_left_up_top_did_not_get_the_side_classifier(self):
        """The batch ternary's failure mode: an overhead camera on side weights."""
        self.assertNotEqual(
            ts.classification_model_for(C.CAMERA_LEFT_UP_TOP),
            ts.SIDE_CLASSIFICATION_MODEL)

    def test_every_camera_is_mapped_exactly_once(self):
        self.assertEqual(set(ts.CAMERA_CLASSIFICATION_MODEL), set(C.ALL_CAMERAS))

    def test_an_unknown_camera_raises_rather_than_defaulting(self):
        with self.assertRaises(KeyError) as cm:
            ts.classification_model_for("LEFT_DOWN_TOP")
        self.assertIn("LEFT_DOWN_TOP", str(cm.exception))

    def test_the_gap_models_are_untouched(self):
        """This task changed CLASSIFICATION only; gap detection is unaffected."""
        self.assertEqual(C.MODEL_TOP_GAP, "top_gap.pt")
        for name in ("right_up_wagon_gap.pt", "left_up_wagon_gap.pt",
                     "top_gap.pt", "side_classification.pt"):
            self.assertIn(name, C.RECON_MODEL_FILES)
        self.assertNotIn("ltop.pt", C.RECON_MODEL_FILES)


# ---------------------------------------------------------------------------
# 2. S3 resolution -- it must be a sync requirement, or it is never fetched
# ---------------------------------------------------------------------------

class TestLtopIsSyncedFromS3(unittest.TestCase):

    def test_ltop_is_a_synced_optional_reconstruction_model(self):
        self.assertIn("ltop.pt", C.RECON_OPTIONAL_MODEL_FILES)

    def test_top_classification_is_still_synced_too(self):
        self.assertIn("top_classification.pt", C.RECON_OPTIONAL_MODEL_FILES)

    def test_it_appears_in_the_requirement_list_with_the_recon_dir(self):
        reqs = [r for r in model_sync.required_models()
                if r.filename == "ltop.pt"]
        self.assertEqual(len(reqs), 1, "ltop.pt is not a sync requirement")
        self.assertEqual(reqs[0].category, "reconstruction")
        self.assertEqual(os.path.abspath(reqs[0].local_dir),
                         os.path.abspath(CFG.RECON_MODELS_DIR))

    def test_its_s3_key_uses_the_configured_bucket_and_prefix(self):
        req = next(r for r in model_sync.required_models()
                   if r.filename == "ltop.pt")
        expected = "/".join(p for p in (C.MODELS_S3_PREFIX, "ltop.pt") if p)
        self.assertEqual(req.s3_key, expected)
        self.assertEqual(req.s3_uri, f"s3://{C.MODELS_S3_BUCKET}/{expected}")

    def test_its_runtime_path_is_under_models_reconstruction(self):
        req = next(r for r in model_sync.required_models()
                   if r.filename == "ltop.pt")
        self.assertEqual(os.path.basename(req.local_path), "ltop.pt")
        self.assertEqual(
            os.path.basename(os.path.dirname(req.local_path)), "reconstruction")


# ---------------------------------------------------------------------------
# 3. Both modes select through the one mapping
# ---------------------------------------------------------------------------

class TestBothModesAgree(unittest.TestCase):

    def test_batch_resolves_a_path_for_every_distinct_model(self):
        """Name-keyed, so a third classifier cannot fall through to the side."""
        src = _src("wagon_count", "run_global_count.py")
        self.assertIn("cls_paths", src)
        self.assertNotIn("want == ts.TOP_CLASSIFICATION_MODEL\n", src)
        self.assertIn("set(ts.CAMERA_CLASSIFICATION_MODEL.values())", src)

    def test_batch_selects_by_name_not_by_a_top_or_side_guess(self):
        src = _src("wagon_count", "run_global_count.py")
        self.assertIn("path = cls_paths.get(want)", src)

    def test_the_sequential_runner_has_no_hardcoded_classifier_choice(self):
        src = _src("orchestrator", "camera_runner.py")
        self.assertNotIn('"top_classification.pt" if is_top', src)
        self.assertIn("classification_model_for", src)

    def test_both_modes_log_the_model_they_chose(self):
        for parts in (("orchestrator", "camera_runner.py"),
                      ("wagon_count", "run_global_count.py")):
            self.assertIn("[MODEL]", _src(*parts), f"{parts} does not log it")

    def test_the_batch_module_still_imports(self):
        mod = _batch_module()
        self.assertTrue(hasattr(mod, "main"))


# ---------------------------------------------------------------------------
# 4. A missing ltop.pt degrades; it never substitutes
# ---------------------------------------------------------------------------

class TestMissingLtopNeverSubstitutes(unittest.TestCase):

    def test_sequential_says_so_and_forbids_substitution(self):
        src = _src("orchestrator", "camera_runner.py")
        self.assertIn("NOT PRESENT", src)
        self.assertIn("never substituted", src)
        self.assertIn("core.model_sync", src,
                      "the message should say how the file is obtained")

    def test_batch_names_the_cameras_that_lose_classification(self):
        src = _src("wagon_count", "run_global_count.py")
        self.assertIn("will not be classified", src)
        self.assertIn("no other camera's classifier is substituted", src)

    def test_a_missing_classifier_does_not_fail_the_run(self):
        """Optional for the same reason top_classification.pt always was:
        RIGHT_UP alone is the counting authority."""
        src = _src("wagon_count", "run_global_count.py")
        self.assertIn("wagon count", src)
        self.assertIn("ltop.pt", str(C.RECON_OPTIONAL_MODEL_FILES))
        self.assertNotIn("ltop.pt", str(C.RECON_MODEL_FILES))

    def test_the_side_classifier_is_not_a_fallback_for_a_top_camera(self):
        src = _src("wagon_count", "run_global_count.py")
        self.assertNotIn("else side_cls_path", src,
                         "a top camera could still fall through to side weights")

    def test_an_absent_optional_model_does_not_make_the_report_fail(self):
        """Synced when present, never fatal when absent."""
        from core.model_sync import ModelReq, ModelStatus, SyncReport
        req_opt = ModelReq("reconstruction", "ltop.pt", "/tmp", optional=True)
        req_req = ModelReq("reconstruction", "top_gap.pt", "/tmp")
        rep = SyncReport(statuses=[
            ModelStatus(req=req_req, present=True),
            ModelStatus(req=req_opt, present=False, error="not in store"),
        ])
        self.assertTrue(rep.ok, "an absent OPTIONAL model failed the report")
        self.assertEqual(rep.missing, [])
        self.assertEqual([s.req.filename for s in rep.missing_optional],
                         ["ltop.pt"])
        self.assertTrue(any("[optional]" in ln for ln in rep.summary_lines()))

    def test_an_absent_required_model_still_fails(self):
        from core.model_sync import ModelReq, ModelStatus, SyncReport
        rep = SyncReport(statuses=[ModelStatus(
            req=ModelReq("reconstruction", "top_gap.pt", "/tmp"),
            present=False, error="not in store")])
        self.assertFalse(rep.ok)
        self.assertEqual([s.req.filename for s in rep.missing], ["top_gap.pt"])
