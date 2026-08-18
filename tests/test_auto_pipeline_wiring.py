"""The auto pipeline's wiring: source selection, batch discovery, extraction.

`--auto` was previously DEAD in this package: `run_auto` imported
`train_batch_manager` from the repo's PARENT directory, a file that does not
exist here, so every `--auto` / `--once` / `--batch` invocation logged
"continuous polling unavailable" and returned 3.  These tests pin the pieces
that make it real:

* the batch manager is importable from inside the package and exposes the exact
  call contract `run_auto` uses;
* `--source raw` vs `trimmed` decides whether this process also PRODUCES its
  trimmed clips, and nothing else;
* discovery is anchored to the operational day (05:00 IST), so a restart at any
  hour still sees today's trains but can never reach into months of archive;
* the extraction manager's lifecycle is start/stop-safe and never runs a sweep
  it was not asked for.

No test here touches S3, ultralytics, or ffmpeg: the extraction sweep is
monkeypatched, so what is asserted is the orchestration, not the CV.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config as CFG                                     # noqa: E402
from core import constants as C                                   # noqa: E402
from core.pipeline_source import PipelineSource                    # noqa: E402
from orchestrator import train_batch_manager as TBM                # noqa: E402


class TestPipelineSource(unittest.TestCase):

    def setUp(self):
        self._saved = os.environ.pop("WAGONEYE_PIPELINE_SOURCE", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("WAGONEYE_PIPELINE_SOURCE", None)
        else:
            os.environ["WAGONEYE_PIPELINE_SOURCE"] = self._saved

    def test_default_is_trimmed_the_pure_consumer(self):
        self.assertIs(PipelineSource.resolve(), PipelineSource.TRIMMED)
        self.assertFalse(PipelineSource.TRIMMED.requires_extraction)

    def test_raw_requires_extraction(self):
        self.assertIs(PipelineSource.resolve("raw"), PipelineSource.RAW)
        self.assertTrue(PipelineSource.RAW.requires_extraction)

    def test_env_selects_raw(self):
        os.environ["WAGONEYE_PIPELINE_SOURCE"] = "raw"
        self.assertIs(PipelineSource.resolve(), PipelineSource.RAW)

    def test_explicit_value_beats_env(self):
        os.environ["WAGONEYE_PIPELINE_SOURCE"] = "raw"
        self.assertIs(PipelineSource.resolve("trimmed"), PipelineSource.TRIMMED)

    def test_typo_falls_back_to_the_safe_consumer_default(self):
        """A typo must never take the service down, nor silently start
        extracting on a box that has no raw bucket access."""
        os.environ["WAGONEYE_PIPELINE_SOURCE"] = "rawww"
        self.assertIs(PipelineSource.resolve(), PipelineSource.TRIMMED)


class TestBatchManagerContract(unittest.TestCase):
    """`run_auto` calls exactly these five names -- pin them."""

    def test_importable_from_inside_the_package(self):
        from orchestrator import train_batch_manager as TBM
        self.assertTrue(hasattr(TBM, "poll_for_batches"))
        self.assertTrue(hasattr(TBM, "select_runnable_batch"))
        self.assertTrue(hasattr(TBM, "load_batch_state"))
        self.assertTrue(hasattr(TBM, "save_batch_state"))
        self.assertIsInstance(TBM.DEFAULT_BATCH_TOLERANCE_SEC, int)

    def test_master_runner_no_longer_imports_from_the_repo_parent(self):
        """The old dead path: `from train_batch_manager import ...` resolved
        against the PARENT of the repo root, which has no such module."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(here, "orchestrator", "master_runner.py"),
                   encoding="utf-8").read()
        self.assertNotIn("sys.path.insert(0, os.path.dirname(_REPO_ROOT))", src)
        self.assertIn("from orchestrator import train_batch_manager", src)

    def test_camera_resolution_is_shared_with_constants(self):
        from orchestrator import train_batch_manager as TBM
        self.assertEqual(TBM._camera_for_key(
            "camera_CCTV_HZBN_DHN_5_RIGHT_TOP/2026-08-19/a.mp4"),
            C.CAMERA_RIGHT_UP_TOP)


class TestOperationalDayAnchor(unittest.TestCase):

    def setUp(self):
        self._saved = os.environ.pop("WAGONEYE_PROCESSOR_START_UTC", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("WAGONEYE_PROCESSOR_START_UTC", None)
        else:
            os.environ["WAGONEYE_PROCESSOR_START_UTC"] = self._saved

    def test_anchor_is_0500_ist_on_the_same_day_after_0500(self):
        now = datetime(2026, 8, 19, 12, 0, tzinfo=CFG.IST)
        start = CFG.operational_day_start_utc(now.astimezone(timezone.utc))
        self.assertEqual(start.astimezone(CFG.IST).hour, 5)
        self.assertEqual(start.astimezone(CFG.IST).date(),
                         datetime(2026, 8, 19).date())

    def test_before_0500_ist_rolls_back_to_yesterdays_anchor(self):
        now = datetime(2026, 8, 19, 3, 0, tzinfo=CFG.IST)
        start = CFG.operational_day_start_utc(now.astimezone(timezone.utc))
        self.assertEqual(start.astimezone(CFG.IST).date(),
                         datetime(2026, 8, 18).date())

    def test_cutoff_is_bounded_to_one_day(self):
        """The property that stops a restart queueing months of archive."""
        now = datetime.now(timezone.utc)
        cutoff = CFG.discovery_cutoff_utc(now)
        self.assertLess(now - cutoff, timedelta(days=1, hours=1))

    def test_processor_start_override_can_only_raise_the_anchor(self):
        now = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
        anchor = CFG.operational_day_start_utc(now)
        # Earlier than the anchor -> ignored.
        os.environ["WAGONEYE_PROCESSOR_START_UTC"] = "2020-01-01T00:00:00+00:00"
        self.assertEqual(CFG.discovery_cutoff_utc(now), anchor)
        # Later than the anchor -> honoured.
        later = (anchor + timedelta(hours=2)).isoformat()
        os.environ["WAGONEYE_PROCESSOR_START_UTC"] = later
        self.assertGreater(CFG.discovery_cutoff_utc(now), anchor)

    def test_malformed_override_is_ignored(self):
        now = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
        os.environ["WAGONEYE_PROCESSOR_START_UTC"] = "not-a-timestamp"
        self.assertEqual(CFG.discovery_cutoff_utc(now),
                         CFG.operational_day_start_utc(now))


class TestConfigValidation(unittest.TestCase):

    def test_trimmed_source_does_not_demand_extraction_models(self):
        """A pure consumer has no use for the extraction classifiers; demanding
        them blocked deployments that consume already-trimmed clips."""
        saved = os.environ.pop("WAGONEYE_PIPELINE_SOURCE", None)
        try:
            import importlib
            importlib.reload(CFG)
            errors = CFG.validate_config(mode="auto", skip_email=True)
            self.assertFalse([e for e in errors if "extraction" in e.lower()],
                             errors)
        finally:
            if saved is not None:
                os.environ["WAGONEYE_PIPELINE_SOURCE"] = saved
            import importlib
            importlib.reload(CFG)

    def test_local_mode_needs_no_s3_configuration(self):
        errors = CFG.validate_config(mode="local")
        self.assertFalse([e for e in errors if "S3" in e or "EMAIL" in e], errors)

    def test_startup_summary_redacts_recipients(self):
        text = CFG.startup_summary(mode="auto")
        self.assertIn("redacted", text)
        for addr in (C.EMAIL_RECEIVER or []):
            self.assertNotIn(addr, text)

    def test_startup_summary_states_the_delivery_target(self):
        text = CFG.startup_summary(mode="auto")
        self.assertIn("dashboard_ingest", text)
        self.assertIn("dashboard_version", text)


class TestExtractionManagerLifecycle(unittest.TestCase):
    """The manager's contract, with the CV sweep stubbed out."""

    def _manager_with_stub(self, record):
        from orchestrator.extraction_manager import ExtractionManager
        from train_extraction import run_extraction_service as RES

        original = RES.sweep_camera

        def fake_sweep(camera, **kw):
            record.append(camera)
            return {"listed": 1, "new": 1, "trains": 1, "errors": 0}

        RES.sweep_camera = fake_sweep
        return ExtractionManager(cameras=list(C.ALL_CAMERAS),
                                 poll_interval=1), RES, original

    def test_run_once_sweeps_every_camera_exactly_once(self):
        record = []
        mgr, RES, original = self._manager_with_stub(record)
        try:
            counts = mgr.run_once()
        finally:
            RES.sweep_camera = original
        self.assertEqual(sorted(record), sorted(C.ALL_CAMERAS))
        self.assertEqual(counts["trains"], 4)
        self.assertEqual(counts["errors"], 0)

    def test_a_crashing_camera_does_not_stop_the_others(self):
        from orchestrator.extraction_manager import ExtractionManager
        from train_extraction import run_extraction_service as RES
        original = RES.sweep_camera
        seen = []

        def flaky(camera, **kw):
            seen.append(camera)
            if camera == C.CAMERA_LEFT_UP:
                raise RuntimeError("model missing")
            return {"listed": 1, "new": 0, "trains": 0, "errors": 0}

        RES.sweep_camera = flaky
        try:
            counts = ExtractionManager(cameras=list(C.ALL_CAMERAS)).run_once()
        finally:
            RES.sweep_camera = original
        self.assertEqual(sorted(seen), sorted(C.ALL_CAMERAS))
        self.assertEqual(counts["errors"], 1)

    def test_not_running_until_started(self):
        from orchestrator.extraction_manager import ExtractionManager
        mgr = ExtractionManager()
        self.assertFalse(mgr.is_running())

    def test_start_then_stop_is_clean_and_idempotent(self):
        record = []
        mgr, RES, original = self._manager_with_stub(record)
        try:
            mgr.start()
            mgr.start()               # idempotent
            self.assertTrue(mgr.is_running())
            mgr.stop(timeout=5)
            self.assertFalse(mgr.is_running())
        finally:
            RES.sweep_camera = original

    def test_stop_before_start_does_not_raise(self):
        from orchestrator.extraction_manager import ExtractionManager
        ExtractionManager().stop(timeout=1)


class TestExtractionBuckets(unittest.TestCase):
    """The producer's output must BE the consumer's input, or nothing flows."""

    def test_trimmed_bucket_is_the_consumer_input_by_default(self):
        self.assertEqual(C.S3_INPUT_BUCKET, C.S3_TRIMMED_VIDEO_BUCKET)

    def test_v4_bucket_names_are_the_defaults(self):
        self.assertEqual(C.S3_RAW_VIDEO_BUCKET, "biro-wagon-raw-video-copy")
        self.assertEqual(C.S3_TRIMMED_VIDEO_BUCKET,
                         "biro-wagon-pre-processed-video-copy")
        self.assertEqual(C.S3_DETECTED_VIDEO_BUCKET,
                         "biro-wagon-processed-video-copy")
        self.assertEqual(C.S3_OUTPUT_BUCKET, "biro-wagon-report-biro-copy")
        self.assertEqual(C.S3_COMBINED_REPORT_BUCKET, "biro-combined-report-copy")
        self.assertEqual(C.S3_REGION, "ap-south-1")

    def test_input_prefixes_are_the_four_camera_folders(self):
        self.assertEqual(C.S3_INPUT_PREFIXES,
                         [C.CAMERA_S3_FOLDER[c] for c in C.ALL_CAMERAS])

    def test_every_camera_folder_round_trips(self):
        for cam in C.ALL_CAMERAS:
            folder = C.CAMERA_S3_FOLDER[cam]
            self.assertEqual(C.S3_FOLDER_TO_CAMERA[folder], cam)
            self.assertEqual(C.camera_from_key(f"{folder}/2026-08-19/x.mp4"), cam)


class TestModelInventory(unittest.TestCase):

    def test_extraction_classifiers_are_flagged_ambiguous(self):
        """`side_classification.pt` and `top_classification.pt` exist in BOTH
        the reconstruction and extraction trees with DIFFERENT weights, so a
        flat bucket cannot tell them apart and must never auto-download them."""
        self.assertEqual(set(C.AMBIGUOUS_MODEL_FILENAMES),
                         {C.MODEL_SIDE_CLASSIFICATION, C.MODEL_TOP_CLASSIFICATION})

    def test_ocr_feature_maps_to_v4s_plate_detector(self):
        self.assertEqual(C.FEATURE_MODEL_BY_KEY["ocr"], C.MODEL_WAGON_NUMBER)
        self.assertEqual(C.FEATURE_MODEL_LEGACY[C.MODEL_WAGON_NUMBER],
                         C.MODEL_WAGON_ID_COUNTING)

    def test_recon_models_keep_this_packages_own_filenames(self):
        """This package's counting engine resolves the long `*_wagon_gap.pt`
        names; renaming them to V4's short form would break Stage 1."""
        self.assertEqual(C.MODEL_RIGHT_UP_GAP, "right_up_wagon_gap.pt")
        self.assertEqual(C.MODEL_LEFT_UP_GAP, "left_up_wagon_gap.pt")
        self.assertEqual(C.RECON_MODEL_LEGACY[C.MODEL_RIGHT_UP_GAP],
                         "right_up_gap.pt")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Regressions found while auditing for historical mode
# ---------------------------------------------------------------------------

class TestBatchAgeTimezone(unittest.TestCase):
    """`TrainBatch.age_seconds()` read the IST filename digits as UTC.

    That made every batch look 5h30m in the FUTURE, so `age_seconds()` returned
    about -19,500 s and `select_runnable_batch`'s `age >= partial_wait` gate
    could never be satisfied: a 2-or-3-camera train was held back for ~5.4 hours
    instead of the intended 30 minutes.
    """

    def _ts(self, minutes_ago):
        from datetime import datetime, timedelta
        return (datetime.now(CFG.IST)
                - timedelta(minutes=minutes_ago)).strftime("%Y%m%d_%H%M%S")

    def test_age_of_a_recent_train_is_small_and_positive(self):
        from core.batch import TrainBatch
        age = TrainBatch(batch_key="x", train_timestamp=self._ts(5),
                         videos={}).age_seconds()
        self.assertGreater(age, 0, "age must never be negative")
        self.assertLess(age, 600, f"5-min-old train reported {age:.0f}s old")

    def test_partial_wait_gate_is_reachable(self):
        from core.batch import TrainBatch
        old = TrainBatch(batch_key="x", train_timestamp=self._ts(45), videos={})
        self.assertGreaterEqual(old.age_seconds(), 30 * 60)
        young = TrainBatch(batch_key="y", train_timestamp=self._ts(2), videos={})
        self.assertLess(young.age_seconds(), 30 * 60)

    def test_select_runnable_batch_holds_then_releases_a_partial(self):
        from core.batch import CameraVideo, TrainBatch
        def partial(ts):
            return TrainBatch(batch_key=ts, train_timestamp=ts, videos={
                C.CAMERA_RIGHT_UP: CameraVideo(
                    camera_id=C.CAMERA_RIGHT_UP, bucket="b", s3_key="k",
                    filename="k", s3_url="", train_timestamp=ts)})
        self.assertIsNone(TBM.select_runnable_batch([partial(self._ts(2))],
                                                    partial_wait_minutes=30.0))
        self.assertIsNotNone(TBM.select_runnable_batch([partial(self._ts(45))],
                                                       partial_wait_minutes=30.0))

    def test_agrees_with_the_producer_timezone(self):
        from datetime import datetime
        from train_extraction.time_utils import parse_timestamp_from_filename
        produced = parse_timestamp_from_filename("CCTV_20260808_103000_train.mp4")
        self.assertEqual(produced.utcoffset(), CFG.IST.utcoffset(datetime.now()))


class TestPollForBatchesIsBounded(unittest.TestCase):
    """`poll_for_batches` applied no recency window, so a first production poll
    queued a batch for EVERY clip in the trimmed bucket -- the ~17,600-batch
    flood `consumer_lookback_minutes` documents."""

    def setUp(self):
        self._saved = os.environ.pop("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["WAGONEYE_CONSUMER_LOOKBACK_MINUTES"] = self._saved

    def _archive_stub(self, days_old=30, trains=25):
        from datetime import datetime, timedelta, timezone
        keys = []
        base = datetime.now(timezone.utc) - timedelta(days=60)
        for i in range(trains):
            ts = (base + timedelta(hours=i * 7)).strftime("%Y%m%d_%H%M%S")
            keys += [f"{C.CAMERA_S3_FOLDER[c]}/CCTV_{ts}_train.mp4"
                     for c in C.ALL_CAMERAS]
        stamp = datetime.now(timezone.utc) - timedelta(days=days_old)

        class Stub:
            def list_objects_v2(self, **kw):
                p = kw.get("Prefix", "")
                return {"Contents": [
                    {"Key": k, "LastModified": stamp, "ETag": '"e"', "Size": 1}
                    for k in keys if k.startswith(p)], "IsTruncated": False}
        return Stub()

    def test_old_archive_is_not_queued_by_continuous_polling(self):
        got = TBM.poll_for_batches(s3_client=self._archive_stub(),
                                   processed_batches={})
        self.assertEqual(got, [], f"queued {len(got)} archive batch(es)")

    def test_explicit_replay_can_still_reach_past_the_window(self):
        got = TBM.poll_for_batches(s3_client=self._archive_stub(),
                                   processed_batches={}, apply_cutoff=False)
        self.assertTrue(got, "--batch replay must still find an older batch")

    def test_a_fresh_clip_is_still_discovered(self):
        got = TBM.poll_for_batches(s3_client=self._archive_stub(days_old=0),
                                   processed_batches={})
        self.assertTrue(got, "today's clips must still be discovered")

    def test_bounded_by_default(self):
        import inspect
        sig = inspect.signature(TBM.poll_for_batches)
        self.assertIs(sig.parameters["apply_cutoff"].default, True)


class TestModelStoreConfig(unittest.TestCase):
    """The operator-designated model store, and the tool that reconciles it."""

    def test_default_store_is_the_designated_bucket_and_prefix(self):
        self.assertEqual(C.MODELS_S3_BUCKET, "complete-train")
        self.assertEqual(C.MODELS_S3_PREFIX, "new_local")
        self.assertEqual(C.MODELS_S3_LAYOUT, "flat")

    def test_keys_resolve_under_the_prefix(self):
        from core import model_sync as MS
        keys = [r.s3_key for r in MS.required_models(["door", "ocr"])]
        self.assertTrue(keys)
        for k in keys:
            self.assertTrue(k.startswith("new_local/"), k)
            self.assertEqual(k.count("/"), 1, f"flat layout expected: {k}")

    def test_v4_location_is_still_reachable_by_env(self):
        """Repointing must never need a code edit."""
        import importlib
        saved = (os.environ.get("WAGONEYE_MODELS_S3_BUCKET"),
                 os.environ.get("WAGONEYE_MODELS_S3_PREFIX"))
        os.environ["WAGONEYE_MODELS_S3_BUCKET"] = "wagon-eye-models"
        os.environ["WAGONEYE_MODELS_S3_PREFIX"] = ""
        try:
            importlib.reload(C)
            self.assertEqual(C.MODELS_S3_BUCKET, "wagon-eye-models")
            self.assertEqual(C.MODELS_S3_PREFIX, "")
        finally:
            for k, v in zip(("WAGONEYE_MODELS_S3_BUCKET",
                             "WAGONEYE_MODELS_S3_PREFIX"), saved):
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(C)

    def test_reconcile_report_marks_present_and_absent(self):
        from core import model_sync as MS

        class Stub:
            def list_objects_v2(self, **kw):
                pre = kw.get("Prefix", "")
                names = ["right_up_wagon_gap.pt", "left_up_gap.pt",
                         "right_gap_1.pt", "notes.txt"]
                return {"Contents": [{"Key": f"{pre}{n}", "Size": 1_000_000}
                                     for n in names], "IsTruncated": False}

        text = MS.reconcile_report(["door"], s3_client=Stub())
        # exact name -> FOUND
        self.assertIn("[FOUND  ] reconstruction/right_up_wagon_gap.pt", text)
        # accepted alternative name -> FOUND, and says which
        self.assertIn("via accepted alt name left_up_gap.pt", text)
        # genuinely absent -> ABSENT, with a suggestion drawn from spare objects
        self.assertIn("[ABSENT ] features/door_state.pt", text)
        self.assertIn("right_gap_1.pt", text)
        # non-.pt objects are not offered as model candidates
        self.assertNotIn("notes.txt", text.split("closest unused")[0]
                         .split("objects in the store")[1])

    def test_reconcile_report_survives_no_credentials(self):
        from core import model_sync as MS
        text = MS.reconcile_report(["door"], s3_client=None)
        self.assertIn("model store:", text)   # never raises
