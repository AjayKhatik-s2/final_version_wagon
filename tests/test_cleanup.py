"""Disk cleanup must only ever touch a train that is genuinely delivered.

The failure this guards against is silent and expensive: reclaiming the
intermediates of a train whose S3 upload or dashboard post failed destroys
exactly what a retry needs. So most of these tests are about cleanup NOT
happening.

Every test builds a real batch tree on disk with real bytes in it and asserts on
what survives, not on a return value alone -- a cleanup that reports success
while deleting a retained artifact would pass a mock-based test.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import config as CFG
from core import constants as C
from delivery import cleanup as CU

CAMS = ("RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP", "LEFT_UP_TOP")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write(path: str, kb: int = 1) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x" * (kb * 1024))


def make_batch(root: str, key: str, *, delivered: bool = True,
               with_videos: bool = True) -> str:
    """A batch tree shaped like the sequential pipeline's own output.

    Temporary: downloads/, wagon_cache/, camera_evidence/<CAM>/camera_cache/.
    Retained: reports/, global_state/, wagon_states/, evidence/, archive/,
    and each camera bundle's manifest / tracking_full / camera PDF /
    engine_frames.
    """
    b = os.path.join(root, key)

    # --- temporary, and deliberately the biggest things here ---
    for cam in CAMS:
        _write(os.path.join(b, CFG.DIR_DOWNLOADS, f"{cam}_clip.mp4"), 40)
        for gw in ("L_1", "L_2"):
            _write(os.path.join(b, CAMERA := CFG.DIR_EVIDENCE, "unused"), 1) \
                if False else None
            _write(os.path.join(b, "camera_evidence", cam, "camera_cache",
                                gw, "frame_000001.jpg"), 60)
    for gw in ("GW_1", "GW_2"):
        _write(os.path.join(b, CFG.DIR_WAGON_CACHE, gw, "right_up",
                            "frame_000001.jpg"), 30)
    if with_videos:
        for cam in CAMS:
            _write(os.path.join(b, CFG.DIR_PROCESSED_VIDEOS,
                                f"{cam}_processed.mp4"), 20)

    # --- retained ---
    _write(os.path.join(b, CFG.DIR_REPORTS, "combined_train_report.pdf"), 2)
    with open(os.path.join(b, CFG.DIR_REPORTS,
                           "combined_train_report.json"), "w",
              encoding="utf-8") as f:
        json.dump({"batch_key": key, "summary": {"total_wagons": 2}}, f)
    _write(os.path.join(b, CFG.DIR_GLOBAL_STATE, "global_train_state.json"), 1)
    _write(os.path.join(b, CFG.DIR_WAGON_STATES, "damage", "GW_1.json"), 1)
    _write(os.path.join(b, CFG.DIR_EVIDENCE, "GW_1", "damage",
                        "track_1__RIGHT_UP_TOP.jpg"), 1)
    _write(os.path.join(b, CFG.DIR_ARCHIVE, "timings.json"), 1)
    for cam in CAMS:
        bd = os.path.join(b, "camera_evidence", cam)
        _write(os.path.join(bd, "manifest.json"), 1)
        _write(os.path.join(bd, "tracking_full.json"), 4)
        _write(os.path.join(bd, "camera_report.json"), 1)
        _write(os.path.join(bd, f"{cam}_report.pdf"), 2)
        _write(os.path.join(bd, "engine_frames", "metadata.json"), 1)

    if delivered:
        from delivery import finalization
        finalization.write(b, {"batch_key": key, "report_revision": 0})
    return b


class _Delivery:
    """A DeliveryResult-shaped stand-in: only the fields the gate reads."""

    def __init__(self, *, uploaded=True, errors=None, archived=None,
                 dashboard=None):
        self.uploaded = uploaded
        self.errors = list(errors or [])
        self.archived = dict(archived if archived is not None else
                             {"reports": 3, "evidence": 12, "global_state": 2,
                              "wagon_states": 4, "processed_videos": 4})
        self.dashboard = dict(dashboard if dashboard is not None else {
            "enabled": True,
            "cameras": {c: {"status": "ingested", "run_id": 1} for c in CAMS},
        })


def good_delivery(**kw):
    return _Delivery(**kw)


def _exists(batch_root: str, *parts) -> bool:
    return os.path.exists(os.path.join(batch_root, *parts))


# ---------------------------------------------------------------------------
# The fixture must be able to fail
# ---------------------------------------------------------------------------

class TestFixtureIsHonest(unittest.TestCase):

    def test_the_temporary_artifacts_are_the_big_ones(self):
        with tempfile.TemporaryDirectory() as root:
            b = make_batch(root, "T1")
            temp = (CU._dir_size(os.path.join(b, CFG.DIR_WAGON_CACHE))
                    + CU._dir_size(os.path.join(b, CFG.DIR_DOWNLOADS))
                    + sum(CU._dir_size(os.path.join(b, "camera_evidence", c,
                                                    "camera_cache"))
                          for c in CAMS))
            kept = CU._dir_size(os.path.join(b, CFG.DIR_REPORTS))
            self.assertGreater(temp, kept * 10,
                               "fixture must model the real size imbalance")

    def test_a_delivered_batch_has_a_marker(self):
        with tempfile.TemporaryDirectory() as root:
            ok, _ = CU.is_delivered_marker(make_batch(root, "T1"))
            self.assertTrue(ok)
            ok2, why = CU.is_delivered_marker(
                make_batch(root, "T2", delivered=False))
            self.assertFalse(ok2)
            self.assertIn("marker", why)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

class TestDeliveryGate(unittest.TestCase):

    def _gate(self, root, **kw):
        b = make_batch(root, "T1")
        return CU.is_delivered(b, good_delivery(**kw))

    def test_a_complete_delivery_passes(self):
        with tempfile.TemporaryDirectory() as root:
            ok, why = self._gate(root)
            self.assertTrue(ok, why)

    def test_no_delivery_result_at_all(self):
        with tempfile.TemporaryDirectory() as root:
            ok, why = CU.is_delivered(make_batch(root, "T1"), None)
            self.assertFalse(ok)
            self.assertIn("not delivered", why)

    def test_any_delivery_error_blocks_it(self):
        with tempfile.TemporaryDirectory() as root:
            ok, why = self._gate(root, errors=["archive evidence: timeout"])
            self.assertFalse(ok)
            self.assertIn("error", why)

    def test_nothing_uploaded_blocks_it(self):
        with tempfile.TemporaryDirectory() as root:
            ok, why = self._gate(root, uploaded=False)
            self.assertFalse(ok)
            self.assertIn("uploaded", why)

    def test_a_missing_combined_report_blocks_it(self):
        with tempfile.TemporaryDirectory() as root:
            b = make_batch(root, "T1")
            os.remove(os.path.join(b, CFG.DIR_REPORTS,
                                   "combined_train_report.json"))
            ok, why = CU.is_delivered(b, good_delivery())
            self.assertFalse(ok)
            self.assertIn("combined_train_report.json", why)

    def test_reports_subtree_uploading_nothing_blocks_it(self):
        with tempfile.TemporaryDirectory() as root:
            ok, why = self._gate(root, archived={"reports": 0, "evidence": 9})
            self.assertFalse(ok)
            self.assertIn("reports", why)

    def test_evidence_subtree_uploading_nothing_blocks_it(self):
        with tempfile.TemporaryDirectory() as root:
            ok, why = self._gate(root, archived={"reports": 3, "evidence": 0})
            self.assertFalse(ok)
            self.assertIn("evidence", why)

    def test_dashboard_failure_blocks_it_when_dashboard_was_enabled(self):
        with tempfile.TemporaryDirectory() as root:
            ok, why = self._gate(root, dashboard={
                "enabled": True,
                "cameras": {c: {"status": "ingest_failed"} for c in CAMS}})
            self.assertFalse(ok)
            self.assertIn("dashboard", why)

    def test_a_run_with_the_dashboard_off_can_still_be_cleaned(self):
        """Disk pressure is real even when nobody asked for a dashboard post."""
        with tempfile.TemporaryDirectory() as root:
            ok, why = self._gate(root, dashboard={"enabled": False})
            self.assertTrue(ok, why)

    def test_dashboard_can_be_demanded_regardless(self):
        with tempfile.TemporaryDirectory() as root:
            b = make_batch(root, "T1")
            cfg = CU.CleanupConfig(require_dashboard=True)
            ok, why = CU.is_delivered(
                b, good_delivery(dashboard={"enabled": False}), cfg)
            self.assertFalse(ok)
            self.assertIn("dashboard", why)

    def test_already_ingested_counts_as_success(self):
        """Re-delivering an unchanged batch is a no-op, not a failure."""
        with tempfile.TemporaryDirectory() as root:
            ok, why = self._gate(root, dashboard={
                "enabled": True,
                "cameras": {c: {"status": "already_ingested"} for c in CAMS}})
            self.assertTrue(ok, why)


# ---------------------------------------------------------------------------
# What gets deleted, and what survives
# ---------------------------------------------------------------------------

class TestCleanupBatch(unittest.TestCase):

    def _run(self, root, key="T1", *, cfg=None, delivery=None):
        b = make_batch(root, key)
        r = CU.cleanup_batch(batch_root=b, batch_key=key,
                             delivery=delivery or good_delivery(),
                             cfg=cfg or CU.DEFAULT_CONFIG, verbose=False)
        return b, r

    def test_successful_delivery_reclaims_the_temporaries(self):
        with tempfile.TemporaryDirectory() as root:
            b, r = self._run(root)
            self.assertTrue(r.performed, r.skipped_reason)
            self.assertFalse(_exists(b, CFG.DIR_DOWNLOADS))
            self.assertFalse(_exists(b, CFG.DIR_WAGON_CACHE))
            for cam in CAMS:
                self.assertFalse(_exists(b, "camera_evidence", cam,
                                         "camera_cache"),
                                 f"{cam} camera_cache survived")
            self.assertGreater(r.freed_bytes, 0)

    def test_every_retained_artifact_survives(self):
        with tempfile.TemporaryDirectory() as root:
            b, _r = self._run(root)
            for parts in (
                (CFG.DIR_REPORTS, "combined_train_report.json"),
                (CFG.DIR_REPORTS, "combined_train_report.pdf"),
                (CFG.DIR_GLOBAL_STATE, "global_train_state.json"),
                (CFG.DIR_WAGON_STATES, "damage", "GW_1.json"),
                (CFG.DIR_EVIDENCE, "GW_1", "damage",
                 "track_1__RIGHT_UP_TOP.jpg"),
                (CFG.DIR_ARCHIVE, "timings.json"),
            ):
                with self.subTest(artifact="/".join(parts)):
                    self.assertTrue(_exists(b, *parts))
            for cam in CAMS:
                for name in ("manifest.json", "tracking_full.json",
                             "camera_report.json", f"{cam}_report.pdf"):
                    with self.subTest(camera=cam, artifact=name):
                        self.assertTrue(_exists(b, "camera_evidence", cam,
                                                name))
                self.assertTrue(_exists(b, "camera_evidence", cam,
                                        "engine_frames", "metadata.json"))

    def test_tracking_full_survives_because_the_resolver_needs_it(self):
        """It is the only place the gap trajectories live."""
        with tempfile.TemporaryDirectory() as root:
            b, _ = self._run(root)
            for cam in CAMS:
                self.assertTrue(_exists(b, "camera_evidence", cam,
                                        "tracking_full.json"))

    def test_processed_videos_are_kept_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = self._run(root)
            self.assertTrue(_exists(b, CFG.DIR_PROCESSED_VIDEOS))

    def test_processed_videos_go_only_when_asked_and_uploaded(self):
        cfg = CU.CleanupConfig(delete_processed_videos=True)
        with tempfile.TemporaryDirectory() as root:
            b, _ = self._run(root, cfg=cfg)
            self.assertFalse(_exists(b, CFG.DIR_PROCESSED_VIDEOS))
        with tempfile.TemporaryDirectory() as root:
            # asked for, but S3 has none of them -> kept
            b, _ = self._run(root, cfg=cfg, delivery=good_delivery(
                archived={"reports": 3, "evidence": 9,
                          "processed_videos": 0}))
            self.assertTrue(_exists(b, CFG.DIR_PROCESSED_VIDEOS))

    def test_the_batch_directory_itself_is_never_removed(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = self._run(root)
            self.assertTrue(os.path.isdir(b))

    def test_failed_delivery_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            b = make_batch(root, "T1")
            r = CU.cleanup_batch(
                batch_root=b, batch_key="T1",
                delivery=good_delivery(errors=["archive reports: 503"]),
                verbose=False)
            self.assertFalse(r.performed)
            self.assertEqual(r.freed_bytes, 0)
            self.assertTrue(_exists(b, CFG.DIR_DOWNLOADS))
            self.assertTrue(_exists(b, CFG.DIR_WAGON_CACHE))
            for cam in CAMS:
                self.assertTrue(_exists(b, "camera_evidence", cam,
                                        "camera_cache"))

    def test_failed_dashboard_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            b = make_batch(root, "T1")
            r = CU.cleanup_batch(
                batch_root=b, batch_key="T1",
                delivery=good_delivery(dashboard={
                    "enabled": True,
                    "cameras": {c: {"status": "upload_failed"} for c in CAMS}}),
                verbose=False)
            self.assertFalse(r.performed)
            self.assertTrue(_exists(b, CFG.DIR_WAGON_CACHE))

    def test_disabled_config_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            b = make_batch(root, "T1")
            r = CU.cleanup_batch(batch_root=b, batch_key="T1",
                                 delivery=good_delivery(),
                                 cfg=CU.CleanupConfig(enabled=False),
                                 verbose=False)
            self.assertFalse(r.performed)
            self.assertTrue(_exists(b, CFG.DIR_WAGON_CACHE))

    def test_it_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            b, first = self._run(root)
            second = CU.cleanup_batch(batch_root=b, batch_key="T1",
                                      delivery=good_delivery(), verbose=False)
            self.assertTrue(second.performed)
            self.assertEqual(second.removed, [], "nothing left to remove")
            self.assertEqual(second.freed_bytes, 0)
            self.assertEqual(second.errors, [])
            self.assertGreater(first.freed_bytes, 0)

    def test_a_missing_batch_directory_is_safe(self):
        with tempfile.TemporaryDirectory() as root:
            r = CU.cleanup_batch(batch_root=os.path.join(root, "nope"),
                                 batch_key="nope",
                                 delivery=good_delivery(), verbose=False)
            self.assertFalse(r.performed)
            self.assertIn("no such batch", r.skipped_reason)

    def test_a_removal_failure_does_not_raise(self):
        with tempfile.TemporaryDirectory() as root:
            b = make_batch(root, "T1")
            with mock.patch("shutil.rmtree",
                            side_effect=OSError("device busy")):
                r = CU.cleanup_batch(batch_root=b, batch_key="T1",
                                     delivery=good_delivery(), verbose=False)
            self.assertTrue(r.performed)
            self.assertTrue(r.errors)
            self.assertEqual(r.freed_bytes, 0)
            self.assertTrue(_exists(b, CFG.DIR_WAGON_CACHE),
                            "a failed removal must leave the tree intact")


class TestScope(unittest.TestCase):

    def test_only_the_named_batch_is_touched(self):
        with tempfile.TemporaryDirectory() as root:
            a = make_batch(root, "T_0100")
            b = make_batch(root, "T_0200")
            CU.cleanup_batch(batch_root=a, batch_key="T_0100",
                             delivery=good_delivery(), verbose=False)
            self.assertFalse(_exists(a, CFG.DIR_WAGON_CACHE))
            self.assertTrue(_exists(b, CFG.DIR_WAGON_CACHE),
                            "a sibling batch must be untouched")
            for cam in CAMS:
                self.assertTrue(_exists(b, "camera_evidence", cam,
                                        "camera_cache"))

    def test_nothing_outside_the_batch_root_is_ever_planned(self):
        with tempfile.TemporaryDirectory() as root:
            b = make_batch(root, "T1")
            for path, _size in CU.plan(b, CU.DEFAULT_CONFIG):
                with self.subTest(path=path):
                    self.assertTrue(CU._inside(path, b))

    def test_the_guard_rejects_an_outside_path(self):
        with tempfile.TemporaryDirectory() as root:
            b = os.path.join(root, "batch")
            os.makedirs(b)
            self.assertFalse(CU._inside(root, b))
            self.assertFalse(CU._inside(b, b), "the root itself is not inside")
            self.assertTrue(CU._inside(os.path.join(b, "downloads"), b))


class TestDryRun(unittest.TestCase):

    def test_dry_run_reports_but_removes_nothing(self):
        cfg = CU.CleanupConfig(dry_run=True)
        with tempfile.TemporaryDirectory() as root:
            b = make_batch(root, "T1")
            r = CU.cleanup_batch(batch_root=b, batch_key="T1",
                                 delivery=good_delivery(), cfg=cfg,
                                 verbose=False)
            self.assertTrue(r.performed)
            self.assertTrue(r.dry_run)
            self.assertGreater(r.freed_bytes, 0)
            self.assertTrue(r.removed)
            self.assertTrue(_exists(b, CFG.DIR_DOWNLOADS))
            self.assertTrue(_exists(b, CFG.DIR_WAGON_CACHE))
            for cam in CAMS:
                self.assertTrue(_exists(b, "camera_evidence", cam,
                                        "camera_cache"))

    def test_dry_run_and_real_run_agree_on_what_would_go(self):
        """The preview must be the truth, or it is worse than no preview."""
        with tempfile.TemporaryDirectory() as root:
            b = make_batch(root, "T1")
            preview = CU.cleanup_batch(
                batch_root=b, batch_key="T1", delivery=good_delivery(),
                cfg=CU.CleanupConfig(dry_run=True), verbose=False)
            real = CU.cleanup_batch(
                batch_root=b, batch_key="T1", delivery=good_delivery(),
                verbose=False)
            self.assertEqual(sorted(preview.removed), sorted(real.removed))
            self.assertEqual(preview.freed_bytes, real.freed_bytes)


class TestLowDiskProtection(unittest.TestCase):

    def _sweep(self, root, free_gb, active="", cfg=None):
        with mock.patch.object(CU, "free_gb", return_value=free_gb):
            return CU.ensure_free_space(
                workspace_root=root, active_batch_key=active,
                cfg=cfg or CU.CleanupConfig(min_free_gb=8.0), verbose=False)

    def test_a_healthy_disk_is_left_alone(self):
        with tempfile.TemporaryDirectory() as root:
            b = make_batch(root, "T1")
            out = self._sweep(root, free_gb=50.0)
            self.assertEqual(out, [])
            self.assertTrue(_exists(b, CFG.DIR_WAGON_CACHE))

    def test_a_low_disk_reclaims_delivered_batches(self):
        with tempfile.TemporaryDirectory() as root:
            a = make_batch(root, "T_0100")
            out = self._sweep(root, free_gb=2.0)
            self.assertTrue(out)
            self.assertFalse(_exists(a, CFG.DIR_WAGON_CACHE))
            self.assertTrue(_exists(a, CFG.DIR_REPORTS,
                                    "combined_train_report.json"))

    def test_the_active_batch_is_never_reclaimed(self):
        with tempfile.TemporaryDirectory() as root:
            active = make_batch(root, "T_0200")
            other = make_batch(root, "T_0100")
            self._sweep(root, free_gb=2.0, active="T_0200")
            self.assertTrue(_exists(active, CFG.DIR_WAGON_CACHE),
                            "the batch about to run must keep everything")
            self.assertFalse(_exists(other, CFG.DIR_WAGON_CACHE))

    def test_an_undelivered_batch_is_never_reclaimed(self):
        with tempfile.TemporaryDirectory() as root:
            bad = make_batch(root, "T_0100", delivered=False)
            self._sweep(root, free_gb=2.0)
            self.assertTrue(_exists(bad, CFG.DIR_WAGON_CACHE),
                            "no marker means delivery never completed")

    def test_a_disabled_config_never_sweeps(self):
        with tempfile.TemporaryDirectory() as root:
            b = make_batch(root, "T1")
            out = self._sweep(root, free_gb=1.0,
                              cfg=CU.CleanupConfig(enabled=False))
            self.assertEqual(out, [])
            self.assertTrue(_exists(b, CFG.DIR_WAGON_CACHE))

    def test_an_unscannable_workspace_does_not_raise(self):
        out = self._sweep(os.path.join(tempfile.gettempdir(), "nope-xyz"),
                          free_gb=1.0)
        self.assertEqual(out, [])


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

class TestWiring(unittest.TestCase):

    def test_the_sequential_branch_gates_on_delivery(self):
        import inspect
        from orchestrator import historical_runner as HR
        src = inspect.getsource(HR.run)
        seq = src[src.index('if mode == "sequential"'):
                  src.index("invoking existing pipeline")]
        self.assertIn("cleanup", seq)
        self.assertIn("delivery=getattr(asm", seq,
                      "cleanup must be handed the DeliveryResult")

    def test_the_pre_train_sweep_runs_before_staging(self):
        import inspect
        from orchestrator import historical_runner as HR
        src = inspect.getsource(HR.run)
        self.assertLess(src.index("ensure_free_space"),
                        src.index("staging inputs"))
        self.assertIn("active_batch_key=batch.batch_key", src)

    def test_cleanup_failure_cannot_fail_a_train(self):
        import inspect
        from orchestrator import historical_runner as HR
        src = inspect.getsource(HR.run)
        block = src[src.index("ensure_free_space") - 400:
                    src.index("staging inputs")]
        self.assertIn("except Exception", block)

    def test_keep_inputs_still_disables_everything(self):
        from orchestrator.historical_runner import _cleanup_config
        self.assertFalse(_cleanup_config(True).enabled)
        self.assertTrue(_cleanup_config(False).enabled)

    def test_the_env_switches_are_read(self):
        from orchestrator.historical_runner import _cleanup_config
        with mock.patch.dict(os.environ, {
                "WAGONEYE_CLEANUP_DRY_RUN": "1",
                "WAGONEYE_CLEANUP_MIN_FREE_GB": "25",
                "WAGONEYE_CLEANUP_PROCESSED_VIDEOS": "true"}):
            cfg = _cleanup_config(False)
        self.assertTrue(cfg.dry_run)
        self.assertEqual(cfg.min_free_gb, 25.0)
        self.assertTrue(cfg.delete_processed_videos)

    def test_the_module_never_touches_s3(self):
        import ast
        src = open(os.path.join(V4_ROOT, "delivery", "cleanup.py"),
                   encoding="utf-8").read()
        names = set()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Import):
                for a in n.names:
                    names.update(a.name.split("."))
            elif isinstance(n, ast.ImportFrom):
                names.update((n.module or "").split("."))
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
        for banned in ("boto3", "s3_upload", "delete_object", "upload_file",
                       "dashboard_ingest"):
            with self.subTest(token=banned):
                self.assertNotIn(banned, names)

    def test_batch_mode_is_untouched(self):
        import inspect
        from orchestrator import master_runner as MR
        self.assertNotIn("delivery.cleanup",
                         inspect.getsource(MR.process_batch))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSweepCanReclaimOverlayVideos(unittest.TestCase):
    """The sweep holds no DeliveryResult, so it needs durable upload proof.

    `videos_uploaded` was derived from `delivery.archived` alone. The sweep
    passes `delivery=None`, so that read always yielded 0 and
    `delete_processed_videos=True` silently did nothing -- it reported
    "freed 0.00 GB from 0 path(s)" on batches whose delivery had plainly
    uploaded four videos. A flag that quietly does nothing is worse than one
    that refuses out loud.
    """

    def _batch(self, tmp, *, marker_counts=None, report_urls=True):
        root = os.path.join(tmp, "20260729_101500")
        os.makedirs(os.path.join(root, CFG.DIR_REPORTS), exist_ok=True)
        vids = os.path.join(root, CFG.DIR_PROCESSED_VIDEOS)
        os.makedirs(vids, exist_ok=True)
        for cam in C.ALL_CAMERAS:
            with open(os.path.join(vids, f"{cam}_processed.mp4"), "wb") as f:
                f.write(b"\x00" * 4096)
        doc = {"batch_key": "20260729_101500", "summary": {"total_wagons": 2}}
        if report_urls:
            doc["train_metadata"] = {"processed_video_urls": {
                cam: f"https://b.s3.ap-south-1.amazonaws.com/p/{cam}.mp4"
                for cam in C.ALL_CAMERAS}}
        with open(os.path.join(root, CFG.DIR_REPORTS,
                               "combined_train_report.json"), "w") as f:
            json.dump(doc, f)
        marker = {"batch_key": "20260729_101500", "uploaded": True,
                  "upload_urls": {"json": "https://x/y.json"}}
        if marker_counts is not None:
            marker["archived"] = marker_counts
        from delivery import finalization
        finalization.write(root, marker)
        return root

    def _run(self, root):
        cfg = CU.CleanupConfig(delete_processed_videos=True)
        return CU.cleanup_batch(batch_root=root, batch_key="20260729_101500",
                                delivery=None, require_delivery=False,
                                cfg=cfg, verbose=False)

    def test_a_recorded_count_lets_the_videos_go(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._batch(tmp, marker_counts={"processed_videos": 4})
            res = self._run(root)
            self.assertGreater(res.freed_bytes, 0,
                               "a recorded upload count must permit reclaim")
            self.assertFalse(os.path.isdir(
                os.path.join(root, CFG.DIR_PROCESSED_VIDEOS)))

    def test_a_zero_count_keeps_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._batch(tmp, marker_counts={"processed_videos": 0})
            self._run(root)
            self.assertTrue(os.path.isdir(
                os.path.join(root, CFG.DIR_PROCESSED_VIDEOS)),
                "nothing uploaded means the local copies are all there is")

    def test_an_older_marker_falls_back_to_the_reports_own_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._batch(tmp, marker_counts=None, report_urls=True)
            res = self._run(root)
            self.assertGreater(res.freed_bytes, 0)

    def test_no_count_and_no_urls_keeps_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._batch(tmp, marker_counts=None, report_urls=False)
            self._run(root)
            self.assertTrue(os.path.isdir(
                os.path.join(root, CFG.DIR_PROCESSED_VIDEOS)),
                "no evidence of upload is not evidence of upload")

    def test_the_default_config_never_touches_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._batch(tmp, marker_counts={"processed_videos": 4})
            CU.cleanup_batch(batch_root=root, batch_key="k", delivery=None,
                             require_delivery=False, verbose=False)
            self.assertTrue(os.path.isdir(
                os.path.join(root, CFG.DIR_PROCESSED_VIDEOS)),
                "reclaiming the videos must stay strictly opt-in")
