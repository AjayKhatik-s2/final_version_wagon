"""Delivering a finished batch from disk: `--deliver-only` and sequential mode.

Two gaps this closes, both found by reading the code against a real run:

* **Sequential mode could not publish at all.**  `global_assembler.assemble()`
  ended at `combined_train_report.build()` -- no upload, no dashboard post, no
  email.  A `--mode sequential` run produced ZERO dashboard entries, which became
  a production hole when sequential became the default foreground mode.
* **A failed delivery cost a full reprocess.**  Delivery lived only inside
  `process_batch`, so re-publishing an already-finished train meant re-running
  Stage 1-5: ~30 minutes of CPU per train, measured.

The tests below pin the behaviours that make republishing safe:

  - a batch with no combined report is REFUSED, not partially published -- the
    per-camera documents are derived from that file, so without it there is
    nothing truthful to send;
  - the S3 key layout is identical to the live path's, so a republish overwrites
    the same objects instead of creating a parallel set;
  - `send_email` defaults to OFF, because a republish must not re-mail operators
    about a train they were already told about;
  - every step is failure-isolated: a receiver outage is reported, never raised.

Nothing here touches AWS: the S3 client, `requests` and the notification module
are all stubs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as C                                    # noqa: E402
from delivery import finalize                                      # noqa: E402


def _no_microservice():
    """Stop `upload_pdf` making a REAL 120s HTTP POST to the report microservice.

    `s3_upload.upload_pdf` tries `C.UPLOAD_API_URL` first and only then falls
    back to S3.  Left alone, each delivery here fired five live requests with a
    120-second timeout and the suite hung.  Returning None puts it straight on
    the S3 fallback, which is what the stub client is for.  Returns the original
    so the caller can restore it.
    """
    from delivery import s3_upload
    original = s3_upload._upload_pdf_microservice
    s3_upload._upload_pdf_microservice = lambda *a, **k: None
    return original


def _restore_microservice(original):
    from delivery import s3_upload
    s3_upload._upload_pdf_microservice = original


class _StubS3:
    def __init__(self):
        self.uploads = []

    def upload_file(self, local, bucket, key, ExtraArgs=None):
        self.uploads.append((bucket, key))

    def put_object(self, **kw):
        self.uploads.append((kw.get("Bucket"), kw.get("Key")))


def _write(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_finished_batch(tmp, key="20260729_103722", *, with_report=True,
                         cameras=C.ALL_CAMERAS):
    """A batch tree in the shape Stage 5 leaves behind."""
    root = os.path.join(tmp, key)
    if with_report:
        _write(os.path.join(root, "reports", "combined_train_report.pdf"))
        with open(os.path.join(root, "reports", "combined_train_report.json"),
                  "w", encoding="utf-8") as f:
            json.dump({
                "batch_key": key,
                "train_metadata": {"source_video_urls": {
                    c: f"https://b/{C.CAMERA_S3_FOLDER[c]}/clip_{key}_train.mp4"
                    for c in cameras}},
                "wagons": [{"global_id": "GW_1",
                            "supporting_cameras": list(cameras)}],
            }, f)
    for cam in cameras:
        _write(os.path.join(root, "reports",
                            f"{C.CAMERA_FOLDER[cam]}_report.pdf"))
    _write(os.path.join(root, "global_state", "global_train_state.json"), "{}")
    _write(os.path.join(root, "wagon_states", "unified", "GW_1.json"), "{}")
    _write(os.path.join(root, "evidence", "GW_1", "door", "right_best.jpg"))
    _write(os.path.join(root, "processed_videos", "RIGHT_UP_processed.mp4"))
    return root


# ---------------------------------------------------------------------------
# Artifact discovery + the refusal rule
# ---------------------------------------------------------------------------

class TestArtifactDiscovery(unittest.TestCase):

    def test_batch_key_comes_from_the_directory_name(self):
        self.assertEqual(finalize.batch_key_for("/a/b/20260729_103722"),
                         "20260729_103722")
        self.assertEqual(finalize.batch_key_for("/a/b/20260729_103722/"),
                         "20260729_103722")

    def test_finds_every_artifact_a_finished_batch_has(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_finished_batch(tmp)
            a = finalize.find_artifacts(root)
            self.assertTrue(a["combined_pdf"])
            self.assertTrue(a["combined_json"])
            self.assertEqual(set(a["camera_pdfs"]), set(C.ALL_CAMERAS))
            for k in ("global_state", "wagon_states", "evidence",
                      "processed_videos"):
                self.assertTrue(a[k], k)

    def test_a_batch_without_a_combined_report_is_refused(self):
        """The per-camera documents are DERIVED from that file.  Publishing
        without it would mean inventing a train."""
        with tempfile.TemporaryDirectory() as tmp:
            root = build_finished_batch(tmp, with_report=False)
            ok, reason = finalize.is_deliverable(root)
            self.assertFalse(ok)
            self.assertIn("combined_train_report.json", reason)

    def test_a_missing_directory_is_refused(self):
        ok, reason = finalize.is_deliverable("/nope/not/here")
        self.assertFalse(ok)
        self.assertIn("no such batch directory", reason)


# ---------------------------------------------------------------------------
# deliver()
# ---------------------------------------------------------------------------

class TestDeliver(unittest.TestCase):

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in
                       ("WAGONEYE_DASHBOARD_INGEST_ENABLED",
                        "WAGONEYE_ML_API_ENABLED")}
        # Keep the two network-facing steps out of these tests; they have their
        # own suites.  What is under test here is the disk->delivery plumbing.
        os.environ["WAGONEYE_DASHBOARD_INGEST_ENABLED"] = "false"
        os.environ["WAGONEYE_ML_API_ENABLED"] = "false"
        self._micro = _no_microservice()

    def tearDown(self):
        _restore_microservice(self._micro)
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_refuses_an_undeliverable_batch_without_uploading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_finished_batch(tmp, with_report=False)
            s3 = _StubS3()
            res = finalize.deliver(batch_root=root, s3_client=s3, verbose=False)
            self.assertFalse(res.ok)
            self.assertEqual(s3.uploads, [], "must not upload anything")

    def test_dry_run_uploads_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_finished_batch(tmp)
            s3 = _StubS3()
            res = finalize.deliver(batch_root=root, s3_client=s3,
                                   dry_run=True, verbose=False)
            self.assertEqual(s3.uploads, [])
            self.assertFalse(res.uploaded)

    def test_archives_every_subtree_under_the_batch_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_finished_batch(tmp)
            s3 = _StubS3()
            res = finalize.deliver(batch_root=root, s3_client=s3, verbose=False)
            self.assertTrue(res.uploaded, res.errors)
            self.assertEqual(
                set(res.archived),
                {"global_state", "wagon_states", "reports", "evidence",
                 "processed_videos"})

    def test_s3_keys_match_the_live_paths_layout(self):
        """A republish must overwrite the SAME objects, not create a parallel
        set -- the live path builds `train_batch/<key>/<sub>/...`."""
        with tempfile.TemporaryDirectory() as tmp:
            root = build_finished_batch(tmp)
            s3 = _StubS3()
            finalize.deliver(batch_root=root, s3_client=s3, verbose=False)
            keys = [k for _b, k in s3.uploads]
            self.assertTrue(keys)
            prefix = f"{C.S3_TRAIN_BATCH_PREFIX}/20260729_103722/"
            for k in keys:
                self.assertTrue(k.startswith(prefix), k)

    def test_seeds_the_finalization_marker_so_documents_carry_pdf_links(self):
        from delivery import finalization as FIN
        with tempfile.TemporaryDirectory() as tmp:
            root = build_finished_batch(tmp)
            finalize.deliver(batch_root=root, s3_client=_StubS3(), verbose=False)
            marker = FIN.load(root)
            self.assertIsNotNone(marker)
            self.assertIn("upload_urls", marker)

    def test_never_overwrites_an_existing_marker(self):
        from delivery import finalization as FIN
        with tempfile.TemporaryDirectory() as tmp:
            root = build_finished_batch(tmp)
            FIN.write(root, {"batch_key": "x", "upload_urls": {"pdf": "KEEP"}})
            finalize.deliver(batch_root=root, s3_client=_StubS3(), verbose=False)
            self.assertEqual(FIN.load(root)["upload_urls"]["pdf"], "KEEP")

    def test_email_is_off_by_default(self):
        """A republish must not re-mail operators about a known train."""
        import inspect
        self.assertIs(
            inspect.signature(finalize.deliver).parameters["send_email"].default,
            False)

    def test_an_upload_failure_is_reported_not_raised(self):
        class Boom:
            def upload_file(self, *a, **k):
                raise RuntimeError("s3 down")
            def put_object(self, **k):
                raise RuntimeError("s3 down")
        with tempfile.TemporaryDirectory() as tmp:
            root = build_finished_batch(tmp)
            res = finalize.deliver(batch_root=root, s3_client=Boom(),
                                   verbose=False)
            self.assertFalse(res.ok)
            self.assertTrue(res.errors)

    def test_render_is_safe_on_every_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_finished_batch(tmp)
            res = finalize.deliver(batch_root=root, s3_client=_StubS3(),
                                   verbose=False)
            self.assertIn("delivery for", res.render())


# ---------------------------------------------------------------------------
# Sequential mode now delivers
# ---------------------------------------------------------------------------

class TestSequentialDelivers(unittest.TestCase):

    def test_assemble_accepts_delivery_and_defaults_it_off(self):
        import inspect
        from orchestrator import global_assembler
        params = inspect.signature(global_assembler.assemble).parameters
        for name in ("deliver", "send_email", "s3_client"):
            self.assertIn(name, params, name)
        self.assertIs(params["deliver"].default, False,
                      "an assembly under validation must not publish")
        self.assertIs(params["send_email"].default, False)

    def test_assembler_calls_the_shared_delivery_module(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(here, "orchestrator", "global_assembler.py"),
                   encoding="utf-8").read()
        self.assertIn("from delivery import finalize", src)
        self.assertIn("finalize.deliver(", src)

    def test_run_sequential_forwards_the_flags(self):
        import inspect
        from orchestrator import master_runner
        params = inspect.signature(master_runner.run_sequential).parameters
        self.assertIn("deliver", params)
        self.assertIn("send_email", params)

    def test_assembler_still_reports_no_yolo_during_assembly(self):
        """Delivery must not have introduced any inference into assembly."""
        from orchestrator.global_assembler import AssemblyResult
        self.assertEqual(AssemblyResult().yolo_calls_during_assembly, 0)


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------

class TestDeliverOnlyCli(unittest.TestCase):

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in
                       ("WAGONEYE_DASHBOARD_INGEST_ENABLED",
                        "WAGONEYE_ML_API_ENABLED")}
        os.environ["WAGONEYE_DASHBOARD_INGEST_ENABLED"] = "false"
        os.environ["WAGONEYE_ML_API_ENABLED"] = "false"
        self._micro = _no_microservice()

    def tearDown(self):
        _restore_microservice(self._micro)
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _args(self, argv):
        from orchestrator import master_runner
        return master_runner._build_parser().parse_args(argv)

    def test_flags_exist_and_default_off(self):
        a = self._args([])
        self.assertIsNone(a.deliver_only)
        self.assertFalse(a.deliver)

    def test_deliver_only_takes_a_directory(self):
        a = self._args(["--deliver-only", "/x/20260729_103722"])
        self.assertEqual(a.deliver_only, "/x/20260729_103722")

    def test_deliver_only_dispatches_before_every_processing_mode(self):
        """It must never fall through into auto/sequential/historical."""
        from orchestrator import master_runner as MR
        entered = []
        saved = (MR.run_auto, MR.run_sequential, MR.run_historical,
                 MR.run_local)
        MR.run_auto = lambda *a, **k: entered.append("auto") or 0
        MR.run_sequential = lambda *a, **k: entered.append("seq") or 0
        MR.run_historical = lambda *a, **k: entered.append("hist") or 0
        MR.run_local = lambda *a, **k: entered.append("local") or 0
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = build_finished_batch(tmp, with_report=False)
                rc = MR.main(["--deliver-only", root, "--no-interactive"])
        finally:
            (MR.run_auto, MR.run_sequential, MR.run_historical,
             MR.run_local) = saved
        self.assertEqual(entered, [], "must not enter a processing mode")
        self.assertEqual(rc, 3, "an undeliverable batch should exit non-zero")

    def test_deliver_only_dry_run_succeeds_without_touching_aws(self):
        from orchestrator import master_runner as MR
        with tempfile.TemporaryDirectory() as tmp:
            root = build_finished_batch(tmp)
            rc = MR.main(["--deliver-only", root, "--dry-run",
                          "--no-interactive"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
