"""The fused report reaches the receiver, not just S3.

The gap this closes: Stage 5 fused the four cameras into
`combined_train_report.json` and Stage 6 uploaded it, and there it stopped.
Nothing told the receiver it existed, so the backend's global endpoint would
have stayed empty however many trains ran.

The two feeds are deliberately NOT one:

    per-camera  POST /inspections/ingest         {camera_id, inspection_s3_uri,
                                                  version}   -- a POINTER
    global      POST /inspections/ingest-global  {camera_id, global_train_data}
                                                             -- the DOCUMENT

Sending a pointer to the global endpoint is a 422: `global_train_data` is
required and is the report itself.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from delivery import global_train_webhook as GTW


class FakeResp:
    def __init__(self, code=200, payload=None, text=""):
        self.status_code, self._p, self.text = code, payload, text

    def json(self):
        if self._p is None:
            raise ValueError("no json")
        return self._p


class FakeRequests:
    """Records every POST so the payload itself can be asserted on."""

    def __init__(self, responses=None):
        self.calls = []
        self._responses = responses or {}

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "body": json, "headers": headers})
        r = self._responses.get(url, self._responses.get("*"))
        if isinstance(r, Exception):
            raise r
        return r or FakeResp(200, {"message": "ok", "run_id": 6801,
                                   "segments_count": 59,
                                   "already_existed": False})


def report(tmp, wagons=3, batch_key="20260724_081227"):
    p = os.path.join(tmp, "combined_train_report.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"schema": "wagon_eye.combined_train_report.v1",
                   "batch_key": batch_key,
                   "summary": {"total_wagons": wagons},
                   "wagons": [{"global_id": f"GW_{i}", "wagon_index": i}
                              for i in range(1, wagons + 1)]}, f)
    return p


class TestTheEndpointIsDerivedNotDuplicated(unittest.TestCase):

    def test_it_is_the_per_camera_endpoint_plus_a_suffix(self):
        """One configured base for both feeds, so the global endpoint cannot
        drift onto a different host than the per-camera one."""
        for base in GTW.global_ingest_urls():
            self.assertTrue(base.endswith("/inspections/ingest-global"), base)

    def test_the_default_is_uat_only_because_prod_has_no_such_endpoint(self):
        """Verified 2026-08-28: the derived PROD path returns 404 -- the
        receiver has only shipped the global endpoint on UAT. Posting there
        anyway would log a failed delivery for every train and teach the
        operator to ignore the line that is meant to matter."""
        self.assertEqual(GTW.global_ingest_urls(),
                         ["https://cctv-wagon-uat-api.suvidhaen.com"
                          "/inspections/ingest-global"])

    def test_prod_can_be_added_when_the_backend_ships_it(self):
        os.environ["WAGONEYE_GLOBAL_INGEST_ALL_RECEIVERS"] = "true"
        try:
            from delivery.dashboard_ingest import ingest_api_urls
            got = GTW.global_ingest_urls()
            self.assertEqual(len(got), len(ingest_api_urls()))
            self.assertTrue(any("ms-pnr-location-notification-api" in u
                                for u in got))
        finally:
            del os.environ["WAGONEYE_GLOBAL_INGEST_ALL_RECEIVERS"]

    def test_the_uat_endpoint_is_the_one_the_backend_gave_us(self):
        self.assertIn(
            "https://cctv-wagon-uat-api.suvidhaen.com/inspections/ingest-global",
            GTW.global_ingest_urls())

    def test_an_explicit_override_wins(self):
        os.environ["WAGONEYE_GLOBAL_INGEST_API_URLS"] = "https://a/x,https://b/y"
        try:
            self.assertEqual(GTW.global_ingest_urls(),
                             ["https://a/x", "https://b/y"])
        finally:
            del os.environ["WAGONEYE_GLOBAL_INGEST_API_URLS"]


class TestTheVirtualFifthCamera(unittest.TestCase):

    def test_the_camera_id_is_global_fused(self):
        """The receiver stores this as a virtual fifth camera under that name.
        Sending one of the four real ids would claim the fused result belongs to
        a single viewpoint, which is the one thing it is not."""
        self.assertEqual(GTW.camera_id(), "GLOBAL_FUSED")
        self.assertEqual(GTW.GLOBAL_FUSED_CAMERA_ID, "GLOBAL_FUSED")
        self.assertNotIn(GTW.camera_id(), C.ALL_CAMERAS)

    def test_it_is_overridable(self):
        os.environ["WAGONEYE_GLOBAL_INGEST_CAMERA_ID"] = "camera_X"
        try:
            self.assertEqual(GTW.camera_id(), "camera_X")
        finally:
            del os.environ["WAGONEYE_GLOBAL_INGEST_CAMERA_ID"]


class TestThePayload(unittest.TestCase):

    def test_it_sends_the_document_not_a_pointer(self):
        """The global endpoint requires `global_train_data` inline. A pointer
        -- which is what the per-camera feed sends -- is a 422 here."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeRequests()
            GTW.publish(report_json_path=report(tmp), requests_mod=fake,
                        verbose=False)
            body = fake.calls[0]["body"]
            self.assertEqual(set(body), {"camera_id", "global_train_data"})
            self.assertNotIn("inspection_s3_uri", body)
            self.assertEqual(body["camera_id"], "GLOBAL_FUSED")
            self.assertEqual(len(body["global_train_data"]["wagons"]), 3)

    def test_the_document_is_sent_verbatim(self):
        """Read from the same file Stage 6 uploaded, so S3 and the dashboard
        cannot disagree about what this train was."""
        with tempfile.TemporaryDirectory() as tmp:
            p = report(tmp)
            fake = FakeRequests()
            GTW.publish(report_json_path=p, requests_mod=fake, verbose=False)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(fake.calls[0]["body"]["global_train_data"],
                                 json.load(f))

    def test_the_result_records_what_the_receiver_said(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = GTW.publish(report_json_path=report(tmp, wagons=59),
                              requests_mod=FakeRequests(), verbose=False)
            self.assertTrue(res.posted)
            self.assertEqual(res.wagons, 59)
            self.assertEqual(res.batch_key, "20260724_081227")
            first = list(res.per_endpoint.values())[0]
            self.assertEqual(first["run_id"], 6801)
            self.assertEqual(first["segments_count"], 59)


class TestFailureIsNeverFatal(unittest.TestCase):
    """The report is already written and already in S3 before this runs. A
    receiver outage costs a dashboard row, not the train."""

    def test_a_connection_error_is_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeRequests({"*": RuntimeError("connection refused")})
            res = GTW.publish(report_json_path=report(tmp), requests_mod=fake,
                              verbose=False)
            self.assertFalse(res.posted)
            self.assertTrue(res.attempted)
            for r in res.per_endpoint.values():
                self.assertFalse(r["ok"])
                self.assertIn("connection refused", r["error"])

    def test_a_422_is_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeRequests({"*": FakeResp(422, {"detail": [{"msg": "x"}]})})
            res = GTW.publish(report_json_path=report(tmp), requests_mod=fake,
                              verbose=False)
            self.assertFalse(res.posted)
            self.assertEqual(list(res.per_endpoint.values())[0]["status_code"],
                             422)

    def test_one_endpoint_succeeding_counts_as_posted(self):
        """Same rule the per-camera feed uses: a document is ingested when at
        least one receiver accepts it, and per-endpoint outcomes stay visible so
        a partial delivery is not hidden."""
        os.environ["WAGONEYE_GLOBAL_INGEST_ALL_RECEIVERS"] = "true"
        self.addCleanup(os.environ.pop, "WAGONEYE_GLOBAL_INGEST_ALL_RECEIVERS",
                        None)
        urls = GTW.global_ingest_urls()
        if len(urls) < 2:
            self.skipTest("only one endpoint configured")
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeRequests({urls[0]: FakeResp(500, None, "boom"),
                                 urls[1]: FakeResp(200, {"run_id": 7})})
            res = GTW.publish(report_json_path=report(tmp), requests_mod=fake,
                              verbose=False)
            self.assertTrue(res.posted)
            self.assertFalse(res.per_endpoint[urls[0]]["ok"])
            self.assertTrue(res.per_endpoint[urls[1]]["ok"])

    def test_a_missing_report_is_skipped_with_a_reason(self):
        res = GTW.publish(report_json_path="/nope/report.json", verbose=False)
        self.assertFalse(res.attempted)
        self.assertIn("no combined report", res.skipped_reason)

    def test_a_corrupt_report_is_skipped_with_a_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "combined_train_report.json")
            with open(p, "w", encoding="utf-8") as f:
                f.write("{not json")
            res = GTW.publish(report_json_path=p, verbose=False)
            self.assertFalse(res.attempted)
            self.assertIn("could not read", res.skipped_reason)

    def test_it_can_be_switched_off(self):
        os.environ["WAGONEYE_GLOBAL_INGEST"] = "false"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                res = GTW.publish(report_json_path=report(tmp), verbose=False)
                self.assertFalse(res.attempted)
                self.assertIn("disabled", res.skipped_reason)
        finally:
            del os.environ["WAGONEYE_GLOBAL_INGEST"]

    def test_it_is_on_by_default(self):
        self.assertTrue(GTW.is_enabled())


class TestBothPipelinesPostIt(unittest.TestCase):

    @staticmethod
    def _src(path):
        return open(os.path.join(V4_ROOT, path), encoding="utf-8").read()

    def test_batch_mode_posts_the_fused_report(self):
        src = self._src("orchestrator/master_runner.py")
        self.assertIn("global_train_webhook as GTW", src)
        self.assertIn("GTW.publish(", src)

    def test_sequential_mode_posts_the_fused_report(self):
        src = self._src("orchestrator/global_assembler.py")
        self.assertIn("global_train_webhook as GTW", src)
        self.assertIn("GTW.publish(", src)

    def test_both_post_it_after_the_per_camera_ingest(self):
        """The virtual GLOBAL_FUSED camera supersedes the four provisional
        per-camera documents, so it has to arrive last rather than race them."""
        src = self._src("orchestrator/master_runner.py")
        lines = src.splitlines()
        per_cam = next(i for i, l in enumerate(lines)
                       if "dashboard_ingest.run(" in l)
        glob = next(i for i, l in enumerate(lines) if "GTW.publish(" in l)
        self.assertLess(per_cam, glob)

    def test_sequential_posts_it_only_when_delivering(self):
        """An assembly run being validated must not publish to the live
        dashboard."""
        src = self._src("orchestrator/global_assembler.py")
        lines = src.splitlines()
        deliver = next(i for i, l in enumerate(lines)
                       if l.strip() == "if deliver:")
        glob = next(i for i, l in enumerate(lines) if "GTW.publish(" in l)
        self.assertLess(deliver, glob)
        self.assertTrue(lines[glob].startswith(" " * 12))


if __name__ == "__main__":
    unittest.main(verbosity=2)
