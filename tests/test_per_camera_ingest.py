"""Per-camera dashboard publishing at the moment a camera seals.

Sequential mode seals each camera independently.  `--deliver-per-camera` posts
that camera's inspection straight away, instead of waiting for the other three
and for global assembly.

This is the V4 model, not a shortcut: V4 runs four independent per-camera
pipelines, each counting its own segments and POSTing its own document, and its
four documents routinely disagree on `total_wagons`.  Measured on 2026-07-29,
local counts were 63 / 56 / 62 / 65 against a fused global 58.

What these tests protect is the honesty of the resulting document, because the
numbering is the one thing a reader can get badly wrong:

* wagon `n` here is THIS camera's nth segment -- not the fused `GW_n`, and not
  necessarily the same physical wagon as another camera's wagon `n`;
* so every document must carry `numbering: camera-local` and
  `superseded_by_assembly: true`;
* and it must land on a DIFFERENT S3 key from the canonical fused document, or an
  early per-camera post would overwrite the authoritative one (or vice versa,
  depending on ordering) and nobody could tell which they were reading;
* a camera that seals must never be un-sealed by a delivery failure.

Nothing here touches AWS: the S3 client and `requests` are stubs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as C                                    # noqa: E402
from core.camera_evidence import CameraEvidenceBundle, LIFECYCLE   # noqa: E402
from delivery import camera_inspection as CI                        # noqa: E402

# The sequential-mode fixture already builds a sealed bundle in the exact shape
# camera_runner writes; reuse it rather than inventing a second one.
from test_camera_report_adapter import _bundle                      # noqa: E402


class _StubS3:
    def __init__(self, fail=False):
        self.uploads = []
        self._fail = fail

    def upload_file(self, local, bucket, key, ExtraArgs=None):
        if self._fail:
            raise RuntimeError("s3 down")
        self.uploads.append((bucket, key))


class _Resp:
    def __init__(self, code=200):
        self.status_code = code
        self.text = '{"run_id": "r-1"}'

    def json(self):
        return {"run_id": "r-1"}


class _StubRequests:
    def __init__(self, code=200):
        self.calls = []
        self._code = code

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        return _Resp(self._code)


def _enable():
    saved = os.environ.get("WAGONEYE_PER_CAMERA_INGEST")
    os.environ["WAGONEYE_PER_CAMERA_INGEST"] = "true"
    return saved


def _restore(saved):
    if saved is None:
        os.environ.pop("WAGONEYE_PER_CAMERA_INGEST", None)
    else:
        os.environ["WAGONEYE_PER_CAMERA_INGEST"] = saved


class TestEnableGate(unittest.TestCase):

    def test_off_by_default(self):
        saved = os.environ.pop("WAGONEYE_PER_CAMERA_INGEST", None)
        try:
            self.assertFalse(CI.is_enabled(),
                             "camera-local numbering must not publish unasked")
        finally:
            _restore(saved)

    def test_env_turns_it_on(self):
        saved = _enable()
        try:
            self.assertTrue(CI.is_enabled())
        finally:
            _restore(saved)

    def test_publish_is_a_no_op_when_disabled(self):
        saved = os.environ.pop("WAGONEYE_PER_CAMERA_INGEST", None)
        try:
            with tempfile.TemporaryDirectory() as root:
                b, _ = _bundle(root)
                s3, req = _StubS3(), _StubRequests()
                res = CI.publish(b, s3_client=s3, requests_mod=req, verbose=False)
                self.assertEqual(res.status, "disabled")
                self.assertEqual(s3.uploads, [])
                self.assertEqual(req.calls, [])
        finally:
            _restore(saved)


class TestDocumentShape(unittest.TestCase):

    def setUp(self):
        self._saved = _enable()

    def tearDown(self):
        _restore(self._saved)

    def _doc(self, cam=C.CAMERA_RIGHT_UP):
        with tempfile.TemporaryDirectory() as root:
            b, segs = _bundle(root, cam=cam)
            return CI.build_document(b, fps=15.0, total_frames=3555,
                                     raw_video_name="clip_20260729_103722_train.mp4"), segs

    def test_carries_the_v4_envelope(self):
        doc, _ = self._doc()
        for k in ("camera_id", "version", "inspection_data"):
            self.assertIn(k, doc)
        self.assertTrue(doc["camera_id"].startswith("camera_"),
                        "v1 keeps the camera_ prefix")

    def test_total_wagons_is_this_cameras_own_segment_count(self):
        doc, segs = self._doc()
        self.assertEqual(doc["inspection_data"]["total_wagons"], len(segs))

    def test_declares_its_numbering_and_that_assembly_supersedes_it(self):
        """The single most important field: without it a consumer would read
        camera-local counts as the train's canonical count."""
        doc, _ = self._doc()
        ad = doc["inspection_data"]["_adapter"]
        self.assertEqual(ad["numbering"], "camera-local")
        self.assertTrue(ad["superseded_by_assembly"])
        self.assertIn("not the fused GW_n", ad["note"].replace("  ", " "))

    def test_never_emits_a_GW_id(self):
        """The PAYLOAD must claim no global ids.

        `_adapter.note` is excluded because it deliberately mentions `GW_n` in
        prose to explain what the numbering is not -- testing the explanation
        instead of the data is what a naive substring check does.
        """
        doc, _ = self._doc()
        data = dict(doc["inspection_data"])
        data.pop("_adapter", None)
        self.assertNotIn("GW_", json.dumps(data),
                         "a pre-assembly document must not claim global ids")
        # positively: it numbers by its own segments
        self.assertEqual([w["segment_id"] for w in data["wagon_segments"]],
                         [1, 2, 3])
        self.assertEqual([w["wagon_count"] for w in data["wagon_segments"]],
                         [1, 2, 3])
        # and its evidence points at camera-local ids
        self.assertIn("L_RIGHT_UP_1", json.dumps(data))

    def test_side_camera_reports_doors(self):
        doc, _ = self._doc(C.CAMERA_RIGHT_UP)
        data = doc["inspection_data"]
        self.assertIn("doors_open", data)
        self.assertIn("doors_closed", data)

    def test_top_camera_reports_load_and_damage(self):
        doc, _ = self._doc(C.CAMERA_RIGHT_UP_TOP)
        data = doc["inspection_data"]
        self.assertIn("wagons_loaded", data)
        self.assertIn("damaged_wagons", data)

    def test_every_camera_builds(self):
        for cam in C.ALL_CAMERAS:
            doc, segs = self._doc(cam)
            self.assertEqual(doc["inspection_data"]["total_wagons"], len(segs),
                             cam)


class TestPublish(unittest.TestCase):

    def setUp(self):
        self._saved = _enable()

    def tearDown(self):
        _restore(self._saved)

    def test_uploads_and_posts_to_every_receiver(self):
        from delivery import dashboard_ingest as DASH
        with tempfile.TemporaryDirectory() as root:
            b, segs = _bundle(root)
            s3, req = _StubS3(), _StubRequests()
            res = CI.publish(b, s3_client=s3, requests_mod=req,
                             raw_video_name="clip_20260729_103722_train.mp4",
                             verbose=False)
            self.assertEqual(res.status, "ingested", res.errors)
            self.assertEqual(res.segments, len(segs))
            self.assertEqual(len(s3.uploads), 1)
            self.assertEqual(len(req.calls), len(DASH.ingest_api_urls()))

    def test_key_is_separate_from_the_canonical_fused_document(self):
        """An early per-camera post must not overwrite the authoritative one."""
        from delivery import dashboard_ingest as DASH
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            s3 = _StubS3()
            CI.publish(b, s3_client=s3, requests_mod=_StubRequests(),
                       raw_video_name="clip_20260729_103722_train.mp4",
                       verbose=False)
            _bucket, key = s3.uploads[0]
            self.assertIn("/per_camera/", key)
            ts = DASH.extract_train_timestamp("clip_20260729_103722_train.mp4")
            canonical = DASH.inspection_s3_key(
                camera=C.CAMERA_RIGHT_UP, date_folder_str="2026-07-29",
                json_name="x.json", ts=ts)
            self.assertNotEqual(key, canonical)

    def test_ingest_payload_is_v4s_three_fields(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            req = _StubRequests()
            CI.publish(b, s3_client=_StubS3(), requests_mod=req, verbose=False)
            for call in req.calls:
                self.assertEqual(set(call["json"]),
                                 {"camera_id", "inspection_s3_uri", "version"})
                self.assertIn("/per_camera/", call["json"]["inspection_s3_uri"])

    def test_writes_the_published_document_beside_the_bundle(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            res = CI.publish(b, s3_client=_StubS3(),
                             requests_mod=_StubRequests(), verbose=False)
            self.assertTrue(os.path.isfile(res.local_json))
            with open(res.local_json, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["inspection_data"]["_adapter"]
                                 ["numbering"], "camera-local")

    def test_dry_run_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            s3, req = _StubS3(), _StubRequests()
            res = CI.publish(b, s3_client=s3, requests_mod=req,
                             dry_run=True, verbose=False)
            self.assertEqual(res.status, "dry_run")
            self.assertEqual(s3.uploads, [])
            self.assertEqual(req.calls, [])

    def test_an_upload_failure_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            res = CI.publish(b, s3_client=_StubS3(fail=True),
                             requests_mod=_StubRequests(), verbose=False)
            self.assertEqual(res.status, "upload_failed")
            self.assertTrue(res.errors)

    def test_a_rejecting_receiver_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            res = CI.publish(b, s3_client=_StubS3(),
                             requests_mod=_StubRequests(code=422), verbose=False)
            self.assertEqual(res.status, "ingest_failed")

    def test_a_broken_bundle_never_raises(self):
        with tempfile.TemporaryDirectory() as root:
            b = CameraEvidenceBundle(root, C.CAMERA_RIGHT_UP)
            os.makedirs(b.dir, exist_ok=True)
            res = CI.publish(b, s3_client=_StubS3(),
                             requests_mod=_StubRequests(), verbose=False)
            self.assertIn(res.status, ("build_failed", "ingested", "uploaded",
                                       "ingest_failed"))


class TestCameraRunnerWiring(unittest.TestCase):

    def test_run_camera_takes_the_flag_and_defaults_it_off(self):
        import inspect
        from orchestrator import camera_runner
        params = inspect.signature(camera_runner.run_camera).parameters
        self.assertIn("deliver_per_camera", params)
        self.assertIs(params["deliver_per_camera"].default, False)

    def test_publishing_happens_after_the_seal(self):
        """A delivery failure must never un-seal a camera."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(here, "orchestrator", "camera_runner.py"),
                   encoding="utf-8").read()
        seal = src.index('bundle.advance("SEALED")')
        pub = src.index("camera_inspection.publish(")
        self.assertLess(seal, pub, "publish must come after the seal")

    def test_run_sequential_forwards_the_flag(self):
        import inspect
        from orchestrator import master_runner
        self.assertIn("deliver_per_camera",
                      inspect.signature(master_runner.run_sequential).parameters)

    def test_cli_flag_exists_and_defaults_off(self):
        from orchestrator import master_runner
        args = master_runner._build_parser().parse_args([])
        self.assertFalse(args.deliver_per_camera)
        args = master_runner._build_parser().parse_args(["--deliver-per-camera"])
        self.assertTrue(args.deliver_per_camera)


if __name__ == "__main__":
    unittest.main()
