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

import inspect
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

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

    def test_key_matches_the_canonical_one_so_assembly_replaces_it(self):
        """The provisional post targets the object assembly will overwrite.

        This test used to assert the OPPOSITE -- a `per_camera/` sidecar that
        could never be overwritten. The requirement changed: a camera should
        publish early and then be REPLACED by the combined result, so the
        dashboard carries one record per camera per train that gets upgraded
        rather than two records that must be reconciled by the reader. The old
        layout is still reachable via `WAGONEYE_PER_CAMERA_KEY_MODE=sidecar`
        and is pinned by the test below.
        """
        from delivery import dashboard_ingest as DASH
        clip = "clip_20260729_103722_train.mp4"
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            s3 = _StubS3()
            CI.publish(b, s3_client=s3, requests_mod=_StubRequests(),
                       raw_video_name=clip, batch_key="20260729_103722",
                       verbose=False)
            _bucket, key = s3.uploads[0]
            self.assertNotIn("/per_camera/", key)
            ts = DASH.extract_train_timestamp("20260729_103722")
            canonical = DASH.inspection_s3_key(
                camera=C.CAMERA_RIGHT_UP,
                date_folder_str=DASH.date_folder(ts),
                json_name=f"{clip.rsplit(".", 1)[0]}_inspection.json", ts=ts)
            self.assertEqual(key, canonical)

    def test_sidecar_mode_still_keeps_the_two_documents_apart(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            s3 = _StubS3()
            with mock.patch.dict(os.environ,
                                 {"WAGONEYE_PER_CAMERA_KEY_MODE": "sidecar"}):
                res = CI.publish(b, s3_client=s3,
                                 requests_mod=_StubRequests(),
                                 raw_video_name="clip_20260729_103722_train.mp4",
                                 batch_key="20260729_103722", verbose=False)
            _bucket, key = s3.uploads[0]
            self.assertIn("/per_camera/", key)
            self.assertEqual(res.key_mode, CI.KEY_MODE_SIDECAR)

    def test_ingest_payload_is_v4s_three_fields(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            req = _StubRequests()
            CI.publish(b, s3_client=_StubS3(), requests_mod=req, verbose=False)
            for call in req.calls:
                self.assertEqual(set(call["json"]),
                                 {"camera_id", "inspection_s3_uri", "version"})
                self.assertTrue(
                    call["json"]["inspection_s3_uri"].endswith(
                        "inspection_data.json"),
                    "the POST must point at the object assembly replaces")

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


# ---------------------------------------------------------------------------
# Replacement: the provisional post and the fused one must share a key
# ---------------------------------------------------------------------------

class TestPerCameraPostIsReplacedInPlace(unittest.TestCase):
    """A camera posts early, assembly overwrites it with the canonical answer.

    That only works if both write the SAME S3 object. The fused path builds its
    key from `extract_train_timestamp(batch_key)` -- one value for all four
    cameras -- so the per-camera path has to do the same. Deriving it from the
    camera's own clip name instead leaves three of four cameras on keys the
    fused pass never touches, and the dashboard keeps a stale camera-local
    record beside the real one.
    """

    #: One real train from 2026-07-31: the four clips are stamped up to 30
    #: seconds apart, which is exactly what breaks a filename-derived key.
    CLIPS = {
        C.CAMERA_RIGHT_UP:     "camera_CCTV_HZBN_DHN_2_RIGHT_UP_20260731_052218_train.mp4",
        C.CAMERA_LEFT_UP:      "camera_CCTV_HZBN_DHN_1_LEFT_UP_20260731_052211_train.mp4",
        C.CAMERA_RIGHT_UP_TOP: "camera_CCTV_HZBN_DHN_5_RIGHT_TOP_20260731_052227_train.mp4",
        C.CAMERA_LEFT_UP_TOP:  "camera_CCTV_HZBN_DHN_6_LEFT_TOP_20260731_052241_train.mp4",
    }
    BATCH_KEY = "20260731_052211"

    def _fused_key(self, cam, raw_video_name):
        """The key Stage 6b will write, reproduced from dashboard_ingest."""
        from delivery import dashboard_ingest as DASH
        ts = DASH.extract_train_timestamp(self.BATCH_KEY)
        json_name = f"{os.path.splitext(raw_video_name)[0]}_inspection.json"
        return DASH.inspection_s3_key(camera=cam,
                                      date_folder_str=DASH.date_folder(ts),
                                      json_name=json_name, ts=ts)

    def _per_camera_key(self, cam, raw_video_name, batch_key):
        from delivery import camera_inspection as CI
        from delivery import dashboard_ingest as DASH

        class _B:
            camera_id = cam
            dir = os.path.join("/tmp", "camera_evidence", cam)

        ts = CI._train_ts(batch_key, raw_video_name, _B())
        json_name = f"{os.path.splitext(raw_video_name)[0]}_inspection.json"
        return DASH.inspection_s3_key(camera=cam,
                                      date_folder_str=DASH.date_folder(ts),
                                      json_name=json_name, ts=ts)

    def test_the_two_keys_are_identical_for_every_camera(self):
        for cam, clip in self.CLIPS.items():
            with self.subTest(camera=cam):
                self.assertEqual(
                    self._per_camera_key(cam, clip, self.BATCH_KEY),
                    self._fused_key(cam, clip),
                    f"{cam}'s provisional post would not be replaced by "
                    f"assembly")

    def test_a_clip_derived_key_would_NOT_collide(self):
        """Documents why the batch key is passed -- the old rule really failed.

        Three of the four cameras have a clip timestamp different from the batch
        anchor, so a filename-derived key misses the fused object.
        """
        missed = []
        for cam, clip in self.CLIPS.items():
            if self._per_camera_key(cam, clip, "") != self._fused_key(cam, clip):
                missed.append(cam)
        self.assertEqual(len(missed), 3,
                         f"expected 3 cameras to miss, got {missed}")
        self.assertNotIn(C.CAMERA_LEFT_UP, missed,
                         "LEFT_UP set the cluster anchor, so it alone matched")

    def test_all_four_cameras_get_distinct_keys(self):
        """Replacement must not make two cameras collide with each other."""
        keys = {self._per_camera_key(cam, clip, self.BATCH_KEY)
                for cam, clip in self.CLIPS.items()}
        self.assertEqual(len(keys), 4)

    def test_replace_is_the_default_mode(self):
        from delivery import camera_inspection as CI
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WAGONEYE_PER_CAMERA_KEY_MODE", None)
            self.assertEqual(CI.key_mode(), CI.KEY_MODE_REPLACE)

    def test_sidecar_mode_restores_the_side_by_side_layout(self):
        from delivery import camera_inspection as CI
        with mock.patch.dict(os.environ,
                             {"WAGONEYE_PER_CAMERA_KEY_MODE": "sidecar"}):
            self.assertEqual(CI.key_mode(), CI.KEY_MODE_SIDECAR)

    def test_the_document_says_it_is_provisional(self):
        from delivery import camera_inspection as CI
        self.assertIn("provisional", inspect.getsource(CI.build_document))
        self.assertIn("replaced_in_place_by_assembly",
                      inspect.getsource(CI.build_document))

    def test_camera_runner_passes_the_train_id_as_the_batch_key(self):
        from orchestrator import camera_runner
        src = inspect.getsource(camera_runner.run_camera)
        block = src[src.index("camera_inspection.publish"):]
        self.assertIn("batch_key=train_id", block)
