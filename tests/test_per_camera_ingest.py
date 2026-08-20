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


def _json_upload(s3):
    """The DOCUMENT upload among s3.uploads.

    `publish()` uploads the referenced evidence JPEGs before the JSON, so the
    document is no longer `uploads[0]`. Selecting it by extension keeps these
    tests indifferent to how many images a fixture happens to reference.
    """
    hits = [(b, k) for b, k in s3.uploads if k.endswith(".json")]
    assert len(hits) == 1, f"expected exactly one JSON upload, got {hits}"
    return hits[0]


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
            # One document, plus however many evidence images it references.
            self.assertEqual(len([k for _b, k in s3.uploads
                                  if k.endswith(".json")]), 1)
            self.assertTrue(all(k.endswith((".json", ".jpg"))
                                for _b, k in s3.uploads))
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
            _bucket, key = _json_upload(s3)
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
            _bucket, key = _json_upload(s3)
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


class TestStagingPrefixDoesNotBreakReplacement(unittest.TestCase):
    """Historical/batch mode stages clips as `<CAM>_<original>`.

    The fused pass names its document from the S3 URL, so it never sees that
    prefix. Under the v1 key layout the filename IS part of the key, so a
    prefixed name would put the provisional document beside the canonical one
    instead of under it.
    """

    ORIGINAL = "camera_CCTV_HZBN_DHN_2_RIGHT_UP_20260731_052218_train.mp4"
    STAGED = f"{C.CAMERA_RIGHT_UP}_{ORIGINAL}"
    BATCH_KEY = "20260731_052211"

    def setUp(self):
        # publish() is a no-op unless per-camera ingest is switched on.
        self._saved = _enable()

    def tearDown(self):
        _restore(self._saved)

    def test_the_staging_prefix_is_stripped(self):
        self.assertEqual(
            CI.source_video_name(self.STAGED, C.CAMERA_RIGHT_UP),
            self.ORIGINAL)

    def test_an_unstaged_name_is_untouched(self):
        """`--local-only` files are never prefixed."""
        self.assertEqual(
            CI.source_video_name(self.ORIGINAL, C.CAMERA_RIGHT_UP),
            self.ORIGINAL)

    def test_the_camera_id_inside_the_name_is_not_damaged(self):
        """A blunt replace would eat `_RIGHT_UP_` from the middle."""
        out = CI.source_video_name(self.STAGED, C.CAMERA_RIGHT_UP)
        self.assertIn("_2_RIGHT_UP_20260731", out,
                      "the camera id occurring INSIDE the name was mangled")
        self.assertEqual(out.count("RIGHT_UP"), 1)

    def test_a_top_camera_prefix_is_handled(self):
        """LEFT_UP is a prefix of LEFT_UP_TOP -- strip only this camera's own."""
        original = "camera_CCTV_HZBN_DHN_6_LEFT_TOP_20260731_052241_train.mp4"
        staged = f"{C.CAMERA_LEFT_UP_TOP}_{original}"
        self.assertEqual(
            CI.source_video_name(staged, C.CAMERA_LEFT_UP_TOP), original)
        # LEFT_UP must NOT strip LEFT_UP_TOP's prefix.
        self.assertEqual(
            CI.source_video_name(staged, C.CAMERA_LEFT_UP),
            "TOP_" + original)

    def test_empty_inputs_are_safe(self):
        self.assertEqual(CI.source_video_name("", C.CAMERA_RIGHT_UP), "")
        self.assertEqual(CI.source_video_name(self.STAGED, ""), self.STAGED)

    def _keys(self, staged_name, layout):
        """(per-camera key, fused key) for one camera under `layout`."""
        from delivery import dashboard_ingest as DASH

        class _B:
            camera_id = C.CAMERA_RIGHT_UP
            dir = "/x/camera_evidence/RIGHT_UP"

        with mock.patch.dict(os.environ,
                             {"WAGONEYE_INSPECTION_KEY_LAYOUT": layout}):
            # per-camera, as publish() now derives it
            name = CI.source_video_name(staged_name, C.CAMERA_RIGHT_UP)
            ts_p = CI._train_ts(self.BATCH_KEY, name, _B())
            jn_p = f"{os.path.splitext(name)[0]}_inspection.json"
            per = DASH.inspection_s3_key(
                camera=C.CAMERA_RIGHT_UP,
                date_folder_str=DASH.date_folder(ts_p),
                json_name=jn_p, ts=ts_p)
            # fused, which names the document from the S3 URL
            ts_f = DASH.extract_train_timestamp(self.BATCH_KEY)
            jn_f = f"{os.path.splitext(self.ORIGINAL)[0]}_inspection.json"
            fused = DASH.inspection_s3_key(
                camera=C.CAMERA_RIGHT_UP,
                date_folder_str=DASH.date_folder(ts_f),
                json_name=jn_f, ts=ts_f)
        return per, fused

    def test_v1_layout_keys_collide_despite_the_staging_prefix(self):
        per, fused = self._keys(self.STAGED, "v1")
        self.assertEqual(per, fused)
        self.assertTrue(per.startswith("Right_up/"),
                        f"expected the dashboard folder layout, got {per}")
        self.assertNotIn("RIGHT_UP_camera_", per)

    def test_v4_layout_keys_collide_too(self):
        per, fused = self._keys(self.STAGED, "v4")
        self.assertEqual(per, fused)
        self.assertTrue(per.endswith("/inspection_data.json"))

    def test_without_the_strip_the_v1_keys_would_differ(self):
        """Proof the fix is load-bearing, not decorative."""
        from delivery import dashboard_ingest as DASH
        with mock.patch.dict(os.environ,
                             {"WAGONEYE_INSPECTION_KEY_LAYOUT": "v1"}):
            ts = DASH.extract_train_timestamp(self.BATCH_KEY)
            df = DASH.date_folder(ts)
            unstripped = DASH.inspection_s3_key(
                camera=C.CAMERA_RIGHT_UP, date_folder_str=df,
                json_name=f"{os.path.splitext(self.STAGED)[0]}_inspection.json",
                ts=ts)
            fused = DASH.inspection_s3_key(
                camera=C.CAMERA_RIGHT_UP, date_folder_str=df,
                json_name=f"{os.path.splitext(self.ORIGINAL)[0]}_inspection.json",
                ts=ts)
        self.assertNotEqual(unstripped, fused)

    def test_the_document_reports_the_source_clip_not_the_staged_copy(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            doc = CI.build_document(b, raw_video_name=self.STAGED,
                                    batch_key=self.BATCH_KEY)
            self.assertEqual(doc["inspection_data"]["raw_video_name"],
                             self.ORIGINAL)

    def test_publish_uploads_under_the_unprefixed_name(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            s3 = _StubS3()
            with mock.patch.dict(os.environ,
                                 {"WAGONEYE_INSPECTION_KEY_LAYOUT": "v1"}):
                CI.publish(b, s3_client=s3, requests_mod=_StubRequests(),
                           raw_video_name=self.STAGED,
                           batch_key=self.BATCH_KEY, verbose=False)
            _bucket, key = _json_upload(s3)
            self.assertNotIn("RIGHT_UP_camera_", key)
            self.assertIn("camera_CCTV_HZBN_DHN_2_RIGHT_UP_20260731_052218",
                          key)


class TestProvisionalAndFusedShareOneIdentity(unittest.TestCase):
    """The provisional post and the canonical one must be the SAME event.

    The receiver snapshots on POST -- it records what it fetched and never
    re-reads the S3 object. So overwriting the object is not enough: unless both
    POSTs carry the same idempotency key, the receiver mints a separate run for
    each and the dashboard ends up holding two records per camera per train.

    Measured on 2026-07-22 before this was fixed: every camera of every train
    produced two runs, and the camera-local one (59 segments) was displayed
    instead of the fused count (54).
    """

    BATCH_KEY = "20260722_050704"

    def test_key_ignores_the_content_hash(self):
        from delivery.dashboard_ingest import ingest_idempotency_key as k
        self.assertEqual(k(self.BATCH_KEY, C.CAMERA_RIGHT_UP, 0, "aaa"),
                         k(self.BATCH_KEY, C.CAMERA_RIGHT_UP, 0, "bbb"))

    def test_key_ignores_the_report_revision(self):
        """A corrected report updates the record; it is not a new record."""
        from delivery.dashboard_ingest import ingest_idempotency_key as k
        self.assertEqual(k(self.BATCH_KEY, C.CAMERA_RIGHT_UP, 0),
                         k(self.BATCH_KEY, C.CAMERA_RIGHT_UP, 7))

    def test_key_still_separates_cameras_and_trains(self):
        from delivery.dashboard_ingest import ingest_idempotency_key as k
        keys = {k(t, c) for t in ("T1", "T2")
                for c in (C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP)}
        self.assertEqual(len(keys), 4)

    def test_the_provisional_post_uses_the_TRAIN_key_not_the_camera_dir(self):
        """The bug that made a match impossible in principle.

        `publish()` passed `os.path.basename(bundle.dir)` as the batch key, and
        `bundle.dir` is `<evidence_root>/<CAMERA>` -- so the provisional post
        identified itself as train "RIGHT_UP" while assembly identified the same
        result as train "20260722_050704".
        """
        import inspect as _inspect
        src = _inspect.getsource(CI.publish)
        block = src[src.index("ingest_idempotency_key"):]
        self.assertIn("batch_key", block.split(")")[0] + ")",
                      "the provisional key is not derived from the train key")

    def test_provisional_and_fused_keys_are_identical(self):
        """End to end: the two code paths must agree on the identity."""
        from delivery.dashboard_ingest import ingest_idempotency_key as k
        # what dashboard_ingest.run() computes for the fused document
        fused = k(self.BATCH_KEY, C.CAMERA_RIGHT_UP, 0, "fused-sha")
        # what camera_inspection.publish() computes for the provisional one
        provisional = k(self.BATCH_KEY, C.CAMERA_RIGHT_UP)
        self.assertEqual(provisional, fused)

    def test_the_local_ledger_still_skips_unchanged_content(self):
        """Removing the sha from the KEY must not break re-delivery skipping.

        `run()` compares `json_sha256` against its own ledger directly, so that
        behaviour is independent of the idempotency key.
        """
        import inspect as _inspect
        from delivery import dashboard_ingest as DASH
        # `run` is a thin never-raises wrapper; the ledger lives in _run_inner.
        src = _inspect.getsource(DASH._run_inner)
        self.assertIn('pj.get("json_sha256") == json_sha', src)
        self.assertIn("already_ingested", src)


class TestProvisionalImagesAreActuallyUploaded(unittest.TestCase):
    """A document must not name an S3 object nobody created.

    `build_document` minted evidence URLs under
    `<folder>/<date>/camera_evidence/...` in the inspection bucket, checking only
    that the file existed LOCALLY -- while `publish()` uploaded the JSON alone.
    Every thumbnail in a provisional document was therefore a 404. Confirmed on
    2026-07-22: the fused document's images returned HTTP 200 from the report
    bucket, while the provisional document referenced camera_evidence objects
    that did not exist.
    """

    def setUp(self):
        self._saved = _enable()

    def tearDown(self):
        _restore(self._saved)

    def test_every_referenced_image_is_uploaded(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            s3 = _StubS3()
            CI.publish(b, s3_client=s3, requests_mod=_StubRequests(),
                       raw_video_name="clip_20260729_103722_train.mp4",
                       batch_key="20260729_103722", verbose=False)

            uploaded = {key for _bucket, key in s3.uploads}
            doc_urls = set()
            with open(
                [k for k in
                 [os.path.join(dp, f) for dp, _d, fs in os.walk(b.dir)
                  for f in fs] if k.endswith("_inspection.json")][0],
                encoding="utf-8") as f:
                import re
                doc_urls = set(re.findall(r'https://[^"]+\.jpg', f.read()))

            missing = [u for u in doc_urls
                       if u.split(".amazonaws.com/", 1)[-1] not in uploaded]
            self.assertEqual(missing, [],
                             f"{len(missing)} referenced image(s) were never "
                             f"uploaded")

    def test_the_result_reports_the_image_count(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            res = CI.publish(b, s3_client=_StubS3(),
                             requests_mod=_StubRequests(),
                             batch_key="20260729_103722", verbose=False)
            self.assertGreaterEqual(res.assets_uploaded, 0)
            self.assertEqual(res.assets_failed, 0, res.errors)

    def test_images_are_uploaded_before_the_json(self):
        """Publishing the document first would show a broken page meanwhile."""
        import inspect as _inspect
        src = _inspect.getsource(CI.publish)
        self.assertLess(src.index("_upload_assets"),
                        src.index("upload_file(res.local_json"))

    def test_duplicate_references_upload_once(self):
        from delivery.camera_inspection import _upload_assets

        class _S3:
            def __init__(self): self.n = 0
            def upload_file(self, *a, **k): self.n += 1

        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "a.jpg")
            open(f, "wb").write(b"x")
            s3 = _S3()
            ok, failed = _upload_assets(s3, "b", [(f, "k/a.jpg")] * 5)
            self.assertEqual((ok, failed, s3.n), (1, 0, 1))

    def test_a_failed_image_does_not_stop_publication(self):
        from delivery.camera_inspection import _upload_assets

        class _S3:
            def upload_file(self, *a, **k): raise RuntimeError("denied")

        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "a.jpg")
            open(f, "wb").write(b"x")
            ok, failed = _upload_assets(_S3(), "b", [(f, "k/a.jpg")])
            self.assertEqual((ok, failed), (0, 1))

    def test_a_missing_local_file_is_counted_not_raised(self):
        from delivery.camera_inspection import _upload_assets
        ok, failed = _upload_assets(_StubS3(), "b", [("/nope/x.jpg", "k.jpg")])
        self.assertEqual((ok, failed), (0, 1))

    def test_build_document_without_assets_still_works(self):
        """The parameter is optional; existing callers are unchanged."""
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            doc = CI.build_document(b, batch_key="20260729_103722")
            self.assertIn("inspection_data", doc)
