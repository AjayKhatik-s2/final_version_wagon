"""Uploads go to one place, and callers learn where the file actually landed.

Ported from the V2 engine so both engines speak one Artifact Upload API
contract. The reason it is a class with a result type rather than a helper:

    "s3"   the URL is COMPUTED from bucket + key + region
    "api"  the BACKEND chooses the bucket and key and returns them, so the URL
           has to be READ OUT of the response

Any caller that computes its own URL is therefore correct in one mode and
silently wrong in the other -- the JSON looks fine, the report publishes, and
the image 404s. These tests pin that the result type is the only source of a
URL, and that `api` mode never quietly falls back to needing AWS credentials.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from delivery import artifact_uploader as AU


class FakeResp:
    def __init__(self, code=200, payload=None, text=""):
        self.status_code, self._p, self.text = code, payload, text

    def json(self):
        if self._p is None:
            raise ValueError("not json")
        return self._p


#: A real 200 from the Artifact Upload API. Named, because a bare `FakeResp()`
#: is a 200 with NO body -- which fails at `.json()` and would make a "success"
#: assertion pass or fail for a reason unrelated to what is being tested.
def ok_resp():
    return FakeResp(200, {
        "bucket": "backend-bucket", "key": "chosen/by/backend.jpg",
        "s3_uri": "s3://backend-bucket/chosen/by/backend.jpg",
        "https_url": "https://backend-bucket.s3.ap-south-1.amazonaws.com/"
                     "chosen/by/backend.jpg",
        "size_bytes": 4, "content_type": "image/jpeg"})


class FakeRequests:
    """Records each POST so the multipart contract can be asserted on."""

    def __init__(self, responses=None):
        self.calls = []
        self._r = responses if isinstance(responses, list) else None
        self._one = None if isinstance(responses, list) else responses

    def post(self, url, headers=None, data=None, files=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "data": data,
                           "files": list((files or {}).keys())})
        r = self._r.pop(0) if self._r else self._one
        if isinstance(r, Exception):
            raise r
        return r or ok_resp()


class FakeS3:
    def __init__(self):
        self.uploads = []

    def upload_file(self, local, bucket, key, ExtraArgs=None):
        self.uploads.append({"local": local, "bucket": bucket, "key": key,
                             "extra": ExtraArgs})


def a_file(tmp, name="track_1__RIGHT_UP_TOP.jpg"):
    p = os.path.join(tmp, name)
    with open(p, "wb") as f:
        f.write(b"\xff\xd8\xff\xd9")
    return p


def api_uploader(requests_mod=None, **kw):
    kw.setdefault("token", "secret-token")
    return AU.ArtifactUploader(mode=AU.MODE_API, requests_mod=requests_mod,
                               verbose=False, **kw)


class TestTheDefaultIsUnchangedBehaviour(unittest.TestCase):
    """The port must change nothing until it is switched on deliberately:
    flipping the transport moves where every artifact in the pipeline lands."""

    def test_the_default_mode_is_s3(self):
        saved = os.environ.pop("WAGONEYE_ARTIFACT_UPLOAD_MODE", None)
        try:
            self.assertEqual(AU.upload_mode(), AU.MODE_S3)
        finally:
            if saved is not None:
                os.environ["WAGONEYE_ARTIFACT_UPLOAD_MODE"] = saved

    def test_the_fallback_is_off_by_default(self):
        """`api` mode exists so the ML host makes no S3 upload calls. A default
        fallback would quietly reintroduce the credential requirement."""
        self.assertFalse(AU.fallback_to_s3())

    def test_an_s3_upload_computes_its_own_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            s3 = FakeS3()
            u = AU.ArtifactUploader(s3_client=s3, verbose=False)
            r = u.upload(a_file(tmp), "wagon_frame", camera_id="RIGHT_UP_TOP",
                         s3_bucket="my-bucket", s3_key="train_batch/k/a.jpg")
            self.assertEqual(r.via, AU.MODE_S3)
            self.assertEqual(r.bucket, "my-bucket")
            self.assertEqual(r.key, "train_batch/k/a.jpg")
            self.assertEqual(r.https_url,
                             f"https://my-bucket.s3.{C.S3_REGION}"
                             f".amazonaws.com/train_batch/k/a.jpg")
            self.assertEqual(len(s3.uploads), 1)


class TestApiModeTakesTheBackendsAnswer(unittest.TestCase):

    def test_the_url_comes_from_the_response_not_from_local_config(self):
        """The whole reason this is a class: the backend picked a different
        bucket AND a different key, and the caller must use those."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeRequests()
            u = api_uploader(fake)
            r = u.upload(a_file(tmp), "wagon_frame", camera_id="RIGHT_UP_TOP",
                         session_ts="20260724_081227",
                         s3_bucket="local-bucket",
                         s3_key="train_batch/k/local.jpg")
            self.assertEqual(r.via, AU.MODE_API)
            self.assertEqual(r.bucket, "backend-bucket")
            self.assertEqual(r.key, "chosen/by/backend.jpg")
            self.assertNotIn("local-bucket", r.https_url)
            self.assertNotIn("local.jpg", r.https_url)

    def test_it_posts_the_documented_multipart_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeRequests()
            api_uploader(fake).upload(
                a_file(tmp), "problem_frame", camera_id="LEFT_UP_TOP",
                session_ts="20260724_081227", s3_bucket="b", s3_key="k/x.jpg")
            call = fake.calls[0]
            self.assertEqual(call["headers"]["X-ML-Upload-Token"],
                             "secret-token")
            self.assertEqual(call["data"]["artifact_type"], "problem_frame")
            self.assertEqual(call["data"]["camera_id"], "LEFT_UP_TOP")
            self.assertEqual(call["data"]["session_ts"], "20260724_081227")
            self.assertEqual(call["data"]["filename"], "x.jpg")
            self.assertEqual(call["files"], ["file"])

    def test_the_filename_comes_from_the_key_not_the_temp_file(self):
        """The local file is often a temp name; the key's basename is the
        convention the backend's ingestion parses. This keeps the API-mode name
        byte-identical to the S3-mode one."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeRequests()
            api_uploader(fake).upload(
                a_file(tmp, "tmp_abc123.jpg"), "wagon_frame",
                camera_id="RIGHT_UP_TOP", session_ts="ts",
                s3_bucket="b", s3_key="a/b/track_2__RIGHT_UP_TOP.jpg")
            self.assertEqual(fake.calls[0]["data"]["filename"],
                             "track_2__RIGHT_UP_TOP.jpg")

    def test_no_s3_client_is_needed_in_api_mode(self):
        """The point of the mode: no AWS credentials on the ML host."""
        with tempfile.TemporaryDirectory() as tmp:
            u = api_uploader(FakeRequests())
            self.assertIsNone(u.s3)
            r = u.upload(a_file(tmp), "inspection_json", camera_id="RIGHT_UP",
                         session_ts="ts", s3_bucket="b", s3_key="k/x.json")
            self.assertEqual(r.via, AU.MODE_API)

    def test_a_response_without_bucket_or_key_is_an_error(self):
        """Without both, nothing downstream can reference the file. A loud
        failure beats a document carrying a link to nowhere."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeRequests(FakeResp(200, {"https_url": "https://x/y"}))
            with self.assertRaises(AU.ArtifactUploadError):
                api_uploader(fake).upload(
                    a_file(tmp), "wagon_frame", camera_id="c",
                    session_ts="ts", s3_bucket="b", s3_key="k/x.jpg")


class TestConfigurationErrorsFailEarly(unittest.TestCase):

    def test_api_mode_without_a_token_is_rejected_at_construction(self):
        """One train publishes hundreds of frames; discovering a missing token
        per upload means discovering it hundreds of times."""
        with self.assertRaises(ValueError):
            AU.ArtifactUploader(mode=AU.MODE_API, token="", verbose=False)

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            AU.ArtifactUploader(mode="ftp", verbose=False)

    def test_an_unknown_artifact_type_fails_before_the_wire(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                api_uploader(FakeRequests()).upload(
                    a_file(tmp), "not_a_type", camera_id="c", session_ts="ts",
                    s3_bucket="b", s3_key="k/x.jpg")

    def test_a_missing_session_ts_is_caught_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                api_uploader(FakeRequests()).upload(
                    a_file(tmp), "wagon_frame", camera_id="c",
                    s3_bucket="b", s3_key="k/x.jpg")

    def test_a_missing_file_is_caught_locally(self):
        with self.assertRaises(FileNotFoundError):
            api_uploader(FakeRequests()).upload(
                "/nope/x.jpg", "wagon_frame", camera_id="c", session_ts="ts",
                s3_bucket="b", s3_key="k/x.jpg")

    def test_the_twelve_documented_types_are_accepted(self):
        self.assertEqual(len(AU.ARTIFACT_TYPES), 12)
        for t in ("trimmed_video", "detected_video", "inspection_pdf",
                  "wagon_frame", "loco_frame", "problem_frame",
                  "wagon_number_frame", "loco_number_frame", "inspection_json",
                  "combined_report_pdf", "pipeline_state", "combiner_state"):
            self.assertIn(t, AU.ARTIFACT_TYPES)


class TestRetryOnlyWhatCouldSucceed(unittest.TestCase):

    def test_a_4xx_is_never_retried(self):
        """The identical request would be rejected identically, so retrying
        only sleeps through a backoff that cannot help."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeRequests(FakeResp(401, None, "bad token"))
            u = api_uploader(fake, max_attempts=3)
            with self.assertRaises(AU.ArtifactUploadError):
                u.upload(a_file(tmp), "wagon_frame", camera_id="c",
                         session_ts="ts", s3_bucket="b", s3_key="k/x.jpg")
            self.assertEqual(len(fake.calls), 1)

    def test_a_5xx_is_retried_then_gives_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeRequests([FakeResp(500, None, "boom")] * 3)
            u = api_uploader(fake, max_attempts=3)
            u._BACKOFF = (0.0,)
            with unittest.mock.patch("time.sleep"):
                with self.assertRaises(AU.ArtifactUploadError):
                    u.upload(a_file(tmp), "wagon_frame", camera_id="c",
                             session_ts="ts", s3_bucket="b", s3_key="k/x.jpg")
            self.assertEqual(len(fake.calls), 3)

    def test_a_transient_failure_then_success_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeRequests([FakeResp(500, None, "boom"), ok_resp()])
            with unittest.mock.patch("time.sleep"):
                r = api_uploader(fake, max_attempts=3).upload(
                    a_file(tmp), "wagon_frame", camera_id="c", session_ts="ts",
                    s3_bucket="b", s3_key="k/x.jpg")
            self.assertEqual(r.via, AU.MODE_API)
            self.assertEqual(len(fake.calls), 2)

    def test_a_4xx_is_classed_permanent_and_a_5xx_transient(self):
        self.assertTrue(issubclass(AU.PermanentUploadError,
                                   AU.ArtifactUploadError))


class TestFallbackIsOptOut(unittest.TestCase):

    def test_without_fallback_a_failed_api_upload_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            s3 = FakeS3()
            fake = FakeRequests(FakeResp(401, None, "no"))
            u = api_uploader(fake, s3_client=s3, fallback=False)
            with self.assertRaises(AU.ArtifactUploadError):
                u.upload(a_file(tmp), "wagon_frame", camera_id="c",
                         session_ts="ts", s3_bucket="b", s3_key="k/x.jpg")
            self.assertEqual(s3.uploads, [], "fell back to S3 unasked")

    def test_with_fallback_it_writes_to_s3_instead(self):
        with tempfile.TemporaryDirectory() as tmp:
            s3 = FakeS3()
            fake = FakeRequests(FakeResp(401, None, "no"))
            u = api_uploader(fake, s3_client=s3, fallback=True)
            r = u.upload(a_file(tmp), "wagon_frame", camera_id="c",
                         session_ts="ts", s3_bucket="b", s3_key="k/x.jpg")
            self.assertEqual(r.via, AU.MODE_S3)
            self.assertEqual(len(s3.uploads), 1)


class TestArtifactsWhoseKeyWeRecompute(unittest.TestCase):
    """An artifact this pipeline READS BACK by recomputing its key cannot go
    through the API: nothing in the contract says what key the backend chose, so
    it could never be found again."""

    def test_state_always_goes_straight_to_s3(self):
        with tempfile.TemporaryDirectory() as tmp:
            s3 = FakeS3()
            fake = FakeRequests()
            u = api_uploader(fake, s3_client=s3)
            for t in ("pipeline_state", "combiner_state"):
                r = u.upload(a_file(tmp, f"{t}.json"), t, camera_id="c",
                             s3_bucket="b", s3_key=f"state/{t}.json")
                self.assertEqual(r.via, AU.MODE_S3, t)
            self.assertEqual(fake.calls, [], "state was sent to the API")
            self.assertEqual(len(s3.uploads), 2)

    def test_state_needs_no_session_ts(self):
        with tempfile.TemporaryDirectory() as tmp:
            u = api_uploader(FakeRequests(), s3_client=FakeS3())
            r = u.upload(a_file(tmp, "pipeline_state.json"), "pipeline_state",
                         camera_id="c", s3_bucket="b", s3_key="s/x.json")
            self.assertEqual(r.via, AU.MODE_S3)


class TestReadingTheBucketBackOutOfAUrl(unittest.TestCase):
    """An API-uploaded artifact lives in whatever bucket the backend chose, and
    its URL is what gets recorded. Reading the bucket back out of that URL keeps
    a later consumer correct even if the bucket changes."""

    def test_virtual_hosted_style(self):
        b, k = AU.parse_s3_object_url(
            "https://my-bucket.s3.ap-south-1.amazonaws.com/a/b/c.jpg")
        self.assertEqual((b, k), ("my-bucket", "a/b/c.jpg"))

    def test_path_style(self):
        b, k = AU.parse_s3_object_url(
            "https://s3.ap-south-1.amazonaws.com/my-bucket/a/b/c.jpg")
        self.assertEqual((b, k), ("my-bucket", "a/b/c.jpg"))

    def test_it_round_trips_an_s3_mode_url(self):
        url = AU.s3_object_url("my-bucket", "train_batch/k/evidence/x.jpg")
        self.assertEqual(AU.parse_s3_object_url(url),
                         ("my-bucket", "train_batch/k/evidence/x.jpg"))

    def test_a_non_s3_url_is_empty_not_an_exception(self):
        for bad in ("", "not a url", "https://example.com/a.jpg", None):
            self.assertEqual(AU.parse_s3_object_url(bad or ""), ("", ""))


if __name__ == "__main__":
    import unittest.mock  # noqa: F401
    unittest.main(verbosity=2)


class TestEveryWagonsFrameGetsItsOwnObject(unittest.TestCase):
    """A collision measured against the live endpoint, not hypothesised.

    The backend builds its key as `<camera_id>/<session_ts>/<type>/<filename>`
    -- with NO wagon id. Per-wagon evidence names repeat by design:
    `door/left_best.jpg`, `ocr/best_frame.jpg` and `load/best_frame.jpg` each
    exist once per wagon. Sending the bare basename made two files land on ONE
    key, so on a 59-wagon train every wagon's door frame would have collapsed
    into a single object, last-write-wins.

    Invisible from outside: every upload returns 200, every link resolves, every
    image renders -- and all 59 wagons show whichever photo landed last.
    """

    def test_two_wagons_with_the_same_evidence_name_get_different_filenames(self):
        from delivery.s3_upload import _unique_filename
        a = _unique_filename("GW_25/door/left_best.jpg")
        b = _unique_filename("GW_7/door/left_best.jpg")
        self.assertNotEqual(a, b)
        self.assertEqual(a, "GW_25__left_best.jpg")
        self.assertEqual(b, "GW_7__left_best.jpg")

    def test_the_original_name_is_kept_as_the_tail(self):
        """Anything parsing a suffix -- `track_1__RIGHT_UP_TOP.jpg` -- still
        sees what it expects."""
        got = _unique = None
        from delivery.s3_upload import _unique_filename
        got = _unique_filename("GW_25/damage/track_1__RIGHT_UP_TOP.jpg")
        self.assertEqual(got, "GW_25__track_1__RIGHT_UP_TOP.jpg")
        self.assertTrue(got.endswith("track_1__RIGHT_UP_TOP.jpg"))

    def test_a_whole_train_of_repeated_names_stays_distinct(self):
        from delivery.s3_upload import _unique_filename
        names = {_unique_filename(f"GW_{i}/door/left_best.jpg")
                 for i in range(1, 60)}
        self.assertEqual(len(names), 59)

    def test_a_flat_path_is_left_alone(self):
        from delivery.s3_upload import _unique_filename
        self.assertEqual(_unique_filename("RIGHT_UP_processed.mp4"),
                         "RIGHT_UP_processed.mp4")

    def test_the_tree_upload_sends_the_unique_name(self):
        import tempfile as _tf
        from delivery import s3_upload
        with _tf.TemporaryDirectory() as tmp:
            for gw in ("GW_7", "GW_25"):
                d = os.path.join(tmp, gw, "door")
                os.makedirs(d)
                with open(os.path.join(d, "left_best.jpg"), "wb") as f:
                    f.write(b"\xff\xd8\xff\xd9")
            fake = FakeRequests()
            up = api_uploader(fake)
            s3_upload.upload_tree_detailed(None, tmp, "K",
                                           sub_prefix="evidence",
                                           session_ts="K", uploader=up)
            sent = sorted(c["data"]["filename"] for c in fake.calls)
            self.assertEqual(sent, ["GW_25__left_best.jpg",
                                    "GW_7__left_best.jpg"])

    def test_the_full_prefixed_camera_id_is_sent(self):
        """Verified live: the camera_id becomes a FOLDER in the backend's key,
        so the short form would file artifacts under a folder that exists
        nowhere else."""
        from delivery.s3_upload import _camera_hint
        self.assertEqual(_camera_hint("GW_25/damage/track_1__LEFT_UP_TOP.jpg"),
                         C.CAMERA_S3_FOLDER[C.CAMERA_LEFT_UP_TOP])
        self.assertTrue(_camera_hint(
            "GW_1/damage/RIGHT_UP/x.jpg").startswith("camera_CCTV"))


class TestNoDirectS3UploadsRemain(unittest.TestCase):
    """The point of api mode: no AWS credentials on the ML host. A single
    surviving `s3_client.upload_file` call defeats it."""

    def test_no_delivery_module_calls_upload_file_directly(self):
        import ast as _ast
        offenders = []
        d = os.path.join(V4_ROOT, "delivery")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py") or fn == "artifact_uploader.py":
                continue
            src = open(os.path.join(d, fn), encoding="utf-8").read()
            for n in _ast.walk(_ast.parse(src)):
                if (isinstance(n, _ast.Call)
                        and isinstance(n.func, _ast.Attribute)
                        and n.func.attr in ("upload_file", "upload_fileobj",
                                            "put_object")):
                    offenders.append(f"{fn}:{n.lineno}")
        self.assertEqual(offenders, [],
                         f"direct S3 upload calls remain: {offenders}")

    def test_the_uploader_is_the_only_module_that_calls_upload_file(self):
        import ast as _ast
        src = open(os.path.join(V4_ROOT, "delivery/artifact_uploader.py"),
                   encoding="utf-8").read()
        calls = [n for n in _ast.walk(_ast.parse(src))
                 if isinstance(n, _ast.Call)
                 and isinstance(n.func, _ast.Attribute)
                 and n.func.attr == "upload_file"]
        self.assertEqual(len(calls), 1, "s3 transport should have exactly one")
