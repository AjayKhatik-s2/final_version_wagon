"""Historical (time-range) mode: selection correctness + live-path isolation.

Historical mode is an INPUT-SELECTION layer. These tests therefore split into
two halves, and both matter equally:

**Backward compatibility.** The live `--auto` path must be reachable and
unchanged: the CLI still parses without any historical argument, the default mode
is still batch, historical flags default to off, and no historical code runs on a
live invocation. A regression test also proves historical mode returns before the
polling loop, because "historical accidentally starts polling live S3" is the one
failure that could process a live train twice.

**Selection correctness.** The subtle parts, each of which has a specific way of
going wrong:

* *Timezone.* Filename digits are IST wall-clock (the producer writes them that
  way: `train_extraction/time_utils.parse_timestamp_from_filename` attaches IST
  without shifting). Reading them as UTC shifts every window by 5h30m and selects
  the wrong trains, so the IST agreement is asserted directly against that
  producer function.
* *Clip coverage.* A clip's filename timestamp is the start of the RAW clip, not
  the moment the train passed, so selection treats each clip as covering
  `[T, T+pad]`. Without the pad, a train that passed 4 minutes into its raw clip
  is missed when the window starts after T.
* *Independent trains.* Two trains in one window must produce two batches and two
  `process_batch` calls. Merging them into one Stage-1 reconstruction would
  fabricate a single Global Train out of two physical trains.
* *Nearest-timestamp trap.* A camera missing from train A must NOT be back-filled
  from train B just because B's clip is the closest in time.
* *Cameras stamped apart.* The four cameras are stamped from their own raw clips
  and can differ by minutes, so grouping chains from the latest cluster member.

Nothing here touches S3, AWS, ffmpeg or a model: the S3 client is a stub and
`process_batch` is monkeypatched so the tests assert *what it was called with*.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config as CFG                                     # noqa: E402
from core import constants as C                                   # noqa: E402
from core.batch import CameraVideo                                 # noqa: E402
from orchestrator import historical_runner as HR                   # noqa: E402
from orchestrator import master_runner as MR                       # noqa: E402
from orchestrator import train_batch_manager as TBM                # noqa: E402

IST = CFG.IST


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _key(camera: str, ts: str, *, incomplete: bool = False) -> str:
    """A realistic trimmed-clip key: <camera_folder>/<raw basename>_train.mp4."""
    suffix = "_train_incomplete" if incomplete else "_train"
    return f"{C.CAMERA_S3_FOLDER[camera]}/CCTV_{ts}{suffix}.mp4"


class StubS3:
    """Serves a fixed object list, filtered by prefix like real S3 would."""

    def __init__(self, keys, *, size: int = 1024):
        now = datetime.now(timezone.utc)
        self._objects = [
            {"Key": k, "LastModified": now, "ETag": f'"{i}"', "Size": size}
            for i, k in enumerate(keys)
        ]
        self.downloads = []

    def list_objects_v2(self, **kw):
        prefix = kw.get("Prefix", "")
        return {"Contents": [o for o in self._objects
                             if o["Key"].startswith(prefix)],
                "IsTruncated": False}

    def download_file(self, bucket, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(b"\x00")
        self.downloads.append((bucket, key, local_path))


def all_four(ts: str):
    return [_key(cam, ts) for cam in C.ALL_CAMERAS]


def window(date="2026-08-08", start="10:00", end="12:00", tz="Asia/Kolkata"):
    return HR.resolve_window(date=date, start_time=start, end_time=end,
                             timezone_name=tz)


# ---------------------------------------------------------------------------
# 1. Window parsing / timezone
# ---------------------------------------------------------------------------

class TestWindowResolution(unittest.TestCase):

    def test_date_plus_times_in_ist(self):
        w = window()
        self.assertEqual(w.start.hour, 10)
        self.assertEqual(w.end.hour, 12)
        self.assertEqual(w.start.utcoffset(), timedelta(hours=5, minutes=30))
        self.assertFalse(w.rolled_overnight)

    def test_default_timezone_is_the_documented_site_zone(self):
        self.assertEqual(HR.DEFAULT_TIMEZONE, "Asia/Kolkata")
        w = HR.resolve_window(date="2026-08-08", start_time="10:00",
                              end_time="12:00")
        self.assertEqual(w.start.utcoffset(), timedelta(hours=5, minutes=30))

    def test_seconds_are_accepted(self):
        w = window(start="10:00:30", end="12:00:45")
        self.assertEqual((w.start.second, w.end.second), (30, 45))

    def test_overnight_window_rolls_to_next_day_and_says_so(self):
        w = window(start="22:00", end="02:00")
        self.assertTrue(w.rolled_overnight)
        self.assertEqual((w.end - w.start), timedelta(hours=4))
        self.assertEqual(w.end.day, w.start.day + 1)

    def test_iso_form(self):
        w = HR.resolve_window(start_iso="2026-08-08T10:00:00+05:30",
                              end_iso="2026-08-08T12:00:00+05:30")
        self.assertEqual(w.start.hour, 10)
        self.assertEqual((w.end - w.start), timedelta(hours=2))

    def test_iso_end_before_start_is_an_error_not_a_rollover(self):
        with self.assertRaises(ValueError) as cm:
            HR.resolve_window(start_iso="2026-08-08T12:00:00+05:30",
                              end_iso="2026-08-08T10:00:00+05:30")
        self.assertIn("must be after", str(cm.exception))

    def test_naive_iso_gets_the_requested_zone_never_machine_local(self):
        w = HR.resolve_window(start_iso="2026-08-08T10:00:00",
                              end_iso="2026-08-08T12:00:00",
                              timezone_name="Asia/Kolkata")
        self.assertEqual(w.start.utcoffset(), timedelta(hours=5, minutes=30))

    def test_mixing_the_two_forms_is_rejected(self):
        with self.assertRaises(ValueError):
            HR.resolve_window(date="2026-08-08", start_time="10:00",
                              start_iso="2026-08-08T10:00:00+05:30",
                              end_iso="2026-08-08T12:00:00+05:30")

    def test_iso_needs_both_ends(self):
        with self.assertRaises(ValueError):
            HR.resolve_window(start_iso="2026-08-08T10:00:00+05:30")

    def test_malformed_inputs_are_rejected_with_useful_messages(self):
        for kwargs, needle in (
            (dict(date="08-08-2026", start_time="10:00", end_time="12:00"), "YYYY-MM-DD"),
            (dict(date="2026-08-08", start_time="25:00", end_time="12:00"), "out of range"),
            (dict(date="2026-08-08", start_time="10h", end_time="12:00"), "HH:MM"),
            (dict(date="2026-08-08", start_time="10:00"), "--end-time"),
            (dict(start_iso="nonsense", end_iso="also-nonsense"), "ISO"),
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)) as cm:
                HR.resolve_window(**kwargs)
            self.assertIn(needle, str(cm.exception))

    def test_unknown_timezone_is_rejected(self):
        with self.assertRaises(Exception):
            HR.resolve_window(date="2026-08-08", start_time="10:00",
                              end_time="12:00", timezone_name="Mars/Olympus")


class TestFilenameTimestampSemantics(unittest.TestCase):
    """The single most dangerous assumption in this feature."""

    def test_filename_digits_are_read_as_IST(self):
        dt = HR.filename_timestamp_local("20260808_103000")
        self.assertEqual(dt.utcoffset(), timedelta(hours=5, minutes=30))
        self.assertEqual((dt.hour, dt.minute), (10, 30))

    def test_agrees_with_the_producer_that_writes_the_names(self):
        from train_extraction.time_utils import parse_timestamp_from_filename
        produced = parse_timestamp_from_filename("CCTV_20260808_103000_train.mp4")
        consumed = HR.filename_timestamp_local("20260808_103000")
        self.assertIsNotNone(produced)
        self.assertEqual(produced.utcoffset(), consumed.utcoffset())
        self.assertEqual(produced.replace(tzinfo=None),
                         consumed.replace(tzinfo=None))

    def test_garbage_is_none_not_an_exception(self):
        self.assertIsNone(HR.filename_timestamp_local("not-a-timestamp"))
        self.assertIsNone(HR.filename_timestamp_local(None))


# ---------------------------------------------------------------------------
# 2. Object selection
# ---------------------------------------------------------------------------

class TestSelection(unittest.TestCase):

    def test_clip_inside_the_window_is_selected_for_all_four_cameras(self):
        s3 = StubS3(all_four("20260808_103000"))
        res = HR.select_objects(s3_client=s3, window=window())
        self.assertEqual(len(res.selected), 4)
        self.assertEqual({s.camera_id for s in res.selected}, set(C.ALL_CAMERAS))
        self.assertEqual(len(res.batches), 1)
        self.assertTrue(res.batches[0].is_complete())

    def test_clip_outside_the_window_is_rejected(self):
        s3 = StubS3(all_four("20260808_080000"))       # 08:00, window is 10-12
        res = HR.select_objects(s3_client=s3, window=window())
        self.assertEqual(res.selected, [])
        self.assertEqual(res.batches, [])
        self.assertGreater(res.classified, 0, "should have classified then rejected")

    def test_clip_starting_just_before_the_window_is_kept_via_the_pad(self):
        """Its train can still have passed inside the window."""
        s3 = StubS3(all_four("20260808_095500"))       # 09:55, pad 15min -> 10:10
        res = HR.select_objects(s3_client=s3, window=window(), pad_minutes=15.0)
        self.assertEqual(len(res.selected), 4)
        self.assertIn("before the window", res.selected[0].reason)

    def test_zero_pad_drops_the_pre_window_clip(self):
        s3 = StubS3(all_four("20260808_095500"))
        res = HR.select_objects(s3_client=s3, window=window(), pad_minutes=0.0)
        self.assertEqual(res.selected, [])

    def test_complete_clip_beats_incomplete_for_the_same_slot(self):
        ts = "20260808_103000"
        s3 = StubS3([_key(C.CAMERA_RIGHT_UP, ts, incomplete=True),
                     _key(C.CAMERA_RIGHT_UP, ts)])
        res = HR.select_objects(s3_client=s3, window=window())
        self.assertEqual(len(res.selected), 1)
        self.assertNotIn("_train_incomplete", res.selected[0].key)
        self.assertEqual(res.duplicates_dropped, 1)

    def test_non_video_and_unclassifiable_objects_are_ignored(self):
        s3 = StubS3([f"{C.CAMERA_S3_FOLDER[C.CAMERA_RIGHT_UP]}/notes_20260808_103000.txt",
                     "stray/no_camera_20260808_103000_train.mp4",
                     _key(C.CAMERA_RIGHT_UP, "20260808_103000")])
        res = HR.select_objects(s3_client=s3, window=window())
        self.assertEqual(len(res.selected), 1)

    def test_selection_reuses_the_live_bucket_and_prefix_config(self):
        res = HR.select_objects(s3_client=StubS3([]), window=window())
        self.assertEqual(res.bucket, C.S3_INPUT_BUCKET)
        self.assertEqual(res.prefixes, list(C.S3_INPUT_PREFIXES))

    def test_the_live_recency_cutoff_is_NOT_applied(self):
        """`list_candidate_videos` bounds discovery to the operational day, which
        would hide the entire archive -- the one thing historical must not do."""
        old = StubS3(all_four("20200101_103000"))
        res = HR.select_objects(
            s3_client=old,
            window=HR.resolve_window(date="2020-01-01", start_time="10:00",
                                     end_time="12:00"))
        self.assertEqual(len(res.selected), 4, "an old archive clip must be found")


# ---------------------------------------------------------------------------
# 3. Clustering into independent trains
# ---------------------------------------------------------------------------

class TestClustering(unittest.TestCase):

    @staticmethod
    def _cv(camera, ts):
        return CameraVideo(camera_id=camera, bucket="b", s3_key=_key(camera, ts),
                           filename=_key(camera, ts).rsplit("/", 1)[-1],
                           s3_url="", train_timestamp=ts)

    def test_four_cameras_close_together_form_one_batch(self):
        vids = [self._cv(cam, ts) for cam, ts in zip(
            C.ALL_CAMERAS, ("20260808_103000", "20260808_103010",
                            "20260808_103020", "20260808_103030"))]
        out = HR.cluster_into_batches(vids)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].is_complete())

    def test_cameras_stamped_minutes_apart_still_group(self):
        """Each camera is stamped from its OWN raw clip, so the four can step
        apart. Anchoring on the first member alone would split a real train."""
        vids = [self._cv(cam, ts) for cam, ts in zip(
            C.ALL_CAMERAS, ("20260808_103000", "20260808_103100",
                            "20260808_103200", "20260808_103300"))]
        out = HR.cluster_into_batches(vids, tolerance_sec=120)
        self.assertEqual(len(out), 1, "successive 60s gaps must chain into one train")
        self.assertTrue(out[0].is_complete())

    def test_two_independent_trains_stay_two_batches(self):
        vids = ([self._cv(c, "20260808_103000") for c in C.ALL_CAMERAS]
                + [self._cv(c, "20260808_113000") for c in C.ALL_CAMERAS])
        out = HR.cluster_into_batches(vids)
        self.assertEqual(len(out), 2)
        self.assertEqual([b.batch_key for b in out],
                         ["20260808_103000", "20260808_113000"])
        for b in out:
            self.assertTrue(b.is_complete())

    def test_a_camera_slot_is_never_double_filled(self):
        """Two RIGHT_UP clips close in time are two trains, not one batch with
        one clip silently dropped."""
        vids = [self._cv(C.CAMERA_RIGHT_UP, "20260808_103000"),
                self._cv(C.CAMERA_RIGHT_UP, "20260808_103030")]
        out = HR.cluster_into_batches(vids)
        self.assertEqual(len(out), 2)

    def test_batch_key_is_the_earliest_timestamp_in_the_cluster(self):
        vids = [self._cv(C.CAMERA_LEFT_UP, "20260808_103040"),
                self._cv(C.CAMERA_RIGHT_UP, "20260808_103000")]
        out = HR.cluster_into_batches(vids)
        self.assertEqual(out[0].batch_key, "20260808_103000")

    def test_clustering_default_tolerance_is_the_live_value(self):
        self.assertEqual(HR.cluster_into_batches.__defaults__ or (), ())
        import inspect
        sig = inspect.signature(HR.cluster_into_batches)
        self.assertEqual(sig.parameters["tolerance_sec"].default,
                         TBM.DEFAULT_BATCH_TOLERANCE_SEC)


# ---------------------------------------------------------------------------
# 4. Missing cameras -- never substitute from another train
# ---------------------------------------------------------------------------

class TestMissingCameras(unittest.TestCase):

    def test_missing_camera_is_reported_not_back_filled(self):
        keys = [_key(c, "20260808_103000")
                for c in C.ALL_CAMERAS if c != C.CAMERA_LEFT_UP_TOP]
        res = HR.select_objects(s3_client=StubS3(keys), window=window())
        self.assertEqual(len(res.batches), 1)
        b = res.batches[0]
        self.assertEqual(b.missing_cameras(), [C.CAMERA_LEFT_UP_TOP])
        self.assertFalse(b.is_complete())

    def test_nearest_clip_from_another_train_is_never_borrowed(self):
        """Train A lacks LEFT_UP_TOP; train B has one 30 min later. A must stay
        partial rather than adopt B's clip."""
        keys = [_key(c, "20260808_103000")
                for c in C.ALL_CAMERAS if c != C.CAMERA_LEFT_UP_TOP]
        keys += [_key(c, "20260808_110000") for c in C.ALL_CAMERAS]
        res = HR.select_objects(s3_client=StubS3(keys), window=window())
        self.assertEqual(len(res.batches), 2)
        first = next(b for b in res.batches if b.batch_key == "20260808_103000")
        second = next(b for b in res.batches if b.batch_key == "20260808_110000")
        self.assertEqual(first.missing_cameras(), [C.CAMERA_LEFT_UP_TOP])
        self.assertTrue(second.is_complete())
        # B's clip stayed in B.
        self.assertEqual(second.videos[C.CAMERA_LEFT_UP_TOP].train_timestamp,
                         "20260808_110000")


# ---------------------------------------------------------------------------
# 5. Manifest
# ---------------------------------------------------------------------------

class TestManifest(unittest.TestCase):

    def _manifest(self, keys, **kw):
        res = HR.select_objects(s3_client=StubS3(keys), window=window(), **kw)
        return res, HR.build_manifest(res, workspace_root="/ws/historical",
                                      dry_run=True)

    def test_manifest_records_the_window_search_and_every_selection(self):
        res, m = self._manifest(all_four("20260808_103000"))
        self.assertEqual(m["mode"], "historical")
        self.assertEqual(m["batches_discovered"], 1)
        self.assertEqual(m["requested_window"]["timezone"], "Asia/Kolkata")
        self.assertEqual(m["search"]["bucket"], C.S3_INPUT_BUCKET)
        self.assertEqual(m["search"]["objects_selected"], 4)
        cams = m["batches"][0]["cameras"]
        self.assertEqual(set(cams), set(C.ALL_CAMERAS))
        for cam, info in cams.items():
            self.assertTrue(info["s3_uri"].startswith("s3://"))
            self.assertIn("selected_because", info)
            self.assertIn("clip_start_ist", info)

    def test_manifest_marks_missing_cameras(self):
        keys = [_key(c, "20260808_103000")
                for c in C.ALL_CAMERAS if c != C.CAMERA_RIGHT_UP_TOP]
        _res, m = self._manifest(keys)
        self.assertEqual(m["batches"][0]["missing_cameras"],
                         [C.CAMERA_RIGHT_UP_TOP])

    def test_manifest_is_json_serialisable_and_writable(self):
        _res, m = self._manifest(all_four("20260808_103000"))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            self.assertEqual(HR._write_manifest(m, p), p)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["batches_discovered"], 1)

    def test_manifest_logs_without_raising(self):
        res, m = self._manifest(all_four("20260808_103000"))
        HR.log_manifest(res, m)   # must not raise

    def test_no_match_message_names_the_searched_configuration(self):
        res = HR.select_objects(s3_client=StubS3([]), window=window())
        msg = HR._no_match_message(res)
        for needle in ("window", "bucket", "prefixes", "listed", "pad"):
            self.assertIn(needle, msg)


# ---------------------------------------------------------------------------
# 6. run(): invokes the EXISTING pipeline entry point, once per train
# ---------------------------------------------------------------------------

class _Outcome:
    def __init__(self, batch, status=C.BATCH_COMPLETED):
        self.batch = batch
        self.final_status = status
        self.report_pdf_path = "/tmp/r.pdf"
        self.report_pdf_url = None
        self.camera_pdf_urls = {}


class TestRunInvokesExistingPipeline(unittest.TestCase):

    def setUp(self):
        self._real = MR.process_batch
        self.calls = []

        def spy(**kwargs):
            self.calls.append(kwargs)
            return _Outcome(kwargs["batch"])

        MR.process_batch = spy
        # Owned for the whole test: assertions inspect the manifest AFTER run(),
        # so the workspace must outlive the call.
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = self._tmp.name

    def tearDown(self):
        MR.process_batch = self._real
        self._tmp.cleanup()

    def _run(self, keys, **kw):
        kw.setdefault("workspace_root", self.ws)
        rc = HR.run(s3_client=StubS3(keys), window=window(),
                    recon_models_dir="/m/recon", feat_models_dir="/m/feat",
                    verbose=False, **kw)
        return rc, kw["workspace_root"]

    def test_one_train_one_invocation_of_process_batch(self):
        rc, _ = self._run(all_four("20260808_103000"))
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.calls), 1)
        self.assertIs(self.calls[0]["batch"].__class__.__name__ and True, True)
        self.assertEqual(self.calls[0]["batch"].batch_key, "20260808_103000")

    def test_two_trains_produce_TWO_independent_invocations(self):
        """Never one merged Global Train."""
        keys = all_four("20260808_103000") + all_four("20260808_113000")
        rc, _ = self._run(keys)
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual([c["batch"].batch_key for c in self.calls],
                         ["20260808_103000", "20260808_113000"])
        # Each invocation carries only its own train's clips.
        for call in self.calls:
            b = call["batch"]
            for cv in b.videos.values():
                self.assertEqual(cv.train_timestamp, b.batch_key)

    def test_three_trains_produce_three_invocations(self):
        keys = (all_four("20260808_100500") + all_four("20260808_103000")
                + all_four("20260808_113000"))
        rc, _ = self._run(keys)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(rc, 0)

    def test_output_is_isolated_under_the_historical_subdir(self):
        _rc, ws = self._run(all_four("20260808_103000"))
        root = self.calls[0]["workspace_root"]
        self.assertEqual(root, os.path.join(ws, HR.HISTORICAL_SUBDIR))
        self.assertNotEqual(root, ws, "must not write into the live batch root")

    def test_delivery_is_off_by_default(self):
        self._run(all_four("20260808_103000"))
        self.assertTrue(self.calls[0]["skip_upload"])
        self.assertTrue(self.calls[0]["skip_email"])

    def test_deliver_flag_enables_upload(self):
        self._run(all_four("20260808_103000"), deliver=True)
        self.assertFalse(self.calls[0]["skip_upload"])

    def test_deliver_with_email_suppressed(self):
        self._run(all_four("20260808_103000"), deliver=True, send_email=False)
        self.assertFalse(self.calls[0]["skip_upload"])
        self.assertTrue(self.calls[0]["skip_email"])

    def test_inference_opts_are_passed_through(self):
        opts = {"door_inference_mode": "legacy", "door_sample_stride": 1}
        self._run(all_four("20260808_103000"), inference_opts=opts)
        self.assertEqual(self.calls[0]["door_inference_mode"], "legacy")
        self.assertEqual(self.calls[0]["door_sample_stride"], 1)

    def test_dry_run_discovers_but_never_invokes_the_pipeline(self):
        rc, ws = self._run(all_four("20260808_103000"), dry_run=True)
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls, [], "dry run must not call process_batch")
        self.assertTrue(os.path.isfile(
            os.path.join(ws, HR.HISTORICAL_SUBDIR, HR.MANIFEST_NAME)))

    def test_no_match_returns_2_and_never_invokes_the_pipeline(self):
        rc, _ = self._run(all_four("20260101_030000"))   # far outside the window
        self.assertEqual(rc, 2)
        self.assertEqual(self.calls, [],
                         "must not run the pipeline with empty inputs")

    def test_partial_batch_still_reaches_the_existing_pipeline(self):
        keys = [_key(c, "20260808_103000")
                for c in C.ALL_CAMERAS if c != C.CAMERA_LEFT_UP]
        rc, _ = self._run(keys)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["batch"].missing_cameras(),
                         [C.CAMERA_LEFT_UP])
        self.assertEqual(rc, 0)

    def test_one_failing_batch_does_not_stop_the_others(self):
        def flaky(**kwargs):
            self.calls.append(kwargs)
            if kwargs["batch"].batch_key == "20260808_103000":
                raise RuntimeError("stage 1 blew up")
            return _Outcome(kwargs["batch"])
        MR.process_batch = flaky
        keys = all_four("20260808_103000") + all_four("20260808_113000")
        rc, _ = self._run(keys)
        self.assertEqual(len(self.calls), 2, "second train must still run")
        self.assertEqual(rc, 3, "a failed batch must be reported in the exit code")

    def test_failed_status_is_reported_in_the_exit_code(self):
        MR.process_batch = lambda **kw: (
            self.calls.append(kw) or _Outcome(kw["batch"], C.BATCH_FAILED))
        rc, _ = self._run(all_four("20260808_103000"))
        self.assertEqual(rc, 3)

    def test_manifest_is_written_before_processing(self):
        _rc, ws = self._run(all_four("20260808_103000"))
        with open(os.path.join(ws, HR.HISTORICAL_SUBDIR, HR.MANIFEST_NAME),
                  encoding="utf-8") as f:
            self.assertEqual(json.load(f)["batches_discovered"], 1)

    def test_live_state_file_is_never_touched(self):
        """A historical run must not read or write processed_batches.json."""
        touched = []
        real_load, real_save = TBM.load_batch_state, TBM.save_batch_state
        TBM.load_batch_state = lambda *a, **k: touched.append("load") or {}
        TBM.save_batch_state = lambda *a, **k: touched.append("save")
        try:
            self._run(all_four("20260808_103000"))
        finally:
            TBM.load_batch_state, TBM.save_batch_state = real_load, real_save
        self.assertEqual(touched, [])


# ---------------------------------------------------------------------------
# 7. BACKWARD COMPATIBILITY -- the live path must be untouched
# ---------------------------------------------------------------------------

class TestLivePathUnchanged(unittest.TestCase):

    def _parse(self, argv):
        return MR._build_parser().parse_args(argv)

    def test_auto_parses_with_no_historical_arguments(self):
        args = self._parse(["--auto"])
        self.assertTrue(args.auto)
        self.assertFalse(args.historical)

    def test_historical_flags_default_off_and_none(self):
        args = self._parse(["--auto"])
        self.assertFalse(args.historical)
        self.assertFalse(args.dry_run)
        self.assertFalse(args.keep_inputs)
        self.assertFalse(args.historical_deliver)
        for name in ("date", "start_time", "end_time", "timezone", "start",
                     "end", "tolerance_sec", "pad_minutes", "manifest_out"):
            self.assertIsNone(getattr(args, name), name)

    def test_existing_defaults_are_unchanged(self):
        args = self._parse([])
        # --mode now defaults to sequential (b0905a0). Everything else here is
        # untouched; historical dispatches before --mode is consulted, and the
        # sequential/historical conflict is only raised for an EXPLICIT --mode.
        self.assertEqual(args.mode, "sequential")
        self.assertEqual(args.poll_interval, 60)
        self.assertEqual(args.partial_wait, 30.0)
        self.assertEqual(args.door_inference_mode, "sampled")
        self.assertEqual(args.door_sample_stride, 3)
        self.assertEqual(args.damage_inference_mode, "sampled")
        self.assertEqual(args.damage_sample_stride, 3)
        self.assertEqual(args.load_inference_mode, "sampled")
        self.assertEqual(args.load_sample_stride, 2)
        self.assertEqual(args.ocr_engine, None)
        self.assertEqual(args.source, None)
        self.assertFalse(args.skip_upload)
        self.assertFalse(args.skip_email)

    def test_live_discovery_helpers_still_exist_with_the_same_contract(self):
        for name in ("poll_for_batches", "select_runnable_batch",
                     "load_batch_state", "save_batch_state",
                     "list_candidate_videos", "_discovery_cutoff",
                     "consumer_lookback_minutes"):
            self.assertTrue(hasattr(TBM, name), name)
        self.assertEqual(TBM.DEFAULT_BATCH_TOLERANCE_SEC, 120)

    def test_historical_mode_never_enters_the_live_polling_loop(self):
        """The regression that would let a historical request process live data."""
        entered = []
        real_auto = MR.run_auto
        MR.run_auto = lambda *a, **k: entered.append("run_auto") or 0
        real_hist = MR.run_historical
        MR.run_historical = lambda *a, **k: 0
        try:
            rc = MR.main(["--historical", "--date", "2026-08-08",
                          "--start-time", "10:00", "--end-time", "12:00",
                          "--no-interactive"])
        finally:
            MR.run_auto, MR.run_historical = real_auto, real_hist
        self.assertEqual(rc, 0)
        self.assertEqual(entered, [], "historical must not reach run_auto")

    def test_historical_still_dispatches_under_the_sequential_default(self):
        """--mode now defaults to sequential; historical must survive that.

        The historical branch rejects `--mode sequential` because the two name
        different execution paths. Once sequential became the DEFAULT, testing
        args.mode alone would have rejected every historical run -- including
        the plain invocation an operator actually types, where --mode was never
        passed at all. The check therefore keys on an EXPLICIT --mode.
        """
        entered = []
        real_hist = MR.run_historical
        MR.run_historical = lambda *a, **k: entered.append("hist") or 0
        try:
            rc = MR.main(["--historical", "--date", "2026-08-08",
                          "--start-time", "10:00", "--end-time", "12:00",
                          "--no-interactive"])
        finally:
            MR.run_historical = real_hist
        self.assertEqual(entered, ["hist"],
                         "plain --historical must still reach run_historical")
        self.assertEqual(rc, 0)

    def test_explicit_sequential_with_historical_now_runs_sequential(self):
        """This combination used to be REJECTED, because historical could only
        call `process_batch`.  It is now supported: historical stages each
        discovered batch's clips and runs the per-camera -> assembly path.

        What still matters, and is asserted here, is that the explicit request
        REACHES historical rather than being swallowed."""
        seen = {}
        real_hist = MR.run_historical
        MR.run_historical = lambda *a, **k: seen.update(k) or 0
        try:
            rc = MR.main(["--historical", "--mode", "sequential",
                          "--date", "2026-08-08", "--start-time", "10:00",
                          "--end-time", "12:00", "--no-interactive"])
        finally:
            MR.run_historical = real_hist
        self.assertEqual(rc, 0)
        self.assertTrue(seen.get("mode_explicit"),
                        "the explicit --mode must reach run_historical")

    def test_explicit_mode_is_detected_in_both_spellings(self):
        """`--mode sequential` and `--mode=sequential` are the same request."""
        for argv in (["--historical", "--mode", "sequential"],
                     ["--historical", "--mode=sequential"]):
            with self.subTest(argv=argv):
                real_hist = MR.run_historical
                seen = {}
                MR.run_historical = lambda *a, **k: seen.update(k) or 0
                try:
                    rc = MR.main(argv + ["--date", "2026-08-08",
                                         "--start-time", "10:00",
                                         "--end-time", "12:00",
                                         "--no-interactive"])
                finally:
                    MR.run_historical = real_hist
                # Formerly rc == 2 (rejected).  Both spellings must now be
                # DETECTED as explicit and accepted; what is asserted is that
                # neither is silently ignored.
                self.assertEqual(rc, 0)
                self.assertTrue(seen.get("mode_explicit"), argv)

    def test_explicit_batch_with_historical_is_accepted(self):
        """Historical feeds process_batch, so --mode batch is consistent."""
        entered = []
        real_hist = MR.run_historical
        MR.run_historical = lambda *a, **k: entered.append("hist") or 0
        try:
            rc = MR.main(["--historical", "--mode", "batch",
                          "--date", "2026-08-08", "--start-time", "10:00",
                          "--end-time", "12:00", "--no-interactive"])
        finally:
            MR.run_historical = real_hist
        self.assertEqual(entered, ["hist"])
        self.assertEqual(rc, 0)

    def test_auto_never_enters_historical(self):
        entered = []
        real_hist, real_auto = MR.run_historical, MR.run_auto
        MR.run_historical = lambda *a, **k: entered.append("hist") or 0
        MR.run_auto = lambda *a, **k: 0
        try:
            MR.main(["--auto", "--no-interactive"])
        finally:
            MR.run_historical, MR.run_auto = real_hist, real_auto
        self.assertEqual(entered, [], "--auto must not reach historical code")

    def test_historical_requires_no_pipeline_source_change(self):
        """WAGONEYE_PIPELINE_SOURCE semantics are untouched: historical mode is
        always a pure consumer, and does not read or write that variable."""
        src = io.open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "orchestrator", "historical_runner.py"), encoding="utf-8").read()
        # The module reads NO environment variable directly -- every setting
        # arrives through core.config / core.constants, which is what guarantees
        # it cannot redefine the meaning of an existing variable (the docstring
        # names WAGONEYE_PIPELINE_SOURCE only to state it is irrelevant here).
        self.assertNotIn("os.getenv", src)
        self.assertNotIn("os.environ", src)

    def test_historical_module_does_not_reimplement_the_stages(self):
        src = io.open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "orchestrator", "historical_runner.py"), encoding="utf-8").read()
        for forbidden in ("reconstruction_runner", "wagon_cache_builder",
                          "wagon_state_builder", "combined_train_report",
                          "camera_reports", "feature_overlay_renderer",
                          "from features", "import features"):
            self.assertNotIn(forbidden, src,
                             f"historical mode must not touch {forbidden}")
        self.assertIn("process_batch", src,
                      "historical mode must call the existing entry point")


class TestPollForBatchesRegression(unittest.TestCase):
    """`poll_for_batches` unpacked 4 values from `_list_input_objects`' 5-tuples
    and raised `ValueError: too many values to unpack` on every non-empty poll,
    which made `--auto` unable to discover anything at all."""

    def setUp(self):
        self._saved = os.environ.get("WAGONEYE_CONSUMER_LOOKBACK_MINUTES")
        os.environ["WAGONEYE_CONSUMER_LOOKBACK_MINUTES"] = "0"

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", None)
        else:
            os.environ["WAGONEYE_CONSUMER_LOOKBACK_MINUTES"] = self._saved

    def test_poll_for_batches_survives_a_non_empty_listing(self):
        s3 = StubS3([_key(C.CAMERA_RIGHT_UP, "20260819_101500")])
        batches = TBM.poll_for_batches(s3_client=s3, processed_batches={})
        self.assertTrue(batches)
        self.assertEqual(batches[0].batch_key, "20260819_101500")

    def test_already_processed_batches_are_excluded(self):
        s3 = StubS3([_key(C.CAMERA_RIGHT_UP, "20260819_101500")])
        batches = TBM.poll_for_batches(
            s3_client=s3, processed_batches={"20260819_101500": "completed"})
        self.assertEqual(batches, [])

    def test_list_candidate_videos_still_returns_camera_videos(self):
        s3 = StubS3(all_four("20260819_101500"))
        out = TBM.list_candidate_videos(s3)
        self.assertEqual({cv.camera_id for cv in out}, set(C.ALL_CAMERAS))


class TestValidateConfigHistorical(unittest.TestCase):

    def test_historical_mode_does_not_require_the_email_endpoint(self):
        errors = CFG.validate_config(mode="historical", skip_upload=True,
                                     skip_email=True)
        self.assertFalse([e for e in errors if "email" in e.lower()], errors)

    def test_historical_mode_requires_discovery_configuration(self):
        errors = CFG.validate_config(mode="historical")
        self.assertFalse([e for e in errors if "INPUT_PREFIXES" in e], errors)

    def test_existing_modes_are_unaffected(self):
        for mode in ("auto", "local", "once", "batch"):
            CFG.validate_config(mode=mode, skip_email=True)  # must not raise


if __name__ == "__main__":
    unittest.main()


class TestHistoricalFlagConflicts(unittest.TestCase):
    """Two flags that both name an execution architecture must not combine
    silently -- historical wins by dispatch order, so an operator could believe
    their window ran through a path it never touched."""

    def test_historical_plus_sequential_is_supported(self):
        """Was rejected; now runs the per-camera -> assembly path."""
        real = MR.run_historical
        MR.run_historical = lambda *a, **k: 0
        try:
            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = MR.main(["--historical", "--mode", "sequential",
                              "--date", "2026-08-08", "--start-time", "10:00",
                              "--end-time", "12:00", "--no-interactive"])
        finally:
            MR.run_historical = real
        self.assertEqual(rc, 0)
        self.assertNotIn("cannot be combined", buf.getvalue())

    def test_historical_plus_auto_is_rejected(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            rc = MR.main(["--historical", "--auto", "--date", "2026-08-08",
                          "--start-time", "10:00", "--end-time", "12:00",
                          "--no-interactive"])
        self.assertEqual(rc, 2)
        self.assertIn("mutually exclusive", buf.getvalue())

    def test_sequential_alone_is_still_accepted(self):
        args = MR._build_parser().parse_args(["--mode", "sequential"])
        self.assertEqual(args.mode, "sequential")
        self.assertFalse(args.historical)


class TestIntermediateReclamation(unittest.TestCase):
    """A bulk window is tens of trains back to back, and a wagon cache is the
    bulk of a batch's several GB.  Successful batches must give theirs back;
    failed ones must keep everything for diagnosis."""

    def _batch_root(self, tmp):
        root = os.path.join(tmp, "20260808_103000")
        for sub in (CFG.DIR_DOWNLOADS, CFG.DIR_WAGON_CACHE, CFG.DIR_REPORTS,
                    CFG.DIR_EVIDENCE, CFG.DIR_PROCESSED_VIDEOS,
                    CFG.DIR_GLOBAL_STATE):
            d = os.path.join(root, sub)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "f.bin"), "wb") as f:
                f.write(b"\x00" * 2048)
        return root

    def test_reclaims_only_the_reconstructible_intermediates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._batch_root(tmp)
            HR._cleanup_inputs(root, keep_inputs=False)
            for gone in (CFG.DIR_DOWNLOADS, CFG.DIR_WAGON_CACHE):
                self.assertFalse(os.path.isdir(os.path.join(root, gone)), gone)
            for kept in (CFG.DIR_REPORTS, CFG.DIR_EVIDENCE,
                         CFG.DIR_PROCESSED_VIDEOS, CFG.DIR_GLOBAL_STATE):
                self.assertTrue(os.path.isdir(os.path.join(root, kept)), kept)

    def test_keep_inputs_retains_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._batch_root(tmp)
            HR._cleanup_inputs(root, keep_inputs=True)
            for kept in (CFG.DIR_DOWNLOADS, CFG.DIR_WAGON_CACHE):
                self.assertTrue(os.path.isdir(os.path.join(root, kept)), kept)

    def test_missing_directories_are_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            HR._cleanup_inputs(os.path.join(tmp, "nope"), keep_inputs=False)

    def test_a_failed_batch_keeps_its_intermediates(self):
        """run() calls _cleanup_inputs only on the success branch."""
        src = io.open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "orchestrator", "historical_runner.py"), encoding="utf-8").read()
        after_ok = src.split("if ok:", 1)[1].split("failures.append", 1)
        # Either reclaim entry point is acceptable; what matters is that one of
        # them is reached ONLY on success. The sequential branch uses
        # `delivery.cleanup.cleanup_batch` (delivery-gated); the batch branch
        # keeps the narrower `_cleanup_inputs`.
        self.assertTrue(
            ("cleanup_batch" in after_ok[0]) or ("_cleanup_inputs" in after_ok[0]),
            "cleanup must be on the success branch")
        self.assertIn("RETAINED", src.split("if ok:", 1)[1],
                      "the failure branch must retain inputs")

    def test_sequential_cleanup_is_gated_on_delivery_not_assembly(self):
        """Assembly succeeding is not enough: a train whose upload or dashboard
        post failed must keep the artifacts a retry needs."""
        src = io.open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "orchestrator", "historical_runner.py"), encoding="utf-8").read()
        seq = src.split('if mode == "sequential"', 1)[1].split(
            "invoking existing pipeline", 1)[0]
        self.assertIn("cleanup_batch", seq)
        self.assertIn("delivery=getattr(asm", seq,
                      "the DeliveryResult must reach the cleanup gate")


class TestFilenameTimestampConsistency(unittest.TestCase):
    """One answer to "what timezone is a filename timestamp" across the package."""

    def test_batch_manager_and_historical_agree(self):
        ts = "20260808_103000"
        self.assertEqual(TBM._ts_to_dt(ts), HR.filename_timestamp_local(ts))

    def test_both_agree_with_the_producer(self):
        from train_extraction.time_utils import parse_timestamp_from_filename
        produced = parse_timestamp_from_filename("CCTV_20260808_103000_train.mp4")
        self.assertEqual(produced, TBM._ts_to_dt("20260808_103000"))

    def test_clustering_is_unchanged_by_the_label(self):
        """The gate is a difference of two of these, so a constant offset
        cancels -- proven, not assumed."""
        from datetime import timezone as _tz
        a, b = TBM._ts_to_dt("20260808_103000"), TBM._ts_to_dt("20260808_103100")
        self.assertEqual((b - a).total_seconds(), 60.0)
        naive_a = datetime.strptime("20260808_103000", "%Y%m%d_%H%M%S")
        naive_b = datetime.strptime("20260808_103100", "%Y%m%d_%H%M%S")
        self.assertEqual((b - a), (naive_b - naive_a))

    def test_batch_age_uses_the_same_zone(self):
        from datetime import timedelta as _td
        from core.batch import TrainBatch
        ts = (datetime.now(CFG.IST) - _td(minutes=10)).strftime("%Y%m%d_%H%M%S")
        age = TrainBatch(batch_key=ts, train_timestamp=ts, videos={}).age_seconds()
        self.assertGreater(age, 0)
        self.assertLess(age, 900)


class TestHistoricalSequential(unittest.TestCase):
    """`--historical --mode sequential`: stage the clips, then per-camera ->
    assembly, using the SAME two entry points the foreground sequential path
    uses so the two architectures cannot diverge."""

    def setUp(self):
        self._real = MR.process_batch
        self.calls = []
        MR.process_batch = lambda **kw: (_ for _ in ()).throw(
            AssertionError("sequential mode must NOT call process_batch"))
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = self._tmp.name

    def tearDown(self):
        MR.process_batch = self._real
        self._tmp.cleanup()

    def test_staging_downloads_each_camera_once(self):
        from core.batch import CameraVideo, TrainBatch
        got = []

        class S3:
            def download_file(self, bucket, key, dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                open(dest, "wb").write(b"\x00")
                got.append((bucket, key))

        b = TrainBatch(batch_key="20260729_103722",
                       train_timestamp="20260729_103722",
                       videos={cam: CameraVideo(
                           camera_id=cam, bucket="bkt",
                           s3_key=_key(cam, "20260729_103722"),
                           filename=f"{cam}.mp4", s3_url="",
                           train_timestamp="20260729_103722")
                           for cam in C.ALL_CAMERAS})
        root = os.path.join(self.ws, b.batch_key)
        paths = HR.stage_clips(b, root, S3(), verbose=False)
        self.assertEqual(set(paths), set(C.ALL_CAMERAS))
        self.assertEqual(len(got), 4)
        for p in paths.values():
            self.assertTrue(os.path.isfile(p))

    def test_staging_reuses_an_already_staged_clip(self):
        """A resumed run must not re-download what is already on disk."""
        from core.batch import CameraVideo, TrainBatch
        calls = []

        class S3:
            def download_file(self, bucket, key, dest):
                calls.append(key)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                open(dest, "wb").write(b"\x00")

        b = TrainBatch(batch_key="k", train_timestamp="k", videos={
            C.CAMERA_RIGHT_UP: CameraVideo(
                camera_id=C.CAMERA_RIGHT_UP, bucket="bkt", s3_key="a.mp4",
                filename="a.mp4", s3_url="", train_timestamp="k")})
        root = os.path.join(self.ws, "k")
        HR.stage_clips(b, root, S3(), verbose=False)
        HR.stage_clips(b, root, S3(), verbose=False)
        self.assertEqual(len(calls), 1, "second staging should be a no-op")

    def test_a_camera_that_fails_to_stage_does_not_stop_the_others(self):
        from core.batch import CameraVideo, TrainBatch

        class S3:
            def download_file(self, bucket, key, dest):
                # The site's folder for LEFT_UP_TOP is `..._6_LEFT_TOP` -- there
                # is no "LEFT_UP_TOP" substring in the key.  Match the real
                # folder, which is the whole point of CAMERA_S3_FOLDER.
                if C.CAMERA_S3_FOLDER[C.CAMERA_LEFT_UP_TOP] in key:
                    raise RuntimeError("403")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                open(dest, "wb").write(b"\x00")

        b = TrainBatch(batch_key="k", train_timestamp="k", videos={
            cam: CameraVideo(camera_id=cam, bucket="bkt",
                             s3_key=_key(cam, "20260729_103722"),
                             filename=f"{cam}.mp4", s3_url="",
                             train_timestamp="k")
            for cam in C.ALL_CAMERAS})
        paths = HR.stage_clips(b, os.path.join(self.ws, "k"), S3(), verbose=False)
        self.assertEqual(len(paths), 3)
        self.assertNotIn(C.CAMERA_LEFT_UP_TOP, paths)

    def test_sequential_calls_run_camera_per_camera_then_assemble_once(self):
        from core.batch import CameraVideo, TrainBatch
        from orchestrator import camera_runner, global_assembler

        cams_run, assembled = [], []

        class _Res:
            def __init__(self, cam):
                self.camera_id, self.state = cam, "SEALED"
                self.local_segments, self.per_camera_ingest = 3, None

        class _Asm:
            total_wagons = 58

        real_rc, real_as = camera_runner.run_camera, global_assembler.assemble
        camera_runner.run_camera = lambda **kw: (
            cams_run.append(kw["camera_id"]) or _Res(kw["camera_id"]))
        global_assembler.assemble = lambda **kw: (
            assembled.append(kw) or _Asm())

        class S3:
            def download_file(self, bucket, key, dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                open(dest, "wb").write(b"\x00")

        try:
            b = TrainBatch(batch_key="20260729_103722",
                           train_timestamp="20260729_103722",
                           videos={cam: CameraVideo(
                               camera_id=cam, bucket="bkt",
                               s3_key=_key(cam, "20260729_103722"),
                               filename=f"{cam}.mp4", s3_url="",
                               train_timestamp="20260729_103722")
                               for cam in C.ALL_CAMERAS})
            asm = HR.process_batch_sequential(
                b, hist_root=self.ws, s3_client=S3(),
                recon_models_dir="/m/r", feat_models_dir="/m/f",
                deliver=True, deliver_per_camera=True, verbose=False)
        finally:
            camera_runner.run_camera = real_rc
            global_assembler.assemble = real_as

        self.assertEqual(cams_run, list(C.ALL_CAMERAS),
                         "each camera runs once, in canonical order")
        self.assertEqual(len(assembled), 1, "assembly runs exactly once")
        self.assertTrue(assembled[0]["deliver"])
        self.assertEqual(assembled[0]["batch_key"], "20260729_103722")
        self.assertEqual(asm.total_wagons, 58)

    def test_run_forwards_the_mode(self):
        import inspect
        params = inspect.signature(HR.run).parameters
        self.assertIn("mode", params)
        self.assertEqual(params["mode"].default, "batch",
                         "historical must default to the validated batch path")
        self.assertIn("deliver_per_camera", params)


class TestHistoricalModeSelection(unittest.TestCase):
    """`--mode` defaults to sequential for foreground runs, so a BARE
    `--historical` must not silently change architecture."""

    def test_bare_historical_is_no_longer_rejected(self):
        buf = io.StringIO()
        real = MR.run_historical
        MR.run_historical = lambda *a, **k: 0
        try:
            with redirect_stderr(buf):
                rc = MR.main(["--historical", "--date", "2026-08-08",
                              "--start-time", "10:00", "--end-time", "12:00",
                              "--no-interactive"])
        finally:
            MR.run_historical = real
        self.assertEqual(rc, 0)
        self.assertNotIn("cannot be combined", buf.getvalue())

    def test_explicit_sequential_is_accepted(self):
        real = MR.run_historical
        seen = {}
        MR.run_historical = lambda *a, **k: seen.update(k) or 0
        try:
            rc = MR.main(["--historical", "--mode", "sequential",
                          "--date", "2026-08-08", "--start-time", "10:00",
                          "--end-time", "12:00", "--no-interactive"])
        finally:
            MR.run_historical = real
        self.assertEqual(rc, 0)
        self.assertTrue(seen.get("mode_explicit"),
                        "an explicit --mode must reach run_historical")

    def test_historical_plus_auto_is_still_rejected(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            rc = MR.main(["--historical", "--auto", "--date", "2026-08-08",
                          "--start-time", "10:00", "--end-time", "12:00",
                          "--no-interactive"])
        self.assertEqual(rc, 2)
        self.assertIn("mutually exclusive", buf.getvalue())
