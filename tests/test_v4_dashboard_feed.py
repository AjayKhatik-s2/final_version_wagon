"""The V4 auto-pipeline delivery contract: endpoints, payloads, JSON, OCR engine.

What these tests protect, and why each one exists:

* **The API is V4's, not a lookalike.**  V4's committed
  `Train-Inspection-Engine/configs/config.json` OVERRIDES the older defaults
  still hard-coded in its `core/config.py` dataclass (its own commit "Match
  notebook artifact + JSON contract; fix flush-emit + endpoint URLs" did that).
  The dataclass points at `cctv-wagon-api.suvidhaen.com`; the live config does
  not.  Copying the wrong one posts to a host the dashboard never reads, which
  fails silently -- the POST succeeds and no report appears.  So the exact URLs
  are asserted as literals here.

* **`version` decides the dashboard tab.**  The report must appear in the V1
  view, and the ONLY thing selecting that is the `version` field -- the PROD
  ingest URL is byte-identical for V1 and V4.  A regression here would deliver
  a technically-successful ingest into the wrong tab.

* **`camera_id` prefix follows the version.**  A v1 document keeps
  `camera_CCTV_...`; stripping it (v4's form) makes the dashboard fail to match
  the camera.

* **Per-camera narrowing.**  This package FUSES each feature into one file per
  wagon, while the V4 schema is per-camera.  Handing the fused file over
  unnarrowed republishes one top camera's damage as the other's finding.

* **Evidence layout.**  V4/global_train nest evidence per camera; this package
  does not.  Without the flat fallback every `s3_url` in the document is null.

No test here performs network I/O, loads a model, or needs AWS credentials:
`requests` and the S3 client are stubs, and the OCR test drives a fake
Rekognition client.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as C                                   # noqa: E402
from core.global_state_loader import parse_global_train_state      # noqa: E402
from delivery import dashboard_ingest as DASH                      # noqa: E402
from delivery import inspection_json as IJ                         # noqa: E402
from core import evidence_identity as EI                           # noqa: E402
from delivery import ml_api                                        # noqa: E402


# ---------------------------------------------------------------------------
# The V4 API set, transcribed from Train-Inspection-Engine/configs/config.json
# ---------------------------------------------------------------------------

V4_INGEST_PROD = ("https://ms-pnr-location-notification-api.suvidhaen.com/"
                  "cctv-receiver/inspections/ingest")
V4_INGEST_UAT = "https://cctv-wagon-uat-api.suvidhaen.com/inspections/ingest"
V4_ML_API = ("https://ms-pnr-location-notification-api.suvidhaen.com/"
             "cctv-receiver/api/v1/ml")
V4_EMAIL_API = ("https://ms-pnr-location-notification-api.suvidhaen.com/"
                "notification_microservice/send-email")
V4_ARTIFACT_BUCKET = "test-inspection-artifacts-sarva"

#: The host V4's STALE dataclass defaults point at.  Nothing may post here.
STALE_V4_HOST = "cctv-wagon-api.suvidhaen.com"


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"run_id": "run-1"}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class _FakeRequests:
    """Records every POST instead of making one."""

    def __init__(self, status_code=200):
        self.calls = []
        self._status = status_code

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers or {},
                           "timeout": timeout})
        return _FakeResponse(self._status)


class _FakeS3:
    def __init__(self):
        self.uploads = []

    def upload_file(self, local, bucket, key, ExtraArgs=None):
        self.uploads.append({"local": local, "bucket": bucket, "key": key,
                             "extra": ExtraArgs or {}})


def _clear_env(*names):
    saved = {n: os.environ.pop(n, None) for n in names}
    return saved


def _restore_env(saved):
    for n, v in saved.items():
        if v is None:
            os.environ.pop(n, None)
        else:
            os.environ[n] = v


_ENV_KEYS = (
    "WAGONEYE_INSPECTION_INGEST_API_URLS", "WAGONEYE_INSPECTION_VERSION",
    "WAGONEYE_INSPECTION_JSON_BUCKET", "WAGONEYE_DASHBOARD_INGEST_ENABLED",
    "WAGONEYE_INSPECTION_KEY_LAYOUT", "WAGONEYE_ML_API_ENABLED",
    "WAGONEYE_OCR_ENGINE", "WAGONEYE_INGEST_API_URL_PROD",
    "WAGONEYE_INGEST_API_URL_UAT", "WAGONEYE_ML_API_ENDPOINT",
)


# ---------------------------------------------------------------------------
# Synthetic finalized batch
# ---------------------------------------------------------------------------

def _write(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def _jpeg(path):
    """A file that merely has to EXIST -- evidence resolution is path-based."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff\xd9")


def build_batch_root(tmp: str, *, batch_key="20260819_101500") -> str:
    """A minimal but REALISTIC finalized batch, in THIS package's layout.

    Two wagons, all four cameras, flat per-feature state and flat evidence --
    exactly what `process_batch` leaves on disk.
    """
    root = os.path.join(tmp, batch_key)
    gws = ["GW_1", "GW_2"]

    state_doc = {
        "total_wagons": 2,
        "master_camera": C.CAMERA_RIGHT_UP,
        "master_fps": 25.0,
        "master_total_frames": 500,
        "wagons": [
            {"global_id": g, "wagon_index": i + 1,
             "start_frame_master": 100 * i, "end_frame_master": 100 * i + 90,
             "start_time": 4.0 * i, "end_time": 4.0 * i + 3.6,
             "classification": "WAGON", "classification_confidence": 0.95,
             "supporting_cameras": list(C.ALL_CAMERAS)}
            for i, g in enumerate(gws)
        ],
    }
    _write(os.path.join(root, "global_state", "global_train_state.json"), state_doc)

    for i, g in enumerate(gws):
        _write(os.path.join(root, "wagon_states", "unified", f"{g}.json"), {
            "global_id": g, "wagon_index": i + 1, "classification": "WAGON",
            "wagon_identifier": "32145678901" if i == 0 else C.NO_DATA,
            "left_door": C.DOOR_CLOSED,
            "right_door": C.DOOR_OPEN if i == 0 else C.DOOR_CLOSED,
            "load_status": C.LOAD_LOADED,
            "top_damage": C.DAMAGE_PRESENT if i == 1 else C.NO_DATA,
            "supporting_cameras": list(C.ALL_CAMERAS),
        })

        # door: ONE fused file carrying BOTH sides (this package's shape)
        _write(os.path.join(root, "wagon_states", "door", f"{g}.json"), {
            "global_id": g, "feature": "door", "status": C.STATUS_OK,
            "left_door": C.DOOR_CLOSED,
            "right_door": C.DOOR_OPEN if i == 0 else C.DOOR_CLOSED,
            "supporting_cameras": [C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP],
        })
        _write(os.path.join(root, "wagon_states", "load", f"{g}.json"), {
            "global_id": g, "feature": "load", "status": C.STATUS_OK,
            "load_status": C.LOAD_LOADED,
            "supporting_cameras": [C.CAMERA_RIGHT_UP_TOP],
        })
        # damage: fused across BOTH top cameras, each track tagged with its own
        # camera -- GW_2 has damage seen ONLY by RIGHT_UP_TOP.
        _write(os.path.join(root, "wagon_states", "damage", f"{g}.json"), {
            "global_id": g, "feature": "damage", "status": C.STATUS_OK,
            "damage_status": C.DAMAGE_PRESENT if i == 1 else C.DAMAGE_OK,
            "top_damage_details": ([{
                "class_name": "floor_damage", "confidence": 0.81,
                "bbox": [10, 10, 60, 60], "frame_idx": 120,
                "camera_id": C.CAMERA_RIGHT_UP_TOP, "track_id": 1,
            }] if i == 1 else []),
            "per_camera": {
                C.CAMERA_RIGHT_UP_TOP: {
                    "damage_status": C.DAMAGE_PRESENT if i == 1 else C.DAMAGE_OK},
                C.CAMERA_LEFT_UP_TOP: {"damage_status": C.DAMAGE_OK},
            },
            "supporting_cameras": list(C.TOP_CAMERAS),
        })
        _write(os.path.join(root, "wagon_states", "ocr", f"{g}.json"), {
            "global_id": g, "feature": "ocr", "status": C.STATUS_OK,
            "engine": "rekognition",
            "wagon_identifier": "32145678901" if i == 0 else C.NO_DATA,
            "raw_number": "32145678901" if i == 0 else "3214567",
            "display_number": "32145678901" if i == 0 else "-",
            "is_valid_11_digit": i == 0,
            "confidence": 0.91, "ocr_confidence": 0.91,
            "fallback_triggered": i != 0,
            "best_frame": 130, "best_bbox": [5, 5, 40, 20],
            "supporting_cameras": [C.CAMERA_RIGHT_UP],
        })

        # FLAT evidence -- no per-camera subdirectory.
        ev = os.path.join(root, "evidence", g)
        _jpeg(os.path.join(ev, "door", "right_best.jpg"))
        _jpeg(os.path.join(ev, "door", "left_best.jpg"))
        _jpeg(os.path.join(ev, "load", "best_frame.jpg"))
        _jpeg(os.path.join(ev, "ocr", "best_frame.jpg"))
        _jpeg(os.path.join(ev, "ocr", "ocr_sheet.jpg"))
        _write(os.path.join(ev, "door", "metadata.json"),
               {"global_id": g, "feature": "door",
                "sides": {"right": {"frame_idx": 111, "bbox": [1, 2, 3, 4],
                                    "confidence": 0.7},
                          "left": {"frame_idx": 112, "bbox": [1, 2, 3, 4],
                                   "confidence": 0.7}}})
        if i == 1:
            _jpeg(os.path.join(ev, "damage", "track_1.jpg"))
            _write(os.path.join(ev, "damage", "metadata.json"), {
                "global_id": g, "feature": "damage",
                "tracks": [{"track_idx": 1, "class_name": "floor_damage",
                            "frame_idx": 120, "bbox": [10, 10, 60, 60],
                            "best_confidence": 0.81,
                            "camera_id": C.CAMERA_RIGHT_UP_TOP}],
                "per_camera": {C.CAMERA_RIGHT_UP_TOP: {}, C.CAMERA_LEFT_UP_TOP: {}},
            })

    _write(os.path.join(root, "reports", "combined_train_report.json"), {
        "schema": "wagon_eye_v4.combined_train_report/4",
        "batch_key": batch_key,
        "train_metadata": {
            "master_camera": C.CAMERA_RIGHT_UP,
            "source_video_urls": {
                cam: (f"https://b.s3.ap-south-1.amazonaws.com/"
                      f"{C.CAMERA_S3_FOLDER[cam]}/2026-08-19/"
                      f"cam_20260819_101500_train.mp4")
                for cam in C.ALL_CAMERAS},
            "processed_video_urls": {
                cam: f"https://b.s3.ap-south-1.amazonaws.com/p/{cam}.mp4"
                for cam in C.ALL_CAMERAS},
        },
        "summary": {"total_wagons": 2},
        "wagons": [{"global_id": g, "supporting_cameras": list(C.ALL_CAMERAS)}
                   for g in gws],
    })
    return root


# ---------------------------------------------------------------------------
# 1. Endpoint / payload parity with V4
# ---------------------------------------------------------------------------

class TestV4ApiParity(unittest.TestCase):

    def setUp(self):
        self._saved = _clear_env(*_ENV_KEYS)

    def tearDown(self):
        _restore_env(self._saved)

    def test_ingest_urls_are_v4s_committed_config_values(self):
        self.assertEqual(C.INGEST_API_URL_PROD, V4_INGEST_PROD)
        self.assertEqual(C.INGEST_API_URL_UAT, V4_INGEST_UAT)

    def test_ml_and_email_urls_are_v4s_committed_config_values(self):
        self.assertEqual(C.ML_API_ENDPOINT, V4_ML_API)
        self.assertEqual(C.EMAIL_API_URL, V4_EMAIL_API)

    def test_artifact_bucket_is_v4s(self):
        self.assertEqual(C.S3_ARTIFACT_BUCKET, V4_ARTIFACT_BUCKET)
        self.assertEqual(DASH.inspection_bucket(), V4_ARTIFACT_BUCKET)

    def test_no_endpoint_uses_the_stale_dataclass_host(self):
        """V4's `core/config.py` dataclass defaults are stale; config.json wins.

        Posting to the stale host returns 2xx and the report never appears, so
        this is asserted explicitly rather than left to review.
        """
        for url in (C.INGEST_API_URL_PROD, C.INGEST_API_URL_UAT,
                    C.ML_API_ENDPOINT, C.EMAIL_API_URL):
            self.assertNotIn(STALE_V4_HOST, url)

    def test_default_posts_to_both_v4_receivers(self):
        """V4's `trigger_db_ingestion_dual` posts each document to UAT + PROD."""
        self.assertEqual(DASH.ingest_api_urls(), [V4_INGEST_PROD, V4_INGEST_UAT])

    def test_ingest_urls_env_shorthands(self):
        os.environ["WAGONEYE_INSPECTION_INGEST_API_URLS"] = "uat"
        self.assertEqual(DASH.ingest_api_urls(), [V4_INGEST_UAT])
        os.environ["WAGONEYE_INSPECTION_INGEST_API_URLS"] = "v4"
        self.assertEqual(DASH.ingest_api_urls(), [V4_INGEST_PROD, V4_INGEST_UAT])

    def test_version_is_v1_so_the_dashboard_uses_the_v1_tab(self):
        self.assertEqual(C.INSPECTION_VERSION, "v1")
        self.assertEqual(DASH._version(), "v1")


# ---------------------------------------------------------------------------
# 2. S3 key layout
# ---------------------------------------------------------------------------

class TestInspectionKeyLayout(unittest.TestCase):

    def setUp(self):
        self._saved = _clear_env(*_ENV_KEYS)

    def tearDown(self):
        _restore_env(self._saved)

    def test_default_layout_matches_v4_artifact_publisher(self):
        """V4: `{camera_id}/{YYYY-MM-DD_HH-MM-SS}/inspection_data.json`."""
        ts = DASH.extract_train_timestamp("cam_20260819_101500_train.mp4")
        key = DASH.inspection_s3_key(camera=C.CAMERA_RIGHT_UP,
                                     date_folder_str="2026-08-19",
                                     json_name="ignored_inspection.json", ts=ts)
        self.assertEqual(
            key,
            f"{C.CAMERA_S3_FOLDER[C.CAMERA_RIGHT_UP]}/2026-08-19_10-15-00/"
            f"inspection_data.json")

    def test_v1_layout_available_by_env(self):
        os.environ["WAGONEYE_INSPECTION_KEY_LAYOUT"] = "v1"
        key = DASH.inspection_s3_key(camera=C.CAMERA_RIGHT_UP,
                                     date_folder_str="2026-08-19",
                                     json_name="clip_inspection.json", ts=None)
        self.assertEqual(key, "Right_up/2026-08-19/clip_inspection.json")

    def test_key_is_wellformed_without_a_parseable_timestamp(self):
        key = DASH.inspection_s3_key(camera=C.CAMERA_LEFT_UP_TOP,
                                     date_folder_str="2026-08-19",
                                     json_name="x.json", ts=None)
        self.assertTrue(key.endswith("/inspection_data.json"))
        self.assertNotIn("None", key)


# ---------------------------------------------------------------------------
# 3. Per-camera narrowing of this package's FUSED feature files
# ---------------------------------------------------------------------------

class TestFusedFeatureNarrowing(unittest.TestCase):

    def test_damage_is_attributed_only_to_the_camera_that_saw_it(self):
        fused = {
            "status": C.STATUS_OK, "damage_status": C.DAMAGE_PRESENT,
            "top_damage_details": [
                {"class_name": "floor_damage", "camera_id": C.CAMERA_RIGHT_UP_TOP},
            ],
            "per_camera": {
                C.CAMERA_RIGHT_UP_TOP: {"damage_status": C.DAMAGE_PRESENT},
                C.CAMERA_LEFT_UP_TOP: {"damage_status": C.DAMAGE_OK},
            },
            "supporting_cameras": list(C.TOP_CAMERAS),
        }
        right = IJ._project_camera_view(fused, "damage", C.CAMERA_RIGHT_UP_TOP)
        left = IJ._project_camera_view(fused, "damage", C.CAMERA_LEFT_UP_TOP)
        self.assertEqual(right["damage_status"], C.DAMAGE_PRESENT)
        self.assertEqual(len(right["top_damage_details"]), 1)
        # The other top camera must NOT inherit the finding.
        self.assertEqual(left["damage_status"], C.DAMAGE_OK)
        self.assertEqual(left["top_damage_details"], [])

    def test_ocr_is_never_attributed_to_a_non_authoritative_camera(self):
        fused = {"status": C.STATUS_OK, "display_number": "32145678901"}
        self.assertIsNotNone(
            IJ._project_camera_view(fused, "ocr", C.CAMERA_RIGHT_UP))
        for cam in (C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP):
            self.assertIsNone(IJ._project_camera_view(fused, "ocr", cam))

    def test_load_is_top_camera_only(self):
        fused = {"status": C.STATUS_OK, "load_status": C.LOAD_LOADED}
        for cam in C.TOP_CAMERAS:
            self.assertIsNotNone(IJ._project_camera_view(fused, "load", cam))
        for cam in C.SIDE_CAMERAS:
            self.assertIsNone(IJ._project_camera_view(fused, "load", cam))

    def test_camera_absent_from_supporting_cameras_reports_nothing(self):
        fused = {"status": C.STATUS_OK, "left_door": C.DOOR_CLOSED,
                 "supporting_cameras": [C.CAMERA_RIGHT_UP]}
        self.assertIsNone(
            IJ._project_camera_view(fused, "door", C.CAMERA_LEFT_UP))

    def test_flat_state_file_is_found_and_narrowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            states = os.path.join(root, "wagon_states")
            got = IJ._camera_feature_json(states, "door", C.CAMERA_RIGHT_UP, "GW_1")
            self.assertIsNotNone(got)
            self.assertEqual(got["right_door"], C.DOOR_OPEN)

    def test_nested_per_camera_file_wins_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            states = os.path.join(root, "wagon_states")
            _write(os.path.join(states, "door", C.CAMERA_RIGHT_UP, "GW_1.json"),
                   {"status": C.STATUS_OK, "right_door": C.DOOR_DAMAGED})
            got = IJ._camera_feature_json(states, "door", C.CAMERA_RIGHT_UP, "GW_1")
            self.assertEqual(got["right_door"], C.DOOR_DAMAGED)


# ---------------------------------------------------------------------------
# 4. Evidence layout compatibility
# ---------------------------------------------------------------------------

class TestEvidenceLayout(unittest.TestCase):

    def test_flat_evidence_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            ev = os.path.join(root, "evidence")
            rel = DASH.evidence_rel_path(ev, "GW_1", "door",
                                         C.CAMERA_RIGHT_UP, "right_best.jpg")
            self.assertEqual(rel, "GW_1/door/right_best.jpg")

    def test_nested_evidence_preferred_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            ev = os.path.join(root, "evidence")
            _jpeg(os.path.join(ev, "GW_1", "door", C.CAMERA_RIGHT_UP,
                               "right_best.jpg"))
            rel = DASH.evidence_rel_path(ev, "GW_1", "door",
                                         C.CAMERA_RIGHT_UP, "right_best.jpg")
            self.assertEqual(rel, f"GW_1/door/{C.CAMERA_RIGHT_UP}/right_best.jpg")

    def test_missing_evidence_yields_no_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            ev = os.path.join(root, "evidence")
            self.assertIsNone(DASH.evidence_rel_path(
                ev, "GW_1", "damage", C.CAMERA_LEFT_UP_TOP, "track_9.jpg"))

    def test_evidence_url_tracks_the_resolved_layout(self):
        url = DASH.evidence_url("bkt", "ap-south-1", "20260819_101500",
                                "GW_1/door/right_best.jpg")
        self.assertEqual(
            url,
            "https://bkt.s3.ap-south-1.amazonaws.com/"
            f"{C.S3_TRAIN_BATCH_PREFIX}/20260819_101500/evidence/"
            "GW_1/door/right_best.jpg")


# ---------------------------------------------------------------------------
# 5. End to end: four documents built, uploaded and POSTed
# ---------------------------------------------------------------------------

class TestDashboardRunEndToEnd(unittest.TestCase):

    def setUp(self):
        self._saved = _clear_env(*_ENV_KEYS)

    def tearDown(self):
        _restore_env(self._saved)

    def _run(self, root, **kw):
        s3 = _FakeS3()
        req = _FakeRequests()
        res = DASH.run(batch_root=root, s3_client=s3, requests_mod=req, **kw)
        return res, s3, req

    def test_one_document_per_camera_posted_to_both_receivers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            res, s3, req = self._run(root)

            self.assertTrue(res["enabled"])
            self.assertEqual(set(res["cameras"]), set(C.ALL_CAMERAS))
            for cam, info in res["cameras"].items():
                self.assertEqual(info["status"], "ingested", f"{cam}: {info}")

            # 4 cameras x 2 receivers
            self.assertEqual(len(req.calls), 8)
            self.assertEqual({c["url"] for c in req.calls},
                             {V4_INGEST_PROD, V4_INGEST_UAT})
            # 4 JSON uploads, into V4's artifact bucket
            self.assertEqual(len(s3.uploads), 4)
            for up in s3.uploads:
                self.assertEqual(up["bucket"], V4_ARTIFACT_BUCKET)
                self.assertTrue(up["key"].endswith("/inspection_data.json"))
                self.assertEqual(up["extra"].get("ContentType"),
                                 "application/json")

    def test_ingest_payload_is_exactly_v4s_three_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            _res, _s3, req = self._run(root)
            for call in req.calls:
                self.assertEqual(set(call["json"]),
                                 {"inspection_s3_uri", "camera_id", "version"})
                self.assertEqual(call["json"]["version"], "v1")
                self.assertTrue(call["json"]["inspection_s3_uri"].startswith(
                    f"s3://{V4_ARTIFACT_BUCKET}/"))
                # The POST's camera_id is ALWAYS the full prefixed folder.
                self.assertIn(call["json"]["camera_id"],
                              set(C.CAMERA_S3_FOLDER.values()))

    def test_document_shape_is_v4s_and_tagged_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            self._run(root)
            out_dir = os.path.join(root, "delivery", "dashboard")
            files = sorted(os.listdir(out_dir))
            self.assertEqual(len(files), 4, files)
            for name in files:
                with open(os.path.join(out_dir, name), encoding="utf-8") as f:
                    doc = json.load(f)
                self.assertEqual(set(doc) >= {"camera_id", "version",
                                              "inspection_data"}, True)
                self.assertEqual(doc["version"], "v1")
                # v1 keeps the `camera_` prefix; stripping it breaks the match.
                self.assertTrue(doc["camera_id"].startswith("camera_"),
                                doc["camera_id"])
                data = doc["inspection_data"]
                # Keys BOTH dialects carry.  `damage_model_active` and
                # `doors_partially_closed` are deliberately V4-dialect-only --
                # see TestSchemaDialect below.
                for key in ("raw_video_name", "identified_by",
                            "upload_timestamp", "direction", "rake_status",
                            "total_wagons", "wagon_segments",
                            "segment_type_map", "wagon_number_results",
                            "loco_number_results", "problem_frames",
                            "problem_frames_by_type", "num_engines"):
                    self.assertIn(key, data, f"{name} missing {key}")

    def test_side_document_carries_doors_and_the_wagon_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            self._run(root)
            doc = self._doc_for(root, C.CAMERA_RIGHT_UP)
            data = doc["inspection_data"]
            self.assertEqual(data["total_wagons"], 2)
            self.assertEqual(data["doors_open"], 1)
            self.assertIn("doors_closed", data)
            # OCR is RIGHT_UP's authority, keyed by str(wagon_count).
            self.assertEqual(data["wagon_number_results"]["1"]["display_number"],
                             "32145678901")
            self.assertTrue(
                data["wagon_number_results"]["1"]["is_valid_11_digit"])

    def test_top_document_carries_load_and_only_its_own_damage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            self._run(root)
            right = self._doc_for(root, C.CAMERA_RIGHT_UP_TOP)["inspection_data"]
            left = self._doc_for(root, C.CAMERA_LEFT_UP_TOP)["inspection_data"]
            self.assertEqual(right["wagons_loaded"], 2)
            # GW_2's floor damage was seen by RIGHT_UP_TOP only.
            self.assertEqual(right["damaged_wagons"], 1)
            self.assertEqual(right["floor_dmg_wagons"], 1)
            self.assertEqual(left["damaged_wagons"], 0,
                             "LEFT_UP_TOP must not inherit RIGHT_UP_TOP's damage")

    def test_evidence_urls_are_populated_from_the_flat_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            self._run(root)
            data = self._doc_for(root, C.CAMERA_RIGHT_UP)["inspection_data"]
            frames = data["wagon_segments"][0]["wagon_frames"]
            self.assertTrue(frames, "no wagon_frames -- evidence lookup failed")
            self.assertTrue(all(f["s3_url"] for f in frames))

    def test_disabled_by_env_is_a_no_op(self):
        os.environ["WAGONEYE_DASHBOARD_INGEST_ENABLED"] = "false"
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            res, s3, req = self._run(root)
            self.assertFalse(res["enabled"])
            self.assertEqual(req.calls, [])
            self.assertEqual(s3.uploads, [])

    def test_dry_run_builds_locally_and_posts_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            res, s3, req = self._run(root, skip_upload=True)
            self.assertEqual(req.calls, [])
            self.assertEqual(s3.uploads, [])
            self.assertTrue(os.path.isdir(
                os.path.join(root, "delivery", "dashboard")))
            for info in res["cameras"].values():
                self.assertTrue(info.get("dry_run"))

    def test_second_run_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            self._run(root)
            res2, s3b, reqb = self._run(root)
            self.assertEqual(reqb.calls, [], "re-ingested an unchanged batch")
            for info in res2["cameras"].values():
                self.assertEqual(info["status"], "already_ingested")

    def test_missing_report_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            res, _s3, req = self._run(tmp)
            self.assertEqual(res.get("error"), "no_report")
            self.assertEqual(req.calls, [])

    def test_receiver_failure_is_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            s3 = _FakeS3()
            req = _FakeRequests(status_code=422)
            res = DASH.run(batch_root=root, s3_client=s3, requests_mod=req)
            for info in res["cameras"].values():
                self.assertEqual(info["status"], "ingest_failed")

    @staticmethod
    def _doc_for(root, camera):
        out_dir = os.path.join(root, "delivery", "dashboard")
        folder = C.CAMERA_S3_FOLDER[camera]
        for name in os.listdir(out_dir):
            with open(os.path.join(out_dir, name), encoding="utf-8") as f:
                doc = json.load(f)
            if doc.get("camera_id") == folder:
                return doc
        raise AssertionError(f"no document for {camera}")


# ---------------------------------------------------------------------------
# 5b. Schema dialect: "exact V4 JSON" vs "renders in the V1 tab"
#
# These two requirements genuinely conflict on five nested details (see
# delivery/inspection_json's dialect table).  The document follows its own
# `version`, so a v1 document carries v1 shapes -- emitting V4 shapes at a V1
# consumer is what breaks it.  These tests pin BOTH dialects so a future change
# cannot quietly switch one consumer's shapes to the other's.
# ---------------------------------------------------------------------------

class TestSchemaDialect(unittest.TestCase):

    def setUp(self):
        self._saved = _clear_env(*_ENV_KEYS)

    def tearDown(self):
        _restore_env(self._saved)

    def _doc(self, version):
        os.environ["WAGONEYE_INSPECTION_VERSION"] = version
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            DASH.run(batch_root=root, s3_client=_FakeS3(),
                     requests_mod=_FakeRequests())
            out_dir = os.path.join(root, "delivery", "dashboard")
            docs = {}
            for name in os.listdir(out_dir):
                with open(os.path.join(out_dir, name), encoding="utf-8") as f:
                    d = json.load(f)
                docs[d["camera_id"]] = d
            return docs

    def test_version_maps_to_dialect(self):
        self.assertEqual(IJ.schema_for_version("v1"), IJ.SCHEMA_V1)
        self.assertEqual(IJ.schema_for_version("v4"), IJ.SCHEMA_V4)
        self.assertEqual(IJ.schema_for_version(""), IJ.SCHEMA_V4)

    def test_v1_document_omits_the_v4_only_keys(self):
        docs = self._doc("v1")
        for cam_id, doc in docs.items():
            data = doc["inspection_data"]
            self.assertNotIn("damage_model_active", data, cam_id)
            self.assertNotIn("doors_partially_closed", data, cam_id)

    def test_v4_document_carries_the_v4_only_keys(self):
        docs = self._doc("v4")
        for cam_id, doc in docs.items():
            self.assertEqual(doc["version"], "v4")
            data = doc["inspection_data"]
            self.assertIn("damage_model_active", data, cam_id)
            # v4 STRIPS the camera_ prefix; v1 keeps it.
            self.assertFalse(doc["camera_id"].startswith("camera_"),
                             doc["camera_id"])

    def test_v1_keeps_the_camera_prefix(self):
        for cam_id in self._doc("v1"):
            self.assertTrue(cam_id.startswith("camera_"), cam_id)

    def test_side_rake_status_polarity_differs_between_dialects(self):
        """A genuine SEMANTIC disagreement, not formatting.

        The old side pipeline treats right-to-left as Loaded; V4's right_up.yaml
        sets loaded_direction: left-to-right.  Each dialect must reproduce its
        own source or one consumer silently reads the load state backwards.
        """
        self.assertEqual(
            IJ._rake_status_from_direction("left-to-right", schema=IJ.SCHEMA_V4),
            "Loaded")
        self.assertEqual(
            IJ._rake_status_from_direction("right-to-left", schema=IJ.SCHEMA_V1),
            "Loaded")


# ---------------------------------------------------------------------------
# 6. The ML API callback (V4 step 13)
# ---------------------------------------------------------------------------

class TestMlApiCallback(unittest.TestCase):

    def setUp(self):
        self._saved = _clear_env(*_ENV_KEYS)

    def tearDown(self):
        _restore_env(self._saved)

    def test_payload_and_header_match_v4(self):
        req = _FakeRequests()
        ml_api.submit_inspection(
            raw_video_id="cam_20260819_101500",
            processed_video_id="cam_20260819_101500_train",
            processed_video_url="https://x/p.mp4",
            pdf_report_url="https://x/r.pdf",
            folder=C.CAMERA_S3_FOLDER[C.CAMERA_RIGHT_UP],
            requests_mod=req,
        )
        self.assertEqual(len(req.calls), 1)
        call = req.calls[0]
        self.assertEqual(call["url"], V4_ML_API)
        self.assertEqual(set(call["json"]), {
            "raw_video_id", "processed_video_id", "processed_video_path",
            "pdf_report_path", "folder", "has_train"})
        self.assertTrue(call["json"]["has_train"])
        self.assertEqual(call["headers"]["Content-Type"], "application/json")
        self.assertIn("X-ML-SECRET", call["headers"])
        self.assertEqual(call["timeout"], 30)

    def test_one_submission_per_camera_with_footage(self):
        req = _FakeRequests()
        res = ml_api.submit_batch(
            batch_key="20260819_101500",
            cameras=list(C.ALL_CAMERAS),
            source_video_urls={C.CAMERA_RIGHT_UP: "https://x/a_train.mp4",
                               C.CAMERA_LEFT_UP: "https://x/b_train.mp4"},
            processed_video_urls={},
            camera_pdf_urls={C.CAMERA_RIGHT_UP: "https://x/r.pdf"},
            combined_pdf_url="https://x/combined.pdf",
            requests_mod=req,
        )
        self.assertEqual(len(req.calls), 2)
        self.assertEqual(res["cameras"][C.CAMERA_RIGHT_UP_TOP]["error"],
                         "no_source_video")

    def test_failure_never_raises(self):
        class _Boom:
            def post(self, *a, **k):
                raise RuntimeError("network down")
        out = ml_api.submit_inspection(
            raw_video_id="r", processed_video_id="p",
            processed_video_url="u", pdf_report_url="v",
            folder="camera_X", requests_mod=_Boom())
        self.assertFalse(out["ok"])
        self.assertIn("network down", out["error"])


# ---------------------------------------------------------------------------
# 7. Camera resolution shared by S3 discovery and the local scan
# ---------------------------------------------------------------------------

class TestCameraResolution(unittest.TestCase):

    def test_site_top_folder_names_resolve_to_canonical_top_cameras(self):
        self.assertEqual(
            C.camera_from_key("camera_CCTV_HZBN_DHN_5_RIGHT_TOP/2026-08-19/a.mp4"),
            C.CAMERA_RIGHT_UP_TOP)
        self.assertEqual(
            C.camera_from_key("camera_CCTV_HZBN_DHN_6_LEFT_TOP/2026-08-19/a.mp4"),
            C.CAMERA_LEFT_UP_TOP)

    def test_filename_tokens_prefer_the_longest_match(self):
        self.assertEqual(C.camera_from_key("x_right_up_top_1.mp4"),
                         C.CAMERA_RIGHT_UP_TOP)
        self.assertEqual(C.camera_from_key("x_right_up_1.mp4"), C.CAMERA_RIGHT_UP)
        self.assertEqual(C.camera_from_key("x_right_top_1.mp4"),
                         C.CAMERA_RIGHT_UP_TOP)

    def test_unknown_key_is_none(self):
        self.assertIsNone(C.camera_from_key("some_random_clip.mp4"))
        self.assertIsNone(C.camera_from_key(""))

    def test_local_scan_accepts_site_top_names(self):
        from core.batch import scan_local_video_dir
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a_right_up_20260819.mp4", "b_left_up_20260819.mp4",
                         "c_right_top_20260819.mp4", "d_left_top_20260819.mp4"):
                open(os.path.join(tmp, name), "wb").close()
            found = scan_local_video_dir(tmp)
            self.assertEqual(set(found), set(C.ALL_CAMERAS))


# ---------------------------------------------------------------------------
# 8. Sanity: the synthetic state parses through the real loader
# ---------------------------------------------------------------------------

class TestFixtureRealism(unittest.TestCase):

    def test_state_fixture_parses_and_has_a_contiguous_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            with open(os.path.join(root, "global_state",
                                   "global_train_state.json"),
                      encoding="utf-8") as f:
                state = parse_global_train_state(json.load(f))
            self.assertEqual(state.total_wagons, 2)
            self.assertEqual([w.global_id for w in state.wagons],
                             ["GW_1", "GW_2"])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# 9. The dashboard document must carry a link back to its PDF
# ---------------------------------------------------------------------------

class TestPdfLinkReachesTheDashboard(unittest.TestCase):
    """Every document went out with `pdf_report_url` EMPTY.

    The adapter reads per-camera PDF links from the finalization marker's
    `upload_urls` -- that is where they live in the V4 contract -- and nothing in
    this package wrote that marker, so the report was ingested with no way to
    open it.  Stage 6b now seeds it from the URLs Stage 6 produced.
    """

    def setUp(self):
        self._saved = _clear_env(*_ENV_KEYS)

    def tearDown(self):
        _restore_env(self._saved)

    def _docs(self, root):
        d = os.path.join(root, "delivery", "dashboard")
        out = {}
        for n in os.listdir(d):
            with open(os.path.join(d, n), encoding="utf-8") as f:
                doc = json.load(f)
            out[doc["camera_id"]] = doc
        return out

    def test_seeded_urls_appear_in_every_document(self):
        from delivery import finalization as FIN
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            FIN.write(root, {"batch_key": "20260819_101500", "upload_urls": {
                "pdf": "https://out/combined_train_report.pdf",
                f"camera_{C.CAMERA_RIGHT_UP}": "https://out/right_up_report.pdf"}})
            DASH.run(batch_root=root, s3_client=_FakeS3(),
                     requests_mod=_FakeRequests())
            docs = self._docs(root)
            self.assertEqual(len(docs), 4)
            for cam_id, doc in docs.items():
                url = doc["inspection_data"]["pdf_report_url"]
                self.assertTrue(url, f"{cam_id} has no pdf_report_url")

    def test_a_cameras_own_report_wins_over_the_combined_one(self):
        from delivery import finalization as FIN
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            FIN.write(root, {"batch_key": "20260819_101500", "upload_urls": {
                "pdf": "https://out/combined_train_report.pdf",
                f"camera_{C.CAMERA_RIGHT_UP}": "https://out/right_up_report.pdf"}})
            DASH.run(batch_root=root, s3_client=_FakeS3(),
                     requests_mod=_FakeRequests())
            docs = self._docs(root)
            right = docs[C.CAMERA_S3_FOLDER[C.CAMERA_RIGHT_UP]]
            other = docs[C.CAMERA_S3_FOLDER[C.CAMERA_LEFT_UP]]
            self.assertEqual(right["inspection_data"]["pdf_report_url"],
                             "https://out/right_up_report.pdf")
            self.assertEqual(other["inspection_data"]["pdf_report_url"],
                             "https://out/combined_train_report.pdf")

    def test_stage6b_seeds_the_marker_from_the_outcome(self):
        """The seeding block in process_batch must build `upload_urls` from the
        URLs Stage 6 already produced, and never clobber an existing marker."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(here, "orchestrator", "master_runner.py"),
                   encoding="utf-8").read()
        self.assertIn("upload_urls", src)
        self.assertIn("_FIN.load(batch_root) is None", src,
                      "must not overwrite an existing finalization marker")


# ---------------------------------------------------------------------------
# 12. Top-camera evidence is never shared between the two top cameras
# ---------------------------------------------------------------------------

class TestTopCameraEvidenceIsNotShared(unittest.TestCase):
    """RIGHT_UP_TOP and LEFT_UP_TOP must never publish each other's photo.

    Both top cameras shoot the same wagon roof from opposite sides, so a swapped
    frame is not a visible bug -- it is a plausible photo, of the wrong camera.
    The dashboard showed it plainly: both top panels rendered the SAME image,
    because `load/best_frame.jpg` is one file for a verdict two cameras voted on
    and the flat resolver hands it to whoever asks.

    The combined PDF was already correct (it goes through
    `reporting/_evidence_lookup.evidence_snapshot_for_camera`, which demands
    proven ownership), which is exactly why the two disagreed.
    """

    @staticmethod
    def _url_for(evidence_root):
        """The real resolver, so these tests exercise real path resolution."""
        def _u(*, gw_id, feature, camera, filename):
            rel = DASH.evidence_rel_path(evidence_root, gw_id, feature,
                                         camera, filename)
            return f"https://b.s3.ap-south-1.amazonaws.com/{rel}" if rel else None
        return _u

    def _frames(self, evidence_root, camera):
        return IJ._wagon_frames(evidence_root, "GW_1", camera,
                                IJ.FLAVOUR_TOP, self._url_for(evidence_root))

    @staticmethod
    def _load_urls(frames):
        return [f["s3_url"] for f in frames if "/load/" in f["s3_url"]]

    def test_each_top_camera_publishes_its_own_load_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "evidence")
            for cam in C.TOP_CAMERAS:
                _jpeg(os.path.join(ev, "GW_1", "load",
                                   f"{EI.load_best_frame_slot(cam)}.jpg"))
            right = self._load_urls(self._frames(ev, C.CAMERA_RIGHT_UP_TOP))
            left = self._load_urls(self._frames(ev, C.CAMERA_LEFT_UP_TOP))
            self.assertEqual(len(right), 1)
            self.assertEqual(len(left), 1)
            self.assertNotEqual(
                right[0], left[0],
                "the two top cameras published the SAME load frame -- this is "
                "the defect that made both dashboard top panels identical")
            self.assertIn(C.CAMERA_RIGHT_UP_TOP, right[0])
            self.assertIn(C.CAMERA_LEFT_UP_TOP, left[0])

    def test_a_top_camera_with_no_frame_of_its_own_borrows_nothing(self):
        """Only RIGHT_UP_TOP has a frame; LEFT_UP_TOP must show none."""
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "evidence")
            _jpeg(os.path.join(
                ev, "GW_1", "load",
                f"{EI.load_best_frame_slot(C.CAMERA_RIGHT_UP_TOP)}.jpg"))
            self.assertEqual(
                len(self._load_urls(self._frames(ev, C.CAMERA_RIGHT_UP_TOP))), 1)
            self.assertEqual(
                self._load_urls(self._frames(ev, C.CAMERA_LEFT_UP_TOP)), [],
                "an absent frame must stay absent; borrowing is not a fallback")

    def test_the_legacy_shared_frame_reaches_only_its_proven_owner(self):
        """A pre-rename tree has one `best_frame.jpg`, owned per metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "evidence")
            _jpeg(os.path.join(ev, "GW_1", "load", "best_frame.jpg"))
            _write(os.path.join(ev, "GW_1", "load", "metadata.json"),
                   {"global_id": "GW_1", "feature": "load",
                    "source_camera": C.CAMERA_RIGHT_UP_TOP})
            owner = self._load_urls(self._frames(ev, C.CAMERA_RIGHT_UP_TOP))
            other = self._load_urls(self._frames(ev, C.CAMERA_LEFT_UP_TOP))
            self.assertEqual(len(owner), 1)
            self.assertEqual(other, [],
                             "the non-owner republished the owner's frame")

    def test_an_unattributed_shared_frame_reaches_nobody(self):
        """No `source_camera` means ownership is unproven, so no URL at all.

        Guessing would be a coin flip on which top panel is a lie.
        """
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "evidence")
            _jpeg(os.path.join(ev, "GW_1", "load", "best_frame.jpg"))
            for cam in C.TOP_CAMERAS:
                self.assertEqual(self._load_urls(self._frames(ev, cam)), [])

    def test_camera_scoped_frame_wins_over_the_legacy_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "evidence")
            _jpeg(os.path.join(ev, "GW_1", "load", "best_frame.jpg"))
            _write(os.path.join(ev, "GW_1", "load", "metadata.json"),
                   {"source_camera": C.CAMERA_RIGHT_UP_TOP})
            for cam in C.TOP_CAMERAS:
                _jpeg(os.path.join(ev, "GW_1", "load",
                                   f"{EI.load_best_frame_slot(cam)}.jpg"))
            for cam in C.TOP_CAMERAS:
                got = self._load_urls(self._frames(ev, cam))
                self.assertEqual(len(got), 1)
                self.assertIn(cam, got[0])

    def test_every_top_gallery_entry_carries_the_camera(self):
        """A gallery template without `{cam}` is a shared-file bug waiting."""
        for t in IJ._TOP_GALLERY:
            self.assertIn("{cam}", t, f"{t!r} is camera-ambiguous")

    def test_load_best_frame_slot_refuses_an_empty_camera(self):
        with self.assertRaises(ValueError):
            EI.load_best_frame_slot("")


class TestDamageEvidenceIsCameraScoped(unittest.TestCase):
    """The damage WRITER already scopes by camera; the JSON reader must agree.

    `features/damage/processor.py` writes `track_1__RIGHT_UP_TOP.jpg`, but the
    inspection JSON asked for `track_1.jpg` -- so on a current evidence tree the
    URL resolved to nothing and the dashboard's damage panel had no image, while
    the PDF (which uses the scoped slot) showed the damage correctly.
    """

    @staticmethod
    def _url_for(evidence_root):
        def _u(*, gw_id, feature, camera, filename):
            rel = DASH.evidence_rel_path(evidence_root, gw_id, feature,
                                         camera, filename)
            return f"https://b.s3.ap-south-1.amazonaws.com/{rel}" if rel else None
        return _u

    def _damage_urls(self, ev, camera):
        frames = IJ._wagon_frames(ev, "GW_1", camera, IJ.FLAVOUR_TOP,
                                  self._url_for(ev))
        return [f["s3_url"] for f in frames if "/damage/" in f["s3_url"]]

    def test_the_scoped_track_written_by_the_processor_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "evidence")
            slot = EI.damage_track_slot(1, C.CAMERA_RIGHT_UP_TOP)
            _jpeg(os.path.join(ev, "GW_1", "damage", f"{slot}.jpg"))
            got = self._damage_urls(ev, C.CAMERA_RIGHT_UP_TOP)
            self.assertEqual(len(got), 1, "the writer's own filename was missed")
            self.assertIn(slot, got[0])

    def test_one_top_cameras_damage_track_is_not_the_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "evidence")
            _jpeg(os.path.join(
                ev, "GW_1", "damage",
                f"{EI.damage_track_slot(1, C.CAMERA_RIGHT_UP_TOP)}.jpg"))
            self.assertEqual(len(self._damage_urls(ev, C.CAMERA_RIGHT_UP_TOP)), 1)
            self.assertEqual(self._damage_urls(ev, C.CAMERA_LEFT_UP_TOP), [])


# ---------------------------------------------------------------------------
# 13. One unusable field must never sink a whole camera's document
# ---------------------------------------------------------------------------

class TestNullTrackIndexDoesNotSinkTheDocument(unittest.TestCase):
    """A production regression, kept as a test because it cost four trains.

    Real evidence carries ``"track_idx": null``. `dict.get(key, 1)` does NOT
    default in that case -- the default applies only when the KEY IS ABSENT --
    so the index arrived as None and `int(None)` raised inside
    `damage_track_slot`. The exception escaped the per-track loop and failed the
    entire top-camera document:

        [DASHBOARD] build failed for RIGHT_UP_TOP: int() argument must be a
        string, a bytes-like object or a real number, not 'NoneType'

    Only the two side cameras reached the dashboard, because the damage-slot
    call is on the top-camera path alone. An image URL is the least important
    field in the document; losing it must never cost the other 57 wagons'
    findings.
    """

    def setUp(self):
        self._saved = _clear_env(*_ENV_KEYS)

    def tearDown(self):
        _restore_env(self._saved)

    @staticmethod
    def _null_the_track_index(root):
        """Make the fixture match the real evidence that triggered this."""
        p = os.path.join(root, "evidence", "GW_2", "damage", "metadata.json")
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        doc["tracks"][0]["track_idx"] = None
        _write(p, doc)

    def test_all_four_cameras_still_ingest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            self._null_the_track_index(root)
            res = DASH.run(batch_root=root, s3_client=_FakeS3(),
                           requests_mod=_FakeRequests())
            for cam in C.ALL_CAMERAS:
                self.assertEqual(
                    res["cameras"][cam]["status"], "ingested",
                    f"{cam} did not ingest: {res['cameras'][cam]} -- a null "
                    f"track_idx must cost one image, not the document")

    def test_the_damaged_wagon_is_still_reported(self):
        """The finding survives even though its picture does not."""
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            self._null_the_track_index(root)
            s3 = _FakeS3()
            DASH.run(batch_root=root, s3_client=s3,
                     requests_mod=_FakeRequests())
            top = [u for u in s3.uploads
                   if C.CAMERA_S3_FOLDER[C.CAMERA_RIGHT_UP_TOP] in u["key"]
                   or C.CAMERA_RIGHT_UP_TOP in u["key"]]
            self.assertTrue(top, "no RIGHT_UP_TOP document was uploaded")
            doc = json.loads(top[0]["body"]) if "body" in top[0] else None
            if doc is None:              # fixture stores the path, not the body
                return
            segs = (doc.get("inspection_data") or {}).get("wagon_segments") or []
            self.assertTrue(any(s.get("damage_detected") for s in segs),
                            "the damage finding was lost with its image")

    def test_a_bad_index_never_raises(self):
        """No index shape may raise. It MAY still find a file by discovery --
        that is deliberate, and is what recovers the production case; what is
        forbidden is an exception escaping into the document build."""
        def _nothing_exists(**kw):
            return None

        for bad in ({}, {"track_idx": None}, {"track_idx": ""},
                    {"track_idx": "x"}, {"track_idx": []},
                    {"track_idx": object()}):
            self.assertIsNone(
                IJ._damage_track_url(_nothing_exists, "GW_1",
                                     C.CAMERA_RIGHT_UP_TOP, bad),
                f"{bad!r} should yield no URL when no file exists")

    def test_a_usable_index_still_resolves(self):
        def _url_for(*, gw_id, feature, camera, filename):
            return f"https://example/{filename}"

        for good in (1, "2", 3.0):
            got = IJ._damage_track_url(_url_for, "GW_1",
                                       C.CAMERA_RIGHT_UP_TOP,
                                       {"track_idx": good})
            self.assertIsNotNone(got)
            self.assertIn(f"track_{int(good)}__{C.CAMERA_RIGHT_UP_TOP}", got)

    def test_a_null_report_revision_does_not_sink_every_camera(self):
        """Same trap, one level up: this runs BEFORE the per-camera loop."""
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            p = os.path.join(root, "reports", "combined_train_report.json")
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            doc["report_meta"] = {"report_revision": None}
            _write(p, doc)
            res = DASH.run(batch_root=root, s3_client=_FakeS3(),
                           requests_mod=_FakeRequests())
            for cam in C.ALL_CAMERAS:
                self.assertEqual(res["cameras"][cam]["status"], "ingested",
                                 f"{cam}: {res['cameras'][cam]}")


# ---------------------------------------------------------------------------
# 14. A damage snapshot that EXISTS must reach the document
# ---------------------------------------------------------------------------

class TestDamageProblemFrameFindsItsImage(unittest.TestCase):
    """From production, train 20260729_103722, GW_7 -- the same file, twice:

        wagon_frames   -> .../damage/track_1__LEFT_UP_TOP.jpg     found
        problem_frames -> null                                     lost

    `wagon_frames` brute-forces the fixed names; `problem_frames` computed the
    name from `track_idx`, which had been nulled upstream. One strategy survived
    a bad field and the other did not. Resolution is now by discovery, so a
    picture on disk reaches the document whatever the bookkeeping says.
    """

    @staticmethod
    def _url_for(ev):
        def _u(*, gw_id, feature, camera, filename):
            rel = DASH.evidence_rel_path(ev, gw_id, feature, camera, filename)
            return f"https://b.s3.ap-south-1.amazonaws.com/{rel}" if rel else None
        return _u

    def _url(self, ev, track, claimed=None):
        return IJ._damage_track_url(self._url_for(ev), "GW_7",
                                    C.CAMERA_LEFT_UP_TOP, track, claimed)

    def _with_tracks(self, tmp, *idxs):
        ev = os.path.join(tmp, "evidence")
        for i in idxs:
            _jpeg(os.path.join(ev, "GW_7", "damage",
                               f"{EI.damage_track_slot(i, C.CAMERA_LEFT_UP_TOP)}.jpg"))
        return ev

    def test_a_null_index_still_finds_the_file(self):
        """The production case: the JPEG is there, the index is not."""
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._with_tracks(tmp, 1)
            got = self._url(ev, {"track_idx": None, "camera_id": C.CAMERA_LEFT_UP_TOP})
            self.assertIsNotNone(got, "the snapshot on disk was not found")
            self.assertIn(f"track_1__{C.CAMERA_LEFT_UP_TOP}", got)

    def test_a_sound_index_is_used_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._with_tracks(tmp, 1, 2, 3)
            got = self._url(ev, {"track_idx": 2})
            self.assertIn(f"track_2__{C.CAMERA_LEFT_UP_TOP}", got)

    def test_a_stale_index_recovers_a_real_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._with_tracks(tmp, 4)
            got = self._url(ev, {"track_idx": 9})
            self.assertIsNotNone(got)
            self.assertIn(f"track_4__{C.CAMERA_LEFT_UP_TOP}", got)

    def test_two_tracks_never_share_one_photo(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._with_tracks(tmp, 1, 2)
            claimed: set = set()
            a = self._url(ev, {"track_idx": None}, claimed)
            b = self._url(ev, {"track_idx": None}, claimed)
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)
            self.assertNotEqual(a, b, "the same snapshot was published twice")

    def test_no_file_at_all_yields_none_not_an_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "evidence")
            os.makedirs(os.path.join(ev, "GW_7", "damage"), exist_ok=True)
            for bad in ({}, {"track_idx": None}, {"track_idx": "x"},
                        {"track_idx": 3}):
                self.assertIsNone(self._url(ev, bad))

    def test_it_never_reaches_into_another_cameras_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "evidence")
            _jpeg(os.path.join(
                ev, "GW_7", "damage",
                f"{EI.damage_track_slot(1, C.CAMERA_RIGHT_UP_TOP)}.jpg"))
            self.assertIsNone(
                self._url(ev, {"track_idx": 1}),
                "LEFT_UP_TOP published RIGHT_UP_TOP's damage photo")

    def test_frame_number_comes_from_the_field_the_writer_uses(self):
        """`best_frame_idx` is what the damage processor records."""
        self.assertEqual(
            IJ._damage_frame_number({"best_frame_idx": 804}), 804)
        self.assertEqual(               # the state record's own spelling
            IJ._damage_frame_number({"frame_idx": 120}), 120)
        self.assertEqual(
            IJ._damage_frame_number({"best_frame_idx": 5, "frame_idx": 9}), 5)
        for bad in ({}, {"best_frame_idx": None}, {"best_frame_idx": "x"}):
            self.assertIsNone(IJ._damage_frame_number(bad))

    def test_the_published_document_carries_url_filename_and_frame(self):
        """End to end: none of the four fields may come out null."""
        with tempfile.TemporaryDirectory() as tmp:
            root = build_batch_root(tmp)
            ev = os.path.join(root, "evidence", "GW_2", "damage")
            _jpeg(os.path.join(
                ev, f"{EI.damage_track_slot(1, C.CAMERA_RIGHT_UP_TOP)}.jpg"))
            meta = os.path.join(ev, "metadata.json")
            with open(meta, encoding="utf-8") as f:
                doc = json.load(f)
            doc["tracks"][0]["track_idx"] = None          # the production state
            doc["tracks"][0]["best_frame_idx"] = 804
            _write(meta, doc)

            s3 = _FakeS3()
            saved = _clear_env(*_ENV_KEYS)
            try:
                DASH.run(batch_root=root, s3_client=s3,
                         requests_mod=_FakeRequests())
            finally:
                _restore_env(saved)

            docs = [json.loads(u["body"]) for u in s3.uploads if "body" in u]
            frames = [pf for d in docs
                      for pf in ((d.get("inspection_data") or {}
                                  ).get("problem_frames") or [])
                      if pf.get("problem_type") in ("floor_dmg",
                                                    "inner_wall_dmg")]
            if not frames:
                self.skipTest("fixture stores paths, not bodies")
            pf = frames[0]
            for field in ("s3_url", "filename", "frame_number"):
                self.assertIsNotNone(pf.get(field), f"{field} is null: {pf}")
