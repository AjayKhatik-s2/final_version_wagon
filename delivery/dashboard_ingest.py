"""delivery/dashboard_ingest.py -- Stage-6-only, read-only legacy dashboard adapter.

Purpose
-------
The pre-migration ("old") pipeline fed a per-camera dashboard by POSTing one
``*_inspection.json`` (schema ``{camera_id, version, inspection_data}``) per
camera angle to an S3 bucket and then calling a ``cctv-receiver/inspections/ingest``
API.  The train-state-native v4 pipeline does NOT produce that per-camera feed --
it emits one combined ``combined_train_report.json`` per train.

This module RE-DERIVES the legacy per-camera dashboard payload from finalized v4
artifacts so the existing dashboard keeps working, WITHOUT changing anything
about how the new system computes results.

Hard guarantees (by construction)
---------------------------------
* **Read-only w.r.t. the pipeline.**  It reads finalized artifacts only:
    <batch_root>/reports/combined_train_report.json
    <batch_root>/evidence/<GW>/<feature>/<CAMERA>/{metadata.json,*.jpg}
    <batch_root>/delivery/finalization.json
  It NEVER imports or mutates GlobalTrainState, feature processors, fusion, or
  the report builders, and it NEVER loads a model or opens a video.
* **Writes only under <batch_root>/delivery/.**  Generated JSON goes to
  ``delivery/dashboard/<CAMERA>_inspection.json``; ingest status is merged into
  ``delivery/finalization.json``.  Nothing else on disk is touched.
* **Enabled by default.**  ``WAGONEYE_DASHBOARD_INGEST_ENABLED`` defaults to
  ``true`` -- every finalized batch posts to the live ingest API (version v1).
  Set it to ``false`` to make ``run()`` a no-op (staging / shadow runs).
* **Failure-isolating.**  ``run()`` never raises; any error is logged and
  recorded.  It cannot corrupt the final report or the sealed batch state.
* **Idempotent across restarts.**  Per-camera ingest status (keyed by the
  generated JSON's sha256 + report revision) is persisted; an already-ingested
  camera is skipped on re-entry -- no duplicate uploads, no duplicate ingest.

Schema
------
The document is the EXACT V4 schema -- see `delivery/inspection_json.py`, a
faithful port of the V4 engine's ``reporting/json_builder.py`` (both flavours,
same keys, same key order, same derivations).  Per-camera authority is respected,
so the four files genuinely differ: RIGHT_UP reports the right door + OCR, LEFT_UP
the left door, and each top camera its own load + damage reads.

Fields whose V4 source does not exist in this pipeline (documented, never invented)
----------------------------------------------------------------------------------
* ``loco_frames`` /
  ``loco_number_results`` /
  ``total_loco_frames``    -> empty.  global_train has no loco-specific frame or
                              5-digit loco-OCR feed; V4's side flavour populates
                              these from a dedicated loco band pass.
* ``floor_dmg_probable``   -> always False on top cameras: the current
                              ``damage.pt`` has no "probable" class (V4's
                              ``V4_top_damage`` does).  Reported, never guessed.
* ``wagon_frames`` gallery -> assembled from whatever per-camera evidence JPEGs
                              exist, named with V4's start/mid1/mid2/end
                              positions; only files actually on disk are
                              referenced (never fabricated).
* ``s3_key`` on problem
  frames                   -> null; this adapter references the already-uploaded
                              ``train_batch/.../evidence/...`` URL rather than
                              re-uploading into the legacy key layout.

``direction`` is NO LONGER degraded: Stage 1 now derives the rake's travel
direction from the master camera's gap trajectories and persists it, so
``direction`` and the side-camera ``rake_status`` derived from it match V4's
vocabulary (see `wagon_count.global_alignment.travel_direction`).

Provenance for each payload is echoed under ``inspection_data._adapter``.

NOTE: this posts to the LIVE dashboard on every run.  Confirm with the dashboard
team that (a) reused ``train_batch/.../evidence/...`` HTTPS URLs are accepted and
(b) the degraded loco/direction/gallery fields are acceptable.  Set
``WAGONEYE_DASHBOARD_INGEST_ENABLED=false`` to disable without a code change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from core import constants as C
from core.logging_setup import get_logger
from delivery import finalization as FIN

log = get_logger("delivery.dashboard")

_IST = timezone(timedelta(hours=5, minutes=30))
_TS_RE = re.compile(r"(\d{8})_(\d{6})")
_DATE_RE = re.compile(r"(\d{8})")

# Local (delivery/) scratch subdir for generated per-camera JSON.
_LOCAL_SUBDIR = os.path.join("delivery", "dashboard")


# -----------------------------------------------------------------------------
# Configuration (all self-contained here; nothing shared is modified).
# Every value defaults to the pre-migration production value and is
# env-overridable so a staging deployment needs no source edit.
# -----------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_json_map(name: str, default: Dict[str, str]) -> Dict[str, str]:
    """Merge a JSON-object env override over `default` (override wins)."""
    raw = os.getenv(name)
    if not raw:
        return dict(default)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            merged = dict(default)
            merged.update({str(k): str(v) for k, v in parsed.items()})
            return merged
    except (ValueError, TypeError):
        log.warning("[DASHBOARD] ignoring malformed %s (not a JSON object)", name)
    return dict(default)


# Full CCTV camera ids (dashboard primary key) == the V4 `camera_id`, resolved
# from the shared registry in core.constants so the extraction buckets, the
# report layout, and this feed can never disagree about a camera's folder name.
_DEFAULT_FULL_IDS = dict(C.CAMERA_S3_FOLDER)

# Dashboard S3 folder (prefix) per camera, inside WAGONEYE_INSPECTION_JSON_BUCKET.
#
# VERIFIED, not assumed.  Each value is the `INSPECTION_JSON_FOLDER` constant from
# that camera's own old per-camera production pipeline
# (`output_test/<CAMERA>/sagemaker_main.py`), which is the process that has been
# populating this dashboard.  Note the two TOP cameras do NOT follow the
# "<side>_up_top" pattern -- they are `Right_Top` / `Left_Top`, with a capital T
# and no "up".  An earlier version of this module guessed `Right_up_top` /
# `Left_up_top`, which would have published both top cameras into folders the
# dashboard never reads.
#
#   camera        old pipeline constant        folder
#   RIGHT_UP      INSPECTION_JSON_FOLDER  ->   Right_up
#   LEFT_UP       INSPECTION_JSON_FOLDER  ->   Left_up
#   RIGHT_UP_TOP  INSPECTION_JSON_FOLDER  ->   Right_Top
#   LEFT_UP_TOP   INSPECTION_JSON_FOLDER  ->   Left_Top
#
# Full key: <folder>/<YYYY-MM-DD>/<raw_basename>_inspection.json, where the date
# uses the 05:00 IST operational-day boundary (see `date_folder`) -- identical to
# the old pipeline's `_train_date_folder`.
_DEFAULT_FOLDERS = {
    C.CAMERA_RIGHT_UP:     "Right_up",
    C.CAMERA_LEFT_UP:      "Left_up",
    C.CAMERA_RIGHT_UP_TOP: "Right_Top",
    C.CAMERA_LEFT_UP_TOP:  "Left_Top",
}


def is_enabled() -> bool:
    # ON by default: every finalized batch posts the legacy per-camera feed to
    # the dashboard ingest API (version v1).  Set WAGONEYE_DASHBOARD_INGEST_ENABLED=false
    # to turn it off (e.g. staging / shadow runs).
    return _env_bool("WAGONEYE_DASHBOARD_INGEST_ENABLED", True)


def inspection_bucket() -> str:
    """Bucket the per-camera inspection JSON is uploaded to.

    Defaults to the V4 engine's own ``ARTIFACT_BUCKET`` (config.json:
    ``test-inspection-artifacts-sarva``) -- the bucket the receiver behind
    INGEST_API_URL_PROD already reads, because that is exactly the pairing V4
    itself uses.  Point ``WAGONEYE_INSPECTION_JSON_BUCKET`` elsewhere (e.g. the
    older ``ankit-version-1-prod``) to publish into a different location.
    """
    return _env("WAGONEYE_INSPECTION_JSON_BUCKET", C.S3_ARTIFACT_BUCKET)


#: The V4 receivers, taken from the V4 engine's COMMITTED
#: `Train-Inspection-Engine/configs/config.json` -- NOT from the stale defaults
#: still hard-coded in its `core/config.py` dataclass.  V4's own commit
#: "Match notebook artifact + JSON contract; fix flush-emit + endpoint URLs"
#: replaced those defaults, so the dataclass points at a host
#: (`cctv-wagon-api.suvidhaen.com`) the live deployment does not post to.
#: Copying the dataclass would silently deliver to a backend the dashboard never
#: reads, which is why these resolve through core.constants instead.
#:
#: Note the PROD receiver is the SAME host+path the V1 dashboard feed uses: the
#: document's `version` field, not the URL, selects which dashboard tab renders
#: the report.
INGEST_URL_PROD = C.INGEST_API_URL_PROD
INGEST_URL_UAT = C.INGEST_API_URL_UAT

#: Back-compat alias: this feed's historical single-endpoint name.
INGEST_URL_V1 = C.INGEST_API_URL_PROD


def _prod_url() -> str:
    return _env("WAGONEYE_INGEST_API_URL_PROD", INGEST_URL_PROD)


def _uat_url() -> str:
    return _env("WAGONEYE_INGEST_API_URL_UAT", INGEST_URL_UAT)


def ingest_api_urls() -> List[str]:
    """Every endpoint each document is posted to, in order.

    Default: BOTH V4 receivers (PROD then UAT), which is exactly what the V4
    engine's ``NotificationService.trigger_db_ingestion_dual`` does -- one
    document, two receivers, success on either.

    ``WAGONEYE_INSPECTION_INGEST_API_URLS`` overrides with a comma-separated
    list; the shorthand ``v4`` expands to both V4 receivers and ``prod`` /
    ``uat`` to one each.
    """
    raw = os.getenv("WAGONEYE_INSPECTION_INGEST_API_URLS")
    if not raw:
        return [_prod_url(), _uat_url()]
    urls: List[str] = []
    for tok in raw.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        low = tok.lower()
        if low == "v4":
            urls.extend([_prod_url(), _uat_url()])
        elif low in ("prod", "v1"):
            urls.append(_prod_url())
        elif low == "uat":
            urls.append(_uat_url())
        else:
            urls.append(tok)
    seen: set = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def _version() -> str:
    """The `version` carried by every document AND every ingest POST.

    The dashboard chooses its tab from this value: "v1" -> the V1 tab.  One
    accessor, so the document body and the POST can never disagree about which
    dashboard view a report belongs to.

    Read from the environment at CALL time (not import time) so
    ``WAGONEYE_INSPECTION_VERSION`` works for a process that sets it after
    import; `C.INSPECTION_VERSION` supplies the default.
    """
    return _env("WAGONEYE_INSPECTION_VERSION", C.INSPECTION_VERSION)


# Back-compat aliases for the previous private names.
_inspection_bucket = inspection_bucket
_ingest_api_urls = ingest_api_urls


def _ingest_api_url() -> str:
    """The single primary ingest endpoint (back-compat accessor)."""
    return ingest_api_urls()[0]


def _model_id() -> str:
    return _env("WAGONEYE_INSPECTION_MODEL_ID", "model-v3")


def _reuse_evidence_urls() -> bool:
    return _env_bool("WAGONEYE_DASHBOARD_REUSE_EVIDENCE_URLS", True)


def full_camera_id(camera: str) -> str:
    return _env_json_map("WAGONEYE_INSPECTION_CAMERA_FULL_IDS",
                         _DEFAULT_FULL_IDS).get(camera, camera)


def folder_for(camera: str) -> str:
    return _env_json_map("WAGONEYE_INSPECTION_FOLDERS",
                         _DEFAULT_FOLDERS).get(camera, C.CAMERA_FOLDER.get(camera, camera))


# -----------------------------------------------------------------------------
# Pure helpers (timestamp / date-folder / URLs) -- fully unit-testable
# -----------------------------------------------------------------------------

def extract_train_timestamp(*texts: Optional[str]) -> Optional[datetime]:
    """First ``YYYYMMDD_HHMMSS`` (or ``YYYYMMDD``) token across `texts`.

    Returns a naive datetime (interpreted as train local/IST wall-clock, exactly
    as the old pipeline treated the filename timestamp)."""
    for t in texts:
        if not t:
            continue
        m = _TS_RE.search(t)
        if m:
            try:
                return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
            except ValueError:
                pass
        m = _DATE_RE.search(t)
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y%m%d")
            except ValueError:
                pass
    return None


def date_folder(dt: Optional[datetime]) -> str:
    """Operational-day folder with the old 05:00 IST boundary: a train recorded
    before 05:00 lands in the PREVIOUS calendar day's folder."""
    if dt is None:
        dt = datetime.now(_IST)
    shifted = (dt - timedelta(days=1)) if dt.hour < 5 else dt
    return shifted.strftime("%Y-%m-%d")


def inspection_s3_key(*, camera: str, date_folder_str: str, json_name: str,
                      ts: Optional[datetime]) -> str:
    """S3 key the per-camera inspection JSON is uploaded under.

    Default (``v4``) reproduces the V4 engine's own ArtifactPublisher layout:

        <camera_id>/<YYYY-MM-DD_HH-MM-SS>/inspection_data.json

    i.e. ``s3://{ARTIFACT_BUCKET}/{camera_folder}/{timestamp}/inspection_data.json``
    -- the exact object V4 hands to the ingest API.  When the train timestamp
    cannot be parsed from the filename the operational-day folder stands in for
    the timestamp, so the key is always well-formed.

    ``WAGONEYE_INSPECTION_KEY_LAYOUT=v1`` selects the older per-camera dashboard
    layout instead (``<Folder>/<YYYY-MM-DD>/<basename>_inspection.json``).
    """
    layout = _env("WAGONEYE_INSPECTION_KEY_LAYOUT", "v4").strip().lower()
    if layout == "v1":
        return f"{folder_for(camera)}/{date_folder_str}/{json_name}"
    stamp = ts.strftime("%Y-%m-%d_%H-%M-%S") if ts is not None else date_folder_str
    return f"{full_camera_id(camera)}/{stamp}/inspection_data.json"


def evidence_rel_path(evidence_root: str, gw: str, feature: str, camera: str,
                      filename: str) -> Optional[str]:
    """Evidence path relative to ``evidence/``, for a JPEG that EXISTS on disk.

    Two layouts are supported, in order:
      1. ``<GW>/<feature>/<CAMERA>/<file>`` -- a per-camera evidence subtree.
      2. ``<GW>/<feature>/<file>``          -- this package's flat layout, where
         the camera is encoded in the filename (``right_best.jpg``) or in the
         sibling ``metadata.json`` (damage tracks carry ``camera_id``).

    Returning the RELATIVE path keeps the S3 URL and the local file in lockstep:
    the Stage-6 tree upload mirrors ``evidence/`` verbatim, so whichever layout
    exists locally is the layout that exists in S3.
    """
    nested = os.path.join(gw, feature, camera, filename)
    if os.path.isfile(os.path.join(evidence_root, nested)):
        return nested.replace("\\", "/")
    flat = os.path.join(gw, feature, filename)
    if os.path.isfile(os.path.join(evidence_root, flat)):
        return flat.replace("\\", "/")
    return None


def evidence_url(output_bucket: str, region: str, batch_key: str,
                 rel_path: str) -> str:
    """Deterministic HTTPS URL for an evidence JPEG already mirrored to S3 by the
    Stage-6 tree upload (``train_batch/<key>/evidence/...``)."""
    key = f"{C.S3_TRAIN_BATCH_PREFIX}/{batch_key}/evidence/{rel_path}"
    return f"https://{output_bucket}.s3.{region}.amazonaws.com/{key}"


def _seg_type(classification: Optional[str]) -> str:
    return {
        C.CLASS_ENGINE:    "engine",
        C.CLASS_WAGON:     "wagon",
        C.CLASS_BRAKE_VAN: "brake_van",
    }.get(classification or "", "wagon")


def _door_side(camera: str) -> Optional[str]:
    if camera == C.CAMERA_RIGHT_UP:
        return "right"
    if camera == C.CAMERA_LEFT_UP:
        return "left"
    return None


# -----------------------------------------------------------------------------
# Evidence reads (read-only)
# -----------------------------------------------------------------------------

def _read_meta(evidence_root: str, gw: str, feature: str, camera: str) -> Dict[str, Any]:
    p = os.path.join(evidence_root, gw, feature, camera, "metadata.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _evidence_file(evidence_root: str, gw: str, feature: str, camera: str,
                   filename: str) -> Optional[str]:
    rel = evidence_rel_path(evidence_root, gw, feature, camera, filename)
    return os.path.join(evidence_root, rel) if rel else None


class _UrlMaker:
    """Turn a local evidence JPEG into a dashboard-usable HTTPS URL.

    Default (reuse=True): reference the already-uploaded train_batch evidence URL
    -- no extra upload.  reuse=False: copy ONLY the referenced JPEG into the
    legacy inspection bucket and return that URL."""

    def __init__(self, *, s3_client, output_bucket: str, region: str,
                 inspection_bucket: str, batch_key: str, folder: str,
                 date_folder_str: str, reuse: bool, skip_upload: bool):
        self.s3 = s3_client
        self.output_bucket = output_bucket
        self.region = region
        self.inspection_bucket = inspection_bucket
        self.batch_key = batch_key
        self.folder = folder
        self.date_folder = date_folder_str
        self.reuse = reuse
        self.skip_upload = skip_upload

    def url(self, evidence_root: str, gw: str, feature: str, camera: str,
            filename: str) -> Optional[str]:
        rel = evidence_rel_path(evidence_root, gw, feature, camera, filename)
        if rel is None:
            return None
        local = os.path.join(evidence_root, rel)
        if self.reuse:
            return evidence_url(self.output_bucket, self.region, self.batch_key,
                                rel)
        # copy-only mode: upload just this JPEG to the legacy bucket
        key = f"{self.folder}/{self.date_folder}/evidence/{rel}"
        if self.skip_upload or self.s3 is None:
            return f"https://{self.inspection_bucket}.s3.{self.region}.amazonaws.com/{key}"
        try:
            self.s3.upload_file(local, self.inspection_bucket, key,
                                ExtraArgs={"ContentType": "image/jpeg"})
        except Exception as e:  # pragma: no cover - network path
            log.warning("[DASHBOARD] evidence copy failed %s: %s", key, e)
            return None
        return f"https://{self.inspection_bucket}.s3.{self.region}.amazonaws.com/{key}"


# -----------------------------------------------------------------------------
# Per-camera payload builder -- EXACT V4 schema
#
# The document itself is produced by `delivery.inspection_json`, which is a
# faithful port of the V4 Train-Inspection-Engine's `reporting/json_builder.py`
# (same two flavours, same keys, same key order, same derivations).  Everything
# here is the ADAPTER around it: locate the finalized artifacts under
# `batch_root`, decide this camera's URLs/timestamps, and hand them over.
# -----------------------------------------------------------------------------

def _strip_prefix_enabled() -> bool:
    """Whether the document's ``camera_id`` drops the ``camera_`` prefix.

    The default FOLLOWS THE VERSION, because the two must agree or the dashboard
    cannot match the document to a camera:

      * ``version=v4`` -> strip (``CCTV_HZBN_DHN_2_RIGHT_UP``).  This is what the
        V4 engine's ``json_builder._strip_camera_prefix`` emits.
      * ``version=v1`` (default) -> keep (``camera_CCTV_HZBN_DHN_2_RIGHT_UP``).
        This is the identifier the existing V1 dashboard feed has always used, so
        the live dashboard keeps resolving these reports.

    Pinning ``WAGONEYE_INSPECTION_STRIP_CAMERA_PREFIX`` overrides the coupling in
    either direction.
    """
    raw = os.getenv("WAGONEYE_INSPECTION_STRIP_CAMERA_PREFIX")
    if raw is None or raw == "":
        return _version().strip().lower() != "v1"
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _load_sealed_state(batch_root: str):
    """The sealed GlobalTrainState -- the canonical wagon sequence all four
    camera documents describe."""
    from core.global_state_loader import load_global_train_state
    return load_global_train_state(
        os.path.join(batch_root, "global_state", "global_train_state.json"))


def _load_unified(batch_root: str, state) -> Dict[str, Any]:
    """``{gw_id -> unified dict}`` from Stage-4 fusion (missing wagon -> {})."""
    unified_dir = os.path.join(batch_root, "wagon_states", "unified")
    out: Dict[str, Any] = {}
    for gw in getattr(state, "wagons", []) or []:
        p = os.path.join(unified_dir, f"{gw.global_id}.json")
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                out[gw.global_id] = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def build_inspection_json(*, camera: str, batch_root: str,
                          report_doc: Dict[str, Any],
                          url_maker: "_UrlMaker",
                          state=None,
                          unified: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build ONE exact-V4 ``{camera_id, version, inspection_data}`` document.

    Reads only finalized artifacts under `batch_root` (sealed GlobalTrainState,
    fused unified states, this camera's own per-feature JSON, and its evidence
    JPEGs) plus the already-loaded `report_doc` for train-level URLs.  Pure: no
    model, no video, no writes.
    """
    from delivery import inspection_json as IJ

    if state is None:
        state = _load_sealed_state(batch_root)
    if unified is None:
        unified = _load_unified(batch_root, state)

    evidence_root = os.path.join(batch_root, "evidence")
    states_root = os.path.join(batch_root, "wagon_states")

    report_meta = report_doc.get("report_meta", {}) or {}
    batch_key = report_doc.get("batch_key", "")
    # This package's combined_train_report.json nests the per-camera video URLs
    # under `train_metadata`; a flatter report (or a future schema) may carry
    # them at the top level.  Accept both rather than silently emitting a
    # document with no trimmed/detected video links.
    train_meta = report_doc.get("train_metadata", {}) or {}
    source_urls = (report_doc.get("source_video_urls")
                   or train_meta.get("source_video_urls") or {})
    processed_urls = (report_doc.get("processed_video_urls")
                      or train_meta.get("processed_video_urls") or {})

    src_url = source_urls.get(camera, "") or ""
    raw_video_name = (os.path.basename(src_url) if src_url
                      else f"{batch_key}_{C.CAMERA_FOLDER.get(camera, camera)}.mp4")
    ts = extract_train_timestamp(raw_video_name, batch_key)

    # Travel direction is Stage-1 derived (sign of the master's gap centre_x
    # drift) and carried in the combined report; 'unknown' for a batch sealed
    # before that field existed.
    direction = (report_doc.get("travel_direction")
                 or getattr(state, "travel_direction", "unknown") or "unknown")

    folder = full_camera_id(camera)
    camera_folder = folder if _strip_prefix_enabled() else f"__keep__{folder}"

    def _url_for(*, gw_id: str, feature: str, camera: str,
                 filename: str) -> Optional[str]:
        return url_maker.url(evidence_root, gw_id, feature, camera, filename)

    doc = IJ.build_inspection_json(
        camera=camera,
        camera_folder=folder,
        raw_video_name=raw_video_name,
        upload_timestamp=ts,
        direction=direction,
        state=state,
        unified=unified,
        states_root=states_root,
        evidence_root=evidence_root,
        url_for=_url_for,
        trimmed_video_url=src_url,
        pdf_report_url=_pdf_url(report_meta, camera),
        detected_video_url=processed_urls.get(camera, "") or "",
        raw_video_urls=[src_url] if src_url else [],
        damage_model_active=_damage_model_active(report_doc),
        version=_version(),
        identified_by=_model_id(),
        # Dialect follows the version: a v1 document must carry v1 shapes
        # (bounding_box dict, "door_open", segment_number, v1 rake polarity) or
        # the V1 dashboard cannot read it.  See inspection_json's module docs.
        schema=IJ.schema_for_version(_version()),
    )
    if not _strip_prefix_enabled():
        doc["camera_id"] = folder

    # Provenance: which global_train run produced this document, and what this
    # camera was authoritative for.  Additive -- never replaces a V4 field.
    doc["inspection_data"]["_adapter"] = {
        "generated_by": "global_train delivery.inspection_json (V4 schema)",
        "source": "sealed global_train_state + fused unified + per-camera state",
        "flavour": IJ.flavour_for(camera),
        "report_revision": report_meta.get("report_revision", 0),
        "report_status": report_meta.get("report_status", ""),
        "global_state_version":
            report_meta.get("generated_from_global_state_version", ""),
        "camera_authority": _camera_authority(camera),
        "direction_estimator": "stage1_gap_centre_x_drift",
    }
    return doc


def _camera_authority(camera: str) -> str:
    if camera == C.CAMERA_RIGHT_UP:
        return "right_door+ocr+classification"
    if camera == C.CAMERA_LEFT_UP:
        return "left_door"
    if camera == C.CAMERA_RIGHT_UP_TOP:
        return "load(primary)+top_damage"
    if camera == C.CAMERA_LEFT_UP_TOP:
        return "load(fallback)+top_damage"
    return "none"


def _damage_model_active(report_doc: Dict[str, Any]) -> bool:
    """False only when the damage feature was explicitly disabled for the run."""
    for wagon in report_doc.get("wagons", []) or []:
        if wagon.get("top_damage") == C.DISABLED_DISPLAY:
            return False
    return True


def _pdf_url(report_meta: Dict[str, Any], camera: str) -> str:
    # Prefer a per-camera PDF url if the finalization marker carried one; the
    # caller injects finalization upload_urls into report_meta before building.
    urls = report_meta.get("_upload_urls", {}) or {}
    return urls.get(f"camera_{camera}") or urls.get("pdf") or ""


# -----------------------------------------------------------------------------
# Ingest (HTTP) with retries -- mirrors the old ingest loop
# -----------------------------------------------------------------------------

def ingest_idempotency_key(batch_key: str, camera: str, report_revision: int,
                           json_sha256: str) -> str:
    raw = f"{batch_key}|{camera}|{report_revision}|{json_sha256}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


#: V4's `trigger_db_ingestion_dual` uses a 30-second timeout per receiver.
INGEST_TIMEOUT_SECONDS = 30


def _post_ingest(*, api_url: str, payload: Dict[str, Any], idem_key: str,
                 max_retries: int = 3, base_delay: float = 15.0,
                 requests_mod=None) -> Dict[str, Any]:
    """POST once (with retries).  Returns {ok, status_code, run_id, error}.

    Retries only on >=500 (transient); 422 is treated as a permanent validation
    failure (no retry).  Never raises."""
    if requests_mod is None:  # pragma: no cover - exercised via injection in tests
        import requests as requests_mod  # type: ignore
    # The BODY is exactly V4's three fields -- `inspection_s3_uri`, `camera_id`,
    # `version` -- and nothing else.  The idempotency key travels as a HEADER
    # only: a header is additive and ignored by a receiver that does not know it,
    # whereas an extra BODY field is a payload divergence a strict validator can
    # reject with 422 (and this receiver does return 422 on validation errors).
    headers = {"Idempotency-Key": idem_key}
    body = dict(payload)
    delay = base_delay
    last: Dict[str, Any] = {"ok": False, "status_code": None, "run_id": None,
                            "error": "not_attempted"}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests_mod.post(api_url, json=body, headers=headers,
                                     timeout=INGEST_TIMEOUT_SECONDS)
            code = getattr(resp, "status_code", None)
            if code == 200:
                data = {}
                try:
                    data = resp.json()
                except Exception:
                    pass
                return {"ok": True, "status_code": 200,
                        "run_id": data.get("run_id"), "error": None}
            if code == 422:
                txt = ""
                try:
                    txt = resp.text[:300]
                except Exception:
                    pass
                return {"ok": False, "status_code": 422, "run_id": None,
                        "error": f"validation: {txt}"}
            last = {"ok": False, "status_code": code, "run_id": None,
                    "error": f"http_{code}"}
            if code is not None and code < 500:
                return last  # non-retryable client error
        except Exception as e:  # network/timeout -> retryable
            last = {"ok": False, "status_code": None, "run_id": None,
                    "error": str(e)}
        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2
    return last


# -----------------------------------------------------------------------------
# finalization.json per-camera status (idempotency ledger)
# -----------------------------------------------------------------------------

_DASH_KEY = "dashboard_ingested"


def _load_status(batch_root: str) -> Dict[str, Any]:
    marker = FIN.load(batch_root) or {}
    return dict(marker.get(_DASH_KEY) or {})


def _record_status(batch_root: str, camera: str, entry: Dict[str, Any]) -> None:
    marker = FIN.load(batch_root) or {}
    block = dict(marker.get(_DASH_KEY) or {})
    block[camera] = entry
    marker[_DASH_KEY] = block
    FIN.write(batch_root, marker)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def run(*, batch_root: str, s3_client=None, skip_upload: bool = False,
        skip_ingest: Optional[bool] = None, requests_mod=None) -> Dict[str, Any]:
    """Generate + (optionally) deliver the legacy per-camera dashboard feed.

    NEVER raises.  Returns a summary dict.  A no-op (returns {'enabled': False})
    unless WAGONEYE_DASHBOARD_INGEST_ENABLED is truthy.

    skip_upload=True (shadow/dry-run) -> build + record locally, do NOT upload
    JSON and do NOT POST ingest.  skip_ingest defaults to skip_upload."""
    result: Dict[str, Any] = {"enabled": is_enabled(), "cameras": {}}
    if not is_enabled():
        return result
    if skip_ingest is None:
        skip_ingest = skip_upload
    try:
        return _run_inner(batch_root=batch_root, s3_client=s3_client,
                          skip_upload=skip_upload, skip_ingest=skip_ingest,
                          requests_mod=requests_mod, result=result)
    except Exception as e:  # absolute isolation: never propagate
        log.error("[DASHBOARD] ingest aborted (non-fatal): %s", e)
        result["error"] = str(e)
        return result


def _run_inner(*, batch_root, s3_client, skip_upload, skip_ingest,
               requests_mod, result) -> Dict[str, Any]:
    # reports/ is the fixed finalized-artifact location (core.config.DIR_REPORTS);
    # hardcoded here to keep the adapter decoupled from config internals.
    report_path = os.path.join(batch_root, "reports", "combined_train_report.json")
    if not os.path.isfile(report_path):
        log.warning("[DASHBOARD] no combined_train_report.json -- nothing to ingest")
        result["error"] = "no_report"
        return result
    with open(report_path, "r", encoding="utf-8") as f:
        report_doc = json.load(f)

    report_meta = report_doc.get("report_meta", {}) or {}
    # inject finalization upload_urls so per-camera pdf urls resolve
    fin_marker = FIN.load(batch_root) or {}
    report_meta = dict(report_meta)
    report_meta["_upload_urls"] = fin_marker.get("upload_urls", {}) or {}
    report_doc = dict(report_doc, report_meta=report_meta)
    report_revision = int(report_meta.get("report_revision", 0))

    present = report_meta.get("cameras_present") or [
        c for c in C.ALL_CAMERAS
        if c in {w0 for w in report_doc.get("wagons", [])
                 for w0 in (w.get("supporting_cameras") or [])}
    ]
    present = [c for c in C.ALL_CAMERAS if c in present]  # canonical order

    evidence_root = os.path.join(batch_root, "evidence")
    local_dir = os.path.join(batch_root, _LOCAL_SUBDIR)
    os.makedirs(local_dir, exist_ok=True)

    output_bucket = C.S3_OUTPUT_BUCKET
    region = C.S3_REGION
    inspection_bucket_name = inspection_bucket()
    api_urls = ingest_api_urls()
    log.info("[DASHBOARD] ingest receivers (%d): %s", len(api_urls),
             ", ".join(api_urls))
    reuse = _reuse_evidence_urls()

    batch_key = report_doc.get("batch_key", "")
    ts = extract_train_timestamp(batch_key)
    df = date_folder(ts)

    prior = _load_status(batch_root)

    # Load the sealed state + fused unified states ONCE and share them across all
    # four camera documents, so every file describes the same wagon sequence with
    # the same GW numbering (and we don't re-read N files per camera).
    try:
        shared_state = _load_sealed_state(batch_root)
        shared_unified = _load_unified(batch_root, shared_state)
    except Exception as e:
        log.error("[DASHBOARD] cannot load sealed state -- nothing to ingest: %s", e)
        result["error"] = f"no_global_state: {e}"
        return result

    for camera in present:
        url_maker = _UrlMaker(
            s3_client=s3_client, output_bucket=output_bucket, region=region,
            inspection_bucket=inspection_bucket_name, batch_key=batch_key,
            folder=folder_for(camera), date_folder_str=df,
            reuse=reuse, skip_upload=skip_upload)
        try:
            doc = build_inspection_json(camera=camera, batch_root=batch_root,
                                        report_doc=report_doc,
                                        url_maker=url_maker,
                                        state=shared_state,
                                        unified=shared_unified)
        except Exception as e:
            log.error("[DASHBOARD] build failed for %s: %s", camera, e)
            result["cameras"][camera] = {"status": "build_failed", "error": str(e)}
            continue

        raw_video_name = doc["inspection_data"]["raw_video_name"]
        json_name = f"{os.path.splitext(raw_video_name)[0]}_inspection.json"
        # The four cameras' clips routinely share a basename (they differ by
        # camera FOLDER, not filename), so `json_name` alone collides and each
        # camera overwrites the previous camera's local audit copy -- leaving one
        # file where there should be four.  The S3 key is already namespaced by
        # camera, so only the local copy needs disambiguating.
        local_name = f"{camera}_{json_name}"
        text = json.dumps(doc, indent=2, default=str)
        json_sha = _sha256_text(text)
        idem = ingest_idempotency_key(batch_key, camera, report_revision, json_sha)

        # ---- idempotency: already ingested this exact payload? ----
        pj = prior.get(camera) or {}
        if pj.get("status") == "ingested" and pj.get("json_sha256") == json_sha:
            log.info("[DASHBOARD] %s already ingested (rev=%s) -- skip",
                     camera, report_revision)
            result["cameras"][camera] = {"status": "already_ingested",
                                         "run_id": pj.get("run_id")}
            continue

        # ---- write local JSON (delivery/ only) ----
        local_json = os.path.join(local_dir, local_name)
        with open(local_json, "w", encoding="utf-8") as f:
            f.write(text)

        s3_key = inspection_s3_key(camera=camera, date_folder_str=df,
                                   json_name=json_name, ts=ts)
        s3_uri = f"s3://{inspection_bucket_name}/{s3_key}"

        entry = {
            "camera_id": full_camera_id(camera),
            "json_sha256": json_sha,
            "idempotency_key": idem,
            "report_revision": report_revision,
            "s3_uri": s3_uri,
            "run_id": None,
            "status": "prepared",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        # ---- upload JSON ----
        if skip_upload or s3_client is None:
            entry["status"] = "prepared_local_only"
        else:
            try:
                s3_client.upload_file(local_json, inspection_bucket_name, s3_key,
                                      ExtraArgs={"ContentType": "application/json"})
                entry["status"] = "uploaded"
            except Exception as e:
                log.error("[DASHBOARD] JSON upload failed %s: %s", s3_uri, e)
                entry["status"] = "upload_failed"
                entry["error"] = str(e)
                _record_status(batch_root, camera, entry)
                result["cameras"][camera] = {"status": entry["status"]}
                continue

        # ---- ingest POST ----
        if skip_ingest:
            entry["status"] = "prepared" if entry["status"] == "prepared_local_only" \
                else entry["status"]
            _record_status(batch_root, camera, entry)
            result["cameras"][camera] = {"status": entry["status"], "dry_run": True}
            continue

        # Same three fields the V4 engine sends (notifications.
        # trigger_db_ingestion_dual): the payload's camera_id is ALWAYS the full
        # prefixed folder, independent of the document's camera_id form.
        payload = {"camera_id": full_camera_id(camera),
                   "inspection_s3_uri": s3_uri, "version": _version()}
        # Post to every configured receiver (default: one).  A document counts as
        # ingested when AT LEAST ONE accepts it; per-endpoint outcomes are all
        # recorded so a partial delivery is visible rather than hidden.
        per_endpoint: Dict[str, Any] = {}
        any_ok = False
        for url in api_urls:
            res = _post_ingest(api_url=url, payload=payload, idem_key=idem,
                               requests_mod=requests_mod)
            per_endpoint[url] = {"ok": res["ok"],
                                 "status_code": res.get("status_code"),
                                 "run_id": res.get("run_id"),
                                 "error": res.get("error")}
            if res["ok"]:
                any_ok = True
                entry["run_id"] = entry.get("run_id") or res.get("run_id")
            else:
                entry["error"] = res.get("error")
                entry["last_status_code"] = res.get("status_code")
        entry["endpoints"] = per_endpoint
        entry["status"] = "ingested" if any_ok else "ingest_failed"
        _record_status(batch_root, camera, entry)
        result["cameras"][camera] = {"status": entry["status"],
                                     "run_id": entry.get("run_id")}

    return result
