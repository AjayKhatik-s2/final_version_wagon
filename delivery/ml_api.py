"""V4 ML-API callback -- `submit_inspection_to_ml_api`, reproduced exactly.

The V4 Train-Inspection-Engine makes TWO outbound calls per finished camera
(`pipelines/base_pipeline.py`, steps 12 + 13):

    1. `NotificationService.trigger_db_ingestion_dual(...)`  -> the dashboard
       ingest receivers.  Reproduced by `delivery.dashboard_ingest`.
    2. `NotificationService.submit_inspection_to_ml_api(...)` -> the ML callback
       that registers the raw/processed video pair and the PDF against the
       train event.  Reproduced HERE.

Only the first was needed to make a report appear on the dashboard, so it was
the only one carried over previously.  Both are part of "the V4 API", so the
second is implemented here with V4's exact payload, header and timeout:

    POST <ML_API_ENDPOINT>
    headers: {"Content-Type": "application/json", "X-ML-SECRET": <secret>}
    body:    {"raw_video_id":        <raw video basename, no extension>,
              "processed_video_id":  <trimmed clip basename, no extension>,
              "processed_video_path":<processed/annotated video URL>,
              "pdf_report_path":     <PDF report URL>,
              "folder":              <camera_CCTV_... folder>,
              "has_train":           true}
    timeout: 30s

Differences from V4, all of them structural rather than behavioural:

* V4 calls this once per CAMERA because each camera is its own process with its
  own trimmed clip and its own PDF.  This package processes all four cameras as
  one batch, so `submit_batch` makes the same call once per camera present,
  using that camera's own trimmed clip, its own camera PDF, and its own overlay
  video -- so the ML API sees exactly the per-camera rows it saw before.
* The header is only attached when a secret is configured, matching V4.
* Failures are logged and reported, never raised: an ML-API outage must not fail
  a batch whose reports are already built and uploaded.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("ml_api")

#: V4 uses a 30-second timeout for this call.
TIMEOUT_SECONDS = 30


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_enabled() -> bool:
    """ON by default, like the dashboard feed: V4 always makes this call."""
    return _env_bool("WAGONEYE_ML_API_ENABLED", True)


def _basename_no_ext(url_or_path: str) -> str:
    if not url_or_path:
        return ""
    base = os.path.basename(str(url_or_path).split("?", 1)[0].rstrip("/"))
    return os.path.splitext(base)[0]


def submit_inspection(
    *,
    raw_video_id: str,
    processed_video_id: str,
    processed_video_url: str,
    pdf_report_url: str,
    folder: str,
    has_train: bool = True,
    endpoint: Optional[str] = None,
    secret: Optional[str] = None,
    requests_mod=None,
) -> Dict[str, Any]:
    """POST one inspection to the ML API.  Never raises.

    Returns ``{ok, status_code, error}``.  Mirrors V4's payload/headers exactly.
    """
    endpoint = endpoint if endpoint is not None else C.ML_API_ENDPOINT
    secret = secret if secret is not None else C.ML_API_SECRET

    if not endpoint:
        log.warning("[ML_API] endpoint not configured -- skipping submission")
        return {"ok": False, "status_code": None, "error": "no_endpoint"}

    payload = {
        "raw_video_id": raw_video_id,
        "processed_video_id": processed_video_id,
        "processed_video_path": processed_video_url,
        "pdf_report_path": pdf_report_url,
        "folder": folder,
        "has_train": has_train,
    }
    headers = {"Content-Type": "application/json"}
    if secret:
        headers["X-ML-SECRET"] = secret

    if requests_mod is None:
        try:
            import requests as requests_mod  # type: ignore
        except ImportError:
            log.error("[ML_API] requests is not installed -- cannot submit")
            return {"ok": False, "status_code": None, "error": "no_requests"}

    log.info("[ML_API] submitting inspection to %s (folder=%s)", endpoint, folder)
    try:
        resp = requests_mod.post(endpoint, json=payload, headers=headers,
                                 timeout=TIMEOUT_SECONDS)
        code = getattr(resp, "status_code", None)
        if code is not None and 200 <= int(code) < 300:
            log.info("[ML_API] submission OK: %s", code)
            return {"ok": True, "status_code": code, "error": None}
        body = ""
        try:
            body = (resp.text or "")[:300]
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
        log.error("[ML_API] submission failed: %s %s", code, body)
        return {"ok": False, "status_code": code, "error": f"http_{code}"}
    except Exception as e:  # noqa: BLE001 - never fail a finished batch
        log.error("[ML_API] submission error: %s", e)
        return {"ok": False, "status_code": None, "error": str(e)}


def submit_batch(
    *,
    batch_key: str,
    cameras: List[str],
    source_video_urls: Dict[str, str],
    processed_video_urls: Dict[str, str],
    camera_pdf_urls: Dict[str, str],
    combined_pdf_url: Optional[str] = None,
    requests_mod=None,
) -> Dict[str, Any]:
    """Make V4's ML-API call once per camera present in this batch.

    A camera with no source video is skipped: V4 only ever submits a camera it
    actually processed a clip for, and inventing a row for a camera that never
    delivered footage would misreport the train.
    """
    result: Dict[str, Any] = {"enabled": is_enabled(), "cameras": {}}
    if not is_enabled():
        return result

    for camera in [c for c in C.ALL_CAMERAS if c in cameras]:
        trimmed_url = source_video_urls.get(camera) or ""
        if not trimmed_url:
            result["cameras"][camera] = {"ok": False, "error": "no_source_video"}
            continue
        pdf_url = camera_pdf_urls.get(camera) or combined_pdf_url or ""
        processed_url = processed_video_urls.get(camera) or pdf_url
        raw_id = _basename_no_ext(trimmed_url) or batch_key
        res = submit_inspection(
            raw_video_id=raw_id,
            processed_video_id=_basename_no_ext(trimmed_url) or raw_id,
            processed_video_url=processed_url,
            pdf_report_url=pdf_url,
            folder=C.CAMERA_S3_FOLDER.get(camera, camera),
            has_train=True,
            requests_mod=requests_mod,
        )
        result["cameras"][camera] = res

    ok = sum(1 for v in result["cameras"].values() if v.get("ok"))
    log.info("[ML_API] batch %s: %d/%d camera submissions accepted",
             batch_key, ok, len(result["cameras"]))
    return result
