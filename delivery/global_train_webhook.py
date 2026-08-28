"""Stage 6c -- post the fused global train report to the dashboard receiver.

Where this sits in the flow the backend described:

    1. each camera seals      -> POST /inspections/ingest          (x4)
    2. Stage 5 fuses them     -> combined_train_report.json -> S3
    3. THIS MODULE            -> POST /inspections/ingest-global   (x1)
    4. the receiver stores it as a virtual 5th camera, GLOBAL_FUSED

Steps 1 and 2 already existed. Step 3 did not: the fused report reached S3 and
stopped there, so the receiver's global endpoint would have stayed empty no
matter how many trains ran.

Why the whole document, not a pointer
-------------------------------------
The per-camera endpoint takes `{camera_id, inspection_s3_uri, version}` and
fetches the body from S3 itself. The global endpoint does NOT: its schema takes
`global_train_data` inline, as the full `combined_train_report.json`. Sending a
pointer here would be rejected as a missing required field, so the two feeds
deliberately do not share a payload builder.

Why `GLOBAL_FUSED`
------------------
`camera_id` is required even though this document describes the whole train, and
the receiver stores it as a virtual fifth camera under that name. Sending one of
the four real camera ids would claim the fused result belongs to a single
viewpoint, which is the one thing it is not. Overridable for the case where the
receiver turns out to link on a real camera id instead.

Failure is never fatal
----------------------
The report is already written and already in S3 before this runs. A receiver
outage must not fail a train whose inspection is complete -- it costs a
dashboard row, not the result -- so every path here returns a record of what
happened instead of raising.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("delivery.global_train_webhook")

_TAG = "[GLOBAL-INGEST]"

#: The receiver stores this document as a virtual fifth camera under this name.
GLOBAL_FUSED_CAMERA_ID = "GLOBAL_FUSED"

#: `/inspections/ingest` -> `/inspections/ingest-global`, so the global endpoint
#: is derived from the SAME configured base the per-camera feed already uses and
#: cannot drift onto a different host.
_GLOBAL_SUFFIX = "-global"


def _env(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def camera_id() -> str:
    return _env("WAGONEYE_GLOBAL_INGEST_CAMERA_ID", GLOBAL_FUSED_CAMERA_ID)


def is_enabled() -> bool:
    """ON by default: the fused report is the accurate one, and a train that
    produced it should reach the dashboard."""
    raw = os.getenv("WAGONEYE_GLOBAL_INGEST")
    if raw is None or raw == "":
        return True
    return raw.strip().lower() in ("1", "true", "yes", "on")


def global_ingest_urls() -> List[str]:
    """Derived from the per-camera ingest endpoints, never configured twice.

    An explicit `WAGONEYE_GLOBAL_INGEST_API_URLS` (comma separated) wins, for
    the case where the receiver moves the global endpoint elsewhere.
    """
    explicit = os.getenv("WAGONEYE_GLOBAL_INGEST_API_URLS")
    if explicit and explicit.strip():
        return [u.strip() for u in explicit.split(",") if u.strip()]

    # UAT only, by default, because that is where the endpoint EXISTS. Verified
    # 2026-08-28: the derived PROD path
    # `.../cctv-receiver/inspections/ingest-global` returns 404 -- the receiver
    # has only shipped the global endpoint on UAT so far. Posting there anyway
    # would log a failed delivery for every train and train the operator to
    # ignore the line that is supposed to mean something.
    #
    # Add PROD the moment the backend confirms it is live, either by listing
    # both in WAGONEYE_GLOBAL_INGEST_API_URLS or by setting
    # WAGONEYE_GLOBAL_INGEST_ALL_RECEIVERS=true.
    from delivery.dashboard_ingest import ingest_api_urls

    def _to_global(u: str) -> str:
        base = u.rstrip("/")
        return (base + _GLOBAL_SUFFIX
                if base.endswith("/inspections/ingest") else base)

    every = (os.getenv("WAGONEYE_GLOBAL_INGEST_ALL_RECEIVERS") or "").strip()
    if every.lower() in ("1", "true", "yes", "on"):
        return [_to_global(u) for u in ingest_api_urls()]
    return [_to_global(C.INGEST_API_URL_UAT)]


@dataclass
class GlobalIngestResult:
    """What happened, per endpoint. Always returned; never raised."""

    attempted: bool = False
    posted: bool = False
    skipped_reason: str = ""
    camera_id: str = ""
    batch_key: str = ""
    wagons: int = 0
    per_endpoint: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "wagon_eye.global_train_ingest.v1",
            "attempted": self.attempted,
            "posted": self.posted,
            "skipped_reason": self.skipped_reason,
            "camera_id": self.camera_id,
            "batch_key": self.batch_key,
            "wagons": self.wagons,
            "per_endpoint": dict(self.per_endpoint),
        }

    def render(self) -> str:
        if not self.attempted:
            return f"{_TAG} skipped: {self.skipped_reason}"
        ok = [u for u, r in self.per_endpoint.items() if r.get("ok")]
        bad = [u for u, r in self.per_endpoint.items() if not r.get("ok")]
        return (f"{_TAG} {self.batch_key} camera_id={self.camera_id} "
                f"wagons={self.wagons} -> "
                f"{'POSTED' if self.posted else 'FAILED'} "
                f"ok={len(ok)} failed={len(bad)}")


def _post(url: str, body: Dict[str, Any], requests_mod=None,
          timeout: int = 60) -> Dict[str, Any]:
    if requests_mod is None:
        try:
            import requests as requests_mod  # type: ignore
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"requests unavailable: {e}"}
    try:
        resp = requests_mod.post(url, json=body,
                                 headers={"Content-Type": "application/json"},
                                 timeout=timeout)
        ok = 200 <= int(getattr(resp, "status_code", 0)) < 300
        out: Dict[str, Any] = {"ok": ok,
                               "status_code": getattr(resp, "status_code", None)}
        try:
            payload = resp.json()
            out["run_id"] = payload.get("run_id")
            out["segments_count"] = payload.get("segments_count")
            out["already_existed"] = payload.get("already_existed")
            out["message"] = payload.get("message")
        except Exception:  # noqa: BLE001 - a non-JSON body is not a crash
            out["body"] = str(getattr(resp, "text", ""))[:300]
        if not ok:
            out["error"] = f"HTTP {out.get('status_code')}"
        return out
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def publish(
    *,
    report_json_path: str,
    batch_key: str = "",
    requests_mod=None,
    verbose: bool = True,
) -> GlobalIngestResult:
    """Read the fused report off disk and POST it to the global endpoint.

    Reads the SAME file Stage 5 wrote and Stage 6 uploaded, so the dashboard and
    S3 cannot disagree about what this train was.
    """
    res = GlobalIngestResult(camera_id=camera_id())

    if not is_enabled():
        res.skipped_reason = "disabled by WAGONEYE_GLOBAL_INGEST"
        if verbose:
            log.info("%s", res.render())
        return res
    if not report_json_path or not os.path.isfile(report_json_path):
        res.skipped_reason = f"no combined report at {report_json_path!r}"
        if verbose:
            log.info("%s", res.render())
        return res
    try:
        with open(report_json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        res.skipped_reason = f"could not read the report: {type(e).__name__}: {e}"
        if verbose:
            log.warning("%s", res.render())
        return res

    res.batch_key = str(doc.get("batch_key") or batch_key or "")
    res.wagons = len(doc.get("wagons") or [])
    urls = global_ingest_urls()
    if not urls:
        res.skipped_reason = "no global ingest endpoint configured"
        if verbose:
            log.info("%s", res.render())
        return res

    body = {"camera_id": res.camera_id, "global_train_data": doc}
    res.attempted = True
    if verbose:
        log.info("%s posting %s (%d wagons) to %d endpoint(s)",
                 _TAG, res.batch_key, res.wagons, len(urls))
    for url in urls:
        r = _post(url, body, requests_mod=requests_mod)
        res.per_endpoint[url] = r
        if r.get("ok"):
            res.posted = True
            if verbose:
                log.info("%s %s -> run_id=%s segments=%s already_existed=%s",
                         _TAG, url, r.get("run_id"), r.get("segments_count"),
                         r.get("already_existed"))
        elif verbose:
            log.warning("%s %s FAILED: %s", _TAG, url, r.get("error"))
    if verbose:
        log.info("%s", res.render())
    return res
