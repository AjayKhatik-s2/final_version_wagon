"""Deliver a FINISHED batch from what is on disk.

    <batch_root>/  ->  S3 archive  ->  dashboard feed  ->  ML API  ->  email

One implementation, three callers:

  * `orchestrator.global_assembler`  -- so `--mode sequential` reaches production
    at all.  Before this, the sequential path ended at
    `combined_train_report.build()`: it uploaded nothing, posted nothing and
    emailed nothing, so a sequential run produced ZERO dashboard entries.  That
    matters more now that sequential is the default for foreground runs.
  * `orchestrator.master_runner --deliver-only <batch_root>` -- republish a batch
    that already finished, in seconds, with no inference re-run.  A failed POST
    or a delivery someone forgot to enable used to cost a full reprocess (~30 min
    per train on CPU).
  * any future caller that has a finished batch tree and nothing else.

Why it reads from DISK rather than taking an in-memory outcome: that is the only
input a republish has.  A batch that finished last week left exactly this tree,
and nothing else survives.  Deriving everything from the tree is therefore both
the general case and the testable one.

Relationship to `master_runner.process_batch`'s inline Stage 6
-------------------------------------------------------------
`process_batch` still contains its own Stage 6/6b block, and this module does
NOT replace it.  That block is the live `--auto` delivery path; it already holds
the report paths and per-camera URLs in memory, and rewiring it would put the
production path through code that has never carried a live batch.  The two are
kept in agreement by `tests/test_finalize.py`, which asserts they produce the
same S3 key layout and call the same delivery functions in the same order.

Consolidating them is worth doing once this module has carried real traffic --
it is recorded here as a deliberate, tested duplication rather than an oversight.

Guarantees
----------
* Reads only finalized artifacts; runs no model and opens no video.
* Never raises.  Every step is failure-isolated and reported in the result, so a
  receiver outage cannot turn a finished batch into a failed one.
* Idempotent for the dashboard feed: `dashboard_ingest` keys its per-camera
  ledger on the document sha256, so re-delivering an unchanged batch is a no-op
  rather than a duplicate post.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("delivery.finalize")


@dataclass
class DeliveryResult:
    batch_key: str = ""
    batch_root: str = ""
    uploaded: bool = False
    report_pdf_url: str = ""
    report_json_url: str = ""
    camera_pdf_urls: Dict[str, str] = field(default_factory=dict)
    archived: Dict[str, int] = field(default_factory=dict)
    dashboard: Dict[str, Any] = field(default_factory=dict)
    ml_api: Dict[str, Any] = field(default_factory=dict)
    email_sent: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.uploaded and not self.errors

    def render(self) -> str:
        lines = [f"delivery for {self.batch_key}:",
                 f"  archived        : {self.archived or '(nothing)'}"]
        if self.report_pdf_url:
            lines.append(f"  combined pdf    : {self.report_pdf_url}")
        for cam, u in sorted(self.camera_pdf_urls.items()):
            lines.append(f"  {cam:<15} : {u}")
        cams = (self.dashboard.get("cameras") or {})
        if self.dashboard.get("enabled"):
            for cam in C.ALL_CAMERAS:
                if cam in cams:
                    info = cams[cam]
                    rid = f"  run_id={info['run_id']}" if info.get("run_id") else ""
                    lines.append(f"  dashboard/{cam:<13} {info.get('status')}{rid}")
        else:
            lines.append("  dashboard       : disabled")
        if self.errors:
            lines.append(f"  errors          : {self.errors}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Locating artifacts in a finished batch tree
# ---------------------------------------------------------------------------

#: Per-camera report filenames Stage 5a writes, keyed by camera.
_CAMERA_PDF = {cam: f"{C.CAMERA_FOLDER[cam]}_report.pdf" for cam in C.ALL_CAMERAS}


def batch_key_for(batch_root: str) -> str:
    """The batch key is the directory name -- the same value Stage 6 uses to
    build every S3 key, so a republish lands on exactly the same objects."""
    return os.path.basename(os.path.abspath(batch_root.rstrip("/")))


def find_artifacts(batch_root: str) -> Dict[str, Any]:
    """What this tree actually holds.  Missing pieces are simply absent."""
    reports = os.path.join(batch_root, "reports")
    out: Dict[str, Any] = {"reports_dir": reports, "camera_pdfs": {}}

    pdf = os.path.join(reports, "combined_train_report.pdf")
    jsn = os.path.join(reports, "combined_train_report.json")
    out["combined_pdf"] = pdf if os.path.isfile(pdf) else ""
    out["combined_json"] = jsn if os.path.isfile(jsn) else ""

    for cam, name in _CAMERA_PDF.items():
        p = os.path.join(reports, name)
        if os.path.isfile(p):
            out["camera_pdfs"][cam] = p

    for key, sub in (("global_state", "global_state"),
                     ("wagon_states", "wagon_states"),
                     ("evidence", "evidence"),
                     ("processed_videos", "processed_videos")):
        d = os.path.join(batch_root, sub)
        out[key] = d if os.path.isdir(d) else ""
    return out


def is_deliverable(batch_root: str) -> tuple:
    """`(ok, reason)` -- a batch is deliverable when it has a combined report.

    That file is what `dashboard_ingest` derives every per-camera document from,
    so without it there is nothing to publish and saying so beats posting a
    partial train.
    """
    if not os.path.isdir(batch_root):
        return False, f"no such batch directory: {batch_root}"
    a = find_artifacts(batch_root)
    if not a["combined_json"]:
        return False, ("no reports/combined_train_report.json -- the batch did "
                       "not reach Stage 5b, so there is nothing to deliver")
    return True, ""


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

def deliver(
    *,
    batch_root: str,
    s3_client=None,
    batch_key: Optional[str] = None,
    final_status: str = C.BATCH_COMPLETED,
    missing_cameras: Optional[List[str]] = None,
    send_email: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> DeliveryResult:
    """Upload, publish and (optionally) email a finished batch.  Never raises.

    `dry_run` resolves and reports what WOULD be delivered without uploading or
    posting -- the same escape hatch `--historical --dry-run` gives for discovery.

    `send_email` defaults to False: a republish must not re-mail the operators
    about a train they were already told about.
    """
    key = batch_key or batch_key_for(batch_root)
    res = DeliveryResult(batch_key=key, batch_root=batch_root)

    ok, reason = is_deliverable(batch_root)
    if not ok:
        res.errors.append(reason)
        log.error("[FINALIZE] %s", reason)
        return res

    art = find_artifacts(batch_root)
    if verbose:
        log.info("[FINALIZE] %s: combined=%s camera_pdfs=%d", key,
                 bool(art["combined_pdf"]), len(art["camera_pdfs"]))

    if dry_run:
        log.info("[FINALIZE] --dry-run: would upload reports + evidence + "
                 "processed videos for %s and post %d per-camera document(s)",
                 key, len(art["camera_pdfs"]) or len(C.ALL_CAMERAS))
        res.uploaded = False
        return res

    if s3_client is None:
        try:
            import boto3
            s3_client = boto3.client("s3", region_name=C.S3_REGION)
        except Exception as e:  # noqa: BLE001
            res.errors.append(f"no S3 client: {e}")
            log.error("[FINALIZE] cannot create an S3 client: %s", e)
            return res

    from delivery import s3_upload

    # ---- reports (microservice-first for PDFs, same as the live path) ----
    try:
        if art["combined_pdf"]:
            res.report_pdf_url = s3_upload.upload_pdf(
                s3_client, art["combined_pdf"], key) or ""
        if art["combined_json"]:
            res.report_json_url = s3_upload.upload_json(
                s3_client, art["combined_json"], key) or ""
        for cam, path in art["camera_pdfs"].items():
            url = s3_upload.upload_pdf(s3_client, path, key)
            if url:
                res.camera_pdf_urls[cam] = url
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"report upload: {e}")
        log.error("[FINALIZE] report upload failed: %s", e)

    # A report that EXISTS but produced no URL is the failure that silently put
    # `pdf_report_url: ""` into live dashboard documents -- the entry ingests and
    # then has no link to open.  Never let that pass unremarked.
    if art["combined_pdf"] and not res.report_pdf_url:
        res.errors.append("combined PDF uploaded nowhere -- dashboard documents "
                          "would carry an empty pdf_report_url")
    for cam in art["camera_pdfs"]:
        if cam not in res.camera_pdf_urls:
            res.errors.append(f"{cam} PDF uploaded nowhere")

    # ---- archive the tree, in the same order and with the same prefixes ----
    for label, path, extra in (
        ("global_state", art["global_state"], {"skip_extensions": {".jpg", ".jpeg"}}),
        ("wagon_states", art["wagon_states"], {}),
        ("reports", art["reports_dir"], {}),
        ("evidence", art["evidence"], {}),
        ("processed_videos", art["processed_videos"], {}),
    ):
        if not path:
            continue
        try:
            res.archived[label] = s3_upload.upload_tree(
                s3_client, path, key, sub_prefix=label, **extra)
        except Exception as e:  # noqa: BLE001
            res.errors.append(f"archive {label}: {e}")
            log.error("[FINALIZE] archive %s failed: %s", label, e)
    # `bool(res.archived)` is TRUE for a dict of zeros, so a total S3 outage
    # reported success: `upload_tree` catches each file's error internally and
    # returns a count, so nothing propagates and every count is simply 0.  Judge
    # on files actually written, and say which subtree wrote none.
    for label, n in sorted(res.archived.items()):
        if n == 0:
            res.errors.append(f"archive {label}: 0 files uploaded")
    res.uploaded = sum(res.archived.values()) > 0
    if verbose:
        log.info("[FINALIZE] archived: %s", res.archived)

    # ---- seed the finalization marker so documents carry their PDF links ----
    # `dashboard_ingest` reads per-camera PDF URLs out of the marker; without it
    # every document publishes with `pdf_report_url` empty.  An existing marker
    # is never overwritten -- whatever recorded a delivery first keeps its record.
    try:
        from delivery import finalization as FIN
        urls = {f"camera_{cam}": u for cam, u in res.camera_pdf_urls.items() if u}
        if res.report_pdf_url:
            urls["pdf"] = res.report_pdf_url
        if res.report_json_url:
            urls["json"] = res.report_json_url
        if urls and FIN.load(batch_root) is None:
            FIN.write(batch_root, {
                "batch_key": key,
                "terminal_status": final_status,
                "upload_urls": urls,
                "uploaded": True,
                # Per-subtree upload COUNTS. Recorded because the pre-train disk
                # sweep runs long after this process is gone and has no
                # DeliveryResult to consult: without a durable count it cannot
                # tell whether the overlay videos reached S3, and so refuses to
                # reclaim them at all.
                "archived": dict(getattr(res, "archived", None) or {}),
                "source": "delivery.finalize",
            })
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"finalization marker: {e}")
        log.warning("[FINALIZE] could not seed the marker: %s", e)

    # ---- the V4 dashboard feed (4 per-camera documents) ----
    try:
        from delivery import dashboard_ingest
        res.dashboard = dashboard_ingest.run(
            batch_root=batch_root, s3_client=s3_client, skip_upload=False)
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"dashboard: {e}")
        log.error("[FINALIZE] dashboard ingest failed: %s", e)

    # ---- the V4 ML API callback ----
    try:
        from delivery import ml_api
        present = sorted(res.camera_pdf_urls) or [
            c for c in C.ALL_CAMERAS
            if c not in set(missing_cameras or ())]
        res.ml_api = ml_api.submit_batch(
            batch_key=key,
            cameras=present,
            source_video_urls=_source_video_urls(art),
            processed_video_urls={},
            camera_pdf_urls=res.camera_pdf_urls,
            combined_pdf_url=res.report_pdf_url or None,
        )
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"ml_api: {e}")
        log.error("[FINALIZE] ML API submission failed: %s", e)

    # ---- email, opt-in only ----
    if send_email:
        try:
            res.email_sent = _send_email(batch_root, art, key,
                                         res, missing_cameras or [])
        except Exception as e:  # noqa: BLE001
            res.errors.append(f"email: {e}")
            log.error("[FINALIZE] email failed: %s", e)

    if verbose:
        log.info("[FINALIZE]\n%s", res.render())
    return res


def _source_video_urls(art: Dict[str, Any]) -> Dict[str, str]:
    """Per-camera source URLs, read back out of the combined report."""
    import json
    if not art["combined_json"]:
        return {}
    try:
        with open(art["combined_json"], "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {}
    meta = doc.get("train_metadata", {}) or {}
    return dict(doc.get("source_video_urls")
                or meta.get("source_video_urls") or {})


def _send_email(batch_root: str, art: Dict[str, Any], key: str,
                res: DeliveryResult, missing: List[str]) -> bool:
    """One email per batch, using the existing notification module unchanged."""
    import json
    from core.unified_wagon_state import UnifiedWagonState, summarize_wagons
    from delivery import notification

    unified_dir = os.path.join(batch_root, "wagon_states", "unified")
    wagons = []
    if os.path.isdir(unified_dir):
        for name in sorted(os.listdir(unified_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(unified_dir, name), encoding="utf-8") as f:
                    d = json.load(f)
            except (OSError, ValueError):
                continue
            wagons.append(UnifiedWagonState(**{
                k: v for k, v in d.items()
                if k in UnifiedWagonState.__dataclass_fields__}))

    notification.send_email(
        batch_key=key,
        report_pdf_url=res.report_pdf_url or None,
        report_json_url=res.report_json_url or None,
        summary=summarize_wagons(wagons),
        cameras_present=[c for c in C.ALL_CAMERAS if c not in set(missing)],
        cameras_missing=list(missing),
        final_status=C.BATCH_COMPLETED,
    )
    return True
