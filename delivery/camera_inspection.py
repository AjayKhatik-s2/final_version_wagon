"""Publish ONE camera's inspection to the dashboard the moment it seals.

    camera seals  ->  build its inspection_data.json  ->  upload  ->  POST

Sequential mode processes and seals each camera independently.  This module lets
that camera's result reach the dashboard immediately, without waiting for the
other three or for global assembly.

Why this is legitimate, not a shortcut
-------------------------------------
The V4 Train-Inspection-Engine works exactly this way: four independent
per-camera pipelines, each counting its own segments and POSTing its own
`{camera_id, version, inspection_data}` document.  Its four documents routinely
disagree about `total_wagons`, because each camera sees a different number of
segments.  Measured on 2026-07-29: local counts were RIGHT_UP 63, LEFT_UP 56,
RIGHT_UP_TOP 62, LEFT_UP_TOP 65 -- against a fused global count of 58.

So a document numbered in ONE camera's own segment space is the V4 contract, and
the dashboard was built to receive it.

What that costs, stated plainly
-------------------------------
Wagon `n` in this document is **this camera's** nth segment.  It is NOT
necessarily the same physical wagon as wagon `n` from another camera, and it is
NOT the fused `GW_n`.  Establishing that correspondence is precisely what global
assembly does.  Consequences a reader must expect:

* the four documents can disagree on `total_wagons`;
* comparing wagon `n` across cameras is not meaningful before assembly;
* the counts here are per-camera observations, not the train's canonical count.

Every document therefore carries `_adapter.numbering = "camera-local"` and
`_adapter.superseded_by_assembly = true`, so a consumer can tell an immediate
per-camera post from the fused, canonical one that global assembly publishes
later over the same S3 key.

Nothing here re-implements a stage: the camera-scoped `GlobalTrainState` and
`UnifiedWagonState`s come from `orchestrator.camera_report_adapter` (which itself
reuses `fusion.wagon_state_builder`), the document comes from
`delivery.inspection_json`, and the upload/POST come from
`delivery.dashboard_ingest`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("delivery.camera_inspection")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


#: How a per-camera post relates to the fused one that assembly publishes.
#:
#: ``replace`` (default) -- the per-camera document is written to the SAME S3
#: key the fused document will later occupy, so the dashboard shows a camera's
#: provisional result within minutes and assembly overwrites it in place with
#: the canonical, GW-numbered one. One record per camera per train, upgraded.
#:
#: ``sidecar`` -- the older behaviour: per-camera documents live under a
#: ``per_camera/`` prefix and both versions stay retrievable side by side.
#: Nothing is ever overwritten, but the dashboard sees two records.
KEY_MODE_REPLACE = "replace"
KEY_MODE_SIDECAR = "sidecar"


def key_mode() -> str:
    raw = (os.getenv("WAGONEYE_PER_CAMERA_KEY_MODE") or "").strip().lower()
    return KEY_MODE_SIDECAR if raw == KEY_MODE_SIDECAR else KEY_MODE_REPLACE


def is_enabled() -> bool:
    """Per-camera immediate publishing.  ON by default since 2026-08-28.

    It was off because it publishes CAMERA-LOCAL numbering -- wagon n is that
    camera's nth segment, not the fused `GW_n` -- and a consumer expecting
    canonical counts would misread it.

    That objection no longer applies: the receiver's flow now expects these four
    provisional documents and then supersedes them with the fused report, stored
    as a virtual fifth camera (`GLOBAL_FUSED`, see
    `delivery.global_train_webhook`). With the global post wired, leaving this
    off meant a train reached the dashboard only ~55 minutes after its first
    camera sealed, and a train that failed before assembly reached it never --
    the four cameras that DID finish were simply discarded.

    Still switchable: `WAGONEYE_PER_CAMERA_INGEST=false` turns it off for a run
    that must not publish provisional numbering.
    """
    return _env_bool("WAGONEYE_PER_CAMERA_INGEST", True)


@dataclass
class CameraIngestResult:
    camera_id: str = ""
    built: bool = False
    status: str = "skipped"
    s3_uri: str = ""
    run_id: Optional[str] = None
    local_json: str = ""
    segments: int = 0
    key_mode: str = ""
    assets_uploaded: int = 0
    assets_failed: int = 0
    errors: List[str] = field(default_factory=list)

    def render(self) -> str:
        bits = [f"[per-camera/{self.camera_id}] {self.status}"]
        if self.segments:
            bits.append(f"segments={self.segments}")
        if self.key_mode:
            bits.append(f"key={self.key_mode}")
        if self.assets_uploaded or self.assets_failed:
            bits.append(f"images={self.assets_uploaded}"
                        + (f"/{self.assets_failed}failed"
                           if self.assets_failed else ""))
        if self.run_id:
            bits.append(f"run_id={self.run_id}")
        if self.errors:
            bits.append(f"errors={self.errors}")
        return "  ".join(bits)


def source_video_name(raw_video_name: str, camera_id: str) -> str:
    """The SOURCE clip's name, with any staging prefix removed.

    Both stagers copy a clip to ``<CAMERA>_<original name>`` --
    `historical_runner.stage_clips` and the batch downloader in
    `master_runner` -- so the local file this camera actually ran on is
    ``RIGHT_UP_camera_CCTV_HZBN_DHN_2_RIGHT_UP_20260731_052218_train.mp4``,
    while the ORIGINAL S3 object is the same name without the leading
    ``RIGHT_UP_``.

    Why that matters here: the fused pass names its document from the S3 URL
    (`dashboard_ingest`: ``raw_video_name = os.path.basename(src_url)``), so it
    never sees the prefix.  Under ``WAGONEYE_INSPECTION_KEY_LAYOUT=v1`` the
    filename is part of the S3 key, so a prefixed name puts the provisional
    document at

        Right_up/2026-07-31/RIGHT_UP_camera_..._train_inspection.json

    while assembly writes

        Right_up/2026-07-31/camera_..._train_inspection.json

    -- two objects, and the stale camera-local one is never replaced.  Under
    the v4 layout the key ignores the filename, so this only changed the
    ``raw_video_name`` field; the bug was invisible there, which is exactly why
    it is worth stripping unconditionally rather than per layout.

    Only an EXACT leading ``<camera_id>_`` is removed.  A blunt replace would
    corrupt the name, because the camera id also occurs INSIDE it
    (``..._DHN_2_RIGHT_UP_2026...``).  A file that was never staged -- the
    ``--local-only`` case -- is returned untouched.
    """
    if not raw_video_name or not camera_id:
        return raw_video_name
    prefix = f"{camera_id}_"
    return (raw_video_name[len(prefix):]
            if raw_video_name.startswith(prefix) else raw_video_name)


def _upload_assets(s3_client, bucket: str,
                   assets: Sequence[Tuple[str, str]],
                   *, uploader=None, camera_id: str = "",
                   session_ts: str = "",
                   urls: Optional[Dict[str, str]] = None,
                   ) -> Tuple[int, int]:
    """Upload the evidence JPEGs a document references. Returns (ok, failed).

    De-duplicated: the same frame is legitimately referenced from several places
    in one document (a wagon gallery and a feature snapshot can be the same
    file), and uploading it once is enough.

    Never raises. A failed image costs one broken thumbnail; it must not stop
    the document from being published, because the document still carries the
    wagon data that is the point of the early post.
    """
    from delivery import artifact_uploader as AU
    if uploader is None:
        uploader = AU.ArtifactUploader(s3_client=s3_client, verbose=False)
    if urls is None:
        urls = {}
    ok = failed = 0
    # dict.fromkeys keeps first-seen order while dropping repeats.
    for local, key in dict.fromkeys(assets):
        if not os.path.isfile(local):
            failed += 1
            continue
        try:
            out = uploader.upload(
                local, AU.artifact_type_for(key, sub_prefix="evidence"),
                camera_id=(camera_id
                            or C.CAMERA_S3_FOLDER.get(C.MASTER_CAMERA,
                                                      C.MASTER_CAMERA)), session_ts=session_ts,
                s3_bucket=bucket, s3_key=key, content_type="image/jpeg")
            urls[key] = out.https_url
            ok += 1
        except Exception as e:  # noqa: BLE001
            log.warning("[PER-CAMERA] evidence upload failed %s: %s", key, e)
            failed += 1
    return ok, failed


def _train_ts(batch_key: str, raw_video_name: str, bundle):
    """The train's timestamp, derived the way the FUSED path derives it.

    `dashboard_ingest.run()` uses `extract_train_timestamp(batch_key)` -- ONE
    value shared by all four cameras. The per-camera path must agree with it,
    because the S3 key is built from this timestamp and the two documents can
    only replace one another if their keys are identical.

    Using each camera's own clip filename here (as this did originally) breaks
    that: the four clips are stamped seconds apart -- 05:22:11, 05:22:18,
    05:22:27, 05:22:41 for one real train -- so three of the four per-camera
    documents would land on keys the fused pass never touches, and the
    dashboard would keep a stale camera-local record beside the canonical one
    instead of showing it replaced.

    The clip name and bundle directory remain as FALLBACKS, for a caller that
    has no batch key to give.
    """
    from delivery import dashboard_ingest as DASH
    return DASH.extract_train_timestamp(
        batch_key, raw_video_name, os.path.basename(bundle.dir))


def build_document(
    bundle,
    *,
    fps: float = 0.0,
    total_frames: int = 0,
    raw_video_name: str = "",
    direction: str = "unknown",
    batch_key: str = "",
    assets: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Build this camera's `{camera_id, version, inspection_data}` document.

    `assets` -- when a list is passed, every (local_path, s3_key) pair this
    document references is appended to it. `publish()` uses that to UPLOAD the
    referenced images, because a document that names an object nobody created is
    a document full of broken thumbnails.

    Uses the camera-scoped state + fused states from
    `orchestrator.camera_report_adapter.adapt()`, so the authority rules, anomaly
    precedence and confidence maths are the proven ones -- pointed at this
    camera's own persisted feature tree.
    """
    from delivery import dashboard_ingest as DASH
    from delivery import inspection_json as IJ
    from orchestrator.camera_report_adapter import adapt

    cam = bundle.camera_id
    state, unified, paths = adapt(bundle, fps=fps, total_frames=total_frames)
    evidence_root = paths["evidence_root"]

    # Report the SOURCE clip, not our staged copy of it -- see
    # `source_video_name`. This is what keeps the document's `raw_video_name`
    # equal to the one assembly will write, and therefore keeps the two S3 keys
    # equal under the v1 layout.
    raw_video_name = source_video_name(raw_video_name, cam)
    ts = _train_ts(batch_key, raw_video_name, bundle)
    folder = DASH.full_camera_id(cam)
    version = DASH._version()

    def _url_for(*, gw_id: str, feature: str, camera: str, filename: str):
        # The camera-local evidence tree is flat per feature, exactly the layout
        # `evidence_rel_path` already resolves.  A URL is only produced for a
        # file that EXISTS LOCALLY -- but that is not enough on its own: the
        # browser fetches these over HTTPS, so the file has to reach S3 too.
        # Every minted URL is therefore recorded in `assets` for `publish()` to
        # upload. Emitting the URL without uploading is what made a provisional
        # document render as a page of broken thumbnails.
        rel = DASH.evidence_rel_path(evidence_root, gw_id, feature, camera,
                                     filename)
        if not rel:
            return None
        bucket = DASH.inspection_bucket()
        key = f"{folder}/{DASH.date_folder(ts)}/camera_evidence/{rel}"
        if assets is not None:
            assets.append((os.path.join(evidence_root, rel), key))
        return f"https://{bucket}.s3.{C.S3_REGION}.amazonaws.com/{key}"

    doc = IJ.build_inspection_json(
        camera=cam,
        camera_folder=folder,
        raw_video_name=raw_video_name or f"{folder}.mp4",
        upload_timestamp=ts,
        direction=direction,
        state=state,
        unified=unified,
        states_root=paths["wagon_states_root"],
        evidence_root=evidence_root,
        url_for=_url_for,
        trimmed_video_url="",
        pdf_report_url="",
        detected_video_url="",
        raw_video_urls=[],
        damage_model_active=True,
        version=version,
        identified_by=DASH._model_id(),
        schema=IJ.schema_for_version(version),
    )
    if version.strip().lower() == "v1":
        doc["camera_id"] = folder

    # Say loudly what this document is, so it can never be mistaken for the
    # fused, canonical one that assembly publishes later over the same key.
    doc["inspection_data"]["_adapter"] = {
        "generated_by": "delivery.camera_inspection (per-camera, pre-assembly)",
        "source": f"sealed camera bundle for {cam}",
        "numbering": "camera-local",
        "superseded_by_assembly": True,
        "provisional": True,
        "replaced_in_place_by_assembly": key_mode() == KEY_MODE_REPLACE,
        "note": ("wagon n is THIS camera's nth segment -- not necessarily the "
                 "same physical wagon as another camera's wagon n, and not the "
                 "fused GW_n.  total_wagons is this camera's own segment count."),
        "camera_authority": ("right_door+ocr" if cam == C.CAMERA_RIGHT_UP
                             else "left_door" if cam == C.CAMERA_LEFT_UP
                             else "load+top_damage"),
    }
    return doc


def publish(
    bundle,
    *,
    s3_client=None,
    fps: float = 0.0,
    total_frames: int = 0,
    raw_video_name: str = "",
    direction: str = "unknown",
    batch_key: str = "",
    dry_run: bool = False,
    requests_mod=None,
    verbose: bool = True,
) -> CameraIngestResult:
    """Build, upload and POST one camera's document.  NEVER raises.

    A failure here must not un-seal a camera or fail its run: the camera's work
    is already persisted, and global assembly will publish the canonical
    document later regardless.
    """
    from delivery import dashboard_ingest as DASH

    cam = getattr(bundle, "camera_id", "?")
    raw_video_name = source_video_name(raw_video_name, cam)
    res = CameraIngestResult(camera_id=cam)

    if not is_enabled():
        res.status = "disabled"
        return res

    try:
        assets: List[Tuple[str, str]] = []
        doc = build_document(bundle, fps=fps, total_frames=total_frames,
                             raw_video_name=raw_video_name, direction=direction,
                             batch_key=batch_key, assets=assets)
        res.built = True
        res.segments = int(doc["inspection_data"].get("total_wagons") or 0)
    except Exception as e:  # noqa: BLE001
        res.status = "build_failed"
        res.errors.append(str(e))
        log.error("[PER-CAMERA/%s] build failed: %s", cam, e, exc_info=True)
        return res

    text = json.dumps(doc, indent=2, default=str)
    folder = DASH.full_camera_id(cam)
    ts = _train_ts(batch_key, raw_video_name, bundle)
    json_name = f"{os.path.splitext(doc['inspection_data']['raw_video_name'])[0]}_inspection.json"

    # Written beside the camera's own artifacts, so a sealed bundle carries the
    # exact document that was published from it.
    try:
        local_dir = os.path.join(bundle.dir, "delivery")
        os.makedirs(local_dir, exist_ok=True)
        res.local_json = os.path.join(local_dir, json_name)
        with open(res.local_json, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        res.errors.append(f"local write: {e}")

    bucket = DASH.inspection_bucket()
    # In `replace` mode the key is built by the SAME function the fused path
    # calls, with the SAME timestamp, so assembly's document lands on exactly
    # this object and upgrades the dashboard record in place. Calling
    # `inspection_s3_key` rather than formatting a path by hand is deliberate:
    # it also keeps the two in step under `WAGONEYE_INSPECTION_KEY_LAYOUT=v1`,
    # where the key includes `json_name`.
    mode = key_mode()
    res.key_mode = mode
    if mode == KEY_MODE_REPLACE:
        key = DASH.inspection_s3_key(camera=cam,
                                     date_folder_str=DASH.date_folder(ts),
                                     json_name=json_name, ts=ts)
    else:
        key = (f"{folder}/{DASH.date_folder(ts)}/per_camera/"
               f"{os.path.splitext(json_name)[0]}.json")
    res.s3_uri = f"s3://{bucket}/{key}"

    if dry_run:
        res.status = "dry_run"
        if verbose:
            log.info("[PER-CAMERA/%s] --dry-run: would publish %d segment(s) -> %s",
                     cam, res.segments, res.s3_uri)
        return res

    if s3_client is None:
        try:
            import boto3
            s3_client = boto3.client("s3", region_name=C.S3_REGION)
        except Exception as e:  # noqa: BLE001
            res.status = "no_s3_client"
            res.errors.append(str(e))
            return res

    # Evidence FIRST, then the JSON. A reader that sees the document must find
    # the images it names; publishing the document before its images would show a
    # broken page for as long as the uploads took.
    res.assets_uploaded, res.assets_failed = _upload_assets(
        s3_client, bucket, assets, camera_id=DASH.full_camera_id(cam),
        session_ts=batch_key)
    if res.assets_failed:
        res.errors.append(f"{res.assets_failed} evidence file(s) failed to "
                          f"upload")

    try:
        from delivery import artifact_uploader as AU
        _up = AU.ArtifactUploader(s3_client=s3_client, verbose=False)
        _out = _up.upload(res.local_json, "inspection_json",
                          camera_id=DASH.full_camera_id(cam), session_ts=batch_key,
                          s3_bucket=bucket, s3_key=key,
                          content_type="application/json")
        # The document's own location comes from the RESULT: in `api` mode the
        # backend chose the key, so `res.s3_uri` computed from bucket + key
        # would name an object that does not exist -- and this URI is what the
        # dashboard ingest POST points at.
        res.s3_uri = _out.s3_uri
        res.status = "uploaded"
    except Exception as e:  # noqa: BLE001
        res.status = "upload_failed"
        res.errors.append(str(e))
        log.error("[PER-CAMERA/%s] upload failed %s: %s", cam, res.s3_uri, e)
        return res

    payload = {"camera_id": folder, "inspection_s3_uri": res.s3_uri,
               "version": DASH._version()}
    # `batch_key`, NOT `basename(bundle.dir)` -- that is the CAMERA folder
    # (`<evidence_root>/RIGHT_UP`), so the provisional post was identifying
    # itself as train "RIGHT_UP" while assembly identified the same result as
    # train "20260722_050704". The two could never have been recognised as the
    # same event, no matter what the receiver does with the key.
    idem = DASH.ingest_idempotency_key(batch_key or
                                       os.path.basename(bundle.dir), cam)

    any_ok = False
    for url in DASH.ingest_api_urls():
        out = DASH._post_ingest(api_url=url, payload=payload, idem_key=idem,
                                requests_mod=requests_mod)
        if out["ok"]:
            any_ok = True
            res.run_id = res.run_id or out.get("run_id")
        else:
            res.errors.append(f"{url}: {out.get('error')}")
    res.status = "ingested" if any_ok else "ingest_failed"
    if verbose:
        log.info("[PER-CAMERA/%s] %s (%d segment(s)) -> %s", cam, res.status,
                 res.segments, res.s3_uri)
    return res
