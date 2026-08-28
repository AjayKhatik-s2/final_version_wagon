"""Stage 6 -- upload `batch_outputs/<key>/` to S3.

Strategy:
    * PDF goes to the report microservice first; falls back to S3.
    * JSON goes directly to S3 with application/json content-type.
    * Everything else (wagon_cache + wagon_states + global_state) is
      recursively uploaded under
        s3://<bucket>/<train_batch_prefix>/<batch_key>/...
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core import constants as C


# -----------------------------------------------------------------------------
# Content-type per extension (very small mapping)
# -----------------------------------------------------------------------------

_CONTENT_TYPES = {
    ".pdf":  "application/pdf",
    ".json": "application/json",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".mp4":  "video/mp4",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
}


def _content_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _CONTENT_TYPES.get(ext, "application/octet-stream")


# -----------------------------------------------------------------------------
# Microservice PDF upload (proven helper preserved from the legacy
# master_runner; same API and product name).
# -----------------------------------------------------------------------------

def _upload_pdf_microservice(pdf_path: str) -> Optional[str]:
    import requests
    ist = timezone(timedelta(hours=5, minutes=30))
    today = datetime.now(ist).strftime("%d-%m-%Y")
    for attempt in range(1, 4):
        try:
            with open(pdf_path, "rb") as f:
                files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
                data  = {"product_name": C.PRODUCT_NAME, "folder_name": today}
                resp = requests.post(C.UPLOAD_API_URL, data=data, files=files,
                                     timeout=120)
            if resp.status_code == 200:
                url = resp.json().get("url")
                if url:
                    print(f"[DELIVERY] PDF microservice URL: {url}")
                    return url
        except Exception as e:
            print(f"[DELIVERY] PDF microservice attempt {attempt}/3 failed: {e}")
        time.sleep(10)
    return None


def _publish(s3_client, local_path: str, artifact_type: str, bucket: str,
             key: str, batch_key: str, *, content_type: str = "",
             camera_id: str = "", uploader=None):
    """Upload one file through the artifact uploader and return its result.

    Every single-file upload site in this module goes through here, so `api`
    mode reaches all of them at once and none is left making a direct
    `s3_client.upload_file` call. The URL is taken from the RESULT rather than
    built from `bucket + key`, because in `api` mode the backend chose a
    different key and a computed URL would point at nothing.
    """
    from delivery import artifact_uploader as AU
    up = uploader or AU.ArtifactUploader(s3_client=s3_client, verbose=False)
    return up.upload(local_path, artifact_type,
                     camera_id=(camera_id
                                or C.CAMERA_S3_FOLDER.get(C.MASTER_CAMERA,
                                                          C.MASTER_CAMERA)),
                     session_ts=batch_key, s3_bucket=bucket, s3_key=key,
                     content_type=content_type or None)


@dataclass
class TreeUploadResult:
    """What one subtree upload produced.

    `count` alone was the old return value, and `bool(dict_of_zeros)` being True
    is how a total S3 outage once reported success. `failed` and `errors` make
    the difference between "nothing to upload" and "nothing uploaded" visible.
    """

    count: int = 0
    failed: int = 0
    via: str = ""
    urls: Dict[str, str] = field(default_factory=dict)
    keys: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def upload_pdf(s3_client, pdf_path: str, batch_key: str) -> Optional[str]:
    """Microservice first; S3 direct fallback."""
    if not os.path.exists(pdf_path):
        return None
    url = _upload_pdf_microservice(pdf_path)
    if url:
        return url
    bucket = C.S3_OUTPUT_BUCKET
    key = f"{C.S3_TRAIN_BATCH_PREFIX}/{batch_key}/reports/combined_train_report.pdf"
    try:
        return _publish(s3_client, pdf_path, "combined_report_pdf",
                        bucket, key, batch_key,
                        content_type="application/pdf").https_url
    except Exception as e:
        print(f"[DELIVERY] PDF upload failed: {e}", file=sys.stderr)
        return None


def upload_json(s3_client, json_path: str, batch_key: str) -> Optional[str]:
    if not os.path.exists(json_path):
        return None
    bucket = C.S3_OUTPUT_BUCKET
    key = f"{C.S3_TRAIN_BATCH_PREFIX}/{batch_key}/reports/combined_train_report.json"
    try:
        url = _publish(s3_client, json_path, "inspection_json", bucket, key,
                       batch_key, content_type="application/json").https_url
        print(f"[DELIVERY] JSON URL: {url}")
        return url
    except Exception as e:
        print(f"[DELIVERY] JSON upload failed: {e}", file=sys.stderr)
        return None


def upload_tree_detailed(
    s3_client, local_dir: str, batch_key: str,
    *, sub_prefix: str = "",
    skip_extensions: Optional[set] = None,
    session_ts: str = "",
    uploader=None,
    default_camera_id: str = "",
) -> "TreeUploadResult":
    """Upload everything under `local_dir` and report WHERE each file landed.

    The URL map is the point. In `s3` mode a caller could compute each URL
    itself from bucket + key + region, and this repo did. In `api` mode it
    cannot: the backend chooses the key, so the only correct URL for a file is
    the one its own upload returned. Returning the map makes the two transports
    interchangeable for callers that publish links -- which is every caller that
    matters, because a report full of computed links is a report full of 404s.

    `urls` is keyed by the path RELATIVE to `local_dir` (`GW_25/damage/x.jpg`),
    which is the same key the report's `evidence_pages` already uses, so the two
    join without either side knowing the other's layout.
    """
    res = TreeUploadResult()
    if not os.path.isdir(local_dir):
        return res
    from delivery import artifact_uploader as AU

    if uploader is None:
        uploader = AU.ArtifactUploader(s3_client=s3_client, verbose=False)
    bucket = C.S3_OUTPUT_BUCKET
    base = f"{C.S3_TRAIN_BATCH_PREFIX}/{batch_key}"
    if sub_prefix:
        base = f"{base}/{sub_prefix.strip('/')}"
    skip = skip_extensions or set()

    for root, _, files in os.walk(local_dir):
        for fn in sorted(files):
            if any(fn.lower().endswith(ext) for ext in skip):
                continue
            local = os.path.join(root, fn)
            rel = os.path.relpath(local, local_dir).replace(os.sep, "/")
            key = f"{base}/{rel}"
            try:
                out = uploader.upload(
                    local, AU.artifact_type_for(rel, sub_prefix=sub_prefix),
                    camera_id=(_camera_hint(rel) or default_camera_id
                               or C.CAMERA_S3_FOLDER.get(C.MASTER_CAMERA,
                                                         C.MASTER_CAMERA)),
                    session_ts=session_ts or batch_key,
                    s3_bucket=bucket, s3_key=key,
                    filename=_unique_filename(rel),
                    content_type=_content_type_for(fn))
                res.count += 1
                res.urls[rel] = out.https_url
                res.keys[rel] = out.key
                res.via = out.via
            except Exception as e:  # noqa: BLE001 - one file must not stop a tree
                res.failed += 1
                res.errors.append(f"{rel}: {e}")
                print(f"[DELIVERY] upload failed {local} -> {key}: {e}",
                      file=sys.stderr)
    return res


def _unique_filename(rel_path: str) -> str:
    """A filename unique across the whole train, from its path in the subtree.

    This is not cosmetic. Verified against the live endpoint: the backend builds
    its object key as `<camera_id>/<session_ts>/<artifact_type>/<filename>` --
    with NO wagon id in it. Per-wagon evidence names repeat by design
    (`door/left_best.jpg`, `ocr/best_frame.jpg`, `load/best_frame.jpg` exist
    once per wagon), so sending the bare basename made 59 wagons' frames
    collapse into ONE object each, last-write-wins.

    Measured before the fix: two files, one backend key.

    That failure is invisible from the outside -- every upload returns 200,
    every link resolves, every image renders -- and the report would show all 59
    wagons sharing whichever photo landed last.

    The wagon id therefore goes in front, and the original name is kept as the
    tail so anything parsing a suffix (`track_1__RIGHT_UP_TOP.jpg`) still sees
    what it expects.
    """
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        return parts[-1] if parts else ""
    # `GW_25/damage/track_1__RIGHT_UP_TOP.jpg` -> `GW_25__track_1__RIGHT_UP_TOP.jpg`
    # Intermediate directories other than the wagon are dropped: the feature is
    # already carried by the artifact_type folder the backend chooses.
    return f"{parts[0]}__{parts[-1]}"


def _camera_hint(rel_path: str) -> str:
    """The camera a file belongs to, when its path says so.

    Evidence names are camera-scoped (`track_1__RIGHT_UP_TOP.jpg`) and some trees
    nest by camera, so the API's `camera_id` can usually be filled in honestly.
    Empty when the path does not say, and the caller then supplies a default --
    the API wants this field populated, and `GW_25/door/left_best.jpg` names no
    camera because it is FUSED evidence belonging to the wagon rather than to one
    viewpoint. Guessing a camera from `left_` would be inventing provenance;
    falling back to the master camera says "this is the train's own evidence",
    which is what it is.
    """
    p = rel_path.replace("\\", "/")
    # Longest first: LEFT_UP_TOP contains LEFT_UP, so a shortest-match loop
    # would label every top-camera file with the side camera.
    #
    # Returns the FULL prefixed id (`camera_CCTV_HZBN_DHN_5_RIGHT_TOP`), not the
    # short one. Verified against the live endpoint: the backend builds the
    # object key as `<camera_id>/<session_ts>/<type>/<filename>`, so sending
    # `RIGHT_UP_TOP` would file every artifact under a folder that exists
    # nowhere else in the system.
    for cam in sorted(C.ALL_CAMERAS, key=len, reverse=True):
        if cam in p:
            return C.CAMERA_S3_FOLDER.get(cam, cam)
    return ""


def upload_tree(
    s3_client, local_dir: str, batch_key: str,
    *, sub_prefix: str = "",
    skip_extensions: Optional[set] = None,
) -> int:
    """Count-only wrapper, kept so existing callers are unchanged."""
    return upload_tree_detailed(
        s3_client, local_dir, batch_key, sub_prefix=sub_prefix,
        skip_extensions=skip_extensions).count
