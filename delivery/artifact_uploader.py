"""One choke point for every artifact this pipeline publishes.

Ported from the V2 engine (`Train-Inspection-Engine`,
`core/artifact_uploader.py`) so both engines speak the same Artifact Upload API
contract. Two transports behind one interface:

    "s3"   boto3 `upload_file` straight into the bucket, and the object URL is
           COMPUTED locally from bucket + key + region. The behaviour this
           pipeline has always had.
    "api"  POST the file as multipart/form-data to the backend's Artifact
           Upload API. The BACKEND decides the bucket and key and returns them,
           so the URL is READ OUT of the response rather than computed.

Why one class rather than a flag at each call site
--------------------------------------------------
In `api` mode the destination is not knowable from local config. Any caller
that builds its own `s3://` URI or HTTPS URL from
`S3_OUTPUT_BUCKET + S3_TRAIN_BATCH_PREFIX + batch_key` is therefore emitting a
link to a place the file is not. That failure is silent: the JSON looks right,
the report publishes, and the image 404s in the dashboard.

So every upload site must take its bucket, key and URL from the `UploadResult`
it gets back. That is what makes the two transports interchangeable instead of
merely parallel, and it is the reason this is a class with a result type rather
than a helper function.

Why the pipeline wants `api` mode
---------------------------------
The ML box then needs no AWS credentials for publishing at all -- the backend
owns the bucket and the keys. That removes the long-lived access key from the
EC2 host, which is the whole point.

What deliberately does NOT route through here
---------------------------------------------
See `LOCAL_KEY_ARTIFACTS`. An artifact whose key this pipeline has to RECOMPUTE
later in order to find the file again cannot go through the API, because nothing
in the contract tells us what key the backend chose. Writing it through the API
would mean it could never be read back.
"""

from __future__ import annotations

import mimetypes
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("delivery.artifact_uploader")

_TAG = "[ARTIFACT]"

MODE_S3 = "s3"
MODE_API = "api"

#: Every `artifact_type` the backend accepts. Validated locally so a typo fails
#: here rather than coming back as a 400 after the file is already on the wire.
ARTIFACT_TYPES = frozenset({
    "trimmed_video",
    "detected_video",
    "inspection_pdf",
    "wagon_frame",
    "loco_frame",
    "problem_frame",
    "wagon_number_frame",
    "loco_number_frame",
    "inspection_json",
    "combined_report_pdf",
    "pipeline_state",
    "combiner_state",
})

#: Artifacts this pipeline READS BACK by recomputing their key locally, and
#: which therefore always go straight to S3 whatever the mode.
#:
#: In API mode the backend chooses the key and the contract does not tell us
#: what it chose, so a file written through the API could not be found again.
#: For state that means the pipeline would start empty every run and reprocess
#: every video; the write and the read have to stay symmetrical.
LOCAL_KEY_ARTIFACTS = frozenset({"pipeline_state", "combiner_state"})

#: `artifact_type`s the API requires a `session_ts` for -- i.e. everything else.
_SESSION_TS_OPTIONAL = LOCAL_KEY_ARTIFACTS

#: Seconds to wait before retry N (index 0 = after the first failure).
_BACKOFF_SECONDS = (1.0, 3.0, 8.0)


class ArtifactUploadError(RuntimeError):
    """An artifact could not be uploaded by any available transport."""


class PermanentUploadError(ArtifactUploadError):
    """A 4xx: the request itself is wrong, so the identical retry fails too.

    400 (unknown artifact_type, missing session_ts, empty file, unresolvable
    filename) and 401 (missing or wrong token) are both contract errors.
    Separated from the transient failures so the retry loop stops immediately
    instead of sleeping through a backoff that cannot help.
    """


@dataclass
class UploadResult:
    """Where an artifact actually landed.

    Same shape whichever transport produced it, so no caller branches on the
    mode. `via` is for logging only.
    """

    bucket: str
    key: str
    s3_uri: str
    https_url: str
    via: str = MODE_S3
    size_bytes: Optional[int] = None
    content_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"bucket": self.bucket, "key": self.key, "s3_uri": self.s3_uri,
                "https_url": self.https_url, "via": self.via,
                "size_bytes": self.size_bytes,
                "content_type": self.content_type}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def upload_mode() -> str:
    """`s3` (default) or `api`.

    Default is `s3` so this port changes nothing until it is switched on
    deliberately: turning it on moves where every artifact in the pipeline
    lands, and that is not a change to make by importing a module.
    """
    m = _env("WAGONEYE_ARTIFACT_UPLOAD_MODE",
             _env("ARTIFACT_UPLOAD_MODE", MODE_S3)).lower()
    return MODE_API if m == MODE_API else MODE_S3


def api_url() -> str:
    return _env("WAGONEYE_ARTIFACT_UPLOAD_API_URL",
                _env("ARTIFACT_UPLOAD_API_URL", C.ARTIFACT_UPLOAD_API_URL))


def api_token() -> str:
    return _env("WAGONEYE_ARTIFACT_UPLOAD_TOKEN",
                _env("ARTIFACT_UPLOAD_TOKEN", ""))


def fallback_to_s3() -> bool:
    """Fall back to direct S3 when an API upload fails.

    OFF by default: the requirement `api` mode exists to satisfy is that the ML
    code makes no S3 upload calls at all. With it off a failed upload is a lost
    artifact, which is loud; with it on the pipeline quietly needs AWS
    credentials again.
    """
    return _env_bool("WAGONEYE_ARTIFACT_UPLOAD_FALLBACK_TO_S3",
                     _env_bool("ARTIFACT_UPLOAD_FALLBACK_TO_S3", False))


def s3_object_url(bucket: str, key: str, region: str = "") -> str:
    """Direct HTTPS URL for an S3 object. Only correct for `s3`-mode uploads --
    in `api` mode the backend's returned URL is the only correct one."""
    from urllib.parse import quote
    b = bucket.split("/", 1)[0]
    return (f"https://{b}.s3.{region or C.S3_REGION}.amazonaws.com/"
            f"{quote(key, safe='/')}")


def parse_s3_object_url(url: str) -> "tuple[str, str]":
    """`(bucket, key)` from an object URL, or `("", "")`.

    Exists because an artifact uploaded through the API lives in whatever bucket
    the backend chose. Reading the bucket back out of the recorded URL keeps a
    later consumer correct even if that bucket changes, instead of depending on
    a local config value staying in step with the backend.

    Both AWS styles are handled:
        https://<bucket>.s3.<region>.amazonaws.com/<key>   (virtual-hosted)
        https://s3.<region>.amazonaws.com/<bucket>/<key>   (path-style)
    """
    from urllib.parse import unquote
    if not url or "amazonaws.com/" not in url:
        return "", ""
    tail = url.split("://", 1)[-1]
    host, _, path = tail.partition("/")
    if not path:
        return "", ""
    if host.startswith("s3.") or host.startswith("s3-"):
        bucket, _, key = path.partition("/")          # path-style
        return bucket, unquote(key)
    return host.split(".s3.", 1)[0], unquote(path)    # virtual-hosted


# ---------------------------------------------------------------------------
# The uploader
# ---------------------------------------------------------------------------

class ArtifactUploader:
    """Upload artifacts either straight to S3 or through the backend API."""

    def __init__(
        self,
        *,
        s3_client=None,
        region: str = "",
        mode: Optional[str] = None,
        url: Optional[str] = None,
        token: Optional[str] = None,
        fallback: Optional[bool] = None,
        timeout: int = 120,
        max_attempts: int = 3,
        requests_mod=None,
        verbose: bool = True,
    ) -> None:
        self.s3 = s3_client
        self.region = region or C.S3_REGION
        self.mode = (mode or upload_mode()).strip().lower()
        self.url = url if url is not None else api_url()
        self.token = token if token is not None else api_token()
        self.fallback = fallback if fallback is not None else fallback_to_s3()
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.requests_mod = requests_mod
        self.verbose = verbose

        if self.mode not in (MODE_S3, MODE_API):
            raise ValueError(f"upload mode must be 's3' or 'api', "
                             f"got {self.mode!r}")
        if self.mode == MODE_API and not self.token:
            # Rejected at construction, not per upload: a missing token is a
            # configuration mistake, and discovering it once per artifact means
            # discovering it hundreds of times per train.
            raise ValueError(
                "artifact upload mode 'api' requires an upload token "
                "(WAGONEYE_ARTIFACT_UPLOAD_TOKEN / ARTIFACT_UPLOAD_TOKEN)")
        if self.verbose:
            extra = (f" endpoint={self.url} fallback_to_s3={self.fallback}"
                     if self.mode == MODE_API else "")
            log.info("%s uploader mode=%s%s", _TAG, self.mode, extra)

    @property
    def api_enabled(self) -> bool:
        return self.mode == MODE_API

    # ---- public ----------------------------------------------------------

    def upload(
        self,
        local_path: str,
        artifact_type: str,
        *,
        camera_id: str = "",
        session_ts: Optional[str] = None,
        s3_bucket: str = "",
        s3_key: str = "",
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> UploadResult:
        """Upload one file and report where it landed.

        `s3_bucket` / `s3_key` are always required: they are the destination in
        `s3` mode and the fallback destination in `api` mode, so a caller can
        never end up with no way to store the file.

        `filename` defaults to the basename of `s3_key`, not of `local_path` --
        the local file is often a temp name while the key's basename is the
        convention the backend's ingestion parses. Passing it explicitly keeps
        the API-mode name identical to the S3-mode one.
        """
        if artifact_type not in ARTIFACT_TYPES:
            raise ValueError(f"unknown artifact_type {artifact_type!r}; "
                             f"expected one of {sorted(ARTIFACT_TYPES)}")
        if not local_path or not os.path.isfile(local_path):
            raise FileNotFoundError(local_path)

        filename = (filename or os.path.basename(s3_key)
                    or os.path.basename(local_path))
        use_api = self.api_enabled and artifact_type not in LOCAL_KEY_ARTIFACTS
        if not use_api:
            return self._via_s3(local_path, s3_bucket, s3_key, content_type)

        if session_ts is None and artifact_type not in _SESSION_TS_OPTIONAL:
            raise ValueError(f"artifact_type={artifact_type!r} requires "
                             f"session_ts in api mode")
        try:
            return self._via_api_with_retries(
                local_path, artifact_type, camera_id=camera_id,
                session_ts=session_ts, filename=filename)
        except Exception as e:  # noqa: BLE001 - transport-agnostic fallback
            if not self.fallback:
                raise ArtifactUploadError(
                    f"api upload failed for {artifact_type} {filename}: {e}"
                ) from e
            log.warning("%s api upload failed for %s %s (%s) -- falling back to "
                        "direct S3", _TAG, artifact_type, filename, e)
            return self._via_s3(local_path, s3_bucket, s3_key, content_type)

    # ---- transports ------------------------------------------------------

    def _via_s3(self, local_path: str, bucket: str, key: str,
                content_type: Optional[str]) -> UploadResult:
        if self.s3 is None:
            raise ArtifactUploadError("no S3 client available for an s3-mode "
                                      "upload")
        if not bucket or not key:
            raise ArtifactUploadError("s3-mode upload needs a bucket and a key")
        ct = content_type or (mimetypes.guess_type(key)[0]
                              or "application/octet-stream")
        self.s3.upload_file(local_path, bucket.split("/", 1)[0], key,
                            ExtraArgs={"ContentType": ct})
        b = bucket.split("/", 1)[0]
        return UploadResult(
            bucket=b, key=key, s3_uri=f"s3://{b}/{key}",
            https_url=s3_object_url(b, key, self.region), via=MODE_S3,
            content_type=ct,
            size_bytes=(os.path.getsize(local_path)
                        if os.path.isfile(local_path) else None))

    def _via_api_with_retries(
        self, local_path: str, artifact_type: str, *, camera_id: str,
        session_ts: Optional[str], filename: str,
    ) -> UploadResult:
        """`_via_api`, retrying only failures that could succeed later.

        Timeouts, connection errors and 5xx are transient -- the API documents
        500 as safe to retry. A 4xx is a contract error and the identical
        request would be rejected identically, so retrying only delays it.

        This matters most with `fallback_to_s3=False`: one train publishes
        hundreds of frames and a single blip would otherwise cost the report.
        """
        last: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._via_api(local_path, artifact_type,
                                     camera_id=camera_id,
                                     session_ts=session_ts, filename=filename)
            except PermanentUploadError:
                raise
            except Exception as e:  # noqa: BLE001 - transient
                last = e
                if attempt >= self.max_attempts:
                    break
                delay = _BACKOFF_SECONDS[min(attempt - 1,
                                             len(_BACKOFF_SECONDS) - 1)]
                log.warning("%s attempt %d/%d failed for %s %s (%s) -- "
                            "retrying in %.0fs", _TAG, attempt,
                            self.max_attempts, artifact_type, filename, e, delay)
                time.sleep(delay)
        raise ArtifactUploadError(f"{last} (after {self.max_attempts} attempt(s))"
                                  ) from last

    def _via_api(
        self, local_path: str, artifact_type: str, *, camera_id: str,
        session_ts: Optional[str], filename: str,
    ) -> UploadResult:
        requests_mod = self.requests_mod
        if requests_mod is None:
            import requests as requests_mod  # type: ignore
        ct = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        data = {"artifact_type": artifact_type, "camera_id": camera_id,
                "filename": filename}
        if session_ts:
            data["session_ts"] = session_ts

        with open(local_path, "rb") as fh:
            resp = requests_mod.post(
                self.url, headers={"X-ML-Upload-Token": self.token},
                data=data, files={"file": (filename, fh, ct)},
                timeout=self.timeout)

        code = int(getattr(resp, "status_code", 0))
        if code != 200:
            err = PermanentUploadError if 400 <= code < 500 else ArtifactUploadError
            raise err(f"HTTP {code}: {str(getattr(resp, 'text', ''))[:300]}")

        body = resp.json()
        bucket, key = body.get("bucket"), body.get("key")
        if not bucket or not key:
            # Without both, nothing downstream can reference the file. Better a
            # loud failure than a document carrying a link to nowhere.
            raise ArtifactUploadError(
                f"api response missing bucket/key: {str(body)[:300]}")
        return UploadResult(
            bucket=bucket, key=key,
            s3_uri=body.get("s3_uri") or f"s3://{bucket}/{key}",
            https_url=(body.get("https_url")
                       or s3_object_url(bucket, key, self.region)),
            via=MODE_API, size_bytes=body.get("size_bytes"),
            content_type=body.get("content_type"))


# ---------------------------------------------------------------------------
# This pipeline's files -> the API's artifact_type vocabulary
# ---------------------------------------------------------------------------
#
# The API accepts twelve types and validates them, so every file this pipeline
# publishes has to be named in its terms. The mapping is explicit rather than
# guessed from the extension, because two files with the same extension mean
# different things to the backend: an OCR crop is a `wagon_number_frame` and a
# damage crop is a `problem_frame`, and both are .jpg.

#: `evidence/<GW>/<feature>/...` -> artifact_type.
EVIDENCE_FEATURE_ARTIFACT_TYPES = {
    "ocr":    "wagon_number_frame",
    "damage": "problem_frame",
    "door":   "wagon_frame",
    "load":   "wagon_frame",
}

#: Used when a feature directory is not one of the four above. `wagon_frame` is
#: the honest default: it is a frame of a wagon, which is true of anything under
#: `evidence/<GW>/`, and it is the type with no special downstream meaning.
DEFAULT_EVIDENCE_ARTIFACT_TYPE = "wagon_frame"


def artifact_type_for(rel_path: str, *, sub_prefix: str = "") -> str:
    """The API's `artifact_type` for one file, from its path in the batch tree.

    `rel_path` is relative to the subtree root, e.g. `GW_25/damage/track_1.jpg`
    inside `evidence/`.
    """
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    name = (parts[-1] if parts else "").lower()
    sub = (sub_prefix or "").strip("/").lower()

    if sub == "processed_videos" or name.endswith("_processed.mp4"):
        return "detected_video"
    if sub == "evidence":
        # engine frames are a TRAIN-level asset, not a wagon's -- they are
        # collected into their own tree and describe the locomotive.
        if any(p.lower().startswith("engine_frames") for p in parts):
            return "loco_frame"
        if len(parts) >= 2:
            feat = parts[-2].lower()
            # `evidence/<GW>/<feature>/<camera>/<file>` also occurs, so the
            # feature is not always the immediate parent.
            for p in parts:
                if p.lower() in EVIDENCE_FEATURE_ARTIFACT_TYPES:
                    return EVIDENCE_FEATURE_ARTIFACT_TYPES[p.lower()]
            return EVIDENCE_FEATURE_ARTIFACT_TYPES.get(
                feat, DEFAULT_EVIDENCE_ARTIFACT_TYPE)
        return DEFAULT_EVIDENCE_ARTIFACT_TYPE
    if name.endswith(".pdf"):
        return ("combined_report_pdf" if "combined" in name
                else "inspection_pdf")
    if name.endswith(".json"):
        if "pipeline_state" in name:
            return "pipeline_state"
        if "combiner_state" in name:
            return "combiner_state"
        return "inspection_json"
    if name.endswith((".mp4", ".avi", ".mov")):
        return "detected_video"
    return DEFAULT_EVIDENCE_ARTIFACT_TYPE
