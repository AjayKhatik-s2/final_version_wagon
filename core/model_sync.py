"""Reconstruction + feature model availability check (+ optional S3 sync).

PRIMARY model source is the repo itself: the `.pt` weights ship WITH THE CODE
(old pipeline: in each camera folder; this repo: Git LFS -- `git lfs pull`).
There is NO production model bucket.  This module's main job is therefore to
VERIFY that every model a run needs is present locally, and fail fast (naming
the exact missing file) if not -- the same guarantee the old `bootstrap.sh`
Step 5 gave.

OPTIONAL S3 sync: if (and only if) `WAGONEYE_MODELS_S3_BUCKET` is set, a model
missing locally is downloaded into the local model dir from:
    reconstruction:  s3://<MODELS_S3_BUCKET>/<MODELS_S3_PREFIX>/reconstruction/<file>
    features:        s3://<MODELS_S3_BUCKET>/<MODELS_S3_PREFIX>/features/<file>
(This mirrors, for the inference models, what train_extraction/model_store.py
already does for the extractor's classify models via s3:// URIs.)

Which models are required:
    * every reconstruction model, ALWAYS
    * one feature model per ENABLED feature (Damage-only -> just damage.pt)

Failures are never silent: each missing/failed model reports the exact filename,
the expected `s3://bucket/key` (when a bucket is set), and the reason (no bucket
configured -> run `git lfs pull`; NoSuchKey/404; AccessDenied/403; no creds).

Downloads are atomic (`<file>.part` -> rename) so a half-download can never be
loaded.  Nothing here loads a model or runs inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core import config as CFG
from core import constants as C
from core.logging_setup import get_logger

log = get_logger("model_sync")


# ---------------------------------------------------------------------------
# What a run needs
# ---------------------------------------------------------------------------

# Filenames that appear in MORE THAN ONE model tree with different weights.  Under
# the flat (V4) S3 layout these cannot be told apart by key, so they are never
# auto-downloaded -- see ModelReq.ambiguous_in_flat_layout.
_AMBIGUOUS_FLAT_FILENAMES = C.AMBIGUOUS_MODEL_FILENAMES


# -----------------------------------------------------------------------------
# Git-LFS pointer detection
#
# `.gitattributes` tracks every *.pt with Git LFS.  A clone made without git-lfs
# on PATH leaves a ~130-byte TEXT stub in place of each model:
#
#     version https://git-lfs.github.com/spec/v1
#     oid sha256:9677c76d...
#     size 197566809
#
# That stub passes `os.path.isfile()`, so a naive existence check reports the
# model PRESENT and the run then dies deep inside ultralytics with an unhelpful
# deserialization error.  Detecting it here turns a confusing mid-run crash into
# a one-line startup message naming the file and the fix.
# -----------------------------------------------------------------------------

_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

#: No real .pt is anywhere near this small; a pointer is ~130 bytes.
_LFS_POINTER_MAX_BYTES = 1024


def is_lfs_pointer(path: str) -> bool:
    """True when `path` is an unpulled Git-LFS pointer rather than real weights."""
    try:
        if os.path.getsize(path) > _LFS_POINTER_MAX_BYTES:
            return False
        with open(path, "rb") as fh:
            return fh.read(len(_LFS_MAGIC)) == _LFS_MAGIC
    except OSError:
        return False


def lfs_pointer_size(path: str) -> Optional[int]:
    """The real byte size a pointer claims, for a more informative message."""
    try:
        with open(path, "rb") as fh:
            for line in fh.read(_LFS_POINTER_MAX_BYTES).splitlines():
                if line.startswith(b"size "):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _lfs_hint() -> str:
    """Actionable remedy, naming the PATH trap when git-lfs isn't runnable."""
    from shutil import which
    if which("git-lfs") is None:
        return ("git-lfs is not on PATH -- install it (brew/apt/yum install "
                "git-lfs) or add its directory to PATH, then run `git lfs pull`. "
                "NOTE: a git-lfs installed only for an interactive shell (e.g. "
                "/opt/homebrew/bin) is invisible to systemd/cron, which is the "
                "usual cause of this.")
    return "run `git lfs pull` in the repo to fetch the real weights"


@dataclass
class ModelReq:
    category: str          # "reconstruction" | "features" | "extraction"
    filename: str          # canonical filename (e.g. damage.pt)
    local_dir: str         # RECON_MODELS_DIR or FEAT_MODELS_DIR
    legacy: Optional[str] = None   # accepted legacy filename fallback

    @property
    def local_path(self) -> str:
        return os.path.join(self.local_dir, self.filename)

    @property
    def s3_key(self) -> str:
        """Where this model lives in the models bucket.

        `flat` (default) mirrors V4, which keeps every .pt at the bucket root:
            s3://wagon-eye-models/<file>
        `nested` adds the category folder, for a mirror organised by stage:
            s3://<bucket>/<prefix>/reconstruction|features|extraction/<file>
        """
        parts = [p for p in (C.MODELS_S3_PREFIX,
                             self.category if C.MODELS_S3_LAYOUT == "nested" else "",
                             self.filename) if p]
        return "/".join(parts)

    @property
    def alt_filename(self) -> Optional[str]:
        """The accepted alternative name for this model, if any.

        A store may hold the same weights under a different name (e.g.
        `load.pt` for `loaded.pt`).  `constants` records those, and both the
        local resolution (`feature_model_path`) and the S3 fetch below honour
        them, so nobody has to rename a production object.
        """
        return (C.FEATURE_MODEL_LEGACY.get(self.filename)
                or C.RECON_MODEL_LEGACY.get(self.filename))

    @property
    def alt_s3_key(self) -> Optional[str]:
        alt = self.alt_filename
        if not alt:
            return None
        parts = [p for p in (C.MODELS_S3_PREFIX,
                             self.category if C.MODELS_S3_LAYOUT == "nested" else "",
                             alt) if p]
        return "/".join(parts)

    @property
    def alt_local_path(self) -> Optional[str]:
        alt = self.alt_filename
        return os.path.join(self.local_dir, alt) if alt else None

    @property
    def s3_uri(self) -> str:
        return f"s3://{C.MODELS_S3_BUCKET}/{self.s3_key}"

    def existing_local(self) -> Optional[str]:
        """Return a present local path (canonical or legacy), else None.

        An unpulled Git-LFS POINTER does not count as present -- see
        `is_lfs_pointer`.  Returning it here would report the model available and
        then fail deep inside the model loader.
        """
        for path in self.candidate_paths():
            if os.path.isfile(path) and not is_lfs_pointer(path):
                return path
        return None

    def candidate_paths(self) -> List[str]:
        """Canonical path first, then the accepted legacy name."""
        paths = [self.local_path]
        if self.legacy:
            paths.append(os.path.join(self.local_dir, self.legacy))
        return paths

    def pointer_path(self) -> Optional[str]:
        """A candidate that exists but is only an LFS pointer, if any."""
        for path in self.candidate_paths():
            if os.path.isfile(path) and is_lfs_pointer(path):
                return path
        return None

    @property
    def ambiguous_in_flat_layout(self) -> bool:
        """True when this filename is used by more than one CATEGORY.

        ``side_classification.pt`` exists in BOTH `reconstruction/` (Stage-1
        segment classifier) and `extraction/` (train-presence classifier) with
        DIFFERENT weights.  Under the flat layout both would resolve to the same
        ``s3://<bucket>/side_classification.pt``, so auto-downloading it would
        silently install the wrong model in one of the two dirs.  We refuse to
        download those and require them locally instead.
        """
        return (C.MODELS_S3_LAYOUT == "flat"
                and self.filename in _AMBIGUOUS_FLAT_FILENAMES)


def required_models(enabled_features: Optional[List[str]] = None,
                    *, include_extraction: Optional[bool] = None) -> List[ModelReq]:
    """Return the ModelReq list for a run.

    `enabled_features` restricts the feature models (default: all four).  The
    reconstruction set is always included.

    `include_extraction` adds the raw->trimmed EXTRACTION classify models.
    Default (`None`) follows the resolved pipeline source: they are required only
    when this process produces its own trimmed clips (`--source raw`), because a
    pure consumer never loads them.
    """
    reqs: List[ModelReq] = [
        ModelReq("reconstruction", f, CFG.RECON_MODELS_DIR,
                 legacy=C.RECON_MODEL_LEGACY.get(f))
        for f in C.RECON_MODEL_FILES
    ]
    if include_extraction is None:
        include_extraction = CFG.PIPELINE_SOURCE.requires_extraction
    if include_extraction:
        reqs.extend(ModelReq("extraction", f, CFG.EXTRACTION_MODELS_DIR)
                    for f in C.EXTRACTION_MODEL_FILES)
    keys = C.FEATURE_MODEL_BY_KEY.keys() if enabled_features is None \
        else [k for k in enabled_features if k in C.FEATURE_MODEL_BY_KEY]
    for k in keys:
        filename = C.FEATURE_MODEL_BY_KEY[k]
        reqs.append(ModelReq("features", filename, CFG.FEAT_MODELS_DIR,
                             legacy=C.FEATURE_MODEL_LEGACY.get(filename)))
    return reqs


# ---------------------------------------------------------------------------
# Result of a verify/sync pass
# ---------------------------------------------------------------------------

@dataclass
class ModelStatus:
    req: ModelReq
    present: bool = False
    downloaded: bool = False
    local_path: Optional[str] = None
    error: Optional[str] = None       # human-readable reason when not present


@dataclass
class SyncReport:
    statuses: List[ModelStatus] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.present for s in self.statuses)

    @property
    def missing(self) -> List[ModelStatus]:
        return [s for s in self.statuses if not s.present]

    def summary_lines(self) -> List[str]:
        out: List[str] = []
        for s in self.statuses:
            if s.present and s.downloaded:
                out.append(f"  [downloaded] {s.req.category}/{s.req.filename}  <- {s.req.s3_uri}")
            elif s.present:
                out.append(f"  [present]    {s.req.category}/{s.req.filename}  ({s.local_path})")
            else:
                out.append(f"  [MISSING]    {s.req.category}/{s.req.filename}  "
                           f"expected s3 {s.req.s3_uri}  -- {s.error}")
        return out


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _make_s3_client():
    """Best-effort default-session S3 client (IAM role or ~/.aws creds)."""
    try:
        import boto3
        return boto3.client("s3", region_name=C.S3_REGION)
    except Exception as e:  # pragma: no cover - boto3 always present via reqs
        log.warning("[MODEL_SYNC] could not create S3 client: %s", e)
        return None


def _download_reason(exc: Exception) -> str:
    """Turn a boto3 exception into an operator-actionable reason."""
    code = getattr(getattr(exc, "response", None) or {}, "get", lambda *_: None)("Error")
    err_code = ""
    try:
        err_code = (exc.response.get("Error", {}) or {}).get("Code", "")  # type: ignore[attr-defined]
    except Exception:
        err_code = ""
    if err_code in ("404", "NoSuchKey", "NoSuchBucket"):
        return f"S3 object/bucket not found ({err_code or 'NoSuchKey'}) -- check the key/prefix/bucket"
    if err_code in ("403", "AccessDenied"):
        return ("access denied (403 AccessDenied) -- the instance IAM role/user "
                "lacks s3:GetObject on this key")
    name = type(exc).__name__
    if name in ("NoCredentialsError", "PartialCredentialsError"):
        return "no AWS credentials found (attach an EC2 IAM role or set ~/.aws/credentials)"
    if name == "EndpointConnectionError":
        return "cannot reach S3 endpoint (network/VPC/region issue)"
    return f"{name}: {exc}"


def _download(s3_client, req: ModelReq) -> ModelStatus:
    """Download one model atomically.  Returns a populated ModelStatus."""
    st = ModelStatus(req=req)
    os.makedirs(req.local_dir, exist_ok=True)

    # Try the canonical name, then the accepted alternative.  The alternative is
    # saved under ITS OWN name, not renamed to the canonical one: the processors
    # resolve either via `constants.feature_model_path`, and keeping the store's
    # name on disk makes it obvious which object a run actually loaded.
    attempts = [(req.s3_key, req.local_path)]
    if req.alt_s3_key:
        attempts.append((req.alt_s3_key, req.alt_local_path))

    last_error = None
    for key, dest in attempts:
        part = dest + ".part"
        try:
            s3_client.download_file(C.MODELS_S3_BUCKET, key, part)
            os.replace(part, dest)
            st.present = True
            st.downloaded = True
            st.local_path = dest
            note = "" if dest == req.local_path else f"  (accepted as {req.filename})"
            log.info("[MODEL_SYNC] downloaded s3://%s/%s -> %s%s",
                     C.MODELS_S3_BUCKET, key, dest, note)
            return st
        except Exception as e:
            if os.path.exists(part):
                try:
                    os.remove(part)
                except OSError:
                    pass
            last_error = e
            if len(attempts) > 1 and key == req.s3_key:
                log.info("[MODEL_SYNC] %s not in the store; trying %s",
                         req.filename, req.alt_filename)

    st.error = _download_reason(last_error) if last_error else "unknown"
    log.error("[MODEL_SYNC] FAILED %s : %s", req.s3_uri, st.error)
    return st


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def verify_and_sync(
    *,
    enabled_features: Optional[List[str]] = None,
    s3_client=None,
    download: bool = True,
    include_extraction: Optional[bool] = None,
) -> SyncReport:
    """Verify every required model is present locally; download missing ones.

    * A model already present locally (canonical or legacy name) is `present`,
      never re-downloaded.
    * A missing model is downloaded when `download` and a models bucket is
      configured; otherwise it is reported MISSING with the exact reason.
    """
    report = SyncReport()
    reqs = required_models(enabled_features, include_extraction=include_extraction)
    bucket_set = bool(C.MODELS_S3_BUCKET)
    client = s3_client
    if download and bucket_set and client is None:
        client = _make_s3_client()

    for req in reqs:
        local = req.existing_local()
        if local:
            report.statuses.append(ModelStatus(req=req, present=True,
                                               local_path=local))
            continue
        # An unpulled LFS pointer is NOT a missing object -- downloading over it
        # from a model bucket would mask a broken checkout, so say what it is.
        pointer = req.pointer_path()
        if pointer is not None:
            size = lfs_pointer_size(pointer)
            expected = f" (expects {size/1e6:.0f} MB)" if size else ""
            report.statuses.append(ModelStatus(
                req=req, present=False, local_path=pointer,
                error=(f"UNPULLED GIT-LFS POINTER{expected}, not real weights -- "
                       f"{_lfs_hint()}")))
            continue
        # missing locally
        if not download:
            report.statuses.append(ModelStatus(
                req=req, present=False,
                error="missing locally (sync disabled)"))
            continue
        if not bucket_set:
            report.statuses.append(ModelStatus(
                req=req, present=False,
                error=("missing locally and WAGONEYE_MODELS_S3_BUCKET is not set "
                       "-- either `git lfs pull` the weights or set the model "
                       "bucket for auto-sync")))
            continue
        if client is None:
            report.statuses.append(ModelStatus(
                req=req, present=False,
                error="missing locally and no S3 client/credentials available"))
            continue
        if req.ambiguous_in_flat_layout:
            report.statuses.append(ModelStatus(
                req=req, present=False,
                error=(f"{req.filename} exists in more than one model category "
                       f"with DIFFERENT weights, and the flat S3 layout cannot "
                       f"tell them apart -- refusing to auto-download it into "
                       f"{req.local_dir}.  Place it there explicitly (see "
                       f"models/extraction/README.md), or set "
                       f"WAGONEYE_MODELS_S3_LAYOUT=nested if your mirror has "
                       f"per-category folders.")))
            continue
        report.statuses.append(_download(client, req))

    return report


def ensure_models_or_report(
    *,
    enabled_features: Optional[List[str]] = None,
    s3_client=None,
    download: bool = True,
    include_extraction: Optional[bool] = None,
) -> SyncReport:
    """verify_and_sync + log a one-block summary.  Caller decides fail-fast."""
    report = verify_and_sync(enabled_features=enabled_features,
                             s3_client=s3_client, download=download,
                             include_extraction=include_extraction)
    header = ("[MODEL_SYNC] model availability "
              f"(bucket={C.MODELS_S3_BUCKET or '<unset>'}, "
              f"prefix={C.MODELS_S3_PREFIX or '<root>'}):")
    log.info("%s\n%s", header, "\n".join(report.summary_lines()))
    if not report.ok:
        log.error("[MODEL_SYNC] %d model(s) unavailable -- see MISSING lines above.",
                  len(report.missing))
    return report


# -----------------------------------------------------------------------------
# Operator tool: reconcile the configured model store against what a run needs
#
#     python -m core.model_sync
#
# `ensure_models_or_report` can only say a file is MISSING; it never says what IS
# in the bucket.  When the store's filenames differ from the ones this package
# resolves -- which is exactly the case between deployments (V4's own bucket
# holds `right_gap_1.pt`, `left_gap_det.pt`, `top_gap_2.pt`, `V4_side_damage.pt`,
# `ltop.pt`; this package resolves `right_up_wagon_gap.pt`, `top_gap.pt`,
# `damage.pt`, ...) -- a MISSING list alone leaves the operator guessing.  This
# lists the store, marks each required model present/absent, and suggests the
# closest unused object for each absent one so the mapping can be settled from
# real data instead of assumed.
#
# Read-only: it lists and compares.  It downloads nothing and writes nothing.
# -----------------------------------------------------------------------------

def list_store(s3_client=None, *, bucket: Optional[str] = None,
               prefix: Optional[str] = None) -> List[dict]:
    """Every object under the configured model prefix.  [] on any failure."""
    bucket = bucket or C.MODELS_S3_BUCKET
    prefix = (prefix if prefix is not None else C.MODELS_S3_PREFIX).strip("/")
    if s3_client is None:
        s3_client = _make_s3_client()
    if s3_client is None:
        log.error("[MODEL_SYNC] no S3 client (credentials?) -- cannot list "
                  "s3://%s/%s", bucket, prefix)
        return []
    out: List[dict] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket}
        if prefix:
            kwargs["Prefix"] = f"{prefix}/"
        if token:
            kwargs["ContinuationToken"] = token
        try:
            resp = s3_client.list_objects_v2(**kwargs)
        except Exception as e:  # noqa: BLE001
            log.error("[MODEL_SYNC] could not list s3://%s/%s: %s",
                      bucket, prefix, e)
            return out
        for item in resp.get("Contents", []):
            out.append({"key": item["Key"],
                        "name": item["Key"].rsplit("/", 1)[-1],
                        "size": int(item.get("Size") or 0)})
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
            if not token:
                break
        else:
            break
    return out


def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def reconcile_report(enabled_features: Optional[List[str]] = None,
                     s3_client=None, include_extraction: Optional[bool] = None
                     ) -> str:
    """Human-readable reconciliation of the store against a run's requirements."""
    objects = list_store(s3_client)
    by_name = {o["name"]: o for o in objects}
    reqs = required_models(enabled_features,
                           include_extraction=include_extraction)

    lines = [f"model store: s3://{C.MODELS_S3_BUCKET}/"
             f"{C.MODELS_S3_PREFIX + '/' if C.MODELS_S3_PREFIX else ''}"
             f"   ({len(objects)} object(s), layout={C.MODELS_S3_LAYOUT})"]
    if not objects:
        lines.append("  (nothing listed -- wrong bucket/prefix, or no credentials)")

    pts = [o for o in objects if o["name"].lower().endswith(".pt")]
    lines.append("")
    lines.append(f"objects in the store ({len(pts)} .pt):")
    for o in sorted(pts, key=lambda x: x["name"]):
        lines.append(f"    {o['name']:36s} {o['size'] / 1e6:8.1f} MB")

    lines.append("")
    lines.append("what this run needs:")
    matched: set = set()
    unresolved: List[ModelReq] = []
    for r in reqs:
        names = [r.filename] + ([C.FEATURE_MODEL_LEGACY[r.filename]]
                                if r.filename in C.FEATURE_MODEL_LEGACY else [])
        names += ([C.RECON_MODEL_LEGACY[r.filename]]
                  if r.filename in C.RECON_MODEL_LEGACY else [])
        hit = next((n for n in names if n in by_name), None)
        if hit:
            matched.add(hit)
            note = "" if hit == r.filename else f"  (via accepted alt name {hit})"
            lines.append(f"    [FOUND  ] {r.category}/{r.filename}{note}")
        else:
            unresolved.append(r)
            lines.append(f"    [ABSENT ] {r.category}/{r.filename}"
                         f"   tried: {', '.join(names)}")

    if unresolved:
        spare = [o["name"] for o in pts if o["name"] not in matched]
        lines.append("")
        lines.append("closest unused objects for each ABSENT model "
                     "(suggestion only -- verify before mapping):")
        for r in unresolved:
            ranked = sorted(spare, key=lambda n: _similarity(r.filename, n),
                            reverse=True)[:3]
            shown = ", ".join(f"{n} ({_similarity(r.filename, n):.0%})"
                              for n in ranked) or "none left"
            lines.append(f"    {r.filename:32s} -> {shown}")
        lines.append("")
        lines.append("If a store name is genuinely the same weights under a "
                     "different name, add it to core.constants "
                     "RECON_MODEL_LEGACY / FEATURE_MODEL_LEGACY -- do NOT rename "
                     "the object, and do NOT map a file you have not confirmed.")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - operator tool
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m core.model_sync",
        description="List the configured model store and reconcile it against "
                    "what a pipeline run requires.  Read-only.")
    ap.add_argument("--features", default="door,load,damage,ocr",
                    help="comma-separated enabled features (default: all)")
    ap.add_argument("--include-extraction", action="store_true",
                    help="also require the extraction classifiers (--source raw)")
    a = ap.parse_args()
    feats = [f.strip() for f in a.features.split(",") if f.strip()]
    print(reconcile_report(feats, include_extraction=a.include_extraction))
