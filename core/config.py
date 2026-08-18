"""Central configuration layer for the auto pipeline.

Single source of truth for every filesystem path and every runtime knob the
continuous (`--auto`) pipeline needs.  Every value is env-override-capable and
defaults to *exactly* the path/behaviour this package already used, so a
deployment that sets no environment variables behaves identically to a
hand-driven `--local-only` run.

Design rules:
    * No module hardcodes an absolute path -- import from here.
    * PROJECT_ROOT is discovered dynamically from this file's location, so the
      project works no matter where it is cloned on the host.
    * Nothing here loads a model, reads a frame, or touches GlobalTrainState.
      It is pure configuration (mirrors core/feature_config.py's discipline).

Ported from the V4-parity `global_train` repo, minus its incremental
batch-lifecycle knobs (per-camera arrival deadlines, interim-report policy,
late-camera policy).  Those belong to a lifecycle runner this package does not
have: batch assembly here is `orchestrator.train_batch_manager` +
`--partial-wait`, so inventing settings for a scheduler that does not exist
would be configuration with nothing reading it.

Environment variables (all optional):
    WAGONEYE_WORKSPACE_ROOT         output root (default <root>/batch_outputs)
    WAGONEYE_MODELS_DIR             models root (default <root>/models)
    WAGONEYE_RECON_MODELS_DIR       reconstruction .pt dir
    WAGONEYE_FEAT_MODELS_DIR        feature .pt dir
    WAGONEYE_EXTRACTION_MODELS_DIR  extraction classify .pt dir (--source raw)
    WAGONEYE_LOCAL_INPUTS_DIR       default --local-inputs folder
    WAGONEYE_LOG_DIR                log directory (default <root>/logs)
    WAGONEYE_LOG_LEVEL              root log level (default INFO)
    WAGONEYE_DEVICE                 force 'cuda' / 'cpu' (default: auto-detect)
    WAGONEYE_PIPELINE_SOURCE        'trimmed' (default) | 'raw'
    WAGONEYE_EXTRACTION_POLL_INTERVAL   raw->trimmed sweep cadence (seconds)
    WAGONEYE_OCR_ENGINE             'rekognition' (default) | 'easyocr'
    WAGONEYE_PROCESSOR_START_UTC    one-time backlog-skip anchor (ISO 8601)
"""

from __future__ import annotations

import os
from datetime import datetime as _datetime, timedelta as _timedelta, timezone as _timezone

# -----------------------------------------------------------------------------
# Project root -- discovered from this file, never hardcoded.
# core/config.py  ->  <PROJECT_ROOT>/core/config.py, so PROJECT_ROOT is two
# levels up.  Works regardless of where the repo is cloned.
# -----------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_path(var: str, default: str) -> str:
    """Return an absolute path from an env var, falling back to `default`.

    A relative env value is resolved against PROJECT_ROOT so the project still
    works no matter what the process working directory is.
    """
    raw = os.getenv(var)
    if not raw:
        return default
    return raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)


def _env_str(var: str, default: str) -> str:
    val = os.getenv(var)
    return val if val else default


def _env_float(var: str, default: float) -> float:
    raw = os.getenv(var)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(var: str, default: bool) -> bool:
    raw = os.getenv(var)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# -----------------------------------------------------------------------------
# Filesystem paths (all overridable; all default to this package's own layout)
# -----------------------------------------------------------------------------

MODELS_DIR       = _env_path("WAGONEYE_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
RECON_MODELS_DIR = _env_path("WAGONEYE_RECON_MODELS_DIR",
                             os.path.join(MODELS_DIR, "reconstruction"))
FEAT_MODELS_DIR  = _env_path("WAGONEYE_FEAT_MODELS_DIR",
                             os.path.join(MODELS_DIR, "features"))

# EXTRACTION classify models (empty_track / wagon / engine / second_track) used
# by the raw->trimmed producer.  Deliberately a SEPARATE tree from
# RECON_MODELS_DIR: `side_classification.pt` and `top_classification.pt` exist in
# both and are DIFFERENT weights (extraction classifier vs Stage-1 segment
# classifier), so a shared dir would silently collide.  Kept in sync with
# train_extraction.driver's default.
EXTRACTION_MODELS_DIR = _env_path("WAGONEYE_EXTRACTION_MODELS_DIR",
                                  os.path.join(MODELS_DIR, "extraction"))

WORKSPACE_ROOT   = _env_path("WAGONEYE_WORKSPACE_ROOT",
                             os.path.join(PROJECT_ROOT, "batch_outputs"))
LOCAL_INPUTS_DIR = _env_path("WAGONEYE_LOCAL_INPUTS_DIR",
                             os.path.join(PROJECT_ROOT, "local_inputs"))
LOG_DIR          = _env_path("WAGONEYE_LOG_DIR", os.path.join(PROJECT_ROOT, "logs"))

# Per-page report logo.
LOGO_PATH = os.path.join(PROJECT_ROOT, "reporting", "assets", "Logo.jpeg")

LOG_LEVEL = _env_str("WAGONEYE_LOG_LEVEL", "INFO")


# -----------------------------------------------------------------------------
# Per-batch output subfolder names (were inline string literals in
# orchestrator/master_runner.py; centralized so renaming is a one-line edit).
# -----------------------------------------------------------------------------

DIR_DOWNLOADS        = "downloads"
DIR_GLOBAL_STATE     = "global_state"
DIR_WAGON_CACHE      = "wagon_cache"
DIR_WAGON_STATES     = "wagon_states"
DIR_EVIDENCE         = "evidence"
DIR_PROCESSED_VIDEOS = "processed_videos"
DIR_REPORTS          = "reports"
DIR_ARCHIVE          = "archive"
DIR_DELIVERY         = "delivery"

BATCH_SUBDIRS = (
    DIR_DOWNLOADS, DIR_GLOBAL_STATE, DIR_WAGON_CACHE, DIR_WAGON_STATES,
    DIR_EVIDENCE, DIR_PROCESSED_VIDEOS, DIR_REPORTS, DIR_ARCHIVE,
)


# -----------------------------------------------------------------------------
# Device resolution (CPU / CUDA) -- centralized so every model load and every
# inference call selects the same device deterministically.
# -----------------------------------------------------------------------------

def resolve_device(force: str | None = None) -> str:
    """Return 'cuda' or 'cpu'.

    Precedence:
        1. `force` argument (explicit caller override).
        2. WAGONEYE_DEVICE env var ('cuda' / 'cpu' / 'auto').
        3. Auto-detect via torch.cuda.is_available().

    Any torch import/detection failure degrades safely to 'cpu' so the pipeline
    never crashes on a box without a working CUDA stack.
    """
    choice = (force or os.getenv("WAGONEYE_DEVICE") or "auto").strip().lower()
    if choice in ("cuda", "gpu"):
        return "cuda"
    if choice == "cpu":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def use_half_precision(device: str | None = None) -> bool:
    """FP16 inference is only safe/beneficial on CUDA."""
    dev = device if device is not None else resolve_device()
    return dev == "cuda"


# -----------------------------------------------------------------------------
# Pipeline source -- WHAT the orchestrator consumes (see core.pipeline_source).
#
#   trimmed (default) : the input prefixes already hold trimmed train clips;
#                       the orchestrator is a pure consumer (two-service
#                       topology or manual uploads).
#   raw               : only raw CCTV exists; the orchestrator owns an
#                       ExtractionManager that discovers raw video, detects
#                       train completion, runs the extractor, and produces the
#                       trimmed clips before consuming them (single service).
#
# This describes the source of the input, not the mechanism -- threads/services
# are the orchestrator's concern.  Override with WAGONEYE_PIPELINE_SOURCE.
# -----------------------------------------------------------------------------

from core.pipeline_source import PipelineSource   # noqa: E402

PIPELINE_SOURCE = PipelineSource.resolve()

# Poll cadence (seconds) for the ExtractionManager's raw->trimmed sweeps (and
# the standalone extraction service).  Only consulted when the source is 'raw'.
EXTRACTION_POLL_INTERVAL = int(_env_float("WAGONEYE_EXTRACTION_POLL_INTERVAL", 60))


# -----------------------------------------------------------------------------
# OCR engine -- 'rekognition' (default, AWS DetectText; V4 parity) | 'easyocr'
# (local, no network).  Resolved here so the startup summary + validation can
# report it; features/ocr/processor.resolve_engine() is the runtime authority
# and reads the same variable.
# -----------------------------------------------------------------------------

OCR_ENGINE = _env_str("WAGONEYE_OCR_ENGINE", "rekognition").strip().lower()
if OCR_ENGINE not in ("rekognition", "easyocr"):
    OCR_ENGINE = "rekognition"


# -----------------------------------------------------------------------------
# OPERATIONAL-DAY DISCOVERY ANCHOR  (the V4 production rule, adopted verbatim)
#
# Everything the pipeline discovers -- raw clips and trimmed clips -- is bounded
# by the START OF THE CURRENT OPERATIONAL DAY: 05:00 IST, rolling back a day
# when the clock is before 05:00.
#
# Why an anchor beats a sliding "last N minutes" window:
#   * A restart at any hour still sees the whole operational day, so stopping
#     overnight and starting at 05:30 loses nothing -- a sliding 10-minute
#     window would skip every train uploaded while the service was down.
#   * It is inherently bounded to ONE day, so it can never reach back into
#     months of archive and queue thousands of batches.
#   * It matches the 05:00 boundary the dashboard already uses for its date
#     folders (delivery.dashboard_ingest.date_folder), so a train and its report
#     always agree about which day they belong to.
#
# WAGONEYE_PROCESSOR_START_UTC (ISO 8601) raises the anchor for a one-time
# backlog skip -- never below the 05:00 anchor, and it self-expires at the next
# day's anchor.
# -----------------------------------------------------------------------------

IST = _timezone(_timedelta(hours=5, minutes=30))

OPERATIONAL_DAY_START_HOUR_IST = int(
    _env_float("WAGONEYE_OPERATIONAL_DAY_START_HOUR_IST", 5))


def operational_day_start_utc(now=None):
    """UTC datetime of the current operational day's start (05:00 IST default)."""
    now_utc = now or _datetime.now(_timezone.utc)
    if getattr(now_utc, "tzinfo", None) is None:
        now_utc = now_utc.replace(tzinfo=_timezone.utc)
    now_ist = now_utc.astimezone(IST)
    start_ist = now_ist.replace(hour=OPERATIONAL_DAY_START_HOUR_IST,
                                minute=0, second=0, microsecond=0)
    if now_ist.hour < OPERATIONAL_DAY_START_HOUR_IST:
        start_ist = start_ist - _timedelta(days=1)
    return start_ist.astimezone(_timezone.utc)


def discovery_cutoff_utc(now=None):
    """The effective "ignore anything older than this" instant for discovery.

    The operational-day anchor, raised by WAGONEYE_PROCESSOR_START_UTC when that
    is set and later.  Returns a tz-aware UTC datetime.
    """
    anchor = operational_day_start_utc(now)
    raw = os.getenv("WAGONEYE_PROCESSOR_START_UTC")
    if raw:
        try:
            ov = _datetime.fromisoformat(raw.strip())
            if ov.tzinfo is None:
                ov = ov.replace(tzinfo=_timezone.utc)
            return max(anchor, ov.astimezone(_timezone.utc))
        except ValueError:
            pass
    return anchor


# -----------------------------------------------------------------------------
# Startup configuration validation + redacted summary
# -----------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised (collected) when the effective configuration is invalid."""


def validate_config(*, mode: str, skip_upload: bool = False,
                    skip_email: bool = False, source=None) -> list:
    """Return a list of human-readable configuration errors (empty = OK).

    `mode` is 'auto' | 'local' | 'once' | 'batch' | 'sequential' | 'historical'.
    The caller fails fast and refuses to poll when this is non-empty.

    `source` is the RESOLVED pipeline source for this run.  It must be passed
    explicitly whenever the caller honoured a `--source` flag: PIPELINE_SOURCE
    below is resolved once at import from the environment, so a CLI flag that
    only lives in the caller's local variable would silently skip the
    extraction-model checks -- validating a topology the run is not using.
    """
    from core import constants as C
    src = source if source is not None else PIPELINE_SOURCE
    errors: list = []

    if EXTRACTION_POLL_INTERVAL <= 0 and src.requires_extraction:
        errors.append("WAGONEYE_EXTRACTION_POLL_INTERVAL must be > 0")

    # Historical mode reads the SAME input bucket/prefixes as the live consumer,
    # so discovery config is required -- but delivery is off by default, so the
    # email endpoint is not.  A separate branch keeps the live modes' checks
    # byte-identical rather than widening their condition.
    if mode == "historical":
        if not C.S3_INPUT_PREFIXES:
            errors.append("WAGONEYE_S3_INPUT_PREFIXES is empty -- historical "
                          "discovery would find nothing.  Set it to the "
                          "camera-video prefix(es).")
        if not C.S3_INPUT_BUCKET:
            errors.append("WAGONEYE_S3_INPUT_BUCKET is required for --historical")

    # S3 discovery for continuous polling
    if mode in ("auto", "once", "batch"):
        if not C.S3_OUTPUT_BUCKET:
            errors.append("WAGONEYE_S3_OUTPUT_BUCKET is required for "
                          "--auto/--once/--batch")
        if not C.S3_INPUT_PREFIXES:
            errors.append("WAGONEYE_S3_INPUT_PREFIXES is empty -- --auto would "
                          "discover nothing.  Set it to the camera-video "
                          "prefix(es).")
        if not skip_email and (not C.EMAIL_API_URL or not C.EMAIL_RECEIVER):
            errors.append("email enabled but EMAIL_API_URL / EMAIL_RECEIVER "
                          "missing (or pass --skip-email)")

    # ---- pipeline source = raw: this process produces its own trimmed clips --
    # Fail fast here instead of letting every per-camera sweep raise
    # FileNotFoundError once a minute for the life of the service.
    if src.requires_extraction:
        if not os.path.isdir(EXTRACTION_MODELS_DIR):
            errors.append(
                f"PIPELINE_SOURCE=raw but the extraction models dir does not "
                f"exist: {EXTRACTION_MODELS_DIR} (set "
                f"WAGONEYE_EXTRACTION_MODELS_DIR, or use --source trimmed)")
        else:
            missing = [f for f in C.EXTRACTION_MODEL_FILES
                       if not os.path.isfile(
                           os.path.join(EXTRACTION_MODELS_DIR, f))]
            if missing:
                errors.append(
                    f"PIPELINE_SOURCE=raw but extraction classify model(s) "
                    f"missing from {EXTRACTION_MODELS_DIR}: "
                    f"{', '.join(missing)}. These are the EXTRACTION "
                    f"classifiers (empty_track/wagon/engine), NOT the Stage-1 "
                    f"reconstruction models.")

    # ---- OCR engine ----
    # Rekognition is the default engine; it needs boto3 + a region.  A missing
    # dependency is reported at startup rather than degrading silently per wagon.
    if mode in ("auto", "once", "batch") and OCR_ENGINE == "rekognition":
        try:
            import boto3  # noqa: F401
        except ImportError:
            errors.append(
                "WAGONEYE_OCR_ENGINE=rekognition (default) requires boto3 "
                "(pip install boto3), or set WAGONEYE_OCR_ENGINE=easyocr")

    # writable dirs
    import tempfile as _tf
    for name, d in (("WORKSPACE_ROOT", WORKSPACE_ROOT), ("LOG_DIR", LOG_DIR),
                    ("TMPDIR", _tf.gettempdir())):
        try:
            os.makedirs(d, exist_ok=True)
            if not os.access(d, os.W_OK):
                errors.append(f"{name} is not writable: {d}")
        except OSError as e:
            errors.append(f"{name} could not be created ({d}): {e}")
    return errors


def startup_summary(*, mode: str, source=None) -> str:
    """A single multi-line summary of the effective settings.
    Secrets and recipient addresses are REDACTED (counts only)."""
    from core import constants as C
    from delivery import dashboard_ingest as DASH
    src = source if source is not None else PIPELINE_SOURCE
    n_to = len(C.EMAIL_RECEIVER or [])
    n_cc = len(C.EMAIL_RECEIVER_CC or [])
    lines = [
        "WagonEye v4 effective configuration:",
        f"  mode                     : {mode}",
        f"  device                   : {resolve_device()}",
        f"  workspace                : {WORKSPACE_ROOT}",
        f"  log_dir                  : {LOG_DIR}",
        f"  pipeline_source          : {src.value}",
        f"  extraction_models_dir    : {EXTRACTION_MODELS_DIR}"
        + ("" if src.requires_extraction else "  (unused: source=trimmed)"),
        f"  extraction_poll_s        : {EXTRACTION_POLL_INTERVAL}",
        f"  ocr_engine               : {OCR_ENGINE}",
        f"  s3_raw_video_bucket      : {C.S3_RAW_VIDEO_BUCKET}",
        f"  s3_trimmed_video_bucket  : {C.S3_TRIMMED_VIDEO_BUCKET}",
        f"  s3_input_bucket          : {C.S3_INPUT_BUCKET}",
        f"  s3_input_prefixes        : {len(C.S3_INPUT_PREFIXES)} configured",
        f"  s3_output_bucket         : {C.S3_OUTPUT_BUCKET}",
        f"  s3_train_batch_prefix    : {C.S3_TRAIN_BATCH_PREFIX}",
        f"  dashboard_ingest         : {DASH.is_enabled()}",
        f"  dashboard_version        : {C.INSPECTION_VERSION}",
        f"  dashboard_ingest_urls    : {len(DASH.ingest_api_urls())} endpoint(s)",
        f"  inspection_json_bucket   : {DASH.inspection_bucket()}",
        f"  email_recipients         : to={n_to} cc={n_cc} (redacted)",
        f"  discovery_cutoff_utc     : {discovery_cutoff_utc().isoformat()}",
    ]
    return "\n".join(lines)
