"""Canonical constants shared across the wagon_eye_v4 pipeline."""

from __future__ import annotations

import os


def _env(name: str, default: str) -> str:
    """Read a WAGONEYE_* override, falling back to the built-in default."""
    val = os.getenv(name)
    return val if val else default


def _env_opt(name: str, default: str) -> str:
    """Like `_env`, but an EXPLICITLY EMPTY value is honoured, not replaced.

    `_env` folds "" into the default, which is correct for a bucket or a URL --
    an empty one is never meaningful.  It is wrong for an optional PREFIX, where
    "" legitimately means "objects sit at the bucket root".  Without this,
    `WAGONEYE_MODELS_S3_PREFIX=` silently kept the built-in prefix, so a
    deployment whose weights are at the root had no way to say so.
    """
    val = os.getenv(name)
    return default if val is None else val


def _env_list(name: str, default: list) -> list:
    """Comma/semicolon separated env override for a list-valued constant."""
    raw = os.getenv(name)
    if not raw:
        return default
    return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]

# -----------------------------------------------------------------------------
# Cameras
# -----------------------------------------------------------------------------

CAMERA_RIGHT_UP     = "RIGHT_UP"
CAMERA_LEFT_UP      = "LEFT_UP"
CAMERA_RIGHT_UP_TOP = "RIGHT_UP_TOP"
CAMERA_LEFT_UP_TOP  = "LEFT_UP_TOP"

ALL_CAMERAS = (
    CAMERA_RIGHT_UP, CAMERA_LEFT_UP,
    CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP,
)
SIDE_CAMERAS = (CAMERA_RIGHT_UP, CAMERA_LEFT_UP)
TOP_CAMERAS  = (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP)
MASTER_CAMERA = CAMERA_RIGHT_UP

# Canonical lowercase cache folder name per camera
CAMERA_FOLDER = {
    CAMERA_RIGHT_UP:     "right_up",
    CAMERA_LEFT_UP:      "left_up",
    CAMERA_RIGHT_UP_TOP: "right_up_top",
    CAMERA_LEFT_UP_TOP:  "left_up_top",
}

# Reverse lookup
CAMERA_FROM_FOLDER = {v: k for k, v in CAMERA_FOLDER.items()}


# -----------------------------------------------------------------------------
# Status sentinel values
# -----------------------------------------------------------------------------

NO_DATA       = "NO_DATA"
STATUS_OK     = "OK"
STATUS_FAILED = "FAILED"
STATUS_NO_FRAMES = "NO_FRAMES"
STATUS_DISABLED  = "DISABLED_BY_USER"   # feature-JSON status when a user toggled it OFF

# Display string carried in UnifiedWagonState fields owned by a disabled
# feature, and rendered verbatim in reports in place of NO_DATA / OK.
DISABLED_DISPLAY = "DISABLED BY USER"

# Batch outcome statuses persisted in processed_batches.json
BATCH_COMPLETED          = "completed"
BATCH_COMPLETED_PARTIAL  = "completed_partial"
BATCH_REPORT_FAILED      = "report_failed"
BATCH_FAILED_NO_GLOBAL   = "failed_no_global_state"
BATCH_FAILED             = "failed"


# -----------------------------------------------------------------------------
# Classification labels (matching wagon_count.global_train_state.SegmentClass)
# -----------------------------------------------------------------------------

CLASS_ENGINE    = "ENGINE"
CLASS_WAGON     = "WAGON"
CLASS_BRAKE_VAN = "BRAKE_VAN"
CLASS_UNKNOWN   = "UNKNOWN"


# -----------------------------------------------------------------------------
# Reconstruction model filenames (in models/reconstruction/)
# -----------------------------------------------------------------------------

# Short names (preferred); the wagon_count package now also accepts these.
# DECLARATIVE ONLY -- documentation of the Stage-1 contract for operators.
# The counting engine (wagon_count/run_global_count.py) resolves these names
# itself under --recon-models-dir; nothing in wagon_eye_v4 reads the constants
# below to load a model.  They are kept in sync with the engine so the expected
# filenames are discoverable from one place.
MODEL_RIGHT_UP_GAP        = "right_up_wagon_gap.pt"
MODEL_LEFT_UP_GAP         = "left_up_wagon_gap.pt"
MODEL_TOP_GAP             = "top_gap.pt"
MODEL_SIDE_CLASSIFICATION = "side_classification.pt"

# OPTIONAL.  Classifies the two TOP cameras so engine / brake-van regions stay
# out of wagon synchronization.  Never a counting authority -- RIGHT_UP alone
# decides the count -- so a missing file degrades capability, never the count.
MODEL_TOP_CLASSIFICATION  = "top_classification.pt"


# -----------------------------------------------------------------------------
# Feature model filenames (in models/features/)
# -----------------------------------------------------------------------------

MODEL_DOOR_STATE        = "door_state.pt"
MODEL_LOADED            = "loaded.pt"
MODEL_DAMAGE            = "damage.pt"
MODEL_WAGON_ID_COUNTING = "wagon_id_counting.pt"

# Wagon-number PLATE DETECTOR for the RIGHT_UP camera (OCR authority).
# `wagon_number_update.pt` is the model the V4 Train-Inspection-Engine's right_up
# pipeline uses (configs/cameras/right_up.yaml:
#   wagon_number_model_path: s3://wagon-eye-models/wagon_number_update.pt)
# and is therefore the canonical detector feeding Rekognition OCR here.
# `wagon_id_counting.pt` is the older name and is accepted as a fallback so an
# existing checkout keeps running without a model swap.
MODEL_WAGON_NUMBER      = "wagon_number_update.pt"


# -----------------------------------------------------------------------------
# Model inventory + S3 sync configuration (see core/model_sync.py).
#
# Every reconstruction model is ALWAYS required.  Feature models are required
# only for the ENABLED features of a run.  When a required model is missing on
# disk it is fetched from s3://<MODELS_S3_BUCKET>/<MODELS_S3_PREFIX>/<file>.
# The layout stays "flat" (no per-category folder): this store, like V4's, keeps
# the .pt files side by side, just under a prefix.
# -----------------------------------------------------------------------------

# The operator-designated model store (2026-08-19):
#     s3://complete-train/new_local/<file>      (ap-south-1)
# V4's own bucket was `wagon-eye-models` with everything at the ROOT; this
# deployment keeps its weights under a prefix in a different bucket instead.
# Both are reachable without a code edit:
#     WAGONEYE_MODELS_S3_BUCKET=wagon-eye-models WAGONEYE_MODELS_S3_PREFIX=
# restores the V4 location exactly.
MODELS_S3_BUCKET = _env("WAGONEYE_MODELS_S3_BUCKET", "complete-train")
MODELS_S3_PREFIX = _env_opt("WAGONEYE_MODELS_S3_PREFIX", "new_local").strip("/")

# "flat"   -> s3://<bucket>/<prefix>/<file>            (V4 layout, default)
# "nested" -> s3://<bucket>/<prefix>/<category>/<file> (reconstruction/features/
#                                                       extraction subfolders)
MODELS_S3_LAYOUT = _env("WAGONEYE_MODELS_S3_LAYOUT", "flat").strip().lower()
if MODELS_S3_LAYOUT not in ("flat", "nested"):
    MODELS_S3_LAYOUT = "flat"

# Reconstruction models: always required by Stage 1.
RECON_MODEL_FILES = (
    MODEL_RIGHT_UP_GAP, MODEL_LEFT_UP_GAP, MODEL_TOP_GAP, MODEL_SIDE_CLASSIFICATION,
)

# Stage 1 uses top_classification.pt when present and skips top-camera semantic
# labelling gracefully when absent, so it is optional -- but it still LIVES in
# models/reconstruction/, which is what makes its filename collide with the
# extraction tree below.
RECON_OPTIONAL_MODEL_FILES = (MODEL_TOP_CLASSIFICATION,)

# EXTRACTION classify models -- required ONLY when the pipeline source is `raw`
# (the orchestrator cuts its own trimmed clips).  Same filenames as two of the
# reconstruction models but DIFFERENT weights (the extraction classifier emits
# empty_track / wagon / engine / second_track), which is why they live in their
# own directory:
#   side_classification.pt -> RIGHT_UP, LEFT_UP
#   top_classification.pt  -> RIGHT_UP_TOP, LEFT_UP_TOP
EXTRACTION_MODEL_FILES = (
    MODEL_SIDE_CLASSIFICATION, MODEL_TOP_CLASSIFICATION,
)

# Filenames that exist in MORE THAN ONE model tree with DIFFERENT weights.
# Anything that resolves a model by filename alone must treat these as ambiguous
# and never auto-download them (see core.model_sync).
AMBIGUOUS_MODEL_FILENAMES = frozenset(
    set(RECON_MODEL_FILES) | set(RECON_OPTIONAL_MODEL_FILES)
) & frozenset(EXTRACTION_MODEL_FILES)

# Accepted alternative filename per reconstruction model.  This repo's counting
# engine resolves the long `*_wagon_gap.pt` names; the V4 engine ships the short
# ones, so either satisfies the requirement.
RECON_MODEL_LEGACY = {
    MODEL_RIGHT_UP_GAP: "right_up_gap.pt",
    MODEL_LEFT_UP_GAP:  "left_up_gap.pt",
}

# Feature key -> feature model filename (required only when the feature is on).
FEATURE_MODEL_BY_KEY = {
    "door":   MODEL_DOOR_STATE,
    "load":   MODEL_LOADED,
    "damage": MODEL_DAMAGE,
    "ocr":    MODEL_WAGON_NUMBER,
}

# Accepted legacy filename per feature model.  A checkout that only has the old
# name still satisfies the requirement, so renaming a model never silently
# invalidates a batch.
# Names verified present in the operator's store, s3://complete-train/new_local/
# (listed 2026-08-19).  `load.pt` and `top_damage.pt` are that store's names for
# the two feature models this package calls `loaded.pt` and `damage.pt`:
#   * load.pt        -> the LOADED/EMPTY classifier (features/load).
#   * top_damage.pt  -> the TOP-camera damage detector.  This package's damage
#                       processor is top-cameras-only (DAMAGE_CLASSES_TOP =
#                       floor_damage / inner_wall_damage), so the top model is
#                       the correct weights for the `damage` slot -- the V4
#                       engine's separate side-damage model has no consumer here.
FEATURE_MODEL_LEGACY = {
    MODEL_WAGON_NUMBER: MODEL_WAGON_ID_COUNTING,
    MODEL_LOADED:       "load.pt",
    MODEL_DAMAGE:       "top_damage.pt",
}


def feature_model_path(models_dir: str, filename: str) -> str:
    """Resolve a feature model to the path that will actually be loaded.

    Returns the canonical path when it exists, else the accepted legacy path
    when THAT exists, else the canonical path (so the caller reports the
    canonical name as missing).  Every consumer -- the processors and
    `core.model_sync` -- resolves through here so they all agree on which file a
    run actually used.
    """
    canonical = os.path.join(models_dir, filename)
    if os.path.isfile(canonical):
        return canonical
    legacy_name = FEATURE_MODEL_LEGACY.get(filename)
    if legacy_name:
        legacy = os.path.join(models_dir, legacy_name)
        if os.path.isfile(legacy):
            return legacy
    return canonical


# -----------------------------------------------------------------------------
# Door state vocabulary (from the trained door_state.pt model)
# -----------------------------------------------------------------------------

DOOR_CLOSED  = "CLOSED"
DOOR_OPEN    = "OPEN"
DOOR_PARTIAL = "PARTIAL"
DOOR_DAMAGED = "DAMAGED"

# Map raw YOLO class names to canonical door states. Anything not in the
# dict is preserved verbatim (uppercased) so downstream can still see it.
DOOR_LABEL_TO_STATE = {
    "open":               DOOR_OPEN,
    "open_door":          DOOR_OPEN,
    "closed":             DOOR_CLOSED,
    "closed_door":        DOOR_CLOSED,
    "closed_with_wire":   DOOR_PARTIAL,
    "partial_closed":     DOOR_PARTIAL,
    "partially_closed":   DOOR_PARTIAL,
    "partial":            DOOR_PARTIAL,
    "damage":             DOOR_DAMAGED,
}


# -----------------------------------------------------------------------------
# Load status vocabulary
# -----------------------------------------------------------------------------

LOAD_LOADED = "LOADED"
LOAD_EMPTY  = "EMPTY"

LOAD_LABEL_TO_STATE = {
    "loaded": LOAD_LOADED,
    "load":   LOAD_LOADED,
    "full":   LOAD_LOADED,
    "empty":  LOAD_EMPTY,
    "unload": LOAD_EMPTY,
}


# -----------------------------------------------------------------------------
# Damage vocabulary (top cameras)
# -----------------------------------------------------------------------------

DAMAGE_PRESENT = "DAMAGE"
DAMAGE_OK      = "OK"

# Top-camera damage classes we COUNT as damage.  Outer-wall damage is
# skipped on top cameras because it is the side cameras' responsibility.
DAMAGE_CLASSES_TOP = {"floor_damage", "inner_wall_damage"}
DAMAGE_CLASSES_NEGATIVE = {"no_damage"}

# PROBABLE (not confirmed) top damage.  V4 reports these separately as
# `probable_damage_wagons` / `floor_dmg_probable_wagons` and must NOT count them
# as confirmed damage.  The double underscore in `floor__probable_damage` is the
# trained model's real class name, not a typo -- matching it exactly is what
# makes probable damage reportable instead of silently unmapped.
DAMAGE_CLASSES_PROBABLE = {"floor__probable_damage", "floor_probable_damage",
                           "floor_dmg_probable"}


def is_probable_damage(class_name: str) -> bool:
    """True for a PROBABLE (not confirmed) top-damage class."""
    return str(class_name or "").strip().lower() in DAMAGE_CLASSES_PROBABLE


# -----------------------------------------------------------------------------
# S3 + email -- preserved from the legacy master_runner constants so the
# new package can drop in without operational changes.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# THE V4 BUCKET SET.
#
# These defaults are the SAME buckets the V4 Train-Inspection-Engine uses
# (`Train-Inspection-Engine/configs/cameras/*.yaml` + `configs/combiner.yaml`),
# so this package drops into the existing production topology with no env file:
#
#   raw_video_bucket         biro-wagon-raw-video-copy/<camera_folder>
#   trimmed_video_bucket     biro-wagon-pre-processed-video-copy/<camera_folder>
#   detected_video_bucket    biro-wagon-processed-video-copy/<camera_folder>
#   inspection_output_bucket biro-wagon-report-biro-copy/<camera_folder>
#   combined_output_bucket   biro-combined-report-copy
#   models                   s3://wagon-eye-models/ (flat, at the bucket root)
#   region                   ap-south-1
#
# Per-camera folder names come from CAMERA_S3_FOLDER below.  Every value is
# overridable via a WAGONEYE_* environment variable (same default) so a staging
# deployment needs no source edit.
# -----------------------------------------------------------------------------

S3_REGION = _env("WAGONEYE_S3_REGION", "ap-south-1")

# Canonical per-camera S3 folder (== the V4 `camera_id`).  SINGLE source of
# truth: the extraction driver, the report layout and the dashboard feed all
# resolve through here, so a rig rename is a one-line edit.
CAMERA_S3_FOLDER = {
    CAMERA_RIGHT_UP:     "camera_CCTV_HZBN_DHN_2_RIGHT_UP",
    CAMERA_LEFT_UP:      "camera_CCTV_HZBN_DHN_1_LEFT_UP",
    CAMERA_RIGHT_UP_TOP: "camera_CCTV_HZBN_DHN_5_RIGHT_TOP",
    CAMERA_LEFT_UP_TOP:  "camera_CCTV_HZBN_DHN_6_LEFT_TOP",
}

#: Reverse lookup: S3 folder -> camera id.  The folder is AUTHORITATIVE, because
#: the rig writes it; a filename is whatever the uploader felt like.
S3_FOLDER_TO_CAMERA = {v: k for k, v in CAMERA_S3_FOLDER.items()}

#: Filename tokens that identify a camera, for keys whose folder is unknown.
#:
#: The site names its TOP rigs `RIGHT_TOP` / `LEFT_TOP` (see CAMERA_S3_FOLDER
#: above) but the canonical camera ids are RIGHT_UP_TOP / LEFT_UP_TOP.  Matching
#: only on the canonical id silently fails for both top cameras: their clips
#: resolve to no camera at all, get dropped at discovery, and every batch forms
#: with just the two side cameras -- no top load classification, no top damage,
#: and reports missing half the rig.
#:
#: Order matters at match time: LONGEST token first, so a `..._RIGHT_UP_TOP_...`
#: name is not claimed by the shorter `right_up`.
CAMERA_FILENAME_TOKENS = {
    "right_up_top": CAMERA_RIGHT_UP_TOP,
    "left_up_top":  CAMERA_LEFT_UP_TOP,
    "right_top":    CAMERA_RIGHT_UP_TOP,
    "left_top":     CAMERA_LEFT_UP_TOP,
    "right_up":     CAMERA_RIGHT_UP,
    "left_up":      CAMERA_LEFT_UP,
}


def camera_from_key(key: str):
    """Resolve a camera id from an S3 key (or a bare filename).  None if unknown.

    Folder first (the rig writes it), then filename tokens.  Shared by S3
    discovery and the local-directory scan so the two can never disagree.
    """
    if not key:
        return None
    k = key.replace("\\", "/")
    for folder, cam in S3_FOLDER_TO_CAMERA.items():
        if f"/{folder}/" in f"/{k}" or k.startswith(f"{folder}/"):
            return cam
    base = k.rsplit("/", 1)[-1].lower()
    for token in sorted(CAMERA_FILENAME_TOKENS, key=len, reverse=True):
        if token in base:
            return CAMERA_FILENAME_TOKENS[token]
    return None


# Reports / evidence / archive (V4: inspection_output_bucket).
S3_OUTPUT_BUCKET = _env("WAGONEYE_S3_OUTPUT_BUCKET", "biro-wagon-report-biro-copy")
S3_TRAIN_BATCH_PREFIX = _env("WAGONEYE_S3_TRAIN_BATCH_PREFIX", "train_batch")
S3_STATE_KEY = _env("WAGONEYE_S3_STATE_KEY", "master_runner/processed_batches.json")

# Raw CCTV (V4: raw_video_bucket).  Consumed only when the pipeline source is
# `raw`; the extraction driver applies the per-camera folder and honours
# WAGONEYE_EXTRACTION_<CAM>_RAW_BUCKET overrides.
S3_RAW_VIDEO_BUCKET = _env("WAGONEYE_S3_RAW_VIDEO_BUCKET",
                           "biro-wagon-raw-video-copy")

# Trimmed per-camera train clips (V4: trimmed_video_bucket).  This is BOTH what
# extraction produces and what `--auto` consumes -- the two halves meet here.
# Defaulting the consumer to the trimmed bucket (not the report bucket) is what
# makes `--auto` discover videos with no env file at all.
S3_TRIMMED_VIDEO_BUCKET = _env("WAGONEYE_S3_TRIMMED_VIDEO_BUCKET",
                               "biro-wagon-pre-processed-video-copy")
S3_INPUT_BUCKET = _env("WAGONEYE_S3_INPUT_BUCKET", S3_TRIMMED_VIDEO_BUCKET)

# Prefixes the poller scans for source videos -- the four camera folders,
# matching the trimmed-bucket layout the extractor writes.
S3_INPUT_PREFIXES = _env_list(
    "WAGONEYE_S3_INPUT_PREFIXES",
    [CAMERA_S3_FOLDER[c] for c in ALL_CAMERAS],
)

# Annotated / overlay videos (V4: detected_video_bucket).  Mirrored per camera
# alongside the train_batch archive copy; empty disables the mirror.
S3_DETECTED_VIDEO_BUCKET = _env("WAGONEYE_S3_DETECTED_VIDEO_BUCKET",
                                "biro-wagon-processed-video-copy")

# Combined 4-camera report (V4 combiner.yaml: combined_output_bucket).
S3_COMBINED_REPORT_BUCKET = _env("WAGONEYE_S3_COMBINED_REPORT_BUCKET",
                                 "biro-combined-report-copy")

# -----------------------------------------------------------------------------
# THE V4 API SET -- the endpoints the V4 Train-Inspection-Engine actually calls.
#
# Source of truth is V4's COMMITTED `Train-Inspection-Engine/configs/config.json`
# (added by its commit "Match notebook artifact + JSON contract; fix flush-emit +
# endpoint URLs"), which OVERRIDES the older hard-coded defaults still sitting in
# V4's `core/config.py` dataclass.  Those stale defaults point at
# `cctv-wagon-api.suvidhaen.com`; the live config does not.  Copying the
# dataclass instead of the config file silently posts to a host the dashboard
# does not read, so the config-file values are what is reproduced here.
# -----------------------------------------------------------------------------

# V4 ARTIFACT_BUCKET -- where the per-camera inspection JSON is uploaded.  The
# ingest API is handed an s3:// URI into this bucket and fetches it from there.
S3_ARTIFACT_BUCKET = _env("WAGONEYE_ARTIFACT_BUCKET", "test-inspection-artifacts-sarva")

# V4 RECEIVER_JSON_INGEST_API_URL_PROD.  Note this is the SAME host+path the V1
# dashboard receiver uses -- the `version` field in the document, not the URL,
# is what selects the dashboard tab.
INGEST_API_URL_PROD = _env(
    "WAGONEYE_INGEST_API_URL_PROD",
    "https://ms-pnr-location-notification-api.suvidhaen.com/"
    "cctv-receiver/inspections/ingest",
)

# V4 RECEIVER_JSON_INGEST_API_URL_UAT.
INGEST_API_URL_UAT = _env(
    "WAGONEYE_INGEST_API_URL_UAT",
    "https://cctv-wagon-uat-api.suvidhaen.com/inspections/ingest",
)

# V4 ML_API_ENDPOINT + ML_API_SECRET (sent as the `X-ML-SECRET` header).
ML_API_ENDPOINT = _env(
    "WAGONEYE_ML_API_ENDPOINT",
    "https://ms-pnr-location-notification-api.suvidhaen.com/"
    "cctv-receiver/api/v1/ml",
)
ML_API_SECRET = _env("WAGONEYE_ML_API_SECRET", "your-ml-callback-secret-key")

# The `version` value carried in each per-camera inspection document.  The
# dashboard chooses which tab renders the report from this: "v1" -> V1 tab.
INSPECTION_VERSION = _env("WAGONEYE_INSPECTION_VERSION", "v1")

UPLOAD_API_URL = _env("WAGONEYE_UPLOAD_API_URL",
                      "https://reports-api.suvidhaen.com/api/upload-pdf")
EMAIL_API_URL = _env(
    "WAGONEYE_EMAIL_API_URL",
    "https://ms-pnr-location-notification-api.suvidhaen.com/"
    "notification_microservice/send-email",
)
PRODUCT_NAME = _env("WAGONEYE_PRODUCT_NAME", "CCTV-WagonEye-CombinedReports")

EMAIL_RECEIVER = _env_list("WAGONEYE_EMAIL_RECEIVER", ["atul.nitt.cse@gmail.com"])
EMAIL_RECEIVER_CC = _env_list("WAGONEYE_EMAIL_RECEIVER_CC", [
    "Shivank.kumar.s2.s2@gmail.com",
    "rithish.sheru.s2@gmail.com",
    "omarbil01.s2@gmail.com",
    "kumarankitiitps2@gmail.com",
    "ajaykhatik6367s2@gmail.com",
    "priyankagp51.s2@gmail.com",
    "aman.freelancer.s2@gmail.com",
    "rajchaudhary01.official@gmail.com",
    "shyambabugupt.s2@gmail.com",
    "contact@suvidhaen.com",
])


# -----------------------------------------------------------------------------
# Misc tunables
# -----------------------------------------------------------------------------

# Confidence floors (inference)
CONF_DOOR    = 0.40
CONF_DAMAGE  = 0.55
CONF_OCR_BOX = 0.40

# JPEG quality for materializer
JPEG_QUALITY = 90

# OCR
WAGON_NUMBER_LENGTH = 11
