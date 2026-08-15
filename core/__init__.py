"""Foundational types + constants for wagon_eye_v4."""

from . import constants
from .global_state_loader import (
    GlobalTrainState, GlobalWagon,
    load_global_train_state, load_per_camera_fps,
    parse_global_train_state,
    RosterImmutabilityError, roster_fingerprint,
    assert_roster_unchanged, verify_roster_integrity,
)
from .unified_wagon_state import UnifiedWagonState, summarize_wagons
from .feature_config import (
    FeatureConfig, FeatureSpec, FEATURE_REGISTRY, FEATURE_KEYS,
    FIELD_TO_FEATURE, parse_disable_arg,
)
from .batch import (
    CameraVideo, TrainBatch,
    parse_train_timestamp,
    build_local_batch, scan_local_video_dir,
)

__all__ = [
    "constants",
    "GlobalTrainState", "GlobalWagon",
    "load_global_train_state", "load_per_camera_fps",
    "parse_global_train_state",
    "RosterImmutabilityError", "roster_fingerprint",
    "assert_roster_unchanged", "verify_roster_integrity",
    "UnifiedWagonState", "summarize_wagons",
    "FeatureConfig", "FeatureSpec", "FEATURE_REGISTRY", "FEATURE_KEYS",
    "FIELD_TO_FEATURE", "parse_disable_arg",
    "CameraVideo", "TrainBatch",
    "parse_train_timestamp",
    "build_local_batch", "scan_local_video_dir",
]
