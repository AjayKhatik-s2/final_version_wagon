"""Damage evidence straight off the raw video, before any wagon exists.

The Phase-1 pilot. Everything else in the feature layer reads frames the
materializer already bucketed -- `iter_wagon_frames(cache_root, gw_id, camera)`
-- which means a feature cannot run until the roster exists, and its
wagon assignment is encoded in the directory a frame came out of rather than in
the evidence itself. A wrong bucket then looks exactly like a right one.

This walks the ORIGINAL camera video and emits timestamped observations. No
`cache_root`, no `gw_id`, no `GlobalTrainState`: it can run at the same moment
as gap detection, on the same video timeline, before a single canonical gap has
been minted. Assignment happens later, in
`core.timeline_evidence.TimelineEvidence.fuse()`, purely from timestamps.

Now a damage-only VIEW of `features/raw_collect.py`, which scores every enabled
detector in one decode pass per camera. The scoring, the confidence floor and
`_filter_detections_for_top` live there and are reused, so there is exactly one
raw-video damage implementation rather than two that could drift.

What it deliberately does not do: decide anything about wagons. It never reads
a roster, never names a `GW_n`, and never drops an observation for being
outside one -- an observation on the locomotive is collected like any other and
is left unassigned by fusion, which is a result rather than a silence.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core import constants as C
from core.master_timeline import CameraClock
from core.timeline_evidence import KIND_DAMAGE, Observation

#: Only the top cameras see the roof, which is where this model looks.
DAMAGE_CAMERAS = C.TOP_CAMERAS

#: Matches the batch path's sampled stride, so the two see the same frames.
DEFAULT_STRIDE = 3

MODEL_PROVENANCE = "damage.pt"


@dataclass
class DamageCollectionResult:
    camera_id: str
    observations: List[Observation] = field(default_factory=list)
    frames_read: int = 0
    frames_scored: int = 0
    detections: int = 0
    elapsed_seconds: float = 0.0
    skipped: str = ""

    @property
    def ok(self) -> bool:
        return not self.skipped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "observations": len(self.observations),
            "frames_read": self.frames_read,
            "frames_scored": self.frames_scored,
            "detections": self.detections,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "skipped": self.skipped,
        }


def collect_damage_observations(
    *,
    camera_id: str,
    video_path: str,
    feature_models_dir: str,
    clock: Optional[CameraClock] = None,
    fps: float = 0.0,
    offset: float = 0.0,
    stride: int = DEFAULT_STRIDE,
    confidence: float = C.CONF_DAMAGE,
    max_frames: Optional[int] = None,
    model: Any = None,
    verbose: bool = True,
) -> DamageCollectionResult:
    """Damage observations for ONE camera, delegating to the shared scorer."""
    from features import raw_collect

    if clock is None:
        clock = CameraClock(camera_id=camera_id, fps=float(fps or 0.0),
                            total_frames=0, offset=float(offset or 0.0))
    raw = raw_collect.collect_camera(
        camera_id=camera_id, video_path=video_path,
        feature_models_dir=feature_models_dir, features=("damage",),
        clock=clock, strides={"damage": stride},
        models=({"damage": model} if model is not None else None),
        max_frames=max_frames, verbose=verbose)
    return DamageCollectionResult(
        camera_id=camera_id, observations=list(raw.observations),
        frames_read=raw.frames_read,
        frames_scored=raw.frames_scored.get("damage", 0),
        detections=raw.detections.get("damage", 0),
        elapsed_seconds=raw.elapsed_seconds, skipped=raw.skipped)


def collect_all_cameras(
    *,
    video_paths: Dict[str, str],
    feature_models_dir: str,
    clocks: Optional[Dict[str, CameraClock]] = None,
    cameras: Sequence[str] = DAMAGE_CAMERAS,
    stride: int = DEFAULT_STRIDE,
    confidence: float = C.CONF_DAMAGE,
    model: Any = None,
    verbose: bool = True,
) -> Dict[str, DamageCollectionResult]:
    """Every damage camera. Independent: one missing video does not stop another."""
    return {cam: collect_damage_observations(
        camera_id=cam, video_path=video_paths.get(cam, ""),
        feature_models_dir=feature_models_dir,
        clock=(clocks or {}).get(cam), stride=stride, confidence=confidence,
        model=model, verbose=verbose) for cam in cameras}
