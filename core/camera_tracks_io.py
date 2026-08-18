"""Lossless persistence of LocalCameraTracks for sequential mode.

`GapEvent.to_dict()` is a REPORTING view: it drops `center_x_trajectory`,
`hit_frames`, `bbox_history` and `class_label`, which global fusion and the
overlay renderer need. Reconstructing tracks from it would silently degrade
assembly.

This module serializes every dataclass field of `LocalCameraTracks` and
`GapEvent` verbatim, so a bundle can be reloaded and fused WITHOUT re-running
Stage 1. All frame numbers stay absolute to the original camera video.

Pure I/O: no inference, no thresholds, nothing under wagon_count/ modified.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WC = os.path.join(_ROOT, "wagon_count")
for _p in (_ROOT, _WC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "wagon_eye.camera_tracks.v1"

#: Every dataclass field, in constructor order. Kept explicit so a change to
#: the upstream dataclass surfaces as a test failure rather than silent loss.
GAP_FIELDS = (
    "track_id", "camera_id", "start_frame", "end_frame", "confidence",
    "hit_count", "center_x_trajectory", "fps", "temporal_consistency_score",
    "hit_frames", "bbox_history", "class_label",
)
TRACK_FIELDS = (
    "camera_id", "video_path", "fps", "total_frames", "width", "height",
)


def _plain(v: Any) -> Any:
    """numpy -> builtin, so json.dump never chokes and floats round-trip."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return v


def serialize_gap(g: Any) -> Dict[str, Any]:
    return {f: _plain(getattr(g, f, None)) for f in GAP_FIELDS}


def serialize_tracks(tracks: Any) -> Dict[str, Any]:
    """Full-fidelity snapshot of one camera's tracker output."""
    return {
        "schema": SCHEMA,
        **{f: _plain(getattr(tracks, f, None)) for f in TRACK_FIELDS},
        "gaps": [serialize_gap(g) for g in (tracks.gaps or [])],
        "classifications": [
            c.to_dict() for c in (tracks.classifications or [])
            if hasattr(c, "to_dict")
        ],
        # frame_idx -> [bbox dicts]; keys become strings in JSON.
        "raw_frame_detections": _plain(tracks.raw_frame_detections or {}),
    }


def reconstruct_gap(d: Dict[str, Any]) -> Any:
    from global_train_state import GapEvent
    kw = {f: d.get(f) for f in GAP_FIELDS}
    # Sequence fields must be lists, never None -- the tracker's own default.
    for f in ("center_x_trajectory", "hit_frames", "bbox_history"):
        if kw.get(f) is None:
            kw[f] = []
    for f in ("track_id", "start_frame", "end_frame", "hit_count"):
        if kw.get(f) is not None:
            kw[f] = int(kw[f])
    for f in ("confidence", "fps", "temporal_consistency_score"):
        if kw.get(f) is not None:
            kw[f] = float(kw[f])
    return GapEvent(**kw)


def reconstruct_tracks(d: Dict[str, Any]) -> Any:
    """Rebuild LocalCameraTracks with the fidelity global fusion needs."""
    from global_train_state import LocalCameraTracks, _MasterClassification

    raw = {}
    for k, v in (d.get("raw_frame_detections") or {}).items():
        try:
            raw[int(k)] = v
        except (TypeError, ValueError):
            continue

    cls: List[Any] = []
    for c in (d.get("classifications") or []):
        try:
            cls.append(_MasterClassification(
                segment_index=int(c["segment_index"]),
                start_frame=int(c["start_frame"]),
                end_frame=int(c["end_frame"]),
                label=str(c["label"]),
                confidence=float(c.get("confidence", 0.0)),
            ))
        except (KeyError, TypeError, ValueError):
            continue

    return LocalCameraTracks(
        camera_id=str(d.get("camera_id") or ""),
        video_path=str(d.get("video_path") or ""),
        fps=float(d.get("fps") or 0.0),
        total_frames=int(d.get("total_frames") or 0),
        width=int(d.get("width") or 0),
        height=int(d.get("height") or 0),
        gaps=[reconstruct_gap(g) for g in (d.get("gaps") or [])],
        classifications=cls,
        raw_frame_detections=raw,
    )


def write_tracks(path: str, tracks: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_tracks(tracks), f, indent=2, default=str)
    return path


def read_tracks(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return reconstruct_tracks(json.load(f))
