"""Resolve evidence-snapshot + cache-frame paths for the reporting layer.

The legacy door report rendered a 4-frame quartile wagon overview
(12.5 / 37.5 / 62.5 / 87.5%); the damage report rendered a single
midpoint snapshot for loaded / no-damage / non-wagon pages.  Both
sourced frames from the per-camera raw videos via cv2.VideoCapture.

In v4 every wagon's per-camera frames are already on disk under
    wagon_cache/<gw_id>/<camera_folder_lower>/frame_NNNNNN.jpg
because the materializer extracts them in a single pass during Stage 2.
This module computes those paths so the report builders can read them
directly without touching any video file.

It also resolves evidence snapshot paths by feature (e.g.
    evidence/<gw_id>/door/left_best.jpg
) so the combined "Damaged Wagon Report" and the camera-wise reports all
share one helper.

Pure path resolution + JSON read.  No model loads, no decoder calls.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C


# -----------------------------------------------------------------------------
# Per-camera local frame range for a given GlobalWagon
# -----------------------------------------------------------------------------

def wagon_local_frames(
    wagon_start_time: float, wagon_end_time: float,
    local_fps: float, local_total_frames: int,
) -> Tuple[int, int]:
    """Same arithmetic as wagon_count/video_segmenter.py:70.

    Returns (start_frame, end_frame) inclusive, clipped into the camera.
    """
    if local_fps <= 0 or local_total_frames <= 0:
        return (0, -1)
    sf = int(round(wagon_start_time * local_fps))
    ef = int(round(wagon_end_time * local_fps)) - 1
    sf = max(0, min(local_total_frames - 1, sf))
    ef = max(0, min(local_total_frames - 1, ef))
    if ef < sf:
        ef = sf
    return (sf, ef)


# -----------------------------------------------------------------------------
# Per-camera tracking JSON read (fps + total_frames per camera)
# -----------------------------------------------------------------------------

def load_per_camera_meta(
    per_camera_tracking_path: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """Return {camera_id -> {fps, total_frames, width, height}}.  Empty if
    the file is missing / unreadable.
    """
    if not per_camera_tracking_path or not os.path.isfile(per_camera_tracking_path):
        return {}
    try:
        with open(per_camera_tracking_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for cam, meta in doc.items():
        if isinstance(meta, dict):
            out[cam] = {
                "fps":          float(meta.get("fps") or 0.0),
                "total_frames": int(meta.get("total_frames") or 0),
                "width":        int(meta.get("width") or 0),
                "height":       int(meta.get("height") or 0),
                "gaps":         list(meta.get("gaps") or []),
            }
    return out


# -----------------------------------------------------------------------------
# Cache frame paths
# -----------------------------------------------------------------------------

def _cache_frame_path(
    cache_root: str, gw_id: str, camera_id: str, frame_idx: int,
) -> str:
    folder = C.CAMERA_FOLDER.get(camera_id, camera_id.lower())
    return os.path.join(
        cache_root, gw_id, folder, f"frame_{int(frame_idx):06d}.jpg",
    )


def quartile_cache_paths(
    *,
    cache_root: Optional[str],
    gw_id: str,
    camera_id: str,
    wagon_start_time: float,
    wagon_end_time: float,
    local_fps: float,
    local_total_frames: int,
) -> List[Optional[str]]:
    """Return four paths (12.5/37.5/62.5/87.5%) into the wagon_cache for
    one (wagon, camera) pair.  Entries that don't exist on disk are
    returned as None so the caller can render placeholders.
    """
    if not cache_root:
        return [None, None, None, None]
    sf, ef = wagon_local_frames(
        wagon_start_time, wagon_end_time, local_fps, local_total_frames,
    )
    if ef <= sf:
        return [None, None, None, None]
    span = ef - sf
    fractions = (0.125, 0.375, 0.625, 0.875)
    paths: List[Optional[str]] = []
    for frac in fractions:
        idx = sf + int(round(frac * span))
        idx = max(sf, min(ef, idx))
        p = _cache_frame_path(cache_root, gw_id, camera_id, idx)
        paths.append(p if os.path.isfile(p) else None)
    return paths


def midpoint_cache_path(
    *,
    cache_root: Optional[str],
    gw_id: str,
    camera_id: str,
    wagon_start_time: float,
    wagon_end_time: float,
    local_fps: float,
    local_total_frames: int,
) -> Optional[str]:
    """Return the single mid-wagon cache frame path.  Mirrors the legacy
    damage report's `_extract_wagon_snapshot` (legacy :952-1004) which
    used `(start + end) // 2`."""
    if not cache_root:
        return None
    sf, ef = wagon_local_frames(
        wagon_start_time, wagon_end_time, local_fps, local_total_frames,
    )
    if ef <= sf:
        return None
    mid = (sf + ef) // 2
    p = _cache_frame_path(cache_root, gw_id, camera_id, mid)
    return p if os.path.isfile(p) else None


# -----------------------------------------------------------------------------
# Evidence snapshot path resolution
# -----------------------------------------------------------------------------

def evidence_snapshot(
    evidence_root: Optional[str], gw_id: str, feature: str, slot: str,
) -> Optional[str]:
    """Resolve a single evidence file path; returns None if it doesn't exist.

    `feature` is one of {door, damage, ocr, load}.  `slot` examples:
        door:   left_best | left_crop | right_best | right_crop
        damage: track_1 | track_1_crop | track_2 | ... | track_3_crop
        ocr:    best_frame | number_crop
        load:   best_frame
    """
    if not evidence_root:
        return None
    p = os.path.join(evidence_root, gw_id, feature, f"{slot}.jpg")
    return p if os.path.isfile(p) else None


def evidence_metadata(
    evidence_root: Optional[str], gw_id: str, feature: str,
) -> Dict[str, Any]:
    if not evidence_root:
        return {}
    p = os.path.join(evidence_root, gw_id, feature, "metadata.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def damage_track_snapshots(
    evidence_root: Optional[str], gw_id: str, max_tracks: int = 3,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Resolve up to `max_tracks` damage track snapshots for one wagon.

    Returns a list of (path, track_metadata) tuples sorted by
    `best_confidence` descending so the report shows the most certain
    damages first.  Missing files / missing metadata yield an empty list.
    """
    meta = evidence_metadata(evidence_root, gw_id, "damage")
    tracks = meta.get("tracks") or []
    out: List[Tuple[str, Dict[str, Any]]] = []
    for tr in tracks:
        if not isinstance(tr, dict):
            continue
        idx = tr.get("track_idx")
        if not idx:
            continue
        p = evidence_snapshot(evidence_root, gw_id, "damage", f"track_{int(idx)}")
        if not p:
            continue
        out.append((p, tr))
    out.sort(key=lambda x: float(x[1].get("best_confidence") or 0.0), reverse=True)
    return out[:max_tracks]


# -----------------------------------------------------------------------------
# Per-wagon raw feature JSON read (for confidences not folded into UWS)
# -----------------------------------------------------------------------------

def read_wagon_feature_json(
    wagon_states_root: Optional[str], feature: str, gw_id: str,
) -> Dict[str, Any]:
    """Read `wagon_states/<feature>/<gw_id>.json`.  Returns {} on any failure."""
    if not wagon_states_root:
        return {}
    p = os.path.join(wagon_states_root, feature, f"{gw_id}.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}
