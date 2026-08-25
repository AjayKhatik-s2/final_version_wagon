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
from core.evidence_identity import (
    damage_track_slot, legacy_damage_track_slot,
)


# -----------------------------------------------------------------------------
# Per-camera local frame range for a given GlobalWagon
# -----------------------------------------------------------------------------

def wagon_local_frames(
    wagon_start_time: float, wagon_end_time: float,
    local_fps: float, local_total_frames: int,
    time_offset: float = 0.0,
) -> Tuple[int, int]:
    """Master time -> this camera's local frames. Inclusive, clipped.

    `time_offset` is the camera's clock delta (`t_global = t_local + delta`) and
    MUST match what the materializer used when it named the cache files:

        materializer/wagon_cache_builder.py:92
            sf = int(round((wagon.start_time - time_offset) * local_fps))

    Omitting it was a real defect. The cache filenames are offset-corrected
    local indices, so a reader computing UN-corrected ones looked for files that
    do not exist and got None -- and the PDF rendered a placeholder for every
    support-camera frame of every wagon, while the frames sat on disk. It failed
    silently and looked like missing evidence rather than a lookup bug.

    Defaults to 0.0, which is correct for the master and for any camera whose
    offset the counter could not resolve, so existing callers keep their
    behaviour exactly.
    """
    if local_fps <= 0 or local_total_frames <= 0:
        return (0, -1)
    sf = int(round((wagon_start_time - time_offset) * local_fps))
    ef = int(round((wagon_end_time - time_offset) * local_fps)) - 1
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
    time_offset: float = 0.0,
) -> List[Optional[str]]:
    """Four paths (12.5/37.5/62.5/87.5%) into the wagon_cache for one
    (wagon, camera) pair.  Entries absent from disk come back as None, so a
    caller renders a placeholder rather than repeating an image it does have.

    `time_offset` is this camera's clock delta and must be supplied for support
    cameras: without it the computed frame indices do not match the names the
    materializer wrote, and every path resolves to None. See
    `wagon_local_frames`.
    """
    if not cache_root:
        return [None, None, None, None]
    sf, ef = wagon_local_frames(
        wagon_start_time, wagon_end_time, local_fps, local_total_frames,
        time_offset,
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


def evidence_snapshot_for_camera(
    evidence_root: Optional[str], gw_id: str, feature: str, slot: str,
    camera_id: str, *, source_key: str = "source_camera",
) -> Optional[str]:
    """`evidence_snapshot`, but ONLY if that file belongs to `camera_id`.

    Some features write ONE file per wagon even though two cameras could have
    produced it.  `load/best_frame.jpg` is the case that matters: the load
    processor picks a single `source_cam` -- RIGHT_UP_TOP when it has data for
    the winning class, else LEFT_UP_TOP -- and saves that camera's frame under
    a filename with no camera in it (features/load/processor.py).  The path
    therefore cannot say who owns the image; only `metadata.json`'s
    `source_camera` can.

    A resolver that skips that check hands the other camera's frame to this
    camera's report, and because the two top views look alike the substitution
    is invisible on inspection.  That is exactly the defect this function
    exists to make impossible.

    Ownership must be PROVEN, not assumed: when the metadata carries no
    attribution at all, this returns None rather than guessing.  None is a safe
    answer -- every caller's remaining fallback is the wagon_cache frame for
    this camera, which is camera-correct by construction (`_cache_frame_path`
    keys on `C.CAMERA_FOLDER[camera_id]`).  Borrowing is never a fallback.
    """
    if not camera_id:
        raise ValueError(
            "evidence_snapshot_for_camera requires camera_id: camera identity "
            "is part of the lookup, not an optional refinement")
    path = evidence_snapshot(evidence_root, gw_id, feature, slot)
    if not path:
        return None
    meta = evidence_metadata(evidence_root, gw_id, feature)
    owner = meta.get(source_key) or meta.get("camera_id")
    if not owner:
        return None                  # unproven ownership is not ownership
    return path if owner == camera_id else None


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


#: Returned instead of a path when a camera has no evidence of its own.
#: Distinct from None so a caller can tell "this camera saw nothing" from
#: "nobody looked", and so no code path is tempted to fill the gap with a
#: different camera's picture.
MISSING_EVIDENCE = None


def damage_track_snapshot(
    evidence_root: Optional[str], gw_id: str, camera_id: str, track_idx: int,
) -> Optional[str]:
    """One camera's damage snapshot for one observation index.

    Tries the camera-scoped slot first (`track_2__RIGHT_UP_TOP`). Falls back to
    the legacy camera-less `track_2` so evidence written before the rename still
    renders -- and that fallback is safe ONLY because every caller has already
    confirmed from `metadata.json` that this track_idx belongs to `camera_id`.
    The fallback therefore cannot cross camera identity; it just finds an
    older filename for a record whose owner is already established.
    """
    slot = damage_track_slot(track_idx, camera_id)
    p = evidence_snapshot(evidence_root, gw_id, "damage", slot)
    if p:
        return p
    return evidence_snapshot(evidence_root, gw_id, "damage",
                             legacy_damage_track_slot(track_idx))


def load_snapshot(
    evidence_root: Optional[str], gw_id: str, camera_id: str,
) -> Optional[str]:
    """The load snapshot, but ONLY if this camera produced it.

    `evidence/<gw>/load/best_frame.jpg` is a single file written from whichever
    top camera won: features/load/processor.py prefers RIGHT_UP_TOP and falls
    back to LEFT_UP_TOP, recording the winner as `source_camera` in the
    sibling metadata. Resolving that file by wagon id alone therefore hands a
    LEFT_UP_TOP frame to a RIGHT_UP_TOP panel whenever the master had no load
    evidence -- the two top cameras look alike, so it reads as correct.

    The provenance is already on disk; this consults it.
    """
    md = evidence_metadata(evidence_root, gw_id, "load")
    if md.get("source_camera") != camera_id:
        return MISSING_EVIDENCE
    return evidence_snapshot(evidence_root, gw_id, "load", "best_frame")


def damage_track_snapshots(
    evidence_root: Optional[str], gw_id: str, camera_id: str,
    max_tracks: int = 3,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Resolve up to `max_tracks` damage track snapshots for ONE camera.

    `camera_id` is required, and that is deliberate. A wagon's
    `evidence/<gw>/damage/` directory holds the tracks of BOTH top cameras
    side by side -- `track_1..track_N` is a single sequence numbered across
    RIGHT_UP_TOP and LEFT_UP_TOP together, and the only thing that says which
    camera a track came from is `camera_id` in `metadata.json`. A resolver
    that skips that filter will happily hand a LEFT_UP_TOP snapshot to a
    RIGHT_UP_TOP report; the two top cameras can look near-identical, so the
    mistake is invisible on inspection.

    Camera identity is therefore part of the lookup, not an optional refinement
    the caller may forget. `camera_reports._camera_damage_tracks` and
    `combined_train_report._top_damage_snapshot` apply the same filter.

    Returns (path, track_metadata) sorted by `best_confidence` descending, so
    the most certain damage shows first. Missing files or metadata yield [].
    """
    meta = evidence_metadata(evidence_root, gw_id, "damage")
    tracks = meta.get("tracks") or []
    out: List[Tuple[str, Dict[str, Any]]] = []
    for tr in tracks:
        if not isinstance(tr, dict):
            continue
        if tr.get("camera_id") != camera_id:
            continue                    # another camera's track -- never ours
        idx = tr.get("track_idx")
        if not idx:
            continue
        p = damage_track_snapshot(evidence_root, gw_id, camera_id, int(idx))
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
