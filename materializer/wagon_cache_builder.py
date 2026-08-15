"""Stage 2 -- single-pass per-video frame extraction.

Takes the authoritative GlobalTrainState (Stage 1 output) plus the 4
source videos and writes per-wagon, per-camera JPEG folders:

    wagon_cache/
        GW_1/
            right_up/
                frame_000023.jpg
                frame_000024.jpg
                ...
            left_up/...
            right_up_top/...
            left_up_top/...
        GW_2/...

For each camera we open `cv2.VideoCapture` ONCE, walk it linearly, and
write JPEGs to whichever GW_n bucket the current local frame falls into.
No video decoding happens downstream of this stage.

Mapping master_time → local_frame applies the camera's own clock offset
(`t_global = t_local + delta`, estimated by the counting engine):

    local_start = round((GW.start_time - delta) * local_fps)
    local_end   = round((GW.end_time   - delta) * local_fps) - 1

clipped into [0, total_frames - 1].  This mirrors
`wagon_count/video_segmenter.build_camera_wagon_frame_map` exactly, so the
cache stays consistent with what the counting engine itself projects.

`delta` is 0.0 for the master camera and for any camera whose offset the
counter could not resolve, so a state carrying no offsets reproduces the
historical shared-`t=0` behaviour precisely.

A wagon whose projected window falls entirely OUTSIDE a camera's footage is
skipped for that camera rather than clamped to the last frame: clamping
fabricates evidence at the end of a shorter video.  The wagon keeps its global
id everywhere -- the camera simply contributes no frames for it, and the
feature processors then report that camera's existing NO_FRAMES status.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2

from core import constants as C
from core.global_state_loader import GlobalTrainState, GlobalWagon


# -----------------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------------

@dataclass
class CacheBuildResult:
    cache_root: str
    frames_written: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # {gw_id -> {camera_id -> n_frames}}

    per_camera_total: Dict[str, int] = field(default_factory=dict)
    missing_cameras: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def total_frames(self) -> int:
        return sum(self.per_camera_total.values())


# -----------------------------------------------------------------------------
# Master-time -> local-frame mapping
# -----------------------------------------------------------------------------

def _wagon_local_range(
    wagon: GlobalWagon, local_fps: float, local_total_frames: int,
    time_offset: float = 0.0,
) -> Tuple[int, int]:
    """Convert a GlobalWagon's master-clock window to inclusive [start, end]
    frame indices in the camera's local timeline.

    `time_offset` is the camera's clock delta (`t_global = t_local + delta`).
    Returns `(0, -1)` -- an empty range -- when the wagon does not overlap this
    camera's footage at all, so the caller writes no frames for it instead of
    clamping onto a frame that shows a different wagon.
    """
    if local_fps <= 0 or local_total_frames <= 0:
        return (0, -1)
    sf = int(round((wagon.start_time - time_offset) * local_fps))
    ef = int(round((wagon.end_time   - time_offset) * local_fps)) - 1
    # Out of this camera's footage entirely -> contribute nothing.
    if ef < 0 or sf > local_total_frames - 1:
        return (0, -1)
    sf = max(0, min(local_total_frames - 1, sf))
    ef = max(0, min(local_total_frames - 1, ef))
    if ef < sf:
        ef = sf
    return (sf, ef)


# -----------------------------------------------------------------------------
# Per-camera worker
# -----------------------------------------------------------------------------

def _extract_one_camera(
    *,
    camera_id: str,
    video_path: str,
    state: GlobalTrainState,
    local_fps: float,
    cache_root: str,
    jpeg_quality: int,
    verbose: bool,
    time_offset: float = 0.0,
) -> Tuple[str, Dict[str, int]]:
    """Open the video once, walk frames sequentially, dispatch by GW range."""

    if not os.path.exists(video_path):
        if verbose:
            print(f"[STAGE2/{camera_id}] video missing: {video_path}")
        return camera_id, {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if verbose:
            print(f"[STAGE2/{camera_id}] cv2 could not open {video_path}")
        return camera_id, {}

    # Reported total; some containers lie -- we just use it for clipping.
    total_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Build a frame_idx -> (gw_id, dst_dir) map
    frame_to_target: Dict[int, Tuple[str, str]] = {}
    counts: Dict[str, int] = {}
    cam_folder = C.CAMERA_FOLDER[camera_id]
    out_of_range: List[str] = []
    for gw in state.wagons:
        sf, ef = _wagon_local_range(gw, local_fps, total_meta or 10**7,
                                    time_offset)
        if ef < sf:
            # Not visible on this camera.  The wagon keeps its global id; this
            # camera just contributes no evidence for it.
            out_of_range.append(gw.global_id)
            continue
        dst = os.path.join(cache_root, gw.global_id, cam_folder)
        os.makedirs(dst, exist_ok=True)
        counts[gw.global_id] = 0
        for f in range(sf, ef + 1):
            # Wagons must not overlap; last-write-wins is a harmless
            # one-frame seam if they happen to touch.
            frame_to_target[f] = (gw.global_id, dst)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    frame_idx = 0
    written = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        target = frame_to_target.get(frame_idx)
        if target is not None:
            gw_id, dst = target
            out_path = os.path.join(dst, f"frame_{frame_idx:06d}.jpg")
            if cv2.imwrite(out_path, frame, encode_params):
                counts[gw_id] = counts.get(gw_id, 0) + 1
                written += 1
        frame_idx += 1
        if verbose and frame_idx % 1000 == 0:
            print(f"  [STAGE2/{camera_id}] scanned {frame_idx} frames, "
                  f"wrote {written}")
    cap.release()
    elapsed = time.time() - t0

    if verbose:
        offset_note = f"  offset={time_offset:+.2f}s" if time_offset else ""
        print(f"[STAGE2/{camera_id}] done in {elapsed:.1f}s  "
              f"frames_written={written}{offset_note}")
        if out_of_range:
            print(f"  [STAGE2/{camera_id}] {len(out_of_range)} wagon(s) outside "
                  f"this camera's footage -> no frames cached "
                  f"({', '.join(out_of_range[:6])}"
                  f"{' ...' if len(out_of_range) > 6 else ''})")

    return camera_id, counts


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def build(
    *,
    state: GlobalTrainState,
    video_paths: Dict[str, str],
    per_camera_fps: Dict[str, float],
    cache_root: str,
    jpeg_quality: int = C.JPEG_QUALITY,
    parallel: bool = True,
    verbose: bool = True,
    camera_offsets: Optional[Dict[str, float]] = None,
) -> CacheBuildResult:
    """Extract per-wagon JPEG folders for ALL cameras.

    Cameras with a missing source video produce an empty cache subtree
    -- they end up in `result.missing_cameras` but the stage still
    succeeds. Each remaining camera that successfully extracts > 0
    frames counts as "present".

    `camera_offsets` are the per-camera clock deltas resolved by the counting
    engine (`{camera_id: seconds}`).  Omitted or absent cameras use 0.0, which
    is the historical shared-`t=0` projection.  This stage NEVER reads the
    roster's ids or ordering for anything but bucketing -- it cannot change the
    count.
    """
    offsets = dict(camera_offsets or {})
    os.makedirs(cache_root, exist_ok=True)

    if verbose:
        print(f"[STAGE2] building wagon_cache at {cache_root}")
        print(f"[STAGE2] wagons={len(state.wagons)}  cameras="
              f"{list(video_paths.keys())}")

    result = CacheBuildResult(cache_root=cache_root)

    workload = []
    for cam in C.ALL_CAMERAS:
        if cam not in video_paths:
            result.missing_cameras.append(cam)
            continue
        local_fps = per_camera_fps.get(cam) or state.master_fps or 25.0
        workload.append((cam, video_paths[cam], local_fps))

    t_start = time.time()

    if parallel:
        with ThreadPoolExecutor(max_workers=min(4, len(workload) or 1)) as ex:
            futs = {
                ex.submit(
                    _extract_one_camera,
                    camera_id=cam, video_path=path,
                    state=state, local_fps=fps,
                    cache_root=cache_root,
                    jpeg_quality=jpeg_quality,
                    verbose=verbose,
                    time_offset=float(offsets.get(cam, 0.0) or 0.0),
                ): cam
                for (cam, path, fps) in workload
            }
            for f in as_completed(futs):
                cam_id, counts = f.result()
                for gw_id, n in counts.items():
                    result.frames_written.setdefault(gw_id, {})[cam_id] = n
                result.per_camera_total[cam_id] = sum(counts.values())
    else:
        for (cam, path, fps) in workload:
            cam_id, counts = _extract_one_camera(
                camera_id=cam, video_path=path,
                state=state, local_fps=fps,
                cache_root=cache_root,
                jpeg_quality=jpeg_quality,
                verbose=verbose,
                time_offset=float(offsets.get(cam, 0.0) or 0.0),
            )
            for gw_id, n in counts.items():
                result.frames_written.setdefault(gw_id, {})[cam_id] = n
            result.per_camera_total[cam_id] = sum(counts.values())

    result.elapsed_seconds = time.time() - t_start

    if verbose:
        print(f"[STAGE2] done in {result.elapsed_seconds:.1f}s  "
              f"total_frames={result.total_frames()}")
        for cam in C.ALL_CAMERAS:
            n = result.per_camera_total.get(cam, 0)
            status = "missing" if cam in result.missing_cameras else f"{n} frames"
            print(f"  {cam:<14} {status}")

    return result
