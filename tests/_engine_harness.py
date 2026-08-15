"""Shared harness for the counting-engine regression tests.

Not a test module (no `test_` prefix), so neither pytest nor
`unittest discover` collects it.

Everything here drives the REAL counting engine in `wagon_count/`:
fragment reassembly, gap validation, fixed-master fusion and the wagon window
are the production functions, not stand-ins.  Only the *input* is synthetic --
GapEvents with explicit trajectories, standing in for what the YOLO tracker
emits, so the tests need no model weights and no video decode.

No wagon count is ever hard-coded: every expectation is a relationship the
engine must satisfy (invariants, id contiguity, cross-camera agreement), so
these tests cannot be satisfied by fabricating a number.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence

V4_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAGON_COUNT_DIR = os.path.join(V4_ROOT, "wagon_count")
# The reference folder the correct-count engine was adopted from.  Optional:
# provenance tests skip when the reviewer has deleted it.
REFERENCE_DIR = os.path.join(V4_ROOT, "wagon_count - Copy_correct_count")
LEGACY_BACKUP_DIR = os.path.join(V4_ROOT, "_legacy_wagon_count_removed")

for _p in (V4_ROOT, WAGON_COUNT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- the production counting engine ----------------------------------------
import fragment_stitching as fstitch          # noqa: E402
import gap_validation as gval                 # noqa: E402
import global_fusion as gf                    # noqa: E402
from global_train_state import (              # noqa: E402
    ALL_CAMERAS, CAMERA_LEFT_UP, CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP,
    CAMERA_RIGHT_UP_TOP, GapEvent, LocalCameraTracks, SegmentClass,
    _MasterClassification,
)

# --- the production v4 boundary --------------------------------------------
from core.global_state_loader import parse_global_train_state    # noqa: E402


FPS = 15.0
FRAME_W = 848
FRAME_H = 480


def moving_gap(
    track_id: int, center_time: float, *, camera_id: str = CAMERA_RIGHT_UP,
    fps: float = FPS, confidence: float = 0.90, n_hits: int = 20,
    x_start: float = 150.0, x_end: float = 700.0,
) -> GapEvent:
    """A gap that genuinely crosses the frame, as a real inter-wagon gap does.

    Trajectory magnitude and speed are in the band the engine's own validation
    tests describe as measured-real (110-615 px at 74-555 px/s), so this passes
    the production motion gates for the right reason rather than by relaxing
    them.
    """
    span = n_hits - 1
    start_frame = int(round(center_time * fps - span / 2.0))
    start_frame = max(0, start_frame)
    frames = [start_frame + i for i in range(n_hits)]
    xs = [x_start + (x_end - x_start) * i / span for i in range(n_hits)]
    return GapEvent(
        track_id=track_id, camera_id=camera_id,
        start_frame=frames[0], end_frame=frames[-1],
        confidence=confidence, hit_count=n_hits,
        center_x_trajectory=xs, fps=fps, temporal_consistency_score=1.0,
        hit_frames=frames,
        bbox_history=[[x - 20.0, 100.0, x + 20.0, 300.0] for x in xs],
    )


def camera_tracks(
    camera_id: str, gap_times: Sequence[float], *,
    duration_s: float = 300.0, fps: float = FPS, confidence: float = 0.90,
) -> LocalCameraTracks:
    """One camera's tracker output, with gaps at the given LOCAL times."""
    gaps = [moving_gap(i, t, camera_id=camera_id, fps=fps, confidence=confidence)
            for i, t in enumerate(sorted(gap_times), start=1)]
    return LocalCameraTracks(
        camera_id=camera_id, video_path=f"/synthetic/{camera_id}.mp4",
        fps=fps, total_frames=int(round(duration_s * fps)),
        width=FRAME_W, height=FRAME_H, gaps=gaps,
    )


def drifting_gap_times(n: int, start: float = 30.0) -> List[float]:
    """Gap times whose spacing drifts, as a real train's do.

    Uniform spacing would make whole-period clock offsets perfect aliases and
    render the synchronization tests vacuous.
    """
    times: List[float] = []
    t = start
    for i in range(n):
        times.append(t)
        t += 4.0 + 2.0 * (i / max(1, n - 1))
    return times


def whole_video_wagon_classification(
    master: LocalCameraTracks,
) -> List[_MasterClassification]:
    """Label the whole master video WAGON.

    Mirrors the engine's own fusion tests.  Classification is a separate model
    (`side_classification.pt`) and is not what these tests exercise -- they
    exercise counting, so every segment is a wagon and the wagon window spans
    the train.
    """
    return [_MasterClassification(0, 0, max(0, master.total_frames - 1),
                                  SegmentClass.WAGON, 1.0)]


def run_counting_engine(
    master_gap_times: Sequence[float],
    support_gap_times: Optional[Dict[str, Sequence[float]]] = None,
    *,
    duration_s: float = 300.0,
    fps: float = FPS,
    verbose: bool = False,
):
    """Run the REAL four-camera counting chain end to end.

    tracker output -> fragment reassembly -> gap validation
                   -> master classification -> fixed-master global fusion

    Returns `(state, tracks)` where `state` is the engine's own
    GlobalTrainState.  This is the same call sequence
    `wagon_count/run_global_count.py` performs, minus the YOLO/video I/O that
    produced the GapEvents.
    """
    support_gap_times = support_gap_times or {}
    tracks: Dict[str, LocalCameraTracks] = {
        CAMERA_RIGHT_UP: camera_tracks(CAMERA_RIGHT_UP, master_gap_times,
                                       duration_s=duration_s, fps=fps),
    }
    for cam in (CAMERA_LEFT_UP, CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP):
        tracks[cam] = camera_tracks(cam, support_gap_times.get(cam, ()),
                                    duration_s=duration_s, fps=fps)

    # STEP 1a -- fragment reassembly (production function)
    stitch_cfg = fstitch.FragmentStitchConfig()
    for cam in ALL_CAMERAS:
        t = tracks[cam]
        t.gaps = fstitch.reassemble_fragments(
            t.gaps, cam, stitch_cfg, frame_width=t.width, fps=t.fps,
            verbose=verbose).events

    # STEP 1b -- gap validation (production function)
    gv_cfg = gval.GapValidationConfig()
    for cam in ALL_CAMERAS:
        t = tracks[cam]
        res = gval.validate_gap_events(t.gaps, cam, gv_cfg, verbose=verbose,
                                       frame_width=t.width, fps=t.fps)
        t.gaps = gval.renumber_gap_events(res.accepted)

    # STEP 3 -- fixed-master fusion (production function)
    master = tracks[CAMERA_RIGHT_UP]
    state = gf.assemble_global_train_state_master_fixed(
        master_tracks=master,
        support_tracks=[tracks[c] for c in ALL_CAMERAS if c != CAMERA_RIGHT_UP],
        initial_classifications=whole_video_wagon_classification(master),
        config=gf.FusionConfig(),
        verbose=verbose,
        wagon_only=True,
    )
    return state, tracks


def as_v4_state(engine_state):
    """Cross the real Stage-1 -> downstream boundary: engine JSON -> v4 state.

    Serializes with the engine's own `to_dict()` and parses with the v4
    adapter, so the JSON contract itself is under test rather than bypassed.
    """
    return parse_global_train_state(engine_state.to_dict())


def write_stage1_outputs(engine_state, tracks, output_dir: str) -> Dict[str, str]:
    """Write the two files Stage 1 hands downstream, exactly as it does."""
    import json

    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, "global_train_state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        f.write(engine_state.to_json())
    tracking_path = os.path.join(output_dir, "per_camera_tracking.json")
    with open(tracking_path, "w", encoding="utf-8") as f:
        json.dump({cam: tracks[cam].to_dict(
            include_classifications=(cam == CAMERA_RIGHT_UP))
            for cam in ALL_CAMERAS}, f, indent=2)
    return {"state": state_path, "tracking": tracking_path}
