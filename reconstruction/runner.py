"""Stage 1 -- subprocess wrapper around wagon_count/run_global_count.py.

The wagon_count package is the SOLE counting authority.  It owns:
    * gap detection per camera
    * fragment reassembly of tracker-split gaps
    * motion / temporal gap validation (a raw YOLO gap is a candidate only)
    * WAGON_ACTIVE recovery of soft-failed candidates inside the wagon region
    * RIGHT_UP master classification + temporal smoothing
    * per-camera clock-offset estimation
    * fixed-master cross-camera fusion (global gaps == RIGHT_UP gaps)
    * the wagon window -- engines / brake vans get no GW id
    * deterministic GW_n id assignment

Nothing downstream of this stage may count, re-segment, renumber or extend
the roster it returns.

We invoke it as a subprocess with:
    --no-frames      the materializer owns frame extraction, and under the
                     current counter this flag also skips wagon_count's own
                     evidence PDF (v4 has its own reporting layer)
    --render-videos  keep emitting the per-camera tracking-overlay mp4s under
                     `<output_dir>/processed_videos/` as Stage-1 debug
                     artifacts (they became opt-in in the current counter).
                     The rich feature-overlay videos are produced separately
                     by `rendering.feature_overlay_renderer`.

Returns the parsed GlobalTrainState (lightweight dataclass from
core.global_state_loader) or raises on failure.  Caller is responsible
for marking the batch as `failed_no_global_state` when this raises.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from core import constants as C
from core.global_state_loader import (
    GlobalTrainState, load_global_train_state, load_per_camera_fps,
    roster_fingerprint, verify_roster_integrity,
)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ReconstructionResult:
    """Outcome of one Stage-1 invocation."""
    state: GlobalTrainState
    per_camera_fps: Dict[str, float]
    state_json_path: str
    per_camera_tracking_path: str
    output_dir: str
    elapsed_seconds: float
    # Per-camera clock offsets the counter resolved (`t_local = t_global -
    # delta`).  Empty / 0.0 means "no decisive offset" -- i.e. the historical
    # shared-t=0 behaviour.  Consumed by Stage 2 and Stage 4b.
    camera_offsets: Dict[str, float] = field(default_factory=dict)
    # Fingerprint of the finalized roster, taken the moment Stage 1 returns.
    # Every later stage is checked against it.
    roster_fingerprint: str = ""


class ReconstructionError(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# Subprocess driver
# -----------------------------------------------------------------------------

def _find_wagon_count_dir(repo_root: str) -> str:
    """Locate the wagon_count subpackage shipped next to this file."""
    candidate = os.path.join(repo_root, "wagon_count")
    if os.path.isfile(os.path.join(candidate, "run_global_count.py")):
        return candidate
    raise ReconstructionError(
        f"wagon_count/ not found under {repo_root}. "
        f"Expected {candidate}/run_global_count.py."
    )


def run(
    *,
    video_paths: Dict[str, str],
    reconstruction_models_dir: str,
    output_dir: str,
    repo_root: str,
    python_executable: Optional[str] = None,
    timeout_seconds: int = 7200,
    verbose: bool = True,
) -> ReconstructionResult:
    """Run Stage 1.

    Args:
        video_paths: {camera_id -> local path}, all 4 cameras required.
        reconstruction_models_dir: path to models/reconstruction/.  Must
            contain the 4 .pt files (short or long names; wagon_count
            accepts both).
        output_dir: where wagon_count writes its outputs; we put it under
            `batch_outputs/<key>/global_state/`.
        repo_root: parent that contains the wagon_count/ subpackage.
        python_executable: defaults to sys.executable.
        timeout_seconds: hard cap on the subprocess.
        verbose: echo subprocess stdout tail on completion.

    Returns:
        ReconstructionResult.

    Raises:
        ReconstructionError on any failure (missing cameras, subprocess
        exit != 0, no JSON produced, zero wagons).
    """
    # Validate cameras up front
    missing = [c for c in C.ALL_CAMERAS if c not in video_paths]
    if missing:
        raise ReconstructionError(
            f"Stage 1 needs all 4 cameras; missing: {missing}"
        )
    for cam, p in video_paths.items():
        if not os.path.exists(p):
            raise ReconstructionError(
                f"Video for {cam} does not exist: {p}"
            )

    if not os.path.isdir(reconstruction_models_dir):
        raise ReconstructionError(
            f"reconstruction_models_dir does not exist: "
            f"{reconstruction_models_dir}"
        )

    wagon_count_dir = _find_wagon_count_dir(repo_root)
    script = os.path.join(wagon_count_dir, "run_global_count.py")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        python_executable or sys.executable,
        script,
        "--right_up",     video_paths[C.CAMERA_RIGHT_UP],
        "--left_up",      video_paths[C.CAMERA_LEFT_UP],
        "--right_up_top", video_paths[C.CAMERA_RIGHT_UP_TOP],
        "--left_up_top",  video_paths[C.CAMERA_LEFT_UP_TOP],
        "--models-dir",   reconstruction_models_dir,
        "--output",       output_dir,
        "--no-frames",      # materializer owns frame extraction
        # Overlay videos are opt-in in the current counter, so ask for them
        # explicitly.  They land at
        # <output_dir>/processed_videos/<CAM>_processed.mp4 and serve as the
        # Stage-1 debug artifact, exactly as before.  The rich feature-overlay
        # videos are produced separately by rendering.feature_overlay_renderer.
        "--render-videos",
    ]

    if verbose:
        print(f"[STAGE1] launching wagon_count: {' '.join(cmd)}")

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=wagon_count_dir,
        capture_output=True, text=True,
        timeout=timeout_seconds,
    )
    elapsed = time.time() - t0

    if verbose:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        print(f"[STAGE1] subprocess exit={proc.returncode} ({elapsed:.1f}s)")
        if tail:
            print("[STAGE1] --- stdout tail ---")
            print(tail)
            print("[STAGE1] ----------------------")
        if proc.returncode != 0 and proc.stderr:
            print("[STAGE1] --- stderr tail ---", file=sys.stderr)
            print(proc.stderr[-2000:], file=sys.stderr)
            print("[STAGE1] ----------------------", file=sys.stderr)

    if proc.returncode != 0:
        raise ReconstructionError(
            f"wagon_count subprocess exited {proc.returncode}"
        )

    state_path = os.path.join(output_dir, "global_train_state.json")
    if not os.path.isfile(state_path):
        raise ReconstructionError(
            f"wagon_count did not produce {state_path}"
        )

    state = load_global_train_state(state_path)
    if state.total_wagons <= 0:
        raise ReconstructionError(
            f"wagon_count returned total_wagons={state.total_wagons}; "
            f"aborting batch"
        )

    # The roster is the immutable backbone of everything downstream, so it is
    # validated the moment it arrives rather than failing obscurely in Stage 3.
    problems = verify_roster_integrity(state)
    if problems:
        raise ReconstructionError(
            "wagon_count produced a malformed global wagon roster: "
            + "; ".join(problems[:5])
        )

    pcf_path = os.path.join(output_dir, "per_camera_tracking.json")
    per_camera_fps = load_per_camera_fps(pcf_path) if os.path.exists(pcf_path) else {}
    camera_offsets = state.camera_time_offsets()

    if verbose:
        print(f"[STAGE1] OK  total_wagons={state.total_wagons}  "
              f"(E:{state.engine_count}  W:{state.regular_wagon_count}  "
              f"B:{state.brake_van_count})  "
              f"master_fps={state.master_fps:.2f}")
        if state.fusion_mode:
            print(f"[STAGE1] fusion_mode={state.fusion_mode}  "
                  f"global_gaps={state.global_gap_count}  "
                  f"master_wagon_count={state.master_wagon_count}")
        inv = state.invariant_checks or {}
        if inv:
            holds = inv.get("invariant_holds")
            print(f"[STAGE1] counting invariants: "
                  f"{'HOLD' if holds else 'VIOLATED'} "
                  f"({inv.get('checks_run', '?')} checks, "
                  f"{len(inv.get('violations') or [])} violation(s))")
        if camera_offsets:
            pretty = "  ".join(f"{c}={d:+.2f}s" for c, d in
                               sorted(camera_offsets.items()))
            print(f"[STAGE1] resolved camera offsets: {pretty}")
        unresolved = [c for c in C.ALL_CAMERAS if c not in camera_offsets]
        if unresolved:
            print(f"[STAGE1] unresolved camera offsets (treated as 0.0, "
                  f"shared t=0): {unresolved}")

    return ReconstructionResult(
        state=state,
        per_camera_fps=per_camera_fps,
        state_json_path=state_path,
        per_camera_tracking_path=pcf_path,
        output_dir=output_dir,
        elapsed_seconds=elapsed,
        camera_offsets=camera_offsets,
        roster_fingerprint=roster_fingerprint(state),
    )
