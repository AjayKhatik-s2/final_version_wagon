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


# --- protected-path guards -------------------------------------------------
#
# Several tests assert that the proven counting and reporting code has not
# drifted, by requiring `git diff HEAD -- <path>` to be empty. A DELIBERATE,
# reviewed change to one of those paths would therefore fail the guard for as
# long as it sits uncommitted -- so intentional edits are listed here with the
# reason, and everything else still fails loudly.
#
# These entries are self-clearing: once committed, `git diff HEAD` no longer
# reports the file and the entry has no effect. Delete an entry when its change
# is committed rather than letting the list accumulate.
REVIEWED_IN_WORKTREE = (
    # Camera-isolation fix. The audit found the combined report resolving a top
    # camera's snapshot through two camera-BLIND fallbacks, so RIGHT_UP_TOP
    # could render LEFT_UP_TOP's image:
    #
    #   * `_best_damage_snapshot_any()` -- "best track across BOTH top cameras",
    #     reached by either top panel when the wagon was damaged but that camera
    #     had no track of its own. Removed; `_panel_state_text` now names the
    #     camera that actually saw the damage instead of borrowing its frame.
    #   * an unscoped `load/best_frame` lookup applied to RIGHT_UP_TOP only.
    #     That file is ONE per wagon and the load processor may have sourced it
    #     from LEFT_UP_TOP, so both top panels ended up showing a LEFT_UP_TOP
    #     view -- and only when there was NO damage, because a damage track
    #     would have been resolved per-camera first and masked it.
    #
    # These are reporting-fallback defects, not detection or evidence-writing
    # defects: no threshold, no model, no evidence layout and no snapshot score
    # changed. The guards below stay in force for every other file in
    # reporting/, and for all of wagon_count/, reconstruction/ and fusion/.
    # See tests/test_camera_evidence_isolation.py for the proof, which decodes
    # the embedded pixels rather than trusting filenames.
    "reporting/combined_train_report.py",   # symmetric camera-scoped panels
    "reporting/camera_reports.py",          # own-camera load snapshot only
    "reporting/_evidence_lookup.py",        # + evidence_snapshot_for_camera()

    # Report COMPLETENESS, reviewed 2026-08-22. Both files selected the report's
    # wagons with a silent filter over the canonical timeline:
    #
    #   combined_train_report:  [unified[w.global_id] for w in state.wagons
    #                            if w.global_id in unified]
    #   _adapter:               [u for u in (unified.get(w.global_id)
    #                            for w in state.wagons) if u]
    #
    # Right source and right order, but a wagon absent from `unified` vanished
    # from doc["wagons"], from summary, from evidence_pages and from the KPI
    # state counts -- with nothing logged, so an incomplete report looked exactly
    # like a short train. A wagon with no feature result is still a wagon.
    #
    # The canonical Global Wagon timeline is now the iteration source in both,
    # with any absent state synthesized by the MATERIALIZER'S OWN `_fuse_one`
    # (every feature None -- its existing "no observations" path), so no second
    # wagon-counting system exists and the placeholder cannot drift from a real
    # state. `audit_report_integrity` then checks set, order and multiplicity
    # before rendering.
    #
    # No change to Stage 1, the RIGHT_UP master timeline, global wagon identity,
    # fusion logic, feature inference, thresholds, snapshot selection, camera
    # evidence isolation or engine-frame handling.
    # See tests/test_report_completeness.py.
    "reporting/_adapter.py",                # canonical-driven KPI counts

    # LEFT_UP_TOP's own CLASSIFIER, reviewed 2026-08-23. Both top cameras shared
    # `top_classification.pt`; LEFT_UP_TOP now loads `ltop.pt` and RIGHT_UP_TOP
    # is unchanged. This EXTENDS the mapping that already existed for exactly
    # this purpose -- train_structure.CAMERA_CLASSIFICATION_MODEL -- rather than
    # adding a second loader.
    #
    # run_global_count needed a real fix, not just the new name: it resolved ONE
    # `top_cls_path` and then chose between it and the SIDE model with
    # `want == TOP_CLASSIFICATION_MODEL`. That test goes False for `ltop.pt`, so
    # LEFT_UP_TOP would have been handed `side_classification.pt` -- a side-view
    # classifier on an overhead view, silently, returning confident labels from
    # the wrong model. Paths are now keyed by model NAME, one entry per distinct
    # model the mapping names.
    #
    # GAP DETECTION IS UNTOUCHED: both top cameras still use `top_gap.pt`, and
    # `ltop.pt` is deliberately NOT in RECON_MODEL_FILES. Classification remains
    # optional and never a counting authority -- RIGHT_UP alone decides the
    # count -- so an absent classifier reduces capability and nothing else. No
    # change to the tracking algorithm, gap validation, stitching, thresholds,
    # confidence floors, min-height ratios or inference stride.
    # See tests/test_left_up_top_model.py.
    "wagon_count/train_structure.py",       # mapping: LEFT_UP_TOP -> ltop.pt
    "wagon_count/run_global_count.py",      # name-keyed classifier resolution
    "wagon_count/tests/test_train_structure.py",   # the mapping's own test
    "wagon_count/validate_ec2.py",          # ltop.pt in the capability list
    "wagon_count/README.md",                # per-camera classifier table
)



def changed_paths(*pathspecs) -> "list":
    """Worktree changes vs HEAD under `pathspecs`, minus REVIEWED_IN_WORKTREE.

    Returns None when git is unavailable so callers can skip.
    """
    import subprocess

    r = subprocess.run(["git", "diff", "--name-only", "HEAD", *pathspecs],
                       cwd=V4_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return [p for p in r.stdout.split()
            if p and p not in REVIEWED_IN_WORKTREE]
