"""Single-camera Stage-1 pipeline for SEQUENTIAL mode.

A LITERAL EXTRACTION of the per-camera portion of
`wagon_count/run_global_count.py`. Every function called here is the proven
implementation, imported unchanged, invoked with the same arguments in the
same order and with the same configuration objects. Nothing under
`wagon_count/` is modified, and no threshold, heuristic or simplification is
introduced.

Why it exists: `run_global_count.main()` tracks all four cameras and then
fuses, in one process. Sequential mode needs one camera to complete on its own
without waiting for the others, so the per-camera steps are re-expressed here
and fusion is deliberately excluded.

Traced order in `run_global_count.main()`:

    STEP 1   GapTracker.process_video                       all cameras
    STEP 1a  fstitch.reassemble_fragments                   all cameras
    STEP 1b  gval.validate_gap_events -> renumber_gap_events all cameras
    STEP 2   segments_from_gaps -> MasterClassifier          MASTER only
    STEP 2a  tcls.apply_temporal_classification              MASTER only
    STEP 2b  segments_from_gaps -> classify -> temporal ->
             ts.build_local_wagon_region                     SUPPORT only
    STEP 2c  _derive_wagon_window -> recover_wagon_active_
             candidates -> renumber -> RE-RUN 2 and 2a       MASTER only
    STEP 3   fusion                                          NOT here

The master and support paths genuinely differ in control flow -- the master
mutates its own gap set after classification and then classifies again -- so
they are two explicit functions rather than one function with a flag.

GLOBAL WAGON IDS ARE NEVER ASSIGNED HERE. Output is camera-local
`L_<CAMERA>_<n>` only; `GW_n` exists solely after global assembly.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
_WAGON_COUNT = os.path.join(_REPO_ROOT, "wagon_count")
for _p in (_REPO_ROOT, _WAGON_COUNT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- the proven Stage-1 components, imported UNCHANGED ---------------------
import fragment_stitching as fstitch          # noqa: E402
import gap_validation as gval                 # noqa: E402
import global_alignment as ga                 # noqa: E402
import temporal_classification as tcls        # noqa: E402
import train_structure as ts                  # noqa: E402
from tracker_engine import (                  # noqa: E402
    GapTracker, MasterClassifier, segments_from_gaps,
)

from core.camera_evidence import LocalSegment, local_segment_id  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults -- IDENTICAL to run_global_count.py's argparse defaults.
# Kept in one place so a drift test can pin them.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraPipelineConfig:
    side_confidence: float = 0.4
    top_confidence: float = 0.4
    side_min_height_ratio: float = 0.35
    top_min_height_ratio: float = 0.05
    classification_samples: int = 5
    keep_raw_detections: bool = True
    # Feature switches mirror run_global_count's `--no-*` flags (all default ON)
    fragment_stitching: bool = True
    gap_validation: bool = True
    temporal_classification: bool = True
    wagon_recovery: bool = True


DEFAULT_CONFIG = CameraPipelineConfig()


@dataclass
class CameraPipelineResult:
    """Everything one camera produced, all camera-local."""
    camera_id: str
    tracks: Any                       # LocalCameraTracks (gaps already final)
    segments: List[LocalSegment] = field(default_factory=list)
    classifications: List[Any] = field(default_factory=list)
    stitch: Any = None
    validation: Any = None
    recovery: Any = None
    wagon_region: Any = None
    reclassified_after_recovery: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def is_master(self) -> bool:
        return self.recovery is not None or self.wagon_region is None


# ---------------------------------------------------------------------------
# Shared steps 1 / 1a / 1b -- identical for every camera
# ---------------------------------------------------------------------------

def _track_stitch_validate(
    *, camera_id: str, video_path: str, gap_model_path: str,
    confidence: float, min_height_ratio: float,
    cfg: CameraPipelineConfig, gv_cfg, stitch_cfg, verbose: bool,
):
    """STEP 1 + 1a + 1b, in run_global_count's exact order."""
    # STEP 1 -- per-camera gap tracking (unchanged GapTracker)
    tracker = GapTracker(
        camera_id=camera_id, model_path=gap_model_path,
        confidence=confidence, min_height_ratio=min_height_ratio,
        verbose=verbose,
    )
    tracks = tracker.process_video(
        video_path, keep_raw_detections=cfg.keep_raw_detections)

    # STEP 1a -- fragment reassembly BEFORE validation
    sres = fstitch.reassemble_fragments(
        tracks.gaps, camera_id, stitch_cfg,
        frame_width=tracks.width, fps=tracks.fps, verbose=verbose)
    tracks.gaps = sres.events

    # STEP 1b -- gap validation, then renumber to a contiguous temporal rank
    raw_n = sum(len(v) for v in (tracks.raw_frame_detections or {}).values())
    vres = gval.validate_gap_events(
        tracks.gaps, camera_id, gv_cfg,
        raw_detection_count=raw_n, verbose=verbose,
        frame_width=tracks.width, fps=tracks.fps,
        absolute_overrides=None)
    tracks.gaps = gval.renumber_gap_events(vres.accepted)
    return tracks, sres, vres


def _segments_to_local(
    camera_id: str, tracks, labels: Optional[Sequence[Any]] = None,
) -> List[LocalSegment]:
    """`segments_from_gaps` output -> camera-local L_<CAM>_<n> records.

    Frame indices and times are the camera's OWN, untouched. No global id is
    assigned and no reordering occurs: segment k keeps position k.
    """
    segs = segments_from_gaps(tracks.gaps, tracks.total_frames)
    fps = float(tracks.fps) or 1.0
    out: List[LocalSegment] = []
    for i, (sf, ef) in enumerate(segs, start=1):
        label, conf = "UNKNOWN", 0.0
        if labels and i - 1 < len(labels):
            lab = labels[i - 1]
            label = str(getattr(lab, "label", lab) or "UNKNOWN")
            conf = float(getattr(lab, "confidence", 0.0) or 0.0)
        out.append(LocalSegment(
            local_id=local_segment_id(camera_id, i), index=i,
            start_frame=int(sf), end_frame=int(ef),
            start_time=round(sf / fps, 4), end_time=round((ef + 1) / fps, 4),
            label=label, confidence=conf))
    return out


# ---------------------------------------------------------------------------
# _derive_wagon_window -- LITERAL extraction from run_global_count.py:452-469
# ---------------------------------------------------------------------------

def derive_wagon_window(master, classifications, verbose: bool = False):
    """Derive the wagon window from the CURRENT master gaps + classifications.

    Literal extraction of `run_global_count._derive_wagon_window`. Same
    imported functions (`ga.build_global_wagons`, `ts.get_master_wagon_window`),
    same arguments, same ordering, same broad `except` returning None. Do not
    "improve" this -- Stage-1 recovery behaviour depends on it exactly.
    """
    if not classifications:
        return None
    try:
        segments = ga.build_global_wagons(
            list(master.gaps),
            master_total_frames=master.total_frames, master_fps=master.fps,
            initial_classifications=list(classifications),
            support_camera_ids=[])
        return ts.get_master_wagon_window(segments, verbose=verbose)
    except Exception:
        return None


def _classify_master(master, side_cls_path: str, num_samples: int,
                     verbose: bool):
    """Literal extraction of `_classify_master_pre_fusion`."""
    pre_segments = segments_from_gaps(master.gaps, master.total_frames)
    if not pre_segments:
        return []
    clf = MasterClassifier(side_cls_path, num_samples=num_samples,
                           verbose=verbose)
    return clf.classify_segments(master.video_path, pre_segments)


# ---------------------------------------------------------------------------
# MASTER path
# ---------------------------------------------------------------------------

def run_master_camera(
    *, camera_id: str, video_path: str, gap_model_path: str,
    side_cls_path: str, cfg: CameraPipelineConfig = DEFAULT_CONFIG,
    gv_cfg=None, stitch_cfg=None, tc_cfg=None, verbose: bool = True,
) -> CameraPipelineResult:
    """STEP 1 -> 2c for the master camera (RIGHT_UP). No fusion, no GW ids."""
    gv_cfg = gv_cfg or gval.GapValidationConfig(enabled=cfg.gap_validation)
    stitch_cfg = stitch_cfg or fstitch.FragmentStitchConfig(
        enabled=cfg.fragment_stitching)
    tc_cfg = tc_cfg or tcls.TemporalClassificationConfig(
        enabled=cfg.temporal_classification)

    notes: List[str] = []
    master, sres, vres = _track_stitch_validate(
        camera_id=camera_id, video_path=video_path,
        gap_model_path=gap_model_path, confidence=cfg.side_confidence,
        min_height_ratio=cfg.side_min_height_ratio, cfg=cfg,
        gv_cfg=gv_cfg, stitch_cfg=stitch_cfg, verbose=verbose)

    # STEP 2 -- master classification
    try:
        classifications = _classify_master(
            master, side_cls_path, cfg.classification_samples, verbose)
    except Exception as e:
        notes.append(f"master_classification_failed:{e}")
        classifications = []

    # STEP 2a -- temporal smoothing
    if classifications:
        try:
            classifications, _tres = tcls.apply_temporal_classification(
                classifications, master.fps, camera_id=camera_id,
                cfg=tc_cfg, verbose=verbose)
        except Exception as e:
            notes.append(f"temporal_classification_failed:{camera_id}:{e}")

    # STEP 2c -- WAGON_ACTIVE recovery, then RE-RUN 2 and 2a if it fired
    recovery = None
    reclassified = False
    if cfg.gap_validation and cfg.wagon_recovery and vres is not None:
        win = derive_wagon_window(master, classifications, verbose=False)
        if win is not None and win.wagon_start_frame is not None:
            recovery = gval.recover_wagon_active_candidates(
                vres.rejected, master.gaps,
                win.wagon_start_frame, win.wagon_end_frame,
                camera_id, gv_cfg,
                frame_width=master.width, fps=master.fps,
                absolute_overrides=None, verbose=verbose)
            if recovery.recovered:
                master.gaps = gval.renumber_gap_events(
                    list(master.gaps) + list(recovery.recovered))
                # The gap sequence changed, so segments AND classification must
                # be rebuilt from it -- production classifies twice here.
                try:
                    classifications = _classify_master(
                        master, side_cls_path, cfg.classification_samples,
                        False)
                    if classifications:
                        classifications, _t2 = tcls.apply_temporal_classification(
                            classifications, master.fps, camera_id=camera_id,
                            cfg=tc_cfg, verbose=False)
                    reclassified = True
                except Exception as e:
                    notes.append(f"reclassification_after_recovery:{e}")

    return CameraPipelineResult(
        camera_id=camera_id, tracks=master,
        segments=_segments_to_local(camera_id, master, classifications),
        classifications=list(classifications), stitch=sres, validation=vres,
        recovery=recovery, wagon_region=None,
        reclassified_after_recovery=reclassified, notes=notes)


# ---------------------------------------------------------------------------
# SUPPORT path
# ---------------------------------------------------------------------------

def run_support_camera(
    *, camera_id: str, video_path: str, gap_model_path: str,
    classifier_path: Optional[str] = None, is_top: bool = False,
    cfg: CameraPipelineConfig = DEFAULT_CONFIG,
    gv_cfg=None, stitch_cfg=None, tc_cfg=None, verbose: bool = True,
) -> CameraPipelineResult:
    """STEP 1 -> 2b for a support camera. No recovery, no fusion, no GW ids."""
    gv_cfg = gv_cfg or gval.GapValidationConfig(enabled=cfg.gap_validation)
    stitch_cfg = stitch_cfg or fstitch.FragmentStitchConfig(
        enabled=cfg.fragment_stitching)
    tc_cfg = tc_cfg or tcls.TemporalClassificationConfig(
        enabled=cfg.temporal_classification)

    notes: List[str] = []
    tracks, sres, vres = _track_stitch_validate(
        camera_id=camera_id, video_path=video_path,
        gap_model_path=gap_model_path,
        confidence=cfg.top_confidence if is_top else cfg.side_confidence,
        min_height_ratio=(cfg.top_min_height_ratio if is_top
                          else cfg.side_min_height_ratio),
        cfg=cfg, gv_cfg=gv_cfg, stitch_cfg=stitch_cfg, verbose=verbose)

    # STEP 2b -- support classification + temporal smoothing + wagon region
    classifications: List[Any] = []
    region = None
    if classifier_path:
        try:
            clf, mapping = ts.load_segment_classifier(
                classifier_path, num_samples=cfg.classification_samples,
                verbose=verbose)
            segs = segments_from_gaps(tracks.gaps, tracks.total_frames)
            labels: List[str] = []
            if segs:
                cls = clf.classify_segments(tracks.video_path, segs)
                cls, _tres = tcls.apply_temporal_classification(
                    cls, tracks.fps, camera_id=camera_id, cfg=tc_cfg,
                    sample_history=getattr(clf, "sample_history", None),
                    verbose=verbose)
                classifications = list(cls)
                labels = [c.label for c in cls]
            region = ts.build_local_wagon_region(
                camera_id, segs, labels, tracks.fps,
                classifier_model=os.path.basename(classifier_path),
                unmapped_classes=mapping.unmapped, verbose=verbose)
        except Exception as e:
            notes.append(f"support_classification_failed:{camera_id}:{e}")
            region = ts.LocalWagonRegion(
                camera_id=camera_id,
                classifier_model=os.path.basename(str(classifier_path)),
                reason=f"classification error: {type(e).__name__}: {e}")
    else:
        region = ts.LocalWagonRegion(
            camera_id=camera_id, classifier_model="(none)",
            reason="no classifier available; camera not classified")

    return CameraPipelineResult(
        camera_id=camera_id, tracks=tracks,
        segments=_segments_to_local(camera_id, tracks, classifications),
        classifications=classifications, stitch=sres, validation=vres,
        recovery=None, wagon_region=region, notes=notes)
