"""Phase 1: per-camera evidence collection, shared by both orchestrators.

Batch enters through `master_runner -> process_batch -> reconstruction/runner
-> run_global_count.py`; sequential through `master_runner -> camera_runner ->
global_assembler`. Those entry points and their different arrival behaviour are
deliberately preserved. What must NOT differ is the collection itself, so both
converge here.

The phase boundary is the point of the design:

    Phase 1  raw video in, timestamped observations out. No roster, no wagon
             cache, no canonical gap. Gap detection and feature collection are
             independent producers on the same video timeline.
    Phase 2  `TimelineEvidence.fuse()` assigns every observation to a `GW_n`
             from timestamps alone, once the WAGON-active interval and
             RIGHT_UP's canonical gaps exist.

Collection therefore cannot bias assignment: it does not know what the wagons
are. And assignment cannot lose evidence collected before it ran, because the
observations are held with their times until fusion.

Door, Damage and Load now score the ORIGINAL video in one decode pass per
camera (`features/raw_collect.py`). Two couplings shape how far that goes, and
both are properties of the existing models rather than of this layer:

  * the per-wagon state machines -- Door's tracker resets per wagon, Damage and
    Load vote within one -- stay in Phase 2, fed by the detections collected
    here. Inference still happens ONCE; Phase 2 re-runs no model.
  * classification samples frames inside a SEGMENT, and segments come from that
    camera's gaps, so within one camera it follows gap detection. The four
    cameras remain independent of each other.

OCR is not migrated: it bands its detections and picks which frames to read
from the wagon's LOAD state, so it needs a wagon first. It also stays disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core import constants as C
from core.master_timeline import CameraClock
from core.timeline_evidence import Observation, TimelineEvidence

#: Features collecting from raw video today. Adding one here is the whole
#: migration step -- the fusion side needs no change.
RAW_VIDEO_FEATURES = ("door", "damage", "load")

#: Not yet migrated. OCR bands its detections and picks frames by the
#: wagon's LOAD state, so it needs a wagon before it can choose what to
#: read; it also remains disabled by default.
POST_ROSTER_FEATURES = ("ocr",)


@dataclass
class CollectionResult:
    """What Phase 1 produced, per camera and per feature."""
    observations: List[Observation] = field(default_factory=list)
    per_feature: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)
    #: LocalCameraTracks per camera, produced by the SAME decode
    #: pass as the feature detections. Stage 1's gap output.
    gap_tracks: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.observations)

    def summary_lines(self) -> List[str]:
        out = [f"[PHASE1] ONE decode per camera -> GAP + features. "
               f"{self.count} timestamped observation(s), none assigned"]
        for cam, t in sorted(self.gap_tracks.items()):
            out.append(f"  gap/{cam:<13} {len(t.gaps)} gap event(s) "
                       f"from {t.total_frames} frame(s)")
        for feature, per_cam in sorted(self.per_feature.items()):
            for cam, stats in sorted(per_cam.items()):
                out.append(f"  {feature}/{cam:<13} "
                           f"obs={stats.get('observations', 0)} "
                           f"frames={stats.get('frames_scored', 0)}"
                           + (f"  SKIPPED {stats['skipped']}"
                              if stats.get("skipped") else ""))
        return out


def _merge_strides(strides: Optional[Dict[str, int]],
                   damage_stride: int) -> Optional[Dict[str, int]]:
    """Every feature's stride, not just damage's.

    `damage_stride` is the long-standing explicit argument and still wins for
    damage when given. The rest come from `strides`. Passing a partial dict
    down used to leave the unlisted features on a stride of 1 -- every frame --
    so the two are merged here rather than one replacing the other.
    """
    out: Dict[str, int] = dict(strides or {})
    if damage_stride:
        out["damage"] = int(damage_stride)
    return out or None


def collect_camera_evidence(
    *,
    video_paths: Dict[str, str],
    feature_models_dir: str,
    clocks: Optional[Dict[str, CameraClock]] = None,
    features: Sequence[str] = RAW_VIDEO_FEATURES,
    damage_stride: int = 3,
    strides: Optional[Dict[str, int]] = None,
    models: Optional[Dict[str, Any]] = None,
    gap_trackers: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> CollectionResult:
    """Run every raw-video collector. One implementation, both modes.

    `clocks` carry fps and, when known, each camera's resolved offset. During a
    true Phase-1 run the offsets are NOT yet known -- they are an output of
    fusion -- so observations are recorded on the camera's own clock with the
    frame index retained, and re-projected during fusion once offsets exist.
    That is why `local_frame` is part of every observation rather than a
    convenience field.
    """
    res = CollectionResult()

    from features import raw_collect

    wanted = [f for f in features if f in RAW_VIDEO_FEATURES]
    for cam, path in sorted(video_paths.items()):
        # ONE decode per camera, every enabled detector scoring the same
        # frames, so they share a coordinate system by construction.
        r = raw_collect.collect_camera(
            camera_id=cam, video_path=path,
            feature_models_dir=feature_models_dir, features=wanted,
            clock=(clocks or {}).get(cam),
            strides=_merge_strides(strides, damage_stride),
            models=models, gap_tracker=(gap_trackers or {}).get(cam),
            verbose=verbose)
        res.observations.extend(r.observations)
        for f in r.detectors_run:
            res.per_feature.setdefault(f, {})[cam] = {
                "observations": r.detections.get(f, 0),
                "frames_scored": r.frames_scored.get(f, 0),
                "frames_read": r.frames_read,
            }
        if r.gap_tracks is not None:
            res.gap_tracks[cam] = r.gap_tracks
        if r.skipped:
            res.skipped[cam] = r.skipped

    if verbose:
        for line in res.summary_lines():
            print(line)
    return res


def reproject(observations: Sequence[Observation],
              clocks: Dict[str, CameraClock]) -> List[Observation]:
    """Re-express observations on the master clock, once offsets are known.

    Collection may have run before fusion resolved the offsets, in which case
    the times recorded were the camera's own. This recomputes them from the
    retained `local_frame` -- which is why that field exists -- and leaves an
    observation untouched when its camera has no clock, rather than shifting it
    by a guess.
    """
    out: List[Observation] = []
    for o in observations:
        clock = clocks.get(o.camera_id)
        if clock is None or clock.fps <= 0 or o.local_frame is None:
            out.append(o)
            continue
        t = clock.to_master_time(float(o.local_frame) / clock.fps)
        out.append(Observation(
            camera_id=o.camera_id, kind=o.kind, t_start=t, t_end=t,
            confidence=o.confidence, local_frame=o.local_frame, bbox=o.bbox,
            model=o.model, label=o.label, detected=o.detected,
            payload={**o.payload, "reprojected_offset": clock.offset}))
    return out


def build_timeline_evidence(
    *,
    collection: CollectionResult,
    mode: str,
    clocks: Optional[Dict[str, CameraClock]] = None,
    canonical_gaps: Sequence[float] = (),
    wagon_active: Optional[Dict[str, Any]] = None,
    camera_offsets: Optional[Dict[str, Dict[str, Any]]] = None,
    extra: Sequence[Observation] = (),
) -> TimelineEvidence:
    """Phase-1 output -> the container Phase 2 fuses. No assignment yet."""
    ev = TimelineEvidence(mode=mode)
    obs = list(collection.observations) + list(extra)
    ev.extend(reproject(obs, clocks) if clocks else obs)
    ev.canonical_gaps = [float(t) for t in canonical_gaps]
    ev.wagon_active = wagon_active
    ev.camera_offsets = dict(camera_offsets or {})
    return ev
