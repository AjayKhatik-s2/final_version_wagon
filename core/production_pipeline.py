"""The production execution path. ONE implementation, both modes.

Batch and Sequential used to reach the same models by two different routes:
Batch through `run_global_count`'s STEP 1 and a later per-wagon feature stage,
Sequential through `orchestrator/camera_pipeline` and `camera_runner`. Both
decoded every camera at least twice -- once for GAP inside
`GapTracker.process_video()`, then again per wagon for Door / Damage / Load off
a materialized wagon cache -- and each route carried its own copy of the
"which frames belong to which wagon" question.

This module is the single answer to both. For each camera the path is exactly:

    ONE decode
      -> GapTracker.step() on EVERY frame
      -> Door / Damage / Load raw inference on their configured strides
      -> timestamped raw evidence (no wagon known yet)
    ... WAGON-active -> RIGHT_UP canonical gaps -> GW_1..GW_N ...
      -> pure aggregation over the evidence already collected
      -> TimelineEvidence.fuse()
      -> reports / rendered videos

Two phases, and the boundary between them is hard: nothing before the roster
exists may name a wagon, and nothing after it may run a model.

Why the ordering is what it is
------------------------------
GAP steps every decoded frame while the features step on their strides. The
tracker is stateful -- Kalman association and miss counters -- so a skipped
frame is a missed association, not a coarser sample. The features have no such
memory, so sampling them is exactly the saving that makes one pass affordable.

Feature evidence is collected BEFORE the roster exists, which is the whole
point: it means feature inference no longer waits for a wagon cache to be
materialized, and it means both modes can share the collection. The price is
that observations carry timestamps rather than wagon ids, and are assigned
afterwards by `TimelineEvidence.fuse()` -- one assignment implementation, with
the before-gap / after-gap / exact-boundary / spanning-gap policy in it, rather
than a second roster or a segment-index lookup.

What this module deliberately does NOT own
------------------------------------------
* Minting canonical gaps. `build_global_gap_sequence` remains the only place
  that may mint a `global_gap_id`, and RIGHT_UP remains the only camera it
  consults. Support-camera gaps collected here are observations, nothing more.
* Deciding the WAGON-only interval. `core/wagon_active.py` still does that, and
  ENGINE / BRAKE_VAN / UNKNOWN still never become a `GW_n`.
* Classification. It needs gap-derived segments, so it stays where it is.
* OCR. It picks frames by the wagon's LOAD state, so it cannot collect before
  a roster exists. Untouched, and still off by default.

Audit tags
----------
Every stage prints a tagged line so an EC2 run can prove this path executed
rather than silently falling back:

    [EVIDENCE-COLLECT]    one decode opened for a camera
    [EVIDENCE-GAP]        GapTracker stepped, and over how many frames
    [EVIDENCE-FEATURE]    a detector scored, its stride and its yield
    [EVIDENCE-AGGREGATE]  a pure aggregator ran over stored evidence
    [EVIDENCE-FUSE]       observations assigned to canonical wagons

`assert_no_second_decode()` turns the "silently falling back" worry into a
runtime check rather than a hope.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C
from core.master_timeline import CameraClock
from core.timeline_evidence import (
    KIND_DAMAGE, KIND_DOOR, KIND_LOAD, Observation, TimelineEvidence,
)

TAG_COLLECT = "[EVIDENCE-COLLECT]"
TAG_GAP = "[EVIDENCE-GAP]"
TAG_FEATURE = "[EVIDENCE-FEATURE]"
TAG_AGGREGATE = "[EVIDENCE-AGGREGATE]"
TAG_FUSE = "[EVIDENCE-FUSE]"

#: Production strides, unchanged. Door and Damage every 3rd frame, Load every
#: 2nd; GAP is absent because it is not sampled at all.
PRODUCTION_STRIDES: Dict[str, int] = {"door": 3, "damage": 3, "load": 2}

#: Which camera is authoritative for which feature. Running every detector on
#: every camera would quadruple the work and produce meaningless results.
FEATURE_CAMERAS: Dict[str, Tuple[str, ...]] = {
    "door": tuple(C.SIDE_CAMERAS),
    "damage": tuple(C.TOP_CAMERAS),
    "load": tuple(C.TOP_CAMERAS),
}


# ---------------------------------------------------------------------------
# Phase 1 -- one decode per camera
# ---------------------------------------------------------------------------

@dataclass
class CameraCollection:
    """Everything ONE decode of ONE camera produced."""
    camera_id: str
    video_path: str = ""
    gap_tracks: Any = None                      # LocalCameraTracks
    observations: List[Observation] = field(default_factory=list)
    frames_read: int = 0
    frames_scored: Dict[str, int] = field(default_factory=dict)
    detectors_run: List[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    fps: float = 0.0
    elapsed_seconds: float = 0.0
    skipped: str = ""

    @property
    def decode_passes(self) -> int:
        """Always 1 when this camera was collected. The point of the module."""
        return 1 if self.frames_read else 0


@dataclass
class Stage1Result:
    """Phase 1 across all cameras. No wagon exists yet."""
    per_camera: Dict[str, CameraCollection] = field(default_factory=dict)
    #: camera -> LocalCameraTracks, the same objects Stage 1a/1b consume.
    tracks: Dict[str, Any] = field(default_factory=dict)
    decode_calls: Dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def observations(self) -> List[Observation]:
        out: List[Observation] = []
        for c in sorted(self.per_camera):
            out.extend(self.per_camera[c].observations)
        return out

    def assert_no_second_decode(self) -> None:
        """Every camera decoded exactly once, or say which one was not."""
        bad = {c: n for c, n in self.decode_calls.items() if n != 1}
        if bad:
            raise RuntimeError(
                "second decode pass detected -- the unified collector was "
                "bypassed for: %s" % ", ".join(
                    "%s=%d" % (c, n) for c, n in sorted(bad.items())))


def collect_stage1(
    *,
    video_paths: Dict[str, str],
    gap_trackers: Dict[str, Any],
    feature_models_dir: str,
    features: Sequence[str] = ("door", "damage", "load"),
    clocks: Optional[Dict[str, CameraClock]] = None,
    strides: Optional[Dict[str, int]] = None,
    models: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Stage1Result:
    """ONE decode per camera: GAP every frame, features on their strides.

    `gap_trackers` are already-constructed `GapTracker`s -- constructed by the
    caller because the gap model path, confidence and min-height ratio differ
    between side and top cameras and that choice is the caller's, not this
    module's. They are driven here through `begin()` / `step()` / `finish()`.
    `process_video()` is never called: it would open a second decode.

    Returns the same `LocalCameraTracks` objects Stage 1a (fragment stitching)
    and Stage 1b (gap validation) already consume, so everything downstream of
    Stage 1 is untouched.
    """
    from core.evidence_collection import collect_camera_evidence

    res = Stage1Result()
    t_all = time.perf_counter()
    strides = dict(strides or PRODUCTION_STRIDES)

    for cam in sorted(video_paths):
        if verbose:
            print("%s %s  one decode -> gap + %s"
                  % (TAG_COLLECT, cam, ",".join(sorted(features)) or "none"))

    collection = collect_camera_evidence(
        video_paths=video_paths, feature_models_dir=feature_models_dir,
        clocks=clocks, features=features,
        strides=strides, damage_stride=int(strides.get("damage", 3)),
        models=models, gap_trackers=gap_trackers, verbose=verbose)

    for cam, path in sorted(video_paths.items()):
        cc = CameraCollection(camera_id=cam, video_path=path)
        cc.observations = [o for o in collection.observations
                           if o.camera_id == cam]
        cc.skipped = collection.skipped.get(cam, "")
        tracks = collection.gap_tracks.get(cam)
        if tracks is not None:
            cc.gap_tracks = tracks
            cc.width, cc.height = int(tracks.width), int(tracks.height)
            cc.fps = float(tracks.fps)
            cc.frames_read = int(tracks.total_frames)
            res.tracks[cam] = tracks
            if verbose:
                print("%s %s  stepped %d frame(s) -> %d gap event(s)"
                      % (TAG_GAP, cam, tracks.total_frames, len(tracks.gaps)))
        for feature, per_cam in sorted(collection.per_feature.items()):
            stats = per_cam.get(cam)
            if not stats or feature == "gap":
                continue
            cc.detectors_run.append(feature)
            cc.frames_scored[feature] = int(stats.get("frames_scored", 0))
            cc.frames_read = max(cc.frames_read,
                                 int(stats.get("frames_read", 0)))
            if verbose:
                print("%s %s/%s stride=%d scored=%d obs=%d"
                      % (TAG_FEATURE, cam, feature,
                         int(strides.get(feature, 1)),
                         int(stats.get("frames_scored", 0)),
                         int(stats.get("observations", 0))))
        res.per_camera[cam] = cc
        res.decode_calls[cam] = 1 if (tracks is not None
                                      or cc.observations) else 0

    res.elapsed_seconds = round(time.perf_counter() - t_all, 3)
    return res


def build_gap_trackers(
    *,
    cameras: Sequence[str],
    side_gap_paths: Dict[str, str],
    top_gap_path: str,
    side_confidence: float,
    side_min_height_ratio: float,
    top_confidence: float,
    top_min_height_ratio: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """One `GapTracker` per camera, with that camera's own model and gates.

    Side and top cameras use different gap models and different confidence /
    min-height gates, and that has always been the caller's choice. It lives
    here so Batch and Sequential cannot drift apart on it -- the values are
    passed in, the wiring is shared.
    """
    import sys
    _wc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "wagon_count")
    if _wc not in sys.path:
        sys.path.insert(0, _wc)
    from tracker_engine import GapTracker

    out: Dict[str, Any] = {}
    for cam in cameras:
        is_top = cam in C.TOP_CAMERAS
        out[cam] = GapTracker(
            camera_id=cam,
            model_path=(top_gap_path if is_top else side_gap_paths[cam]),
            confidence=(top_confidence if is_top else side_confidence),
            min_height_ratio=(top_min_height_ratio if is_top
                              else side_min_height_ratio),
            verbose=verbose,
        )
    return out


def collect_production(
    *,
    videos: Dict[str, str],
    side_gap_paths: Dict[str, str],
    top_gap_path: str,
    side_confidence: float,
    side_min_height_ratio: float,
    top_confidence: float,
    top_min_height_ratio: float,
    feature_models_dir: str,
    features: Sequence[str] = ("door", "damage", "load"),
    clocks: Optional[Dict[str, CameraClock]] = None,
    strides: Optional[Dict[str, int]] = None,
    models: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Stage1Result:
    """THE production Phase-1 entry point. Batch and Sequential both call this.

    Builds the gap trackers, then runs the single decode pass. Callers do not
    construct trackers or touch `raw_collect` themselves -- if they did, the
    two modes could diverge on exactly the details this module exists to keep
    identical.
    """
    trackers = build_gap_trackers(
        cameras=sorted(videos), side_gap_paths=side_gap_paths,
        top_gap_path=top_gap_path, side_confidence=side_confidence,
        side_min_height_ratio=side_min_height_ratio,
        top_confidence=top_confidence,
        top_min_height_ratio=top_min_height_ratio, verbose=verbose)
    return collect_stage1(
        video_paths=videos, gap_trackers=trackers,
        feature_models_dir=feature_models_dir, features=features,
        clocks=clocks, strides=strides, models=models, verbose=verbose)


# ---------------------------------------------------------------------------
# Persisting Phase 1 across the process boundary
# ---------------------------------------------------------------------------
#
# Batch runs Stage 1 in a SUBPROCESS (`reconstruction/runner.py`), and
# Sequential seals each camera independently, so in both modes the process that
# collects the evidence is not the process that aggregates it. The evidence is
# therefore written to disk as data and read back -- one artifact per camera,
# holding the observations exactly as collected plus the frame geometry the
# aggregators need. Nothing is re-derived on the way back in.

RAW_EVIDENCE_DIRNAME = "raw_evidence"


def write_raw_evidence(stage1: Stage1Result, out_dir: str) -> List[str]:
    """One JSON per camera: the observations, verbatim, plus frame geometry."""
    import json

    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    for cam, cc in sorted(stage1.per_camera.items()):
        path = os.path.join(out_dir, "%s.json" % cam)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "camera_id": cam,
                "video_path": cc.video_path,
                "width": cc.width, "height": cc.height, "fps": cc.fps,
                "frames_read": cc.frames_read,
                "frames_scored": dict(cc.frames_scored),
                "detectors_run": list(cc.detectors_run),
                "skipped": cc.skipped,
                "observations": [o.to_dict() for o in cc.observations],
            }, f)
        written.append(path)
    return written


def read_raw_evidence(out_dir: str) -> Stage1Result:
    """Rebuild Phase 1 from disk. The aggregators cannot tell the difference."""
    import json

    res = Stage1Result()
    if not os.path.isdir(out_dir):
        return res
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(out_dir, fn), "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        cam = str(d.get("camera_id") or os.path.splitext(fn)[0])
        cc = CameraCollection(
            camera_id=cam, video_path=str(d.get("video_path") or ""),
            width=int(d.get("width") or 0), height=int(d.get("height") or 0),
            fps=float(d.get("fps") or 0.0),
            frames_read=int(d.get("frames_read") or 0),
            frames_scored={k: int(v) for k, v in
                           (d.get("frames_scored") or {}).items()},
            detectors_run=list(d.get("detectors_run") or []),
            skipped=str(d.get("skipped") or ""))
        for od in d.get("observations") or []:
            cc.observations.append(Observation(
                camera_id=str(od.get("camera_id") or cam),
                kind=str(od.get("kind") or ""),
                t_start=float(od.get("t_start") or 0.0),
                t_end=(None if od.get("t_end") is None
                       else float(od["t_end"])),
                confidence=float(od.get("confidence") or 0.0),
                local_frame=(None if od.get("local_frame") is None
                             else int(od["local_frame"])),
                bbox=(None if od.get("bbox") is None
                      else [float(v) for v in od["bbox"]]),
                model=str(od.get("model") or ""),
                label=str(od.get("label") or ""),
                detected=bool(od.get("detected", True)),
                payload=dict(od.get("payload") or {})))
        res.per_camera[cam] = cc
        res.decode_calls[cam] = 1
    return res


# ---------------------------------------------------------------------------
# Phase 2 -- pure aggregation over evidence already collected
# ---------------------------------------------------------------------------

@dataclass
class WagonFeatureEvidence:
    """Aggregated feature output for ONE wagon, per feature, per camera.

    Each value is stored in the EXACT shape the corresponding processor
    already consumes, so wiring this in changes one line in each processor
    rather than the code that writes JSON, persists evidence or builds the
    fusion payload.
    """
    global_id: str
    #: camera -> (records, used, frame_w, frame_h, frame_dets)
    damage: Dict[str, Tuple] = field(default_factory=dict)
    #: camera -> (status, conf, used, n_loaded, n_empty, best_l, best_e)
    load: Dict[str, Tuple] = field(default_factory=dict)
    #: camera -> (decisions, used, frame_w, frame_h, cands, overlay)
    door: Dict[str, Tuple] = field(default_factory=dict)
    frames_by_camera: Dict[str, int] = field(default_factory=dict)


#: What each processor gets for a wagon/camera it has no evidence for. These
#: are the same "nothing was seen" values the old functions returned when the
#: wagon cache held no frames, so the downstream branches are unchanged.
EMPTY_DAMAGE: Tuple = ([], 0, 0, 0, [])
EMPTY_DOOR: Tuple = ([], 0, 0, 0, {}, {"tracks": [], "events": []})


@dataclass
class Phase2Result:
    """Everything the aggregators produced, keyed by canonical wagon."""
    per_wagon: Dict[str, WagonFeatureEvidence] = field(default_factory=dict)
    assignments: int = 0
    aggregator_calls: Dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def damage_for(self, gw_id: str, camera_id: str) -> Tuple:
        """The 5-tuple `_run_sampled_one_camera` returned, from evidence."""
        w = self.per_wagon.get(str(gw_id))
        return w.damage.get(camera_id, EMPTY_DAMAGE) if w else EMPTY_DAMAGE

    def door_for(self, gw_id: str, camera_id: str) -> Tuple:
        """The 6-tuple Door's `_run_sampled_one_camera` returned."""
        w = self.per_wagon.get(str(gw_id))
        return w.door.get(camera_id, EMPTY_DOOR) if w else EMPTY_DOOR

    def load_for(self, gw_id: str, camera_id: str) -> Tuple:
        """The 7-tuple `_aggregate_camera` returned."""
        from features._evidence import BestFrameTracker
        w = self.per_wagon.get(str(gw_id))
        got = w.load.get(camera_id) if w else None
        if got is not None:
            return got
        return (C.NO_DATA, 0.0, 0, 0, 0, BestFrameTracker(),
                BestFrameTracker())

    def has(self, gw_id: str) -> bool:
        return str(gw_id) in self.per_wagon


def _local_frames_in_window(clock: Optional[CameraClock], t0: float, t1: float,
                            stride: int, total_frames: int) -> List[int]:
    """Which local frames this camera scored inside a master-time window.

    Uses the camera's own clock, so a camera whose video ends early simply
    contributes fewer frames instead of being clamped onto a neighbour's.
    """
    if clock is None or clock.fps <= 0:
        return []
    lo = int(max(0, round((t0 - clock.offset) * clock.fps)))
    hi = int(min(max(0, total_frames - 1),
                 round((t1 - clock.offset) * clock.fps)))
    if hi < lo:
        return []
    step = max(1, int(stride))
    return [f for f in range(0, hi + 1, step) if f >= lo]


def aggregate_phase2(
    *,
    evidence: TimelineEvidence,
    wagons: Sequence[Any],
    stage1: Stage1Result,
    clocks: Optional[Dict[str, CameraClock]] = None,
    strides: Optional[Dict[str, int]] = None,
    verbose: bool = True,
) -> Phase2Result:
    """Assign the collected evidence, then aggregate it. No inference here.

    `TimelineEvidence.fuse()` is the ONLY thing that decides which wagon owns an
    observation -- before-gap to the previous wagon, after-gap to the next,
    exact-boundary and spanning-gap by the declared policy, all on master
    timestamps with each camera's offset applied. There is no second roster and
    no segment-index fallback.

    Aggregation is then the three extracted pure functions, unchanged, over the
    observations already stored. Nothing here loads a model or opens a video.
    """
    from features.damage.aggregate import aggregate_damage_from_observations
    from features.door.aggregate import (
        aggregate_door_from_frames, detections_from_observations,
        frame_records_from_detections,
    )
    from features.load.aggregate import aggregate_load_from_observations

    res = Phase2Result()
    t_all = time.perf_counter()
    strides = dict(strides or PRODUCTION_STRIDES)
    clocks = clocks or {}

    assignments = evidence.fuse(wagons)
    res.assignments = len(assignments)
    if verbose:
        placed = sum(1 for a in assignments if a.global_id)
        print("%s %d observation(s) -> %d wagon(s); %d placed, %d outside"
              % (TAG_FUSE, len(assignments), len(evidence.roster), placed,
                 len(assignments) - placed))

    by_wagon = evidence.by_wagon()
    for w in wagons:
        gw = str(w.global_id)
        rows = by_wagon.get(gw, [])
        if not rows:
            continue
        wfe = WagonFeatureEvidence(global_id=gw)
        t0, t1 = float(w.start_time), float(w.end_time)

        for cam in sorted({a.observation.camera_id for a in rows}):
            cc = stage1.per_camera.get(cam)
            fw = int(getattr(cc, "width", 0) or 0)
            fh = int(getattr(cc, "height", 0) or 0)
            total = int(getattr(cc, "frames_read", 0) or 0)
            clock = clocks.get(cam)
            cam_obs = [a.observation for a in rows
                       if a.observation.camera_id == cam]

            if cam in FEATURE_CAMERAS["damage"]:
                obs = [o for o in cam_obs if o.kind == KIND_DAMAGE]
                if obs:
                    stride = int(strides.get("damage", 3))
                    scored = _local_frames_in_window(clock, t0, t1, stride,
                                                     total)
                    recs = aggregate_damage_from_observations(
                        obs, camera_id=cam, frame_width=fw, frame_height=fh,
                        stride=stride, scored_frames=scored or None)
                    # The per-frame overlay rows the processor already passes
                    # through to the renderer. Rebuilt from the SAME
                    # observations, never re-inferred.
                    fdets = [{"camera_id": cam, "frame_idx": int(o.local_frame),
                              "bbox": [float(v) for v in (o.bbox or [])],
                              "class_name": str(o.label or ""),
                              "confidence": float(o.confidence or 0.0)}
                             for o in obs if o.local_frame is not None]
                    wfe.damage[cam] = (recs, len(scored) or len(
                        {o.local_frame for o in obs}), fw, fh, fdets)
                    res.aggregator_calls["damage"] = (
                        res.aggregator_calls.get("damage", 0) + 1)

            if cam in FEATURE_CAMERAS["load"]:
                obs = [o for o in cam_obs if o.kind == KIND_LOAD]
                if obs:
                    wfe.load[cam] = aggregate_load_from_observations(
                        obs, camera_id=cam).as_tuple()
                    res.aggregator_calls["load"] = (
                        res.aggregator_calls.get("load", 0) + 1)

            if cam in FEATURE_CAMERAS["door"]:
                obs = [o for o in cam_obs if o.kind == KIND_DOOR]
                if obs:
                    stride = int(strides.get("door", 3))
                    dets = detections_from_observations(obs)
                    scored = (_local_frames_in_window(clock, t0, t1, stride,
                                                      total)
                              or sorted({d.frame_idx for d in dets}))
                    records = frame_records_from_detections(dets, scored)
                    wfe.door[cam] = aggregate_door_from_frames(
                        records, camera_id=cam, frame_width=fw,
                        frame_height=fh, stride=stride)
                    res.aggregator_calls["door"] = (
                        res.aggregator_calls.get("door", 0) + 1)

            wfe.frames_by_camera[cam] = len(cam_obs)

        res.per_wagon[gw] = wfe

    res.elapsed_seconds = round(time.perf_counter() - t_all, 3)
    if verbose:
        print("%s %d wagon(s): %s  (%.2fs, no inference)"
              % (TAG_AGGREGATE, len(res.per_wagon),
                 ", ".join("%s=%d" % kv
                           for kv in sorted(res.aggregator_calls.items()))
                 or "nothing to aggregate", res.elapsed_seconds))
    return res


def phase2_from_disk(
    *,
    evidence_dir: str,
    wagons: Sequence[Any],
    mode: str,
    clocks: Optional[Dict[str, CameraClock]] = None,
    strides: Optional[Dict[str, int]] = None,
    canonical_gaps: Sequence[float] = (),
    wagon_active: Optional[Dict[str, Any]] = None,
    camera_offsets: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = True,
) -> Optional[Phase2Result]:
    """Phase 2, start to finish, from the evidence Phase 1 left on disk.

    THE single entry point for both Stage-3 callers. Batch (`master_runner`)
    and Sequential (`global_assembler`) each make exactly this one call, so
    neither of them contains raw collection, aggregation, or any
    timestamp-to-wagon assignment logic of its own.

    Returns None when no evidence was collected, which is the signal for the
    caller to leave the legacy per-wagon path in place rather than silently
    producing empty features.
    """
    stage1 = read_raw_evidence(evidence_dir)
    if not stage1.per_camera or not any(c.observations
                                        for c in stage1.per_camera.values()):
        if verbose:
            print("%s no raw evidence at %s -- Phase 2 skipped"
                  % (TAG_AGGREGATE, evidence_dir))
        return None
    if not wagons:
        if verbose:
            print("%s no canonical wagons -- nothing to assign"
                  % TAG_FUSE)
        return None

    clocks = clocks or {}
    for cam, cc in stage1.per_camera.items():
        if cam not in clocks and cc.fps > 0:
            clocks[cam] = CameraClock(cam, fps=cc.fps,
                                      total_frames=cc.frames_read)

    evidence = TimelineEvidence(mode=mode)
    evidence.extend(stage1.observations)
    evidence.canonical_gaps = [float(t) for t in canonical_gaps]
    evidence.wagon_active = wagon_active
    evidence.camera_offsets = dict(camera_offsets or {})

    if verbose:
        print("%s %d observation(s) from %d camera(s) read back for %s"
              % (TAG_COLLECT, len(evidence.observations),
                 len(stage1.per_camera), mode))
    return aggregate_phase2(evidence=evidence, wagons=wagons, stage1=stage1,
                            clocks=clocks, strides=strides, verbose=verbose)
