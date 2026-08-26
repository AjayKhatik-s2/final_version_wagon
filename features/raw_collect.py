"""One decode pass per camera, every detector scoring the same frames.

The unified Phase-1 collector. It opens the ORIGINAL camera video once and
hands each frame to every enabled detector, so Door, Damage and Load all
observe the same frame indices and the same timestamps by construction rather
than by agreement. No wagon cache, no `GW_n`, no roster.

Two couplings in the existing design shape this, and both are real rather than
incidental:

  the trackers are per-wagon, not per-video
    `features/door/processor.py` resets its tracker for every `(gw, camera)`
    -- "Wagons are independent, so each one resets the tracker" -- and Damage
    and Load likewise vote within a wagon. Running those state machines across
    a whole video would not be a change of frame source, it would be a change
    of model behaviour: door tracks would persist across couplings and the
    hysteresis would smear one wagon's state into the next.

    So Phase 1 collects per-frame DETECTIONS and Phase 2 runs the existing
    per-wagon aggregation over them, grouped by timestamp. The model still sees
    exactly one wagon's frames; they are selected by time instead of by
    directory. Inference happens ONCE -- Phase 2 re-runs no model.

  classification needs segments, and segments come from gaps
    `MasterClassifier.classify_segments` samples frames inside a segment and
    votes. There are no segments before that camera's gaps exist, so within a
    single camera classification follows gap detection. Across cameras the four
    pipelines are independent and can run concurrently.

The honest ordering is therefore per camera:

    decode once -> {gap tracker, door/damage/load per-frame detections}
                -> segments from this camera's gaps -> classification

and only then, across all four cameras, the WAGON-active interval, the
canonical gaps and the roster.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from core import constants as C
from core.master_timeline import CameraClock
from core.timeline_evidence import (
    KIND_DAMAGE, KIND_DOOR, KIND_LOAD, Observation,
)

#: Which cameras each detector is authoritative for -- unchanged from the
#: existing feature plan. A detector is never run on a camera whose view it was
#: not trained for.
DETECTOR_CAMERAS: Dict[str, Sequence[str]] = {
    "door": C.SIDE_CAMERAS,
    "damage": C.TOP_CAMERAS,
    "load": C.TOP_CAMERAS,
}

#: Sampling strides, matching the batch path so both see the same frames.
DEFAULT_STRIDES = {"door": 3, "damage": 3, "load": 2}

MODEL_FILES = {"door": C.MODEL_DOOR_STATE, "damage": C.MODEL_DAMAGE,
               "load": C.MODEL_LOADED}


@dataclass
class RawCameraEvidence:
    """One camera's Phase-1 output. Timestamped, unassigned."""
    camera_id: str
    observations: List[Observation] = field(default_factory=list)
    frames_read: int = 0
    frames_scored: Dict[str, int] = field(default_factory=dict)
    detections: Dict[str, int] = field(default_factory=dict)
    fps: float = 0.0
    total_frames: int = 0
    elapsed_seconds: float = 0.0
    skipped: str = ""
    detectors_run: List[str] = field(default_factory=list)
    gap_tracks: Any = None          # LocalCameraTracks, when GAP ran
    gap_events: int = 0

    @property
    def ok(self) -> bool:
        return not self.skipped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id, "fps": self.fps,
            "total_frames": self.total_frames,
            "observations": len(self.observations),
            "frames_read": self.frames_read,
            "frames_scored": dict(self.frames_scored),
            "detections": dict(self.detections),
            "detectors_run": list(self.detectors_run),
            "gap_events": self.gap_events,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "skipped": self.skipped,
        }


def _load(feature: str, models_dir: str, verbose: bool):
    from features._common import load_yolo
    path = os.path.join(models_dir, MODEL_FILES[feature])
    if not os.path.isfile(path):
        return None
    return load_yolo(path, verbose=verbose)


def _score_door(model, frame, fi, cam, clock, conf):
    """Per-frame door detections, in the shape the extracted aggregator eats.

    This is the collection HALF of `_run_sampled_one_camera`, lifted verbatim:
    raw YOLO boxes, the same `confs >= closed_confidence_threshold` gate, the
    raw class name lowercased, and `detection_quality(frame, bbox)`.

    Two details are deliberate. The gate is the tracker config's
    `closed_confidence_threshold` (0.68), NOT `C.CONF_DOOR` (0.40) -- the
    sampled path has always gated on the former, and collecting at the looser
    threshold would hand the aggregator boxes the old path never saw. And
    `crop_quality` is computed HERE because it is the only value in the whole
    Door path that reads pixels; everything downstream of it is pure and lives
    in `features/door/aggregate.py`.

    An observation is emitted per kept box, so grouping back into per-frame
    records is exact. Frames scored with nothing kept produce no observation;
    that is the same "declared empty" case the aggregator handles.
    """
    from core.frame_quality import detection_quality

    try:
        res = model(frame, verbose=False)[0]
    except Exception:
        return []
    if res.boxes is None or len(res.boxes) == 0:
        return []
    h, w = frame.shape[:2]
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    keep = confs >= conf
    boxes, confs, clss = boxes[keep], confs[keep], clss[keep]
    if len(boxes) == 0:
        return []
    names = getattr(model, "names", {}) or {}
    t = clock.to_master_time(fi / clock.fps)
    out = []
    for bb, cf, ci in zip(boxes, confs, clss):
        bl = [float(x) for x in bb]
        out.append(Observation(
            camera_id=cam, kind=KIND_DOOR, t_start=t, t_end=t,
            confidence=float(cf), local_frame=int(fi), bbox=bl,
            model=MODEL_FILES["door"],
            label=str(names.get(int(ci), "")).lower(),
            payload={"local_time": round(fi / clock.fps, 4),
                     "crop_quality": float(detection_quality(frame, bl)),
                     "frame_width": int(w), "frame_height": int(h)}))
    return out


def _score_damage(model, frame, fi, cam, clock, conf):
    """Per-frame damage detections, through the batch top-camera filter."""
    from features.damage.processor import _filter_detections_for_top

    try:
        res = model(frame, verbose=False)[0]
    except Exception:
        return []
    if res.boxes is None or len(res.boxes) == 0:
        return []
    h, w = frame.shape[:2]
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    names = getattr(model, "names", {}) or {}
    boxes, confs, clss = _filter_detections_for_top(
        boxes, confs, clss, names, w, h, conf)
    t = clock.to_master_time(fi / clock.fps)
    return [Observation(
        camera_id=cam, kind=KIND_DAMAGE, t_start=t, t_end=t,
        confidence=float(cf), local_frame=int(fi),
        bbox=[float(b) for b in bb], model=MODEL_FILES["damage"],
        label=str(names.get(int(ci), "")).lower(),
        payload={"local_time": round(fi / clock.fps, 4)})
        for bb, cf, ci in zip(boxes, confs, clss)]


def _score_load(model, frame, fi, cam, clock, conf):
    """Per-frame load classification. The LOADED/EMPTY vote happens in Phase 2.

    Uses `run_classification`, which is exactly what `_aggregate_camera` calls
    -- it reads `res.probs` and falls back to the highest-confidence box for
    the "classification" models that still emit boxes. Reimplementing that
    fallback here would be a second, drifting copy.

    EVERY scored frame emits ONE observation, including a frame whose label is
    empty or canonicalises to NO_DATA. That is not tidiness: the legacy vote
    divides the loaded count by `frames_used`, i.e. every frame the classifier
    LOOKED at, so dropping unvotable frames here would shrink the denominator
    and flip wagons from EMPTY to LOADED. `conf` is deliberately unused for the
    same reason -- the old path applied no confidence gate before voting.

    The RAW label travels; canonicalisation happens once, in the aggregator.
    """
    from features._common import run_classification

    try:
        cls, cf = run_classification(model, frame)
    except Exception:
        return []
    t = clock.to_master_time(fi / clock.fps)
    return [Observation(
        camera_id=cam, kind=KIND_LOAD, t_start=t, t_end=t,
        confidence=float(cf), local_frame=int(fi), bbox=None,
        model=MODEL_FILES["load"], label=str(cls or ""),
        payload={"raw_class": str(cls or ""),
                 "local_time": round(fi / clock.fps, 4)})]


_SCORERS: Dict[str, Callable] = {"door": _score_door, "damage": _score_damage,
                                 "load": _score_load}
def _door_min_conf() -> float:
    """The gate the sampled Door path actually applies.

    `_run_sampled_one_camera` reads `tracker_config.closed_confidence_threshold`
    (0.68), not `C.CONF_DOOR` (0.40). Taken from TrackerConfig rather than
    hard-coded so the two cannot drift apart.
    """
    try:
        from features.inference_lib.door_tracker import TrackerConfig
        return float(TrackerConfig().closed_confidence_threshold)
    except Exception:
        return float(C.CONF_DOOR)


#: Load applies NO gate before voting -- see `_score_load`.
_CONF = {"door": _door_min_conf(), "damage": C.CONF_DAMAGE, "load": 0.0}


def collect_camera(
    *,
    camera_id: str,
    video_path: str,
    feature_models_dir: str,
    features: Sequence[str] = ("door", "damage", "load"),
    clock: Optional[CameraClock] = None,
    strides: Optional[Dict[str, int]] = None,
    models: Optional[Dict[str, Any]] = None,
    gap_tracker: Any = None,
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> RawCameraEvidence:
    """Score one camera's raw video with every enabled detector, in one pass.

    `gap_tracker` is an already-constructed `GapTracker`. When given, the SAME
    decoded frame that feeds Door / Damage / Load is also handed to
    `GapTracker.step()`, so a camera is decoded exactly once for everything.
    GAP steps on EVERY frame regardless of the feature strides -- its Kalman
    association and miss counters need an unbroken frame sequence, and skipping
    frames would change gap detection rather than merely sample it.

    Feature flags are honoured here: a disabled detector is simply not built,
    and the decode still happens for the others. With no feature enabled and
    only a gap tracker supplied, this becomes exactly the old gap pass -- one
    decode, same result.
    """
    import cv2

    res = RawCameraEvidence(camera_id=camera_id)
    t0 = time.time()

    if not video_path or not os.path.exists(video_path):
        res.skipped = f"video unavailable: {video_path!r}"
        return res

    wanted = [f for f in features
              if f in _SCORERS and camera_id in DETECTOR_CAMERAS.get(f, ())]
    if not wanted and gap_tracker is None:
        res.skipped = (f"no enabled detector applies to {camera_id} "
                       f"(requested {list(features)})")
        return res

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        res.skipped = f"cv2 could not open {video_path!r}"
        return res

    fps = float(clock.fps) if clock and clock.fps > 0 else float(
        cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        cap.release()
        res.skipped = "no fps available; timestamps would be meaningless"
        return res
    clock = CameraClock(camera_id=camera_id, fps=fps, total_frames=total,
                        offset=(clock.offset if clock else 0.0),
                        offset_status=(clock.offset_status if clock else ""))
    res.fps, res.total_frames = fps, total

    built = {}
    for f in wanted:
        m = (models or {}).get(f) or _load(f, feature_models_dir, verbose)
        if m is None:
            res.skipped = f"{res.skipped} {f}:model-missing".strip()
            continue
        built[f] = m
        res.frames_scored[f] = 0
        res.detections[f] = 0
    res.detectors_run = sorted(built)
    if gap_tracker is not None:
        res.detectors_run = sorted(res.detectors_run + ["gap"])
    if not built and gap_tracker is None:
        cap.release()
        res.skipped = res.skipped or "no detector could be loaded"
        return res

    # MERGED over the defaults, not substituted for them. A caller passing
    # only {"damage": 3} used to leave door and load on `.get(f, 1)` -- every
    # single frame -- which is 3x the intended Door work and 2x the Load work,
    # silently, with no wrong answer to notice.
    step = {f: max(1, int({**DEFAULT_STRIDES, **(strides or {})}.get(f, 1)))
            for f in built}
    if gap_tracker is not None:
        gap_tracker.begin(keep_raw_detections=True)

    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res.frames_read += 1
            if max_frames is not None and fi >= max_frames:
                break

            # GAP first, and on EVERY frame: the tracker is stateful and a
            # skipped frame is a missed association, not a coarser sample.
            if gap_tracker is not None:
                gap_tracker.step(fi, frame, frame_h=frame.shape[0])

            for f, model in built.items():
                if fi % step[f]:
                    continue
                res.frames_scored[f] += 1
                obs = _SCORERS[f](model, frame, fi, camera_id, clock, _CONF[f])
                # Provenance every observation must carry, stamped in one place
                # so no detector can forget a field.
                for o in obs:
                    o.payload.update({"fps": clock.fps, "stride": step[f],
                                      "source_video": video_path,
                                      "offset_applied": clock.offset,
                                      "detector": f})
                res.observations.extend(obs)
                res.detections[f] += len(obs)
            fi += 1
    finally:
        cap.release()

    if gap_tracker is not None:
        from core.timeline_evidence import observations_from_gaps

        res.gap_tracks = gap_tracker.finish(
            video_path=video_path, fps=fps, width=int(
                cap_w or 0), height=int(cap_h or 0),
            total_frames_meta=total)
        res.gap_events = len(res.gap_tracks.gaps)
        # Gap events as timestamped observations, on the same clock as the
        # feature detections from the same frames. No wagon is named: these
        # are evidence, and RIGHT_UP's authority is applied later.
        res.observations.extend(observations_from_gaps(
            res.gap_tracks.gaps, camera_id, clock=clock, detected=True,
            model="gap"))

    res.elapsed_seconds = time.time() - t0
    if verbose:
        print(f"[PHASE1/{camera_id}] one decode pass, {res.frames_read} "
              f"frame(s): " + ", ".join(
                  f"{f}={res.detections[f]}det/{res.frames_scored[f]}scored"
                  for f in res.detectors_run if f in res.detections)
              + (f", gap={res.gap_events}events"
                 if gap_tracker is not None else "")
              + f"  ({res.elapsed_seconds:.1f}s)")
    return res
