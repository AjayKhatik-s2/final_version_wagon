"""Engine (locomotive) frame extraction -- a TRAIN-LEVEL evidence stream.

An engine is not a wagon. The counting engine already knows this: master
classification labels each master segment, and `train_structure`'s wagon window
keeps ENGINE / BRAKE_VAN segments OUT of the roster, recording them as
`NonWagonObject`s under `wagon_window` instead. They never receive a `GW_n`.

This module reuses that existing decision rather than re-deriving it. It reads
the ENGINE segments the counting engine already identified, pulls the best
frames for them from the two SIDE cameras, and writes them to their own tree:

    engine_frames/RIGHT_UP/engine_001.jpg ... + manifest.json
    engine_frames/LEFT_UP/engine_001.jpg  ... + manifest.json

Deliberately kept out of everything wagon-shaped. These frames are never given
a wagon id, never appended to a local or global timeline, never counted, never
fused, never mapped, and never handed to the Door / Damage / Load processors.
The only thing that changes anywhere else is that this directory now exists.

No inference runs here, and no second full-video pass: the engine occupies a
short span at one end of the train, so only that span is decoded. Ranking uses
the existing `core.frame_quality.detection_quality` brightness-plus-Laplacian
scorer, applied to the whole frame, so the choice is deterministic and shares
the pipeline's notion of a good frame.

Shared by BOTH modes -- `master_runner.process_batch` (batch) and
`global_assembler.assemble` (sequential) call this same function, so the two
produce the same engine evidence for the same input.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

from core import constants as C
from core.frame_quality import detection_quality

#: Engines are photographed by the SIDE cameras -- the top cameras look down at
#: the roof, where a locomotive number never appears.
ENGINE_CAMERAS: Tuple[str, ...] = C.SIDE_CAMERAS

#: Per side. Five each, so at most ten per train.
MAX_FRAMES_PER_SIDE = 5

DIR_NAME = "engine_frames"
MANIFEST_NAME = "manifest.json"
SCHEMA = "wagon_eye.engine_frames.v1"


@dataclass
class EngineFrame:
    """One stored engine frame, with everything a later OCR pass needs."""
    camera_id: str
    rank: int                       # 1 = best
    source_frame_index: int         # absolute index in THIS camera's video
    timestamp: float                # seconds into this camera's video
    master_time: float              # same instant on the master clock
    score: float
    path: str
    source_video: str
    segment_index: int              # which non-wagon segment it came from
    classification: str             # ENGINE
    classification_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "rank": self.rank,
            "source_frame_index": self.source_frame_index,
            "timestamp": round(self.timestamp, 4),
            "master_time": round(self.master_time, 4),
            "score": round(self.score, 6),
            "path": self.path,
            "source_video": self.source_video,
            "segment_index": self.segment_index,
            "classification": self.classification,
            "classification_confidence": round(self.classification_confidence, 4),
        }


@dataclass
class EngineFrameResult:
    frames_by_camera: Dict[str, List[EngineFrame]] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)
    engine_segments: int = 0
    root: str = ""

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "root": self.root,
            "engine_segments": self.engine_segments,
            "counts": dict(self.counts),
            "total": self.total,
            "skipped": dict(self.skipped),
        }


def engine_segments(state: Any) -> List[Dict[str, Any]]:
    """The ENGINE segments the counting engine already identified.

    Read straight out of `wagon_window`, where `get_master_wagon_window()`
    records every segment it excluded from the roster. Nothing is
    re-classified and no threshold is applied -- if the counter did not call it
    an engine, it is not one here either.
    """
    window = getattr(state, "wagon_window", None) or {}
    out: List[Dict[str, Any]] = []
    for bucket in ("leading_non_wagon_objects", "interior_non_wagon_objects",
                   "trailing_non_wagon_objects"):
        for obj in (window.get(bucket) or []):
            if not isinstance(obj, dict):
                continue
            if str(obj.get("classification") or "") != C.CLASS_ENGINE:
                continue
            out.append(obj)
    out.sort(key=lambda o: (float(o.get("start_time") or 0.0),
                            int(o.get("segment_index") or 0)))
    return out


def _local_range(seg: Dict[str, Any], fps: float, offset: float,
                 total_frames: int) -> Tuple[int, int]:
    """Master segment -> this camera's own frame range.

    Same arithmetic the materializer uses for a wagon
    (`local = round((t_master - delta) * local_fps)`), so an engine span is
    projected onto a camera exactly the way a wagon span is.
    """
    if fps <= 0:
        return (0, -1)
    sf = int(round((float(seg.get("start_time") or 0.0) - offset) * fps))
    ef = int(round((float(seg.get("end_time") or 0.0) - offset) * fps)) - 1
    if total_frames > 0:
        sf = max(0, min(total_frames - 1, sf))
        ef = max(0, min(total_frames - 1, ef))
    else:
        sf, ef = max(0, sf), max(0, ef)
    return (sf, ef)


def _score_frame(frame) -> float:
    """Whole-frame quality, via the pipeline's existing scorer.

    `detection_quality` wants a bbox; the bbox here is the frame itself, which
    reduces it to the brightness-and-Laplacian-texture term. Blurry or blown-out
    frames sink; a sharp, well-exposed one rises. Deterministic for a given
    frame, so the chosen five never wobble between runs.
    """
    if frame is None:
        return 0.0
    h, w = frame.shape[:2]
    return float(detection_quality(frame, [0, 0, w, h], pad=0))


def _collect_one_camera(
    *, camera_id: str, video_path: str, fps: float, offset: float,
    segments: Sequence[Dict[str, Any]], out_dir: str,
    max_frames: int, jpeg_quality: int, verbose: bool,
) -> Tuple[List[EngineFrame], Optional[str]]:
    """Best `max_frames` engine frames from ONE camera. Returns (frames, skip)."""
    if not video_path or not os.path.exists(video_path):
        return [], f"video unavailable: {video_path!r}"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], f"cv2 could not open {video_path!r}"
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # frame_idx -> the segment that claims it. Overlapping engine segments
        # cannot double-count a frame.
        claim: Dict[int, Dict[str, Any]] = {}
        for seg in segments:
            sf, ef = _local_range(seg, fps, offset, total)
            if ef < sf:
                continue
            for f in range(sf, ef + 1):
                claim.setdefault(f, seg)
        if not claim:
            return [], "no engine frames land inside this camera's timeline"

        lo, hi = min(claim), max(claim)
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        scored: List[Tuple[float, int, Dict[str, Any], Any]] = []
        idx = lo
        while idx <= hi:
            ok, frame = cap.read()
            if not ok:
                break
            seg = claim.get(idx)
            if seg is not None:
                scored.append((_score_frame(frame), idx, seg, frame))
            idx += 1
        if not scored:
            return [], "no frame decoded in the engine span"

        # Highest score first; ties broken by the earlier frame so the choice is
        # reproducible rather than dependent on decode order.
        scored.sort(key=lambda t: (-t[0], t[1]))
        keep = scored[:max_frames]

        os.makedirs(out_dir, exist_ok=True)
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        out: List[EngineFrame] = []
        for rank, (score, frame_idx, seg, frame) in enumerate(keep, start=1):
            name = f"engine_{rank:03d}.jpg"
            path = os.path.join(out_dir, name)
            if not cv2.imwrite(path, frame, params):
                continue
            out.append(EngineFrame(
                camera_id=camera_id, rank=rank,
                source_frame_index=int(frame_idx),
                timestamp=(frame_idx / fps) if fps > 0 else 0.0,
                master_time=((frame_idx / fps) + offset) if fps > 0 else 0.0,
                score=float(score), path=path, source_video=video_path,
                segment_index=int(seg.get("segment_index") or 0),
                classification=str(seg.get("classification") or C.CLASS_ENGINE),
                classification_confidence=float(
                    seg.get("classification_confidence") or 0.0),
            ))
        if verbose:
            print(f"[ENGINE/{camera_id}] {len(out)} frame(s) from "
                  f"{len(scored)} candidate(s) in span {lo}-{hi}")
        return out, None
    finally:
        cap.release()


def extract(
    *,
    state: Any,
    video_paths: Dict[str, str],
    per_camera_fps: Dict[str, float],
    output_root: str,
    camera_offsets: Optional[Dict[str, float]] = None,
    cameras: Sequence[str] = ENGINE_CAMERAS,
    max_frames_per_side: int = MAX_FRAMES_PER_SIDE,
    jpeg_quality: int = C.JPEG_QUALITY,
    verbose: bool = True,
) -> EngineFrameResult:
    """Store the best engine frames per SIDE camera, as train-level evidence.

    Each camera is handled on its own: a missing or unreadable video for one
    side is recorded in `skipped` and the other side still runs, and the top
    cameras being absent entirely is irrelevant here. Fewer than
    `max_frames_per_side` candidates stores fewer frames and reports the real
    count -- nothing is ever duplicated to reach five.
    """
    res = EngineFrameResult(root=os.path.join(output_root, DIR_NAME))
    segs = engine_segments(state)
    res.engine_segments = len(segs)
    if not segs:
        if verbose:
            print("[ENGINE] no ENGINE segment in the wagon window -- "
                  "nothing to extract")
        return res

    offsets = dict(camera_offsets or {})
    for cam in cameras:
        frames, skip = _collect_one_camera(
            camera_id=cam, video_path=video_paths.get(cam, ""),
            fps=float(per_camera_fps.get(cam) or 0.0),
            offset=float(offsets.get(cam) or 0.0),
            segments=segs, out_dir=os.path.join(res.root, cam),
            max_frames=int(max_frames_per_side),
            jpeg_quality=jpeg_quality, verbose=verbose)
        if skip:
            res.skipped[cam] = skip
            res.counts[cam] = 0
            if verbose:
                print(f"[ENGINE/{cam}] skipped -- {skip}")
            continue
        res.frames_by_camera[cam] = frames
        res.counts[cam] = len(frames)
        _write_manifest(os.path.join(res.root, cam), cam, frames,
                        max_frames_per_side)

    if verbose:
        print(f"[ENGINE] stored {res.total} frame(s) "
              f"{dict(res.counts)} under {res.root}")
    return res


def _write_manifest(out_dir: str, camera_id: str, frames: List[EngineFrame],
                    requested: int) -> str:
    """Per-camera manifest. `count` is what was actually stored, not the target."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "schema": SCHEMA,
            "camera_id": camera_id,
            "requested": int(requested),
            "count": len(frames),
            "complete": len(frames) >= int(requested),
            "frames": [fr.to_dict() for fr in frames],
        }, f, indent=2, default=str)
    return path


def read_manifest(root: str, camera_id: str) -> Dict[str, Any]:
    """Read one camera's engine frames back. Retrieval is per camera by design."""
    p = os.path.join(root, camera_id, MANIFEST_NAME)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}
