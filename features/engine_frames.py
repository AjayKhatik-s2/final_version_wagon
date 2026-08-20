"""Capture the best ENGINE/LOCO frames as a TRAIN-LEVEL asset.

Up to five frames from RIGHT_UP and five from LEFT_UP -- ten per train when
enough valid frames exist -- kept for future locomotive-number detection.

What these frames are NOT
------------------------
They are not wagons, and nothing here can make them one:

* no `GW_` id and no `L_<CAM>_<n>` id is minted;
* nothing is written under `wagon_cache/` or `camera_cache/`;
* no `LocalSegment` is created, renamed, reclassified or removed;
* `bundle.write_segments()` is never called;
* `as_feature_wagons()` is never called, so no feature processor ever sees
  these frames;
* Stage 1 reconstruction, gap validation, fusion and wagon counting are never
  invoked and receive nothing from this module.

The collector is read-only with respect to every wagon structure: it takes an
already-final segment list, looks at the ones the classifier labelled ENGINE,
and writes JPEGs into its own directory.  Remove this module and the wagon
count is bit-identical -- which is exactly the property the tests assert.

Where they go
-------------
    <bundle.dir>/engine_frames/
        <camera_id>/
            engine_000123.jpg          named by ORIGINAL frame index
            ...
        metadata.json                  both cameras, one file per train

Separate from `evidence/` (per-wagon feature evidence) and from
`camera_cache/` (the wagon materializer's output), so no wagon-oriented reader
can pick them up by walking a directory.

Scoring
-------
Reuses the repository's existing deterministic frame-quality scorer,
`features.inference_lib.snapshot_selector.SnapshotSelector` -- the same one the
door pipeline uses -- rather than inventing a second notion of "best frame".
No new score is defined and its weights are not changed.

`score_candidate` needs a bbox, and an ENGINE segment is a time span rather
than a tracked detection, so the bbox passed is a fixed CENTRAL region of the
frame.  Two consequences, both deliberate:

* the bbox-derived terms (completeness, size, center, balance) are identical
  for every candidate, so they contribute a constant offset and cannot affect
  the ranking;
* the ranking is therefore driven by Laplacian sharpness over that central
  region -- which is precisely the area a locomotive number is read from.  The
  repo already relies on this geometry: `rekognition_wagon_number` selects the
  middle triplet for loco plates because "a loco's plate sits centred on its
  face".

So the ordering means "sharpest view of where the number will be", which is
the right criterion for the downstream task.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C

#: Frames kept per camera.  Five each from RIGHT_UP and LEFT_UP -> ten a train.
MAX_FRAMES_PER_CAMERA = 5

#: Only the side cameras see a locomotive's face.  The top cameras look down at
#: the roof, where there is no number plate, so they are not collected from.
ENGINE_FRAME_CAMERAS: Tuple[str, ...] = C.SIDE_CAMERAS

#: Candidates scored per camera.  An engine span at 25 fps is a few hundred
#: frames; scoring every one buys nothing once the stride is below motion blur
#: scale, so the span is sampled evenly to bound the cost.
MAX_CANDIDATES_PER_CAMERA = 60

#: Central region used as the scoring bbox, as a fraction of frame size.
_CENTRAL_BOX = 0.5


@dataclass
class EngineFrame:
    """One saved engine frame and why it was chosen."""
    train_id: str
    camera_id: str
    frame_idx: int
    timestamp: float                 # camera-local seconds
    score: float
    reason: str
    path: str = ""
    s3_uri: str = ""
    segment_id: str = ""
    segment_label: str = ""
    segment_confidence: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_id": self.train_id,
            "camera_id": self.camera_id,
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 4),
            "score": round(self.score, 6),
            "reason": self.reason,
            "path": self.path,
            "s3_uri": self.s3_uri,
            "segment_id": self.segment_id,
            "segment_label": self.segment_label,
            "segment_confidence": round(self.segment_confidence, 4),
            "score_breakdown": {k: round(float(v), 6)
                                for k, v in self.score_breakdown.items()},
            "rank": self.rank,
        }


@dataclass
class EngineFrameResult:
    train_id: str = ""
    camera_id: str = ""
    frames: List[EngineFrame] = field(default_factory=list)
    engine_segments: int = 0
    candidates_scored: int = 0
    status: str = "NO_ENGINE"
    note: str = ""

    @property
    def count(self) -> int:
        return len(self.frames)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_id": self.train_id,
            "camera_id": self.camera_id,
            "status": self.status,
            "note": self.note,
            "engine_segments": self.engine_segments,
            "candidates_scored": self.candidates_scored,
            "frames_saved": self.count,
            "max_per_camera": MAX_FRAMES_PER_CAMERA,
            "frames": [f.to_dict() for f in self.frames],
        }

    def render(self) -> str:
        return (f"[ENGINE/{self.camera_id}] {self.status} "
                f"saved={self.count}/{MAX_FRAMES_PER_CAMERA} "
                f"segments={self.engine_segments} "
                f"scored={self.candidates_scored}"
                + (f"  ({self.note})" if self.note else ""))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def engine_frames_dir(bundle_dir: str, camera_id: str = "") -> str:
    """`<bundle>/engine_frames[/<camera>]`.  Created on demand."""
    p = os.path.join(bundle_dir, "engine_frames")
    if camera_id:
        p = os.path.join(p, camera_id)
    os.makedirs(p, exist_ok=True)
    return p


def is_engine_segment(seg: Any) -> bool:
    """True for a segment the CLASSIFIER labelled ENGINE.

    Reads the existing `label` field written by the Stage-1 classifier; it does
    not re-derive, re-detect or widen the definition.  BRAKE_VAN is excluded --
    it is not a locomotive and carries no loco number.
    """
    return str(getattr(seg, "label", "") or "").upper() == C.CLASS_ENGINE


def engine_segments(segments: Sequence[Any]) -> List[Any]:
    return [s for s in segments if is_engine_segment(s)]


def _central_bbox(width: int, height: int, frac: float = _CENTRAL_BOX):
    """Fixed central box -- see the module docstring for why it is fixed."""
    import numpy as np
    bw, bh = width * frac, height * frac
    x1 = (width - bw) / 2.0
    y1 = (height - bh) / 2.0
    return np.array([x1, y1, x1 + bw, y1 + bh], dtype=float)


def _sample_indices(spans: Sequence[Tuple[int, int]], budget: int) -> List[int]:
    """Evenly sample up to `budget` frame indices across the given spans.

    Deterministic: no randomness, and the same spans always yield the same
    indices, so a rerun selects the same frames.
    """
    total = sum(max(0, e - s + 1) for s, e in spans)
    if total <= 0:
        return []
    if total <= budget:
        out: List[int] = []
        for s, e in spans:
            out.extend(range(s, e + 1))
        return sorted(set(out))
    # Proportional allocation, at least one frame per span.
    picks: List[int] = []
    for s, e in spans:
        n = max(0, e - s + 1)
        if n <= 0:
            continue
        k = max(1, int(round(budget * (n / total))))
        if k == 1:
            picks.append((s + e) // 2)
            continue
        step = (n - 1) / float(k - 1)
        picks.extend(int(round(s + i * step)) for i in range(k))
    return sorted(set(picks))


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

def collect(
    *,
    train_id: str,
    camera_id: str,
    video_path: str,
    segments: Sequence[Any],
    output_dir: str,
    fps: float = 0.0,
    max_frames: int = MAX_FRAMES_PER_CAMERA,
    jpeg_quality: int = 92,
    verbose: bool = True,
) -> EngineFrameResult:
    """Save the best engine frames for ONE camera.  Never raises.

    Returns a result whose `frames` list is EMPTY when this camera saw no
    engine, when the video is unreadable, or when no frame could be decoded --
    a missing locomotive is a normal outcome, not a failure, and it must never
    affect the camera's own lifecycle.

    Fewer than `max_frames` valid frames yields fewer saved frames.  A frame is
    never duplicated to reach the target.
    """
    res = EngineFrameResult(train_id=train_id, camera_id=camera_id)

    if camera_id not in ENGINE_FRAME_CAMERAS:
        res.status = "SKIPPED"
        res.note = (f"{camera_id} is not a side camera; a locomotive number is "
                    f"only visible to {list(ENGINE_FRAME_CAMERAS)}")
        if verbose:
            print(res.render())
        return res

    segs = engine_segments(segments)
    res.engine_segments = len(segs)
    if not segs:
        res.status = "NO_ENGINE"
        res.note = "no segment classified ENGINE on this camera"
        if verbose:
            print(res.render())
        return res

    try:
        import cv2
        import numpy as np
        from features.inference_lib.snapshot_selector import (
            SnapshotCandidate, SnapshotSelector,
        )
    except Exception as e:  # noqa: BLE001
        res.status = "UNAVAILABLE"
        res.note = f"{type(e).__name__}: {e}"
        if verbose:
            print(res.render())
        return res

    spans = [(int(s.start_frame), int(s.end_frame)) for s in segs
             if int(s.end_frame) >= int(s.start_frame)]
    wanted = _sample_indices(spans, MAX_CANDIDATES_PER_CAMERA)
    if not wanted:
        res.status = "NO_FRAMES"
        res.note = "engine segment(s) carry an empty frame range"
        if verbose:
            print(res.render())
        return res

    # Which segment each wanted index belongs to, for the metadata trail.
    def _owner(idx: int):
        for s in segs:
            if int(s.start_frame) <= idx <= int(s.end_frame):
                return s
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        res.status = "UNREADABLE"
        res.note = f"cannot open {video_path}"
        if verbose:
            print(res.render())
        return res

    local_fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 0.0)
    selector = SnapshotSelector()
    scored: List[Tuple[float, int, Dict[str, float], Any]] = []
    frames_by_idx: Dict[int, Any] = {}

    try:
        # ONE sequential pass, `grab()` to skip and `retrieve()` only on the
        # frames we want -- the technique wagon_count/evidence_report.py uses.
        target = set(wanted)
        last = max(wanted)
        idx = 0
        while idx <= last:
            if not cap.grab():
                break
            if idx in target:
                ok, frame = cap.retrieve()
                if ok and frame is not None and frame.size:
                    h, w = frame.shape[:2]
                    seg = _owner(idx)
                    conf = float(getattr(seg, "confidence", 0.0) or 0.0)
                    bbox = _central_bbox(w, h)
                    total, breakdown = selector.score_candidate(
                        frame, bbox, conf)
                    scored.append((float(total), idx, breakdown, seg))
                    frames_by_idx[idx] = frame.copy()
            idx += 1
    except Exception as e:  # noqa: BLE001
        res.note = f"decode stopped early: {type(e).__name__}: {e}"
    finally:
        cap.release()

    res.candidates_scored = len(scored)
    if not scored:
        res.status = "NO_FRAMES"
        res.note = res.note or "no engine frame could be decoded"
        if verbose:
            print(res.render())
        return res

    # Highest score first; frame index breaks ties so the choice is stable.
    scored.sort(key=lambda t: (-t[0], t[1]))
    keep = scored[:max(0, int(max_frames))]

    cam_dir = engine_frames_dir(output_dir, camera_id)
    for rank, (total, idx, breakdown, seg) in enumerate(keep, start=1):
        path = os.path.join(cam_dir, f"engine_{idx:06d}.jpg")
        frame = frames_by_idx.get(idx)
        if frame is None:
            continue
        try:
            ok = bool(cv2.imwrite(path, frame,
                                  [int(cv2.IMWRITE_JPEG_QUALITY),
                                   int(jpeg_quality)]))
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            continue
        res.frames.append(EngineFrame(
            train_id=train_id, camera_id=camera_id, frame_idx=idx,
            timestamp=(idx / local_fps) if local_fps > 0 else 0.0,
            score=float(total),
            reason=(f"rank {rank} of {len(scored)} scored engine frame(s) by "
                    f"SnapshotSelector (sharpness-dominated over the central "
                    f"region; bbox-derived terms are constant)"),
            path=path,
            segment_id=str(getattr(seg, "local_id", "") or ""),
            segment_label=str(getattr(seg, "label", "") or ""),
            segment_confidence=float(getattr(seg, "confidence", 0.0) or 0.0),
            score_breakdown=dict(breakdown),
            rank=rank,
        ))

    res.status = "OK" if res.frames else "NO_FRAMES"
    if res.frames and len(res.frames) < max_frames:
        res.note = (f"only {len(res.frames)} valid engine frame(s) available; "
                    f"not padded to {max_frames}")
    if verbose:
        print(res.render())
    return res


def write_metadata(output_dir: str,
                   results: Sequence[EngineFrameResult]) -> str:
    """Write `engine_frames/metadata.json` for a train.  Returns the path.

    Merges with what is already there, so cameras that seal minutes apart both
    end up in one file without the later one erasing the earlier.
    """
    root = engine_frames_dir(output_dir)
    path = os.path.join(root, "metadata.json")
    doc: Dict[str, Any] = {
        "schema": "wagon_eye.engine_frames.v1",
        "purpose": ("train-level locomotive frames for future loco-number "
                    "detection; NOT wagons, never in any wagon timeline"),
        "max_frames_per_camera": MAX_FRAMES_PER_CAMERA,
        "cameras": {},
    }
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f) or {}
            if isinstance(existing.get("cameras"), dict):
                doc["cameras"] = existing["cameras"]
            if existing.get("train_id"):
                doc["train_id"] = existing["train_id"]
        except (OSError, ValueError):
            pass

    for r in results:
        if r.train_id:
            doc["train_id"] = r.train_id
        doc["cameras"][r.camera_id] = r.to_dict()

    doc["total_frames"] = sum(int(c.get("frames_saved") or 0)
                              for c in doc["cameras"].values())
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)
    return path


def load_metadata(output_dir: str) -> Dict[str, Any]:
    """Read a train's engine-frame metadata.  `{}` when absent."""
    path = os.path.join(output_dir, "engine_frames", "metadata.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}
