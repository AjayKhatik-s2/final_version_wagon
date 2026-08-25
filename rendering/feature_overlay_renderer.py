"""Feature overlay renderer  --  visualization-only, LEGACY-PARITY.

Produces one overlay mp4 per camera (4 total) whose visual appearance clones
the legacy WagonEye `_tracked.mp4` output as closely as possible:

    Side cameras (RIGHT_UP / LEFT_UP) reproduce the legacy DOOR annotation
    (old_system/RIGHT_UP/door_processor.py `_annotate_frame`):
        * per confirmed track: a state-coloured box (3px stroke when OPEN,
          else 2px), a filled label bar with black "Door {id}: {STATE}"
          text, the raw last-frame confidence printed below the box, and a
          velocity arrow when the door is moving.
        * a single-frame red "EVENT: ... - Track N" banner top-left on the
          exact frame a door-level event fired.

    Top cameras (RIGHT_UP_TOP / LEFT_UP_TOP) reproduce the legacy DAMAGE
    annotation (old_system/RIGHT_UP_TOP/damage_processor.py `_annotate_frame`):
        * per raw per-frame detection: a class-coloured 2px box with a filled
          label bar and white "{class}: {conf}" text.
        * a green top-left info block: "Frame: N", "Damages: K", "Type: CLASS".

On top of that legacy layer, a v4 HUD is drawn on EVERY camera:
    * cyan gap boxes, replayed from the Stage-1 tracking JSON;
    * a magenta wagon-boundary flash + "GW_BOUNDARY" banner at each canonical
      wagon start, matching the batch counting renderer;
    * a translucent panel: frame, wagon-active-region state, the canonical
      GW id / classification / position out of the total, the fused LOAD
      verdict, live gap count and damage count;
    * an "ACTIVE-REGION START/END" banner on the boundary frames.

Every element is REPLAYED from artifacts already on disk. There is no second
count and no second tracker: gap boxes come from the tracking JSON, wagon ids
and boundaries from the canonical roster, the region from the master's own
`wagon_window`, and the load verdict from the fused state. The gap interpolator
is the counting engine's `_interp_gap_bbox`, CALLED rather than copied.

One limitation, stated rather than papered over: LOAD has no per-frame boxes
because the load processor never persisted per-frame detections -- only
`load/best_frame.jpg` and a fused per-wagon status. Drawing per-frame load boxes
would mean running inference again, which this module must never do, so the load
verdict is shown per wagon in the panel instead.

This module NEVER invokes any detector / YOLO / OCR model.  Every box is
replayed from artifacts Stage 3 already persisted:

    * evidence/<gw>/door/overlay.json    per-frame Kalman-smoothed door track
                                         trajectories + door-level events
    * evidence/<gw>/damage/overlay.json  raw per-frame damage detections
    * per_camera_tracking.json           per-camera fps / total_frames / dims

The four cameras render in parallel threads (OpenCV decode releases the GIL).

Output layout (one mp4 per camera):

    <output_dir>/<CAMERA_ID>_processed.mp4
"""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

from core import constants as C
from core.global_state_loader import GlobalTrainState, GlobalWagon
from core.unified_wagon_state import UnifiedWagonState


# -----------------------------------------------------------------------------
# Legacy colour maps (BGR) -- verbatim from old_system
# -----------------------------------------------------------------------------

# old_system/RIGHT_UP/door_processor.py STATE_COLORS_BGR
_DOOR_STATE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "open_door":        (0, 0, 255),     # Red
    "open":             (0, 0, 255),     # Red
    "closed_door":      (0, 255, 0),     # Green
    "closed":           (0, 255, 0),     # Green
    "closed_with_wire": (0, 255, 255),   # Yellow
    "partial_closed":   (0, 255, 255),   # Yellow
    "partially_closed": (0, 255, 255),   # Yellow
    "damage":           (0, 0, 255),     # Red
    "other":            (255, 165, 0),   # Orange
    "unknown":          (128, 128, 128), # Gray
}

# old_system/RIGHT_UP_TOP/damage_processor.py DAMAGE_COLORS
_DAMAGE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "floor_damage":      (0, 0, 255),     # Red
    "inner_wall_damage": (0, 165, 255),   # Orange
    "outer_wall_damage": (0, 255, 255),   # Yellow
    "no_damage":         (0, 255, 0),     # Green
    "unknown":           (128, 128, 128), # Gray
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Last render's per-camera audit, keyed by camera id. Written beside each mp4
#: as `<CAMERA>_processed_render_audit.json` too, so the annotations on a video
#: can be traced back to their source camera, tracks and canonical wagons.
RENDER_AUDITS: Dict[str, Any] = {}


# -----------------------------------------------------------------------------
# Per-camera overlay registry (replays persisted trajectories)
# -----------------------------------------------------------------------------

class _OverlayRegistry:
    """Per-frame draw items + events for ONE camera.

    Builds two O(1) lookups:
        boxes_by_frame[frame_idx]  -> list of door/damage draw items
        events_by_frame[frame_idx] -> list of door events (side cameras)
    """

    def __init__(
        self, *, camera_id: str, evidence_root: str, wagons: List[GlobalWagon],
        enabled_features: Optional[set] = None,
    ) -> None:
        self.camera_id = camera_id
        self.boxes_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        self.events_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        # Explicit gate: a feature NOT in enabled_features is never ingested, so
        # a DISABLED feature can never render -- even if a stale overlay.json
        # from a previous run still sits on disk.  When None (legacy default)
        # all features are eligible and the "no overlay.json -> no boxes"
        # fallback applies.
        self._door_enabled = enabled_features is None or "door" in enabled_features
        self._damage_enabled = enabled_features is None or "damage" in enabled_features
        if not evidence_root or not os.path.isdir(evidence_root):
            return
        for gw in wagons:
            ev_gw_dir = os.path.join(evidence_root, gw.global_id)
            if not os.path.isdir(ev_gw_dir):
                continue
            if self._door_enabled:
                self._ingest_door(ev_gw_dir)
            if self._damage_enabled:
                self._ingest_damage(ev_gw_dir)

    # --- internal -------------------------------------------------------

    @staticmethod
    def _load_json(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _push_box(self, frame_idx: Any, item: Dict[str, Any]) -> None:
        try:
            fi = int(frame_idx)
        except (TypeError, ValueError):
            return
        if fi < 0:
            return
        self.boxes_by_frame.setdefault(fi, []).append(item)

    def _ingest_door(self, ev_gw_dir: str) -> None:
        data = self._load_json(os.path.join(ev_gw_dir, "door", "overlay.json"))
        if not data:
            return
        for tr in data.get("tracks") or []:
            if not isinstance(tr, dict) or tr.get("camera_id") != self.camera_id:
                continue
            tid = tr.get("track_id")
            for fr in tr.get("frames") or []:
                bbox = fr.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                self._push_box(fr.get("frame_idx"), {
                    "kind":       "door",
                    "bbox":       list(bbox),
                    "track_id":   tid,
                    "state_raw":  str(fr.get("state_raw") or ""),
                    "last_class": str(fr.get("last_class") or ""),
                    "confidence": float(fr.get("confidence") or 0.0),
                    "velocity":   fr.get("velocity") or [0.0, 0.0],
                })
        for ev in data.get("events") or []:
            if not isinstance(ev, dict) or ev.get("camera_id") != self.camera_id:
                continue
            try:
                fi = int(ev.get("frame_idx", -1))
            except (TypeError, ValueError):
                continue
            if fi < 0:
                continue
            self.events_by_frame.setdefault(fi, []).append({
                "event":    str(ev.get("event", "")),
                "track_id": ev.get("track_id"),
            })

    def _ingest_damage(self, ev_gw_dir: str) -> None:
        data = self._load_json(os.path.join(ev_gw_dir, "damage", "overlay.json"))
        if not data:
            return
        for det in data.get("detections") or []:
            if not isinstance(det, dict) or det.get("camera_id") != self.camera_id:
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            self._push_box(det.get("frame_idx"), {
                "kind":       "damage",
                "bbox":       list(bbox),
                "class_name": str(det.get("class_name") or ""),
                "confidence": float(det.get("confidence") or 0.0),
            })


# -----------------------------------------------------------------------------
# Legacy draw primitives (cloned pixel-for-pixel from old_system)
# -----------------------------------------------------------------------------

def _draw_door_track(frame, item: Dict[str, Any]) -> None:
    """Clone of old_system door_processor `_annotate_frame` per-track block."""
    try:
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
    except (TypeError, ValueError, KeyError):
        return

    state_raw = str(item.get("state_raw") or "")
    last_class = str(item.get("last_class") or "")
    state_name = state_raw.lower()

    color = _DOOR_STATE_COLORS.get(state_name, (255, 255, 255))
    if state_raw == "UNKNOWN":
        color = _DOOR_STATE_COLORS.get(last_class.lower(), (128, 128, 128))

    thickness = 3 if state_raw == "OPEN" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if state_raw == "UNKNOWN":
        display_state = last_class.upper() if last_class else state_raw
    else:
        display_state = state_raw
    label = f"Door {item.get('track_id')}: {display_state}"
    conf_label = f"{float(item.get('confidence') or 0.0):.2f}"

    font_scale = 0.6
    text_thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, _FONT, font_scale, text_thickness)
    label_y = max(y1 - 5, text_h + 5)
    cv2.rectangle(frame, (x1, label_y - text_h - 5),
                  (x1 + text_w + 5, label_y + 2), color, -1)
    cv2.putText(frame, label, (x1 + 2, label_y - 2), _FONT, font_scale,
                (0, 0, 0), text_thickness)

    cv2.putText(frame, conf_label, (x1, y2 + 20), _FONT, 0.5, color, 1)

    vel = item.get("velocity") or [0.0, 0.0]
    try:
        vx, vy = float(vel[0]), float(vel[1])
    except (TypeError, ValueError, IndexError):
        vx, vy = 0.0, 0.0
    if abs(vx) > 1 or abs(vy) > 1:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        end_x, end_y = int(cx + vx * 3), int(cy + vy * 3)
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2)


def _draw_damage_det(frame, item: Dict[str, Any]) -> None:
    """Clone of old_system damage_processor `_annotate_frame` per-detection block."""
    try:
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
    except (TypeError, ValueError, KeyError):
        return
    class_name = str(item.get("class_name") or "")
    conf = float(item.get("confidence") or 0.0)
    color = _DAMAGE_COLORS.get(class_name, _DAMAGE_COLORS["unknown"])

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{class_name}: {conf:.2f}"
    (lw, lh), _ = cv2.getTextSize(label, _FONT, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), _FONT, 0.6, (255, 255, 255), 2)


def _draw_event_banner(frame, events: List[Dict[str, Any]]) -> None:
    """Clone of the legacy single-frame red event banner (top-left)."""
    y_offset = 30
    for ev in events:
        text = f"EVENT: {ev.get('event','')} - Track {ev.get('track_id')}"
        cv2.putText(frame, text, (10, y_offset), _FONT, 0.8, (0, 0, 255), 2)
        y_offset += 30


def _draw_damage_info(frame, frame_idx: int, n_damages: int, frame_class: str) -> None:
    """Clone of the legacy green top-left damage info block."""
    info_lines = [
        f"Frame: {frame_idx}",
        f"Damages: {n_damages}",
        f"Type: {frame_class}",
    ]
    for i, line in enumerate(info_lines):
        y = 30 + i * 25
        cv2.putText(frame, line, (10, y), _FONT, 0.7, (0, 255, 0), 2)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

#: Codecs to try, in order. H.264 FIRST, because this video's whole purpose is
#: to be played in a browser from the dashboard.
#:
#: `mp4v` (MPEG-4 Part 2) writes anywhere and plays nowhere: no browser can
#: decode it in an HTML5 <video>, so the dashboard rendered a player that sat at
#: 0:00 on a black frame. The file was fine and in S3 -- it simply was not a
#: format any browser speaks. `avc1` is H.264 in an MP4 container, which every
#: browser plays.
#:
#: mp4v stays last so a build without H.264 still produces a video rather than
#: failing the train; `_transcode_to_h264` then rescues playability via ffmpeg.
_CODEC_PREFERENCE = ("avc1", "H264", "mp4v")

#: Not browser-playable; a file written with this needs transcoding.
_FALLBACK_CODEC = "mp4v"


def _open_browser_playable_writer(output_path: str, fps: float,
                                  width: int, height: int):
    """`(writer, codec)` -- the best codec this OpenCV build will actually open.

    Asking for a fourcc is not the same as getting it: OpenCV happily returns a
    writer that never opened, or silently substitutes. So each candidate is
    verified with `isOpened()` before use, and the codec that won is returned so
    the caller knows whether a transcode is still needed.
    """
    for codec in _CODEC_PREFERENCE:
        try:
            w = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec),
                                fps, (width, height))
        except Exception:                                        # noqa: BLE001
            continue
        if w is not None and w.isOpened():
            return w, codec
        if w is not None:
            w.release()
    return None, ""


def _transcode_to_h264(path: str, *, verbose: bool = False) -> bool:
    """Re-encode a non-playable overlay video in place. True if it now is H.264.

    Only reached when the OpenCV build could not write H.264 at all. Uses
    `train_extraction.video_io.compress_video`, which exists for exactly this --
    its docstring says "e.g. a bulky mp4v-codec overlay video" -- and was never
    wired to a caller. It tries NVENC and falls back to libx264, and its size cap
    shrinks the file as a side benefit.

    Failure is not fatal: an unplayable video is worse than a playable one but
    better than a failed train, so the original is left in place.
    """
    import logging
    import shutil
    import tempfile

    if not shutil.which("ffmpeg"):
        if verbose:
            print(f"  [{os.path.basename(path)}] no ffmpeg: staying mp4v "
                  f"(will not play in a browser)")
        return False
    try:
        from train_extraction.video_io import compress_video
        tmp = os.path.join(tempfile.mkdtemp(), os.path.basename(path))
        out = compress_video(path, tmp, logging.getLogger("wagon_eye.rendering"))
        if out and os.path.isfile(out) and os.path.getsize(out) > 0:
            shutil.move(out, path)
            return True
    except Exception as e:                                       # noqa: BLE001
        if verbose:
            print(f"  [{os.path.basename(path)}] H.264 transcode failed: {e}")
    return False


#: v4 HUD colours. Cyan gaps and a magenta boundary flash match the batch
#: counting renderer (`wagon_count/video_segmenter.py`) so the two look alike.
_GAP_COLOR = (255, 255, 0)
_BOUNDARY_COLOR = (255, 0, 255)
_REGION_COLOR = (0, 200, 255)
_PANEL_BG = (0, 0, 0)


_NONWAGON_COLOR = (0, 140, 255)
_UNRESOLVED_COLOR = (0, 165, 255)


def _non_wagon_spans(state: Any, src_fps: float, total: int,
                     time_offset: float) -> List[Tuple[int, int, str, str]]:
    """`(start_frame, end_frame, classification, position)` per non-wagon object.

    Read from the master's `wagon_window`, which is where leading and trailing
    ENGINE / BRAKE_VAN objects are already preserved with their frame ranges.
    They are drawn so the video shows them as real, identified vehicles -- and
    labelled as non-wagon, because they must never carry a GW id.
    """
    win = dict(getattr(state, "wagon_window", None) or {})
    out: List[Tuple[int, int, str, str]] = []
    if src_fps <= 0:
        return out
    for key, position in (("leading_non_wagon_objects", "leading"),
                          ("trailing_non_wagon_objects", "trailing")):
        for o in (win.get(key) or []):
            if not isinstance(o, dict):
                continue
            st_t, en_t = o.get("start_time"), o.get("end_time")
            if st_t is None or en_t is None:
                continue
            # `end_time` is EXCLUSIVE -- the pipeline writes it as
            # (end_frame + 1) / fps -- so the last frame is one before it.
            # Treating it as inclusive stretched every span by one frame and
            # made an ENGINE appear to overlap the first wagon.
            sf = int(round((float(st_t) - time_offset) * src_fps))
            ef = int(round((float(en_t) - time_offset) * src_fps)) - 1
            sf, ef = max(0, sf), min(max(0, total - 1), ef)
            if ef < sf:
                continue
            out.append((sf, ef, str(o.get("classification") or "NON_WAGON"),
                        position))
    return out


def _canonical_gap_view(global_gaps: Optional[Sequence[Dict[str, Any]]],
                        camera_id: str) -> Tuple[Dict[Any, int],
                                                 List[Dict[str, Any]],
                                                 Dict[str, Any]]:
    """How this camera relates to each CANONICAL gap.

    Returns `(local_track_id -> global_gap_id, unobserved, stats)`.

    The canonical sequence is the counting engine's own `state.global_gaps`:
    every entry has a RIGHT_UP master observation, plus a
    `support_observations[camera]` for each camera that saw the same boundary.
    That mapping is what lets a support camera's marker be labelled with the
    CANONICAL id -- `GAP_25` -- instead of its own local track number, which is
    a per-camera counter and is not the same number.

    `unobserved` are canonical gaps this camera never observed. They carry no
    local frames at all, so they are reported in the audit and NOT drawn: the
    only way to place them would be to project the master time through an offset
    that may be unresolved, and a marker at a guessed moment is worse than an
    honest absence.
    """
    by_track: Dict[Any, int] = {}
    unobserved: List[Dict[str, Any]] = []
    for g in (global_gaps or ()):
        if not isinstance(g, dict):
            continue
        gid = g.get("global_gap_id")
        if gid is None:
            continue
        obs = (g.get("support_observations") or {}).get(camera_id)
        if isinstance(obs, dict) and obs.get("local_track_id") is not None:
            by_track[obs.get("local_track_id")] = int(gid)
            continue
        if camera_id == str(g.get("master_camera") or ""):
            mt = g.get("master_track_id")
            if mt is not None:
                by_track[mt] = int(gid)
                continue
        unobserved.append({"global_gap_id": int(gid),
                           "reason": "this camera never observed this boundary"})
    stats = {"canonical_gaps": len([g for g in (global_gaps or ())
                                    if isinstance(g, dict)]),
             "mapped_to_local_track": len(by_track),
             "not_observed_by_this_camera": len(unobserved)}
    return by_track, unobserved, stats


def _gap_tracks_for(tracking: Dict[str, Any], camera_id: str,
                    track_to_global: Optional[Dict[Any, int]] = None,
                    ) -> List[Dict[str, Any]]:
    """This camera's gap tracks, from the tracking JSON Stage 1 already wrote.

    Replayed, never recomputed: no gap model runs here. Only gaps carrying the
    full-fidelity `hit_frames` + `bbox_history` arrays can be drawn per frame --
    `GapEvent.to_dict()` is a REPORTING view that drops them, so a gap
    serialized through that path is skipped rather than guessed at.
    """
    cam = (tracking or {}).get(camera_id) or {}
    out: List[Dict[str, Any]] = []
    for g in (cam.get("gaps") or []):
        if not isinstance(g, dict):
            continue
        # A gap WITHOUT the trajectory arrays is kept and marked unresolved,
        # not dropped. Silently omitting it would make the video look like the
        # reconstruction found no boundary there, which is a different claim
        # from "the boundary is known but its image position is not".
        g = dict(g)
        g["_resolved"] = bool(g.get("hit_frames") and g.get("bbox_history"))
        # Canonical identity where the fusion established one. The local track
        # id stays on the record for traceability; the LABEL prefers this.
        gid = (track_to_global or {}).get(g.get("track_id"))
        if gid is not None:
            g["global_gap_id"] = gid
        out.append(g)
    return out


def _gap_neighbour_pair(boundary_map: Dict[int, Tuple[str, str]],
                        gap: Dict[str, Any]) -> Tuple[str, str]:
    """`(left_global_wagon, right_global_wagon)` for a gap, or `("", "")`.

    Taken from the canonical roster's own frame ranges -- the wagons the
    reconstruction actually placed either side of this boundary -- never from
    wagon numbering or an assumed spacing.
    """
    for f in (gap.get("start_frame"), gap.get("end_frame")):
        if f is not None and int(f) in boundary_map:
            return boundary_map[int(f)]
    return ("", "")


def _gap_neighbours(boundary_map: Dict[int, Tuple[str, str]],
                    gap: Dict[str, Any]) -> str:
    """`GW_n | GAP_id | GW_n+1` for a gap, or just the gap id.

    The adjacency comes from the canonical roster's own frame ranges, so the
    label names the wagons the reconstruction actually put either side of this
    boundary. It is NOT derived from wagon arithmetic or an assumed spacing.
    """
    # The CANONICAL boundary id when the fusion mapped this track to one --
    # `GAP_25` means the same physical coupling on all four cameras, whereas a
    # local track id is a per-camera counter and differs between them.
    gid = gap.get("global_gap_id")
    if gid is None:
        gid = gap.get("track_id", "?")
    prev_gw, next_gw = _gap_neighbour_pair(boundary_map, gap)
    if prev_gw and next_gw:
        return f"{prev_gw} | GAP_{gid} | {next_gw}"
    return f"GAP_{gid}"


class _GapView:
    """Attribute view over a gap dict, for the engine's own interpolator.

    `video_segmenter._interp_gap_bbox` reads `gap.hit_frames`,
    `gap.bbox_history`, `gap.start_frame`, `gap.end_frame` -- it takes a
    `GapEvent`, not a mapping. The tracking JSON gives us dicts, and passing one
    straight in raised `AttributeError` on the first attribute access, which the
    caller's `except` swallowed into `bbox = None`. The visible effect was that
    interpolation never happened at all: a marker appeared only on frames that
    were exact recorded hits and vanished in between, which looks like a
    flickering detector rather than a type mismatch.

    Reusing the engine's interpolator through this shim keeps ONE interpolation
    algorithm in the repository, which is the point -- reimplementing it here
    would have hidden the same bug behind a second copy of the maths.
    """

    __slots__ = ("hit_frames", "bbox_history", "start_frame", "end_frame")

    def __init__(self, g: Dict[str, Any]) -> None:
        self.hit_frames = list(g.get("hit_frames") or [])
        self.bbox_history = list(g.get("bbox_history") or [])
        self.start_frame = g.get("start_frame")
        self.end_frame = g.get("end_frame")


def _draw_gap_boxes(frame, gaps: List[Dict[str, Any]], frame_idx: int,
                    boundary_map: Optional[Dict[int, Tuple[str, str]]] = None,
                    drawn_provenance: Optional[List[Dict[str, Any]]] = None,
                    camera_id: str = "") -> int:
    """Cyan box for every gap live on this frame. Returns how many were drawn.

    The geometry is the gap's OWN recorded trajectory -- `bbox_history` at a
    recorded hit, otherwise the counting engine's `_interp_gap_bbox`, CALLED not
    copied. Nothing is placed at an assumed x-coordinate and nothing is derived
    from wagon numbers, so the marker tracks the real object across frames and
    each camera uses its own image-plane geometry.
    """
    drawn = 0
    try:
        from video_segmenter import _interp_gap_bbox        # type: ignore
    except Exception:                                        # noqa: BLE001
        _interp_gap_bbox = None                              # type: ignore
    for g in gaps:
        sf, ef = g.get("start_frame"), g.get("end_frame")
        if sf is None or ef is None or not (int(sf) <= frame_idx <= int(ef)):
            continue
        if not g.get("_resolved", True):
            # Known boundary, unknown image position. Say exactly that.
            cv2.putText(frame,
                        f"GAP {g.get('track_id', '?')} UNRESOLVED "
                        f"(no trajectory)", (20, frame.shape[0] - 60),
                        _FONT, 0.55, _UNRESOLVED_COLOR, 2, cv2.LINE_AA)
            continue
        bbox = None
        hits = g.get("hit_frames") or []
        hist = g.get("bbox_history") or []
        if frame_idx in hits:
            bbox = hist[hits.index(frame_idx)]
        elif _interp_gap_bbox is not None:
            try:
                bbox = _interp_gap_bbox(_GapView(g), frame_idx)
            except Exception:                                # noqa: BLE001
                bbox = None
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), _GAP_COLOR, 2)
        label = _gap_neighbours(boundary_map or {}, g)
        cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                    _FONT, 0.5, _GAP_COLOR, 1, cv2.LINE_AA)
        drawn += 1
        if drawn_provenance is not None:
            _l, _r = _gap_neighbour_pair(boundary_map or {}, g)
            drawn_provenance.append({
                "camera_id": camera_id,
                "local_gap_track_id": g.get("track_id"),
                "global_gap_id": g.get("global_gap_id",
                                       g.get("master_track_id")),
                "frame": frame_idx,
                "bbox": [round(float(v), 2) for v in bbox[:4]],
                "center_x": round((float(bbox[0]) + float(bbox[2])) / 2.0, 2),
                "label": label,
                # recorded_hit = the gap's own bbox at this exact frame;
                # interpolated = between recorded hits, via the counting
                # engine's own `_interp_gap_bbox`. Never an assumed position.
                "geometry_source": ("recorded_hit" if frame_idx in hits
                                    else "interpolated"),
                "geometry": ("recorded_hit" if frame_idx in hits
                             else "interpolated"),
                "left_global_wagon": _l,
                "right_global_wagon": _r,
                "resolved": True,
                "is_physical_wagon_gap": True,
                "is_active_region_boundary": False,
            })
    return drawn


def _draw_boundary_flash(frame, boundary_frames: Sequence[int],
                         frame_idx: int) -> None:
    """Magenta flash + GW_BOUNDARY banner within +/-3 frames of a boundary."""
    for b in boundary_frames:
        if abs(frame_idx - int(b)) > 3:
            continue
        h, w = frame.shape[:2]
        cv2.line(frame, (0, 0), (w, 0), _BOUNDARY_COLOR, 4)
        cv2.line(frame, (0, h - 1), (w, h - 1), _BOUNDARY_COLOR, 4)
        label = "GW_BOUNDARY"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.7, 2)
        tx, ty = (w - tw) // 2, 40
        cv2.rectangle(frame, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8),
                      _BOUNDARY_COLOR, -1)
        cv2.putText(frame, label, (tx, ty), _FONT, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        return


def _draw_v4_panel(frame, lines: Sequence[tuple]) -> None:
    """Translucent top-left panel, same shape as the batch renderer's."""
    if not lines:
        return
    panel_w, panel_h = 430, 22 * len(lines) + 16
    ov = frame.copy()
    cv2.rectangle(ov, (10, 10), (10 + panel_w, 10 + panel_h), _PANEL_BG, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    y = 30
    for text, color in lines:
        cv2.putText(frame, text, (20, y), _FONT, 0.55, color, 1, cv2.LINE_AA)
        y += 22


def _map_wagon_to_local_frames(
    wagon: GlobalWagon, local_fps: float, local_total_frames: int,
    time_offset: float = 0.0,
) -> Tuple[int, int]:
    """Mirror of wagon_count/video_segmenter.build_camera_wagon_frame_map.

    `time_offset` is this camera's clock delta (`t_global = t_local + delta`)
    as resolved by the counting engine, so a projected wagon window lands on
    the frames that actually show it.  0.0 = historical shared-t=0 behaviour.
    A wagon outside this camera's footage yields an empty range instead of
    being clamped onto an unrelated frame.
    """
    if local_fps <= 0 or local_total_frames <= 0:
        return (0, -1)
    sf = int(round((wagon.start_time - time_offset) * local_fps))
    ef = int(round((wagon.end_time - time_offset) * local_fps)) - 1
    if ef < 0 or sf > local_total_frames - 1:
        return (0, -1)
    sf = max(0, min(local_total_frames - 1, sf))
    ef = max(0, min(local_total_frames - 1, ef))
    if ef < sf:
        ef = sf
    return (sf, ef)


def _load_camera_tracking(per_camera_tracking_path: str) -> Dict[str, Any]:
    if not per_camera_tracking_path or not os.path.isfile(per_camera_tracking_path):
        return {}
    try:
        with open(per_camera_tracking_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# Single-camera render
# -----------------------------------------------------------------------------

def _render_one_camera(
    *,
    camera_id: str,
    video_path: str,
    output_path: str,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],   # kept for API symmetry; not drawn
    evidence_root: str,
    camera_meta: Dict[str, Any],
    enabled_features: Optional[set] = None,
    verbose: bool = True,
    time_offset: float = 0.0,
    camera_tracking: Optional[Dict[str, Any]] = None,
    camera_regions: Optional[Dict[str, Any]] = None,
    global_gaps: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    # `unified` IS drawn now: the per-wagon LOAD verdict comes from it. Load
    # never persisted per-frame detections -- only `load/best_frame.jpg` and a
    # fused per-wagon status -- so a per-frame load BOX is not available without
    # re-running inference, which this module must never do. The verdict is
    # shown per wagon in the panel instead, which is the evidence that exists.

    if not os.path.isfile(video_path):
        raise RuntimeError(f"raw video missing for {camera_id}: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for {camera_id}: {video_path}")

    src_fps = float(camera_meta.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total   = int(camera_meta.get("total_frames")
                  or cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width   = int(camera_meta.get("width")
                  or cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height  = int(camera_meta.get("height")
                  or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    writer, codec = _open_browser_playable_writer(
        output_path, src_fps if src_fps > 0 else 25.0, width, height)
    if writer is None:
        cap.release()
        raise RuntimeError(f"cannot open writer for {output_path}")

    is_side = camera_id in C.SIDE_CAMERAS
    is_top  = camera_id in C.TOP_CAMERAS

    # frame -> canonical wagon, for EVERY camera now (the legacy block used it
    # only on the tops). This is the existing projection -- master time mapped
    # through this camera's offset -- not a new local->global matcher.
    frame_to_wagon: Dict[int, GlobalWagon] = {}
    boundary_frames: List[int] = []
    for w in state.wagons:
        sf, ef = _map_wagon_to_local_frames(w, src_fps, total, time_offset)
        if ef < sf:
            continue            # wagon not visible on this camera
        for f in range(sf, ef + 1):
            frame_to_wagon[f] = w
        if sf > 0:
            boundary_frames.append(sf)
    boundary_frames = sorted(set(boundary_frames))

    # boundary frame -> (previous GW, next GW), from the canonical roster's own
    # frame ranges. This is what lets a gap be labelled "GW_25 | GAP_x | GW_26"
    # using the wagons the reconstruction actually placed either side of it --
    # never inferred from wagon numbering or an assumed spacing.
    _boundary_map: Dict[int, Tuple[str, str]] = {}
    _ordered = []
    for w in state.wagons:
        sf, ef = _map_wagon_to_local_frames(w, src_fps, total, time_offset)
        if ef >= sf:
            _ordered.append((sf, ef, w.global_id))
    _ordered.sort()
    for _i in range(1, len(_ordered)):
        _prev, _next = _ordered[_i - 1], _ordered[_i]
        # A gap sits between the previous wagon's end and the next one's start;
        # index it under both so a gap track touching either is labelled.
        for _f in (_prev[1], _next[0]):
            _boundary_map[int(_f)] = (_prev[2], _next[2])

    # ---- Wagon-ACTIVE-REGION edges, in THIS camera's frames ---------------
    #
    # Two sources, and the order matters.
    #
    # PREFERRED: this camera's OWN `LocalWagonRegion` -- built by
    # `train_structure.build_local_wagon_region` from this camera's own
    # temporally-smoothed labels, in its own clock, and persisted as
    # `wagon_region.json`. It already knows where the wagons are on THIS
    # footage, and a single noisy WAGON prediction cannot open it because
    # `apply_temporal_classification` ran first.
    #
    # FALLBACK: the master's `wagon_window` projected by `time_offset`.
    #
    # The fallback alone was the bug. `camera_time_offsets()` returns 0.0 for a
    # camera whose offset the counter could NOT resolve -- deliberately, since
    # guessing a shift is worse than assuming none -- so on a camera whose
    # footage is genuinely displaced the master's region projects as though the
    # clocks agreed, and lands over the leading ENGINE. The video then showed
    # WAGON_REGION_ACTIVE while an engine filled the frame.
    #
    # This is DISPLAY ONLY. It selects which frames a marker is drawn on; it
    # cannot add, remove or renumber a canonical wagon, and it never touches
    # `state.wagon_window`. The canonical timeline stays with RIGHT_UP.
    _win = dict(getattr(state, "wagon_window", None) or {})
    _region_frames: Dict[str, Optional[int]] = {"start": None, "end": None}
    _region_source = "master_window_projected"
    _region_reason = "master wagon_window projected by this camera's offset"

    _local = (camera_regions or {}).get(camera_id)
    _lf = getattr(_local, "start_frame", None) if _local is not None else None
    _le = getattr(_local, "end_frame", None) if _local is not None else None
    if (_local is not None and getattr(_local, "found", False)
            and _lf is not None and _le is not None and int(_le) >= int(_lf)):
        _region_frames["start"] = max(0, min(max(0, total - 1), int(_lf)))
        _region_frames["end"] = max(0, min(max(0, total - 1), int(_le)))
        _region_source = f"{camera_id}_local_wagon_region"
        _region_reason = str(getattr(_local, "reason", "") or
                             "this camera's own classified wagon region")
        _region_found = True
    else:
        for _key, _t in (("start", _win.get("wagon_start_time")),
                         ("end", _win.get("wagon_end_time"))):
            if _t is None or src_fps <= 0:
                continue
            _f = int(round((float(_t) - time_offset) * src_fps))
            if _key == "end":
                # EXCLUSIVE: the window's end_time is (end_frame + 1) / fps, so
                # the last frame is one before it. Off by one here would put the
                # END marker on the first frame AFTER the region.
                _f -= 1
            if 0 <= _f < max(1, total):
                _region_frames[_key] = _f
        _region_found = bool(_win.get("found"))
        if _local is not None and not getattr(_local, "found", False):
            _region_reason = (
                f"{camera_id} has no local wagon region "
                f"({getattr(_local, 'reason', '') or 'unclassified'}); "
                f"fell back to the projected master window")
    _total_wagons = len(state.wagons)

    if verbose:
        print(f"[ACTIVE-REGION] {camera_id}  START frame="
              f"{_region_frames['start']}  END frame={_region_frames['end']}  "
              f"source={_region_source}  reason={_region_reason}")

    overlay = _OverlayRegistry(
        camera_id=camera_id, evidence_root=evidence_root, wagons=state.wagons,
        enabled_features=enabled_features,
    )

    if verbose:
        n_boxes = sum(len(v) for v in overlay.boxes_by_frame.values())
        print(f"[RENDER/{camera_id}] writing -> {output_path}  "
              f"({total} frames, {n_boxes} box-instances)")

    # Canonical boundary identity for THIS camera, so a marker can say GAP_25
    # (the physical coupling) rather than this camera's own track number.
    _track_to_gid, _unobserved_gaps, _canon_stats = _canonical_gap_view(
        global_gaps, camera_id)
    gap_tracks = _gap_tracks_for(camera_tracking or {}, camera_id,
                                 _track_to_gid)
    # Non-wagon spans come from the master window projected by the offset. When
    # this camera supplied its OWN region, anything outside that region is this
    # camera's non-wagon footage, so the spans are clipped to it -- otherwise the
    # ENGINE label and the region marker could disagree on the same frame.
    non_wagon_spans = _non_wagon_spans(state, src_fps, total, time_offset)
    _frame_to_nonwagon: Dict[int, Tuple[str, str]] = {}
    for _sf, _ef, _cls, _pos in non_wagon_spans:
        for _f in range(_sf, _ef + 1):
            _frame_to_nonwagon[_f] = (_cls, _pos)

    # Per-camera audit, so every annotation is traceable to its source.
    _gap_provenance: List[Dict[str, Any]] = []
    _audit: Dict[str, Any] = {
        "camera_id": camera_id,
        "output": output_path,
        "total_frames": total,
        "fps": src_fps,
        "time_offset_sec": round(float(time_offset), 4),
        "canonical_wagons_total": len(state.wagons),
        "global_wagon_ids_shown": sorted(
            {w.global_id for w in state.wagons
             if _map_wagon_to_local_frames(w, src_fps, total,
                                           time_offset)[1] >= 0},
            key=lambda g: int(str(g).split("_")[-1]) if str(g).split("_")[-1].isdigit() else 0),
        "gap_tracks_total": len(gap_tracks),
        "canonical_gap_mapping": dict(_canon_stats),
        # Canonical boundaries this camera never observed. Reported, never
        # drawn: placing them would mean guessing a moment from an offset that
        # may be unresolved.
        "canonical_gaps_not_observed": [d["global_gap_id"]
                                        for d in _unobserved_gaps],
        "gap_tracks_resolved": sum(1 for g in gap_tracks if g.get("_resolved")),
        "gap_tracks_unresolved_detail": [
            {"local_gap_track_id": g.get("track_id"),
             "global_gap_id": g.get("global_gap_id"),
             "start_frame": g.get("start_frame"),
             "end_frame": g.get("end_frame"),
             "reason": "no hit_frames / bbox_history for this track"}
            for g in gap_tracks if not g.get("_resolved")],
        "gap_tracks_unresolved": sum(1 for g in gap_tracks
                                     if not g.get("_resolved")),
        "boundary_frames": list(boundary_frames),
        "active_region_found": _region_found,
        "active_region_start_frame": _region_frames["start"],
        "active_region_end_frame": _region_frames["end"],
        "active_region": {
            "start": _region_frames["start"],
            "end": _region_frames["end"],
            "source": _region_source,
            "reason": _region_reason,
            # An active-region edge is NOT a wagon gap. Recorded separately so
            # nothing downstream can count one as the other.
            "is_physical_wagon_gap": False,
            "start_reason": ("first_canonical_wagon" if _region_found
                             else "no_wagon_evidence_anywhere"),
            "end_reason": ("last_canonical_wagon" if _region_found
                           else "no_wagon_evidence_anywhere"),
            "found": _region_found,
        },
        "unresolved_gaps": [
            {"local_gap_track_id": g.get("track_id"),
             "start_frame": g.get("start_frame"),
             "end_frame": g.get("end_frame"),
             "status": "UNRESOLVED",
             "reason": "serialized without bbox_history/hit_frames "
                       "(GapEvent.to_dict is a reporting view)"}
            for g in gap_tracks if not g.get("_resolved")],
        "non_wagon_objects": [
            {"start_frame": a, "end_frame": b, "classification": c,
             "position": d, "global_wagon_id": None}
            for (a, b, c, d) in non_wagon_spans],
        "gap_markers_drawn": 0,
        "door_detections_drawn": 0,
        "damage_detections_drawn": 0,
        "load_status_frames": 0,
        "load_per_frame_boxes": "unavailable: load persists no per-frame "
                                "detections; drawing them would require a "
                                "second inference pass",
        "features_enabled": sorted(enabled_features) if enabled_features else "all",
    }
    if verbose:
        print(f"  [{camera_id}] HUD: {len(gap_tracks)} gap track(s), "
              f"{len(boundary_frames)} boundary frame(s), "
              f"region={_region_frames['start']}..{_region_frames['end']}")

    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        n_damages = 0
        n_doors = 0
        for item in overlay.boxes_by_frame.get(frame_idx, []):
            if item.get("kind") == "door":
                _draw_door_track(frame, item)
                n_doors += 1
            else:
                _draw_damage_det(frame, item)
                n_damages += 1
        _audit["door_detections_drawn"] += n_doors
        _audit["damage_detections_drawn"] += n_damages

        if is_side:
            evs = overlay.events_by_frame.get(frame_idx)
            if evs:
                _draw_event_banner(frame, evs)
        elif is_top:
            w = frame_to_wagon.get(frame_idx)
            frame_class = str(w.classification).upper() if w else "WAGON"
            _draw_damage_info(frame, frame_idx, n_damages, frame_class)

        # ---- v4 HUD: gaps, boundaries, canonical GW, active region -------
        # Every element is REPLAYED from artifacts already on disk. No model,
        # no tracker, no second count: gaps come from the Stage-1 tracking JSON,
        # wagon ids and boundaries from the canonical roster, the region from the
        # master's own `wagon_window`, and the load verdict from the fused state.
        n_gaps = _draw_gap_boxes(frame, gap_tracks, frame_idx,
                                 boundary_map=_boundary_map,
                                 drawn_provenance=_gap_provenance,
                                 camera_id=camera_id)
        _audit["gap_markers_drawn"] += n_gaps
        _draw_boundary_flash(frame, boundary_frames, frame_idx)

        _cur = frame_to_wagon.get(frame_idx)
        if _region_frames["start"] is not None and frame_idx < _region_frames["start"]:
            _state_name = "BEFORE_WAGON_REGION"
        elif _region_frames["end"] is not None and frame_idx > _region_frames["end"]:
            _state_name = "AFTER_WAGON_REGION"
        elif _region_found:
            _state_name = "WAGON_REGION_ACTIVE"
        else:
            _state_name = "BEFORE_WAGON_REGION"

        _lines = [(f"Frame: {frame_idx}", (255, 255, 255)),
                  (f"Region: {_state_name}", _REGION_COLOR)]
        if _cur is not None:
            _u = (unified or {}).get(_cur.global_id)
            _load = str(getattr(_u, "load_status", "") or "-")
            _lines.append((f"{_cur.global_id}  {_cur.classification}  "
                           f"({_cur.wagon_index}/{_total_wagons})",
                           (0, 255, 0)))
            _lines.append((f"Load: {_load}", (0, 255, 0)))
            if _load and _load != "-":
                _audit["load_status_frames"] += 1
        else:
            _nw = _frame_to_nonwagon.get(frame_idx)
            if _nw is not None:
                # A real, identified vehicle -- shown as such, and explicitly
                # WITHOUT a GW id, which is the whole point of the region gate.
                _lines.append((f"{_nw[0]}  ({_nw[1]} non-wagon)  GW: none",
                               _NONWAGON_COLOR))
            else:
                _lines.append((f"No canonical wagon (outside region)  "
                               f"total={_total_wagons}", (180, 180, 180)))
        if n_gaps:
            _lines.append((f"Gaps live: {n_gaps}", _GAP_COLOR))
        if n_damages:
            _lines.append((f"Damages: {n_damages}", (0, 0, 255)))
        _draw_v4_panel(frame, _lines)

        for _k, _f in _region_frames.items():
            if _f is not None and abs(frame_idx - _f) <= 3:
                _txt = f"ACTIVE-REGION {_k.upper()}"
                (_tw, _th), _ = cv2.getTextSize(_txt, _FONT, 0.8, 2)
                _h, _w = frame.shape[:2]
                _x, _y = (_w - _tw) // 2, _h - 24
                cv2.rectangle(frame, (_x - 10, _y - _th - 10),
                              (_x + _tw + 10, _y + 10), _REGION_COLOR, -1)
                cv2.putText(frame, _txt, (_x, _y), _FONT, 0.8, (0, 0, 0), 2,
                            cv2.LINE_AA)

        writer.write(frame)
        written += 1
        frame_idx += 1
        if verbose and frame_idx % 500 == 0:
            print(f"  [{camera_id}] {frame_idx} frames")

    cap.release()
    writer.release()
    # The file must be closed before ffmpeg reads it.
    if codec == _FALLBACK_CODEC:
        _transcode_to_h264(output_path, verbose=verbose)
    # Per-marker provenance, capped so a long train's audit stays readable.
    # Every marker is included in the counts; the sample proves which ACTUAL
    # local gap track produced a marker and at which frame.
    _audit["gap_marker_provenance_sample"] = _gap_provenance[:200]
    _audit["gap_marker_provenance_total"] = len(_gap_provenance)
    _audit["distinct_gap_tracks_drawn"] = sorted(
        {str(p["local_gap_track_id"]) for p in _gap_provenance})
    _audit["frames_written"] = written
    _audit["codec"] = codec
    try:
        _ap = os.path.splitext(output_path)[0] + "_render_audit.json"
        with open(_ap, "w", encoding="utf-8") as _f:
            json.dump(_audit, _f, indent=2, default=str)
        _audit["audit_path"] = _ap
    except Exception as e:                                       # noqa: BLE001
        print(f"[RENDER/{camera_id}] audit not written: {e}")
    if verbose:
        print(f"[RENDER/{camera_id}] done ({written} frames, codec={codec}); "
              f"GW shown={len(_audit['global_wagon_ids_shown'])} "
              f"gaps={_audit['gap_markers_drawn']} "
              f"doors={_audit['door_detections_drawn']} "
              f"damages={_audit['damage_detections_drawn']} "
              f"non-wagon={len(_audit['non_wagon_objects'])}")
    RENDER_AUDITS[camera_id] = _audit
    return output_path


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def render_all_cameras(
    *,
    camera_regions: Optional[Dict[str, Any]] = None,
    global_gaps: Optional[Sequence[Dict[str, Any]]] = None,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    evidence_root: str,
    video_paths: Dict[str, str],
    per_camera_tracking_path: str,
    output_dir: str,
    enabled_features: Optional[set] = None,
    verbose: bool = True,
    camera_offsets: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """Render four camera overlay videos in parallel.

    Returns ``{camera_id -> output_mp4_path}`` for every camera that rendered
    successfully.  Cameras that fail produce no entry; the error is logged.

    `camera_offsets` are the counting engine's resolved per-camera clock
    deltas; cameras absent from it use 0.0.  Visualization only -- this module
    still never runs a detector and never touches the roster.
    """
    os.makedirs(output_dir, exist_ok=True)
    tracking = _load_camera_tracking(per_camera_tracking_path)
    offsets = dict(camera_offsets or {})

    jobs: Dict[str, Dict[str, Any]] = {}
    for cam in C.ALL_CAMERAS:
        vp = video_paths.get(cam)
        if not vp:
            if verbose:
                print(f"[RENDER/{cam}] SKIP -- no raw video path")
            continue
        out_mp4 = os.path.join(output_dir, f"{cam}_processed.mp4")
        jobs[cam] = {
            "video_path":  vp,
            "output_path": out_mp4,
            "camera_meta": tracking.get(cam, {}) or {},
        }

    if not jobs:
        return {}

    t0 = time.time()
    results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as ex:
        futs = {
            ex.submit(
                _render_one_camera,
                camera_id=cam,
                video_path=cfg["video_path"],
                output_path=cfg["output_path"],
                state=state,
                unified=unified,
                evidence_root=evidence_root,
                camera_meta=cfg["camera_meta"],
                enabled_features=enabled_features,
                verbose=verbose,
                time_offset=float(offsets.get(cam, 0.0) or 0.0),
                camera_tracking=tracking,
                camera_regions=camera_regions,
                global_gaps=global_gaps,
            ): cam
            for cam, cfg in jobs.items()
        }
        for f in as_completed(futs):
            cam = futs[f]
            try:
                results[cam] = f.result()
            except Exception as e:
                print(f"[RENDER/{cam}] FAILED: {type(e).__name__}: {e}")
                if verbose:
                    traceback.print_exc(limit=3)

    if verbose:
        print(f"[RENDER] done {len(results)}/{len(jobs)} cameras  "
              f"({time.time() - t0:.1f}s)")
    return results
