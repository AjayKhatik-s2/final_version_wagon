"""
evidence_report.py
==================

Phase-1 lightweight evidence reporting.

This module REPLACES the old "write every frame of every wagon to disk"
behaviour with a compact, self-cleaning evidence report:

  * For every global event (GW_n produced by the existing fusion logic) and
    for every one of the four cameras, four representative frames are
    selected at 20%, 40%, 60% and 80% **through that camera's own valid
    evidence interval for that event** -- never through the whole video.
  * Maximum evidence for one event is therefore 4 percentages x 4 cameras
    = 16 frames.  Fewer frames are reported when a camera genuinely has no
    valid evidence (synchronization boundary, shorter video, undecodable
    frame, or an interval too short to yield four distinct frames).
  * The frames are written to a temporary directory, composed into
    ``combined_report.pdf``, and the temporary directory is deleted once
    the PDF has been written and verified.

IMPORTANT -- what this module does NOT do
-----------------------------------------
It performs NO detection, NO tracking, NO alignment, NO fusion and NO
counting.  It is a pure consumer of the finished ``GlobalTrainState`` and
the per-camera ``LocalCameraTracks`` produced by the existing pipeline.
Every number it prints is read straight out of those objects.

The per-event, per-camera evidence interval is obtained from the project's
existing mapping function --
``video_segmenter.map_global_wagon_to_local_frames`` -- which is the same
function the previous full-frame extraction used.  No new timing or
alignment method is introduced.

Dependencies
------------
cv2 + numpy (already hard requirements) for frame IO and resizing, and
PIL (Pillow) for writing the multi-page PDF.  Pillow and matplotlib are
already installed as dependencies of ``ultralytics``; matplotlib is used
only to locate its bundled DejaVu TrueType fonts.  Nothing here needs a
display -- there is no cv2.imshow and no GUI backend.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from global_train_state import (
    ALL_CAMERAS,
    GlobalTrainState,
    GlobalWagon,
    LocalCameraTracks,
)
from video_segmenter import map_global_wagon_to_local_frames

# =============================================================================
# Configuration
# =============================================================================

#: The four representative sampling points, as percentages of each camera's
#: valid evidence interval for one event.  Fixed by the report contract.
EVIDENCE_PERCENTAGES: Tuple[int, int, int, int] = (20, 40, 60, 80)

#: The one and only final deliverable filename.
PDF_FILENAME = "combined_report.pdf"

#: Temporary directory (under the output root) holding the extracted frames
#: until the PDF has been written.  Deleted afterwards.
TEMP_EVIDENCE_DIRNAME = ".evidence_tmp"

#: Page geometry: A4 landscape.
_A4_LANDSCAPE_INCHES = (11.69, 8.27)
DEFAULT_REPORT_DPI = 150
DEFAULT_JPEG_QUALITY = 92
DEFAULT_PAGE_JPEG_QUALITY = 88

# Palette (RGB -- this module works in RGB, unlike video_segmenter's BGR)
_INK = (24, 24, 28)
_INK_SOFT = (90, 94, 104)
_PAPER = (255, 255, 255)
_RULE = (196, 200, 210)
_BAND = (238, 241, 246)
_MISSING_BG = (243, 244, 247)
_MISSING_INK = (152, 24, 32)
_ACCENT = (16, 78, 150)
_CLASS_INK = {
    "ENGINE": (176, 92, 0),
    "WAGON": (18, 108, 56),
    "BRAKE_VAN": (168, 24, 36),
    "UNKNOWN": (110, 110, 118),
}


# =============================================================================
# Data holders
# =============================================================================

@dataclass
class EvidenceSlot:
    """One (event, camera, percentage) evidence cell."""
    camera_id: str
    global_id: str
    percentage: int
    frame_index: Optional[int] = None
    image_path: Optional[str] = None
    available: bool = False
    reason: str = ""

    @property
    def label(self) -> str:
        """The caption drawn in the PDF, e.g. ``RIGHT_UP - 20%``."""
        return f"{self.camera_id} - {self.percentage}%"


@dataclass
class CameraEvidence:
    """All four percentage slots for one camera on one event."""
    camera_id: str
    interval: Optional[Tuple[int, int]] = None     # inclusive local frame range
    span_frames: int = 0
    note: str = ""
    slots: List[EvidenceSlot] = field(default_factory=list)

    @property
    def available_count(self) -> int:
        return sum(1 for s in self.slots if s.available)


@dataclass
class EventEvidence:
    """Evidence plan for one global event (one GW_n)."""
    wagon: GlobalWagon
    cameras: Dict[str, CameraEvidence] = field(default_factory=dict)

    @property
    def slots(self) -> List[EvidenceSlot]:
        out: List[EvidenceSlot] = []
        for cam in ALL_CAMERAS:
            ce = self.cameras.get(cam)
            if ce:
                out.extend(ce.slots)
        return out

    @property
    def available_count(self) -> int:
        return sum(1 for s in self.slots if s.available)

    @property
    def total_slots(self) -> int:
        return len(self.slots)


# =============================================================================
# STEP A -- select the 20 / 40 / 60 / 80 % frames
# =============================================================================

def _percentage_frame(start: int, end: int, percentage: int) -> int:
    """Frame index at ``percentage`` through the inclusive range [start, end].

    The percentage is relative to THIS interval only -- the caller passes the
    camera's valid evidence interval for one event, never the whole video.
    """
    span = end - start
    if span <= 0:
        return start
    idx = start + int(round((percentage / 100.0) * span))
    return max(start, min(end, idx))


def select_event_evidence(
    state: GlobalTrainState,
    tracks: Dict[str, LocalCameraTracks],
    camera_offsets: Optional[Dict[str, float]] = None,
    unresolved_cameras: Optional[Dict[str, str]] = None,
) -> List[EventEvidence]:
    """Plan the evidence frames for every global event, for every camera.

    Reads the event list straight from ``state.wagons`` -- the global ids and
    time windows the fusion stage produced.  Nothing is recomputed and no wagon
    is ever added or removed here: the number of report pages always follows the
    global wagon roster.

    Parameters
    ----------
    camera_offsets :
        Optional per-camera clock offset in seconds (``t_global = t_local +
        delta``).  When supplied, a global wagon's master time window is shifted
        into each camera's own clock before projection, so the sampled frames
        show the same physical wagon in every camera.  Omitting it reproduces the
        historical behaviour (all cameras assumed to share t=0).
    unresolved_cameras :
        Cameras whose offset could not be resolved, mapped to the reason.  Their
        evidence is reported as unavailable rather than sampled at a guessed
        offset.
    """
    events: List[EventEvidence] = []
    camera_offsets = camera_offsets or {}
    unresolved_cameras = unresolved_cameras or {}

    for wagon in state.wagons:
        ev = EventEvidence(wagon=wagon)

        for cam in ALL_CAMERAS:
            tr = tracks.get(cam)
            ce = CameraEvidence(camera_id=cam)

            if cam in unresolved_cameras:
                ce.note = unresolved_cameras[cam]
                ce.slots = [
                    EvidenceSlot(cam, wagon.global_id, p, available=False,
                                 reason="camera offset unresolved")
                    for p in EVIDENCE_PERCENTAGES
                ]
                ev.cameras[cam] = ce
                continue

            if tr is None:
                ce.note = "camera was not processed in this run"
                ce.slots = [
                    EvidenceSlot(cam, wagon.global_id, p, available=False,
                                 reason="camera not processed")
                    for p in EVIDENCE_PERCENTAGES
                ]
                ev.cameras[cam] = ce
                continue

            if tr.fps <= 0 or tr.total_frames <= 0:
                ce.note = "camera reported no usable video metadata"
                ce.slots = [
                    EvidenceSlot(cam, wagon.global_id, p, available=False,
                                 reason="no usable video metadata")
                    for p in EVIDENCE_PERCENTAGES
                ]
                ev.cameras[cam] = ce
                continue

            # Shift the event's master time window into THIS camera's clock.
            # delta == 0.0 reproduces the historical shared-t=0 assumption.
            delta = float(camera_offsets.get(cam, 0.0))
            local_start_time = wagon.start_time - delta
            local_end_time = wagon.end_time - delta

            # Does this event's window overlap this camera at all?
            # The unclamped projection tells us honestly; the clamped interval
            # (from the project's existing mapping function) is what we sample.
            cam_duration = tr.total_frames / tr.fps
            raw_start = int(round(local_start_time * tr.fps))
            raw_end = int(round(local_end_time * tr.fps)) - 1
            if raw_end < 0 or raw_start > tr.total_frames - 1:
                ce.note = (f"event window {wagon.start_time:.2f}-{wagon.end_time:.2f}s "
                           f"(local {local_start_time:.2f}-{local_end_time:.2f}s at "
                           f"delta={delta:+.2f}s) lies outside this camera's "
                           f"{cam_duration:.2f}s of video")
                ce.slots = [
                    EvidenceSlot(cam, wagon.global_id, p, available=False,
                                 reason="event outside this camera's video length")
                    for p in EVIDENCE_PERCENTAGES
                ]
                ev.cameras[cam] = ce
                continue

            sf, ef = map_global_wagon_to_local_frames(
                dataclasses.replace(wagon, start_time=local_start_time,
                                    end_time=local_end_time) if delta else wagon,
                tr.fps, tr.total_frames,
            )
            if ef < sf:
                ce.note = "no valid frame range for this event"
                ce.slots = [
                    EvidenceSlot(cam, wagon.global_id, p, available=False,
                                 reason="empty evidence interval")
                    for p in EVIDENCE_PERCENTAGES
                ]
                ev.cameras[cam] = ce
                continue

            ce.interval = (sf, ef)
            ce.span_frames = ef - sf + 1
            if raw_start < 0 or raw_end > tr.total_frames - 1:
                ce.note = (f"evidence interval clipped to this camera's video "
                           f"({cam_duration:.2f}s)")

            # One frame per percentage.  If the interval is too short for four
            # distinct frames we do NOT fabricate duplicates -- the frame is
            # used once and the remaining slots are reported as unavailable.
            used: Dict[int, int] = {}
            for p in EVIDENCE_PERCENTAGES:
                idx = _percentage_frame(sf, ef, p)
                if idx in used:
                    ce.slots.append(EvidenceSlot(
                        cam, wagon.global_id, p, available=False,
                        reason=(f"interval too short ({ce.span_frames} frame(s)); "
                                f"{p}% resolves to the same frame as {used[idx]}%"),
                    ))
                else:
                    used[idx] = p
                    ce.slots.append(EvidenceSlot(
                        cam, wagon.global_id, p, frame_index=idx, available=True,
                    ))
            if ce.span_frames < len(EVIDENCE_PERCENTAGES) and not ce.note:
                ce.note = (f"only {ce.span_frames} valid frame(s) in this "
                           f"camera's evidence interval")

            ev.cameras[cam] = ce

        events.append(ev)

    return events


# =============================================================================
# STEP B -- extract the selected frames to a temporary directory
# =============================================================================

def _safe_name(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in text)


def extract_evidence_frames(
    events: Sequence[EventEvidence],
    tracks: Dict[str, LocalCameraTracks],
    temp_dir: str,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    verbose: bool = True,
) -> int:
    """Write every planned evidence frame into ``temp_dir``.

    One sequential decode pass per camera (``grab()`` to skip, ``retrieve()``
    only on wanted frames) -- no seeking, deterministic, and far cheaper than
    decoding whole wagons.  Frames are stored at their ORIGINAL resolution so
    nothing is distorted or degraded before the PDF stage.

    Filenames are deterministic and collision-free:
        ``{CAMERA}__{GW_id}__p{pct}__f{frame:06d}.jpg``

    Returns the number of frames actually written.
    """
    os.makedirs(temp_dir, exist_ok=True)

    # camera -> frame_index -> slots wanting that frame
    wanted: Dict[str, Dict[int, List[EvidenceSlot]]] = {}
    for ev in events:
        for cam in ALL_CAMERAS:
            ce = ev.cameras.get(cam)
            if not ce:
                continue
            for slot in ce.slots:
                if slot.available and slot.frame_index is not None:
                    wanted.setdefault(cam, {}).setdefault(slot.frame_index, []).append(slot)

    written = 0
    for cam in ALL_CAMERAS:
        per_frame = wanted.get(cam)
        tr = tracks.get(cam)
        if not per_frame or tr is None:
            continue

        targets = sorted(per_frame.keys())
        cap = cv2.VideoCapture(tr.video_path)
        if not cap.isOpened():
            for slots in per_frame.values():
                for slot in slots:
                    slot.available = False
                    slot.reason = "camera video could not be reopened for evidence"
            if verbose:
                print(f"  [EVIDENCE/{cam}] WARNING cannot open {tr.video_path}")
            continue

        try:
            ptr = 0
            frame_idx = 0
            while ptr < len(targets):
                if not cap.grab():
                    break
                if frame_idx == targets[ptr]:
                    ok, frame = cap.retrieve()
                    slots = per_frame[targets[ptr]]
                    if ok and frame is not None:
                        for slot in slots:
                            fname = (f"{_safe_name(cam)}__{_safe_name(slot.global_id)}"
                                     f"__p{slot.percentage:02d}__f{frame_idx:06d}.jpg")
                            path = os.path.join(temp_dir, fname)
                            if cv2.imwrite(path, frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]):
                                slot.image_path = path
                                written += 1
                            else:
                                slot.available = False
                                slot.reason = "frame could not be written to disk"
                    else:
                        for slot in slots:
                            slot.available = False
                            slot.reason = "frame could not be decoded"
                    ptr += 1
                frame_idx += 1

            # Anything past the real end of the stream is genuinely absent.
            while ptr < len(targets):
                for slot in per_frame[targets[ptr]]:
                    slot.available = False
                    slot.reason = "frame index beyond the end of this camera's video"
                ptr += 1
        finally:
            cap.release()

        if verbose:
            # Count SLOTS, not unique frame indices -- two slots may legitimately
            # resolve to the same frame index and each still gets its own file.
            n_slots = sum(len(v) for v in per_frame.values())
            got = sum(1 for slots in per_frame.values() for s in slots if s.available)
            print(f"  [EVIDENCE/{cam}] {got}/{n_slots} representative frame(s) "
                  f"extracted in one pass over {frame_idx} frames")

    return written


# =============================================================================
# STEP C -- PDF composition (Pillow; JPEG-compressed pages, no GUI)
# =============================================================================

class _Fonts:
    """TrueType font resolver with a safe fallback chain.

    Prefers the DejaVu faces bundled inside matplotlib (a guaranteed
    ultralytics dependency), then common Linux/Windows system paths, then
    Pillow's built-in bitmap font as a last resort.
    """

    def __init__(self) -> None:
        from PIL import ImageFont
        self._ImageFont = ImageFont
        self._cache: Dict[Tuple[str, int], Any] = {}
        self._sans = self._first_existing(self._sans_candidates())
        self._mono = self._first_existing(self._mono_candidates())
        self.degraded = self._sans is None

    @staticmethod
    def _mpl_font_dir() -> Optional[str]:
        try:
            import matplotlib
            d = os.path.join(os.path.dirname(matplotlib.__file__),
                             "mpl-data", "fonts", "ttf")
            return d if os.path.isdir(d) else None
        except Exception:
            return None

    def _sans_candidates(self) -> List[str]:
        out: List[str] = []
        d = self._mpl_font_dir()
        if d:
            out += [os.path.join(d, "DejaVuSans.ttf")]
        out += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        return out

    def _mono_candidates(self) -> List[str]:
        out: List[str] = []
        d = self._mpl_font_dir()
        if d:
            out += [os.path.join(d, "DejaVuSansMono.ttf")]
        out += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "C:/Windows/Fonts/consola.ttf",
        ]
        return out

    @staticmethod
    def _first_existing(paths: Sequence[str]) -> Optional[str]:
        for p in paths:
            if p and os.path.isfile(p):
                return p
        return None

    def get(self, size: int, mono: bool = False, bold: bool = False):
        path = self._mono if (mono and self._mono) else self._sans
        key = (path or "default", size)
        if key not in self._cache:
            if path is None:
                self._cache[key] = self._ImageFont.load_default()
            else:
                try:
                    self._cache[key] = self._ImageFont.truetype(path, size)
                except Exception:
                    self._cache[key] = self._ImageFont.load_default()
        return self._cache[key]


class _Page:
    """One PDF page being composed as an RGB raster."""

    def __init__(self, width: int, height: int, fonts: _Fonts):
        from PIL import Image, ImageDraw
        self.img = Image.new("RGB", (width, height), _PAPER)
        self.draw = ImageDraw.Draw(self.img)
        self.w = width
        self.h = height
        self.fonts = fonts

    # -- text helpers ----------------------------------------------------
    def text(self, xy: Tuple[int, int], s: str, size: int,
             fill=_INK, mono: bool = False, anchor: Optional[str] = None) -> int:
        f = self.fonts.get(size, mono=mono)
        self.draw.text(xy, s, font=f, fill=fill, anchor=anchor)
        return self.text_height(s, size, mono=mono)

    def text_width(self, s: str, size: int, mono: bool = False) -> int:
        f = self.fonts.get(size, mono=mono)
        box = self.draw.textbbox((0, 0), s, font=f)
        return int(box[2] - box[0])

    def text_height(self, s: str, size: int, mono: bool = False) -> int:
        f = self.fonts.get(size, mono=mono)
        box = self.draw.textbbox((0, 0), s or "X", font=f)
        return int(box[3] - box[1])

    def centered(self, cx: int, y: int, s: str, size: int,
                 fill=_INK, mono: bool = False) -> None:
        self.draw.text((cx - self.text_width(s, size, mono) // 2, y), s,
                       font=self.fonts.get(size, mono=mono), fill=fill)

    def rule(self, x0: int, y: int, x1: int, fill=_RULE, width: int = 1) -> None:
        self.draw.line([(x0, y), (x1, y)], fill=fill, width=width)

    def truncate(self, s: str, size: int, max_w: int, mono: bool = False) -> str:
        if self.text_width(s, size, mono) <= max_w:
            return s
        ell = "..."
        lo, hi = 0, len(s)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.text_width(s[:mid] + ell, size, mono) <= max_w:
                lo = mid + 1
            else:
                hi = mid
        return s[:max(0, lo - 1)] + ell


def _fit_preserving_aspect(img_w: int, img_h: int,
                           box_w: int, box_h: int) -> Tuple[int, int]:
    """Largest (w, h) fitting the box with the ORIGINAL aspect ratio intact."""
    if img_w <= 0 or img_h <= 0 or box_w <= 0 or box_h <= 0:
        return (0, 0)
    scale = min(box_w / img_w, box_h / img_h)
    return (max(1, int(round(img_w * scale))), max(1, int(round(img_h * scale))))


class _ReportBuilder:
    """Composes the multi-page combined evidence report."""

    def __init__(self, dpi: int = DEFAULT_REPORT_DPI):
        self.dpi = int(dpi)
        self.w = int(round(_A4_LANDSCAPE_INCHES[0] * self.dpi))
        self.h = int(round(_A4_LANDSCAPE_INCHES[1] * self.dpi))
        self.margin = int(round(0.040 * self.w))
        self.fonts = _Fonts()
        self.pages: List[_Page] = []
        #: every caption drawn, for verification / logging
        self.labels_drawn: List[str] = []
        #: camera_id -> frame aspect ratio (w/h), so "no evidence" placeholders
        #: occupy the same footprint as real frames and the grid stays uniform
        self.camera_aspect: Dict[str, float] = {}

        # Font ladder scaled to the page height
        self.s_title = max(11, self.h // 30)
        self.s_h2 = max(9, self.h // 46)
        self.s_body = max(8, self.h // 62)
        self.s_small = max(7, self.h // 76)
        self.s_label = max(7, self.h // 70)
        self.s_tiny = max(6, self.h // 92)

    # ------------------------------------------------------------------
    def set_camera_aspects(self, tracks: Dict[str, LocalCameraTracks]) -> None:
        """Record each camera's frame aspect ratio from the existing tracks."""
        for cam in ALL_CAMERAS:
            tr = tracks.get(cam)
            if tr and tr.width > 0 and tr.height > 0:
                self.camera_aspect[cam] = tr.width / tr.height
        if self.camera_aspect:
            fallback = sum(self.camera_aspect.values()) / len(self.camera_aspect)
        else:
            fallback = 16.0 / 9.0
        for cam in ALL_CAMERAS:
            self.camera_aspect.setdefault(cam, fallback)

    def new_page(self) -> _Page:
        p = _Page(self.w, self.h, self.fonts)
        self.pages.append(p)
        return p

    def _header(self, p: _Page, title: str, subtitle: str = "") -> int:
        """Draw the page header band; return the y where content may start."""
        y = self.margin
        p.text((self.margin, y), title, self.s_h2, fill=_ACCENT)
        if subtitle:
            p.text((self.w - self.margin, y), subtitle, self.s_small,
                   fill=_INK_SOFT, anchor="ra")
        y += int(self.s_h2 * 1.45)
        p.rule(self.margin, y, self.w - self.margin, fill=_ACCENT, width=2)
        return y + int(self.s_body * 0.9)

    def _footer(self, p: _Page, index: int, total: int, note: str = "") -> None:
        y = self.h - self.margin + int(self.s_tiny * 0.4)
        p.rule(self.margin, y - int(self.s_tiny * 0.8),
               self.w - self.margin, fill=_RULE)
        p.text((self.margin, y), f"{PDF_FILENAME}{('  ·  ' + note) if note else ''}",
               self.s_tiny, fill=_INK_SOFT)
        p.text((self.w - self.margin, y), f"page {index} of {total}",
               self.s_tiny, fill=_INK_SOFT, anchor="ra")

    # ------------------------------------------------------------------
    # Summary section
    # ------------------------------------------------------------------
    def add_summary(
        self,
        state: GlobalTrainState,
        tracks: Dict[str, LocalCameraTracks],
        events: Sequence[EventEvidence],
    ) -> None:
        d = state.to_dict()          # canonical values, straight from the pipeline
        p = self.new_page()

        # Title block
        y = self.margin
        p.text((self.margin, y), "GLOBAL WAGON COUNT", self.s_title, fill=_INK)
        y += int(self.s_title * 1.15)
        p.text((self.margin, y), "Combined evidence report", self.s_h2, fill=_ACCENT)
        y += int(self.s_h2 * 1.5)
        p.rule(self.margin, y, self.w - self.margin, fill=_ACCENT, width=3)
        y += int(self.s_body * 1.2)
        p.text((self.margin, y),
               f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}   ·   "
               f"schema {d.get('schema', '-')}", self.s_small, fill=_INK_SOFT)
        y += int(self.s_small * 2.2)

        col_w = (self.w - 2 * self.margin) // 2
        left_x = self.margin
        right_x = self.margin + col_w + int(0.02 * self.w)
        y_left = y_right = y

        # ---- Global train state ----
        y_left = self._kv_block(
            p, left_x, y_left, col_w - int(0.03 * self.w),
            "GLOBAL TRAIN STATE",
            [
                ("master camera", str(d.get("master_camera", "-"))),
                ("master fps", f"{d.get('master_fps', 0)}"),
                ("master total frames", f"{d.get('master_total_frames', 0)}"),
                ("FINAL WAGON COUNT", str(d.get("total_wagons", 0))),
                ("regular wagons", str(d.get("regular_wagon_count", 0))),
                ("engines", str(d.get("engine_count", 0))),
                ("brake vans", str(d.get("brake_van_count", 0))),
                ("corrections applied", str(len(d.get("corrections_applied", [])))),
                ("fallback used", str(d.get("fallback_used", False))),
            ] + ([("fallback reason", str(d.get("fallback_reason", "")))]
                 if d.get("fallback_used") else []),
            emphasise={"FINAL WAGON COUNT"},
        )

        # ---- Per-camera local counts ----
        rows: List[List[str]] = []
        for cam in ALL_CAMERAS:
            tr = tracks.get(cam)
            rows.append([
                cam,
                str(d.get("per_camera_local_counts", {}).get(cam, "-")),
                str(d.get("per_camera_gap_counts", {}).get(cam, "-")),
                f"{tr.fps:.2f}" if tr else "-",
                str(tr.total_frames) if tr else "-",
                f"{tr.total_frames / tr.fps:.1f}s" if (tr and tr.fps > 0) else "-",
                str(d.get("per_camera_status", {}).get(cam, "-")),
            ])
        y_right = self._table_block(
            p, right_x, y_right, self.w - self.margin - right_x,
            "PER-CAMERA LOCAL COUNTS (from the pipeline)",
            ["camera", "wagons", "gaps", "fps", "frames", "duration", "status"],
            rows,
        )

        y = max(y_left, y_right) + int(self.s_body * 1.0)

        # ---- Train structure: engines and brake vans are NOT wagons ----
        ww = getattr(state, "wagon_window", None) or {}
        if ww:
            def _fmt(classes: Dict[str, int]) -> str:
                if not classes:
                    return "none"
                return ", ".join(f"{n} x {c}" for c, n in sorted(classes.items()))

            y = self._kv_block(
                p, self.margin, y, self.w - 2 * self.margin,
                "TRAIN STRUCTURE  (only WAGONs are counted)",
                [
                    ("leading non-wagon",
                     _fmt(ww.get("leading_non_wagon_classes", {}))),
                    ("WAGON region",
                     f"{ww.get('first_wagon') or '-'} .. {ww.get('last_wagon') or '-'}"
                     f"   frames {ww.get('wagon_start_frame')}-"
                     f"{ww.get('wagon_end_frame')}"),
                    ("trailing non-wagon",
                     _fmt(ww.get("trailing_non_wagon_classes", {}))),
                ] + ([("excluded inside region",
                       _fmt(ww.get("interior_non_wagon_classes", {})))]
                     if ww.get("interior_non_wagon_count") else []) + [
                    ("TOTAL WAGONS", str(ww.get("master_wagon_count",
                                                d.get("total_wagons", 0)))),
                    ("not counted",
                     "ENGINE and BRAKE_VAN are real train objects but never "
                     "receive a GW id"),
                ],
                emphasise={"TOTAL WAGONS"},
            )

        # ---- Evidence policy ----
        avail = sum(e.available_count for e in events)
        total = sum(e.total_slots for e in events)
        y = self._kv_block(
            p, self.margin, y, self.w - 2 * self.margin,
            "EVIDENCE POLICY",
            [
                ("sampling",
                 "20% / 40% / 60% / 80% through each camera's own valid evidence "
                 "interval for each event"),
                ("maximum per event",
                 f"{len(EVIDENCE_PERCENTAGES)} percentages x {len(ALL_CAMERAS)} cameras "
                 f"= {len(EVIDENCE_PERCENTAGES) * len(ALL_CAMERAS)} frames"),
                ("evidence frames in this report", f"{avail} of {total} possible"),
                ("full frame sequences",
                 "not stored -- temporary frames are deleted after this PDF is written"),
            ],
        )
        y += int(self.s_body * 0.4)

        # ---- Corrections / insertions ----
        corrections = d.get("corrections_applied", [])
        if corrections:
            crows = [[
                f"{c.get('inserted_at_master_time', 0):.2f}s",
                str(c.get("inserted_at_master_frame", "-")),
                "/".join(c.get("supporting_cameras", [])),
                f"{c.get('mean_confidence', 0):.2f}",
                f"{c.get('time_spread_sec', 0):.2f}s",
            ] for c in corrections]
            y = self._table_block(
                p, self.margin, y, self.w - 2 * self.margin,
                f"GLOBAL CORRECTIONS / INSERTIONS  ({len(corrections)})",
                ["master time", "master frame", "supporting cameras",
                 "mean conf", "spread"],
                crows, max_rows=6,
            )
        else:
            y = self._kv_block(
                p, self.margin, y, self.w - 2 * self.margin,
                "GLOBAL CORRECTIONS / INSERTIONS",
                [("count", "0 -- no gaps were inserted into the master timeline")],
            )

        # ---- Global wagon roster ----
        self._wagon_roster(state, events)

    # ------------------------------------------------------------------
    def _kv_block(self, p: _Page, x: int, y: int, w: int, title: str,
                  pairs: Sequence[Tuple[str, str]],
                  emphasise: Optional[set] = None) -> int:
        emphasise = emphasise or set()
        p.text((x, y), title, self.s_body, fill=_ACCENT)
        y += int(self.s_body * 1.5)
        # Measure the widest key at its own font size so a larger emphasised
        # key can never collide with its value.
        key_w = 0
        for k, _v in pairs:
            size = self.s_body if k in emphasise else self.s_small
            key_w = max(key_w, p.text_width(k, size))
        key_w = min(int(w * 0.46), key_w + int(self.s_body * 1.2))
        for k, v in pairs:
            big = k in emphasise
            size = self.s_body if big else self.s_small
            p.text((x, y), k, size, fill=_INK_SOFT)
            p.text((x + key_w, y), p.truncate(v, size, w - key_w, mono=True),
                   size, fill=_INK if not big else _ACCENT, mono=True)
            y += int(size * 1.62)
        return y + int(self.s_small * 0.6)

    def _table_block(self, p: _Page, x: int, y: int, w: int, title: str,
                     headers: Sequence[str], rows: Sequence[Sequence[str]],
                     max_rows: Optional[int] = None) -> int:
        p.text((x, y), title, self.s_body, fill=_ACCENT)
        y += int(self.s_body * 1.5)

        ncol = len(headers)
        # Size each column to its widest cell (headers included) instead of
        # splitting the width evenly -- otherwise long values such as
        # 'RIGHT_UP_TOP' get truncated while numeric columns waste space.
        natural = []
        for i in range(ncol):
            wid = p.text_width(str(headers[i]), self.s_tiny)
            for r in rows:
                if i < len(r):
                    wid = max(wid, p.text_width(str(r[i]), self.s_small, mono=True))
            natural.append(wid + int(self.s_small * 1.1))
        total = sum(natural) or 1
        if total > w:                       # shrink proportionally if too wide
            col_ws = [max(int(self.s_small * 2), int(n * w / total)) for n in natural]
        else:                               # distribute the slack evenly
            extra = (w - total) // ncol
            col_ws = [n + extra for n in natural]
        col_x = [x + sum(col_ws[:i]) for i in range(ncol)]

        band_h = int(self.s_small * 1.7)
        p.draw.rectangle([x, y - int(self.s_small * 0.35),
                          x + w, y - int(self.s_small * 0.35) + band_h], fill=_BAND)
        for i, hh in enumerate(headers):
            p.text((col_x[i] + 4, y),
                   p.truncate(str(hh), self.s_tiny, col_ws[i] - 8),
                   self.s_tiny, fill=_INK_SOFT)
        y += band_h

        shown = rows if max_rows is None else rows[:max_rows]
        for r in shown:
            for i, cell in enumerate(r):
                if i >= ncol:
                    break
                p.text((col_x[i] + 4, y),
                       p.truncate(str(cell), self.s_small, col_ws[i] - 8, mono=True),
                       self.s_small, fill=_INK, mono=True)
            y += int(self.s_small * 1.62)
        if max_rows is not None and len(rows) > max_rows:
            p.text((x + 4, y), f"... and {len(rows) - max_rows} more "
                               f"(see the machine-readable global_train_state.json)",
                   self.s_tiny, fill=_INK_SOFT)
            y += int(self.s_tiny * 1.8)
        return y + int(self.s_small * 0.6)

    def _wagon_roster(self, state: GlobalTrainState,
                      events: Sequence[EventEvidence]) -> None:
        """Paginated table of every global wagon id and its evidence coverage."""
        by_id = {e.wagon.global_id: e for e in events}
        rows: List[List[str]] = []
        for w in state.wagons:
            ev = by_id.get(w.global_id)
            cov = f"{ev.available_count}/{ev.total_slots}" if ev else "-"
            per_cam = " ".join(
                f"{cam.split('_')[0][0]}{('T' if cam.endswith('TOP') else '')}"
                f":{ev.cameras[cam].available_count if (ev and cam in ev.cameras) else 0}"
                for cam in ALL_CAMERAS
            )
            rows.append([
                w.global_id,
                str(w.wagon_index),
                w.classification,
                f"{w.classification_confidence:.2f}",
                f"{w.start_frame_master}-{w.end_frame_master}",
                f"{w.start_time:.2f}-{w.end_time:.2f}s",
                f"{w.duration:.2f}s",
                cov,
                per_cam,
            ])

        headers = ["global id", "idx", "classification", "conf",
                   "master frames", "master time", "duration",
                   "evidence", "per camera (R/L/RT/LT)"]

        if not rows:
            p = self.new_page()
            y = self._header(p, "GLOBAL WAGON ROSTER")
            p.text((self.margin, y),
                   "No global wagons were produced by the pipeline for this input.",
                   self.s_body, fill=_MISSING_INK)
            return

        usable_h = self.h - 2 * self.margin - int(self.s_h2 * 3.2)
        row_h = int(self.s_small * 1.62)
        per_page = max(6, (usable_h - int(self.s_small * 3)) // row_h)

        chunks = [rows[i:i + per_page] for i in range(0, len(rows), per_page)]
        for ci, chunk in enumerate(chunks, start=1):
            p = self.new_page()
            y = self._header(
                p, "GLOBAL WAGON ROSTER",
                f"part {ci} of {len(chunks)}" if len(chunks) > 1 else "",
            )
            self._table_block(p, self.margin, y, self.w - 2 * self.margin,
                              f"{len(rows)} global wagon(s) - ids and evidence coverage",
                              headers, chunk)

    # ------------------------------------------------------------------
    # Event evidence pages: 4 cameras x 4 percentages
    # ------------------------------------------------------------------
    def add_event_page(self, ev: EventEvidence) -> None:
        w_obj = ev.wagon
        p = self.new_page()

        # ---- header ----
        y = self.margin
        cls_ink = _CLASS_INK.get(w_obj.classification, _INK)
        p.text((self.margin, y), f"{w_obj.global_id}", self.s_title, fill=_INK)
        gid_w = p.text_width(f"{w_obj.global_id}", self.s_title)
        p.text((self.margin + gid_w + int(0.012 * self.w),
                y + int(self.s_title * 0.30)),
               f"{w_obj.classification}  (conf {w_obj.classification_confidence:.2f})",
               self.s_h2, fill=cls_ink)
        p.text((self.w - self.margin, y + int(self.s_title * 0.30)),
               f"evidence {ev.available_count} / {ev.total_slots} frames",
               self.s_small, fill=_INK_SOFT, anchor="ra")
        y += int(self.s_title * 1.30)
        p.rule(self.margin, y, self.w - self.margin, fill=_ACCENT, width=2)
        y += int(self.s_small * 0.9)

        # ---- metadata line(s): all values read from the existing state ----
        lead = w_obj.leading_gap or {}
        trail = w_obj.trailing_gap or {}

        def _gap_str(g: Dict[str, Any]) -> str:
            src = g.get("source", "-")
            if "center_time" in g:
                return (f"{src} (cam {g.get('camera_id', '-')}, "
                        f"track {g.get('track_id', '-')}, "
                        f"t={g.get('center_time', 0)}s)")
            return str(src)

        meta_pairs = [
            ("wagon index", str(w_obj.wagon_index)),
            ("master frames", f"{w_obj.start_frame_master} - {w_obj.end_frame_master}"),
            ("master time", f"{w_obj.start_time:.3f}s - {w_obj.end_time:.3f}s "
                            f"(duration {w_obj.duration:.3f}s)"),
            ("supporting cameras", ", ".join(w_obj.supporting_cameras) or "-"),
            ("leading gap", _gap_str(lead)),
            ("trailing gap", _gap_str(trail)),
        ]
        if w_obj.split_from_global_id:
            meta_pairs.append(("split from", str(w_obj.split_from_global_id)))

        mid = (len(meta_pairs) + 1) // 2
        col_w = (self.w - 2 * self.margin) // 2
        key_w = int(col_w * 0.26)
        for ci, group in enumerate((meta_pairs[:mid], meta_pairs[mid:])):
            yy = y
            xx = self.margin + ci * col_w
            for k, v in group:
                p.text((xx, yy), k, self.s_tiny, fill=_INK_SOFT)
                p.text((xx + key_w, yy),
                       p.truncate(v, self.s_tiny, col_w - key_w - 12, mono=True),
                       self.s_tiny, fill=_INK, mono=True)
                yy += int(self.s_tiny * 1.62)
        y += int(self.s_tiny * 1.62) * mid + int(self.s_small * 0.7)
        p.rule(self.margin, y, self.w - self.margin, fill=_RULE)
        y += int(self.s_small * 0.8)

        # ---- 4 x 4 evidence grid ----
        grid_x0 = self.margin
        grid_x1 = self.w - self.margin
        grid_y0 = y
        grid_y1 = self.h - self.margin - int(self.s_tiny * 2.6)

        ncols = len(EVIDENCE_PERCENTAGES)
        nrows = len(ALL_CAMERAS)
        gut_x = int(0.010 * self.w)
        gut_y = int(0.012 * self.h)
        cell_w = (grid_x1 - grid_x0 - (ncols - 1) * gut_x) // ncols
        cell_h = (grid_y1 - grid_y0 - (nrows - 1) * gut_y) // nrows
        label_h = int(self.s_label * 1.55)

        for r, cam in enumerate(ALL_CAMERAS):
            ce = ev.cameras.get(cam)
            for c, pct in enumerate(EVIDENCE_PERCENTAGES):
                x0 = grid_x0 + c * (cell_w + gut_x)
                y0 = grid_y0 + r * (cell_h + gut_y)
                slot = None
                if ce:
                    slot = next((s for s in ce.slots if s.percentage == pct), None)
                self._draw_cell(p, x0, y0, cell_w, cell_h, label_h,
                                cam, pct, slot,
                                camera_note=(ce.note if ce else "camera missing"))

    def _draw_cell(self, p: _Page, x0: int, y0: int, cw: int, ch: int,
                   label_h: int, cam: str, pct: int,
                   slot: Optional[EvidenceSlot], camera_note: str = "") -> None:
        label = f"{cam} - {pct}%"
        self.labels_drawn.append(label)

        # caption
        p.text((x0, y0), label, self.s_label, fill=_INK)
        img_y0 = y0 + label_h
        img_h = ch - label_h
        box = [x0, img_y0, x0 + cw, img_y0 + img_h]

        if slot is not None and slot.available and slot.image_path \
                and os.path.isfile(slot.image_path):
            bgr = cv2.imread(slot.image_path, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ih, iw = rgb.shape[:2]
                tw, th = _fit_preserving_aspect(iw, ih, cw, img_h)
                interp = cv2.INTER_AREA if (tw < iw) else cv2.INTER_LINEAR
                resized = cv2.resize(rgb, (tw, th), interpolation=interp)
                from PIL import Image
                px = x0 + (cw - tw) // 2
                py = img_y0 + (img_h - th) // 2
                p.img.paste(Image.fromarray(resized), (px, py))
                p.draw.rectangle([px, py, px + tw, py + th], outline=_RULE, width=1)
                if slot.frame_index is not None:
                    p.text((x0 + cw, y0), f"frame {slot.frame_index}",
                           self.s_tiny, fill=_INK_SOFT, anchor="ra")
                return

        # ---- placeholder: no valid evidence ----
        # Sized to this camera's real frame aspect so the 4x4 grid stays even.
        aspect = self.camera_aspect.get(cam, 16.0 / 9.0)
        pw, ph = _fit_preserving_aspect(int(round(aspect * 1000)), 1000, cw, img_h)
        px = x0 + (cw - pw) // 2
        py = img_y0 + (img_h - ph) // 2
        box = [px, py, px + pw, py + ph]
        p.draw.rectangle(box, fill=_MISSING_BG, outline=_RULE, width=1)
        cx = px + pw // 2
        cy = py + ph // 2
        msg = "No valid evidence available"
        p.centered(cx, cy - int(self.s_small * 1.3), msg, self.s_small,
                   fill=_MISSING_INK)
        reason = (slot.reason if (slot and slot.reason) else camera_note) or ""
        if reason:
            words = reason.split()
            lines: List[str] = []
            cur = ""
            for wd in words:
                trial = (cur + " " + wd).strip()
                if p.text_width(trial, self.s_tiny) > pw - 16 and cur:
                    lines.append(cur)
                    cur = wd
                else:
                    cur = trial
            if cur:
                lines.append(cur)
            yy = cy + int(self.s_small * 0.2)
            for ln in lines[:3]:
                p.centered(cx, yy, ln, self.s_tiny, fill=_INK_SOFT)
                yy += int(self.s_tiny * 1.4)

    # ------------------------------------------------------------------
    def save(self, out_path: str,
             jpeg_quality: int = DEFAULT_PAGE_JPEG_QUALITY) -> str:
        """Write all composed pages as a single multi-page PDF."""
        if not self.pages:
            raise RuntimeError("no pages composed; nothing to save")

        total = len(self.pages)
        for i, p in enumerate(self.pages, start=1):
            self._footer(p, i, total)

        images = [p.img for p in self.pages]
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        # Pillow encodes RGB pages as JPEG (DCTDecode) inside the PDF, which
        # keeps a photo-heavy evidence report small.
        images[0].save(
            out_path, "PDF", save_all=True, append_images=images[1:],
            resolution=float(self.dpi), quality=int(jpeg_quality), optimize=True,
        )
        for im in images:
            try:
                im.close()
            except Exception:
                pass
        self.pages.clear()
        return out_path


# =============================================================================
# STEP D -- verification + cleanup
# =============================================================================

def verify_pdf(path: str) -> Tuple[bool, str]:
    """Cheap structural check that the file really is a readable PDF."""
    try:
        size = os.path.getsize(path)
        if size < 1024:
            return False, f"suspiciously small ({size} bytes)"
        with open(path, "rb") as f:
            head = f.read(5)
            f.seek(max(0, size - 2048))
            tail = f.read()
        if not head.startswith(b"%PDF"):
            return False, "missing %PDF header"
        if b"%%EOF" not in tail:
            return False, "missing %%EOF trailer"
        return True, f"{size / (1024 * 1024):.2f} MB"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def cleanup_temp_evidence(temp_dir: str, verbose: bool = True) -> int:
    """Delete the temporary evidence directory. Returns files removed."""
    if not os.path.isdir(temp_dir):
        return 0
    n = sum(len(files) for _, _, files in os.walk(temp_dir))
    shutil.rmtree(temp_dir, ignore_errors=True)
    if os.path.isdir(temp_dir):          # retry once, then report honestly
        shutil.rmtree(temp_dir, ignore_errors=True)
    if verbose:
        if os.path.isdir(temp_dir):
            print(f"  [CLEANUP] WARNING could not fully remove {temp_dir}")
        else:
            print(f"  [CLEANUP] removed {n} temporary evidence frame(s) "
                  f"from {os.path.basename(temp_dir)}/")
    return n


def warn_about_legacy_frame_dirs(output_root: str, verbose: bool = True) -> None:
    """Notice (but never delete) heavyweight output from previous runs."""
    for name in ("frames", "processed_videos"):
        d = os.path.join(output_root, name)
        if not os.path.isdir(d):
            continue
        n = sum(len(files) for _, _, files in os.walk(d))
        if n and verbose:
            print(f"  [NOTE] legacy '{name}/' from an earlier run holds {n} file(s): {d}")
            print(f"         The pipeline no longer writes there. Delete it manually "
                  f"if you no longer need it.")


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def camera_offsets_from_state(
    state: GlobalTrainState,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Split ``state.camera_offsets`` into usable deltas and unresolved reasons.

    An unresolved camera is never sampled at a guessed offset; its evidence is
    reported as unavailable instead.
    """
    deltas: Dict[str, float] = {}
    unresolved: Dict[str, str] = {}
    for cam, off in (getattr(state, "camera_offsets", None) or {}).items():
        status = off.get("status")
        if status in ("REFERENCE", "RESOLVED"):
            deltas[cam] = float(off.get("delta", 0.0))
        else:
            unresolved[cam] = (off.get("reason")
                               or f"camera offset {status or 'unknown'}")
    return deltas, unresolved


def build_combined_report(
    *,
    state: GlobalTrainState,
    tracks: Dict[str, LocalCameraTracks],
    output_root: str,
    dpi: int = DEFAULT_REPORT_DPI,
    keep_evidence_frames: bool = False,
    verbose: bool = True,
    use_camera_offsets: bool = True,
) -> Dict[str, Any]:
    """Select evidence, build ``combined_report.pdf``, clean up temp frames.

    Strictly downstream of counting: this runs after the global train state is
    final, only reads it, and produces exactly one page per global wagon plus the
    summary and roster.  Extra support-camera detections cannot add pages, and
    nothing here can create or renumber a GW id.

    Returns a summary dict:
        pdf_path, pages, events, slots_available, slots_total,
        frames_extracted, frames_cleaned, verified, detail, labels
    """
    t0 = time.time()
    temp_dir = os.path.join(output_root, TEMP_EVIDENCE_DIRNAME)
    pdf_path = os.path.join(output_root, PDF_FILENAME)

    result: Dict[str, Any] = {
        "pdf_path": pdf_path, "pages": 0, "events": len(state.wagons),
        "slots_available": 0, "slots_total": 0, "frames_extracted": 0,
        "frames_cleaned": 0, "verified": False, "detail": "", "labels": [],
    }

    # A leftover temp dir from a crashed previous run would pollute this one.
    if os.path.isdir(temp_dir):
        cleanup_temp_evidence(temp_dir, verbose=False)

    # ---- select ----
    deltas, unresolved = ({}, {})
    if use_camera_offsets:
        deltas, unresolved = camera_offsets_from_state(state)
        if verbose and deltas:
            shown = ", ".join(f"{c}{d:+.2f}s" for c, d in sorted(deltas.items()) if d)
            if shown:
                print(f"  [EVIDENCE] applying camera clock offsets: {shown}")
        if verbose and unresolved:
            print(f"  [EVIDENCE] offset unresolved for {', '.join(sorted(unresolved))} "
                  f"-> their evidence is reported unavailable, not guessed")
    events = select_event_evidence(state, tracks, camera_offsets=deltas,
                                  unresolved_cameras=unresolved)
    result["slots_total"] = sum(e.total_slots for e in events)
    if verbose:
        planned = sum(1 for e in events for s in e.slots if s.available)
        print(f"  [EVIDENCE] {len(events)} event(s) x {len(ALL_CAMERAS)} camera(s) "
              f"x {len(EVIDENCE_PERCENTAGES)} percentages -> "
              f"{planned}/{result['slots_total']} frame(s) to extract")

    # ---- extract ----
    try:
        result["frames_extracted"] = extract_evidence_frames(
            events, tracks, temp_dir, verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"  [EVIDENCE] WARNING extraction problem: {type(e).__name__}: {e}")

    result["slots_available"] = sum(e.available_count for e in events)

    # ---- compose + save ----
    builder = _ReportBuilder(dpi=dpi)
    if builder.fonts.degraded and verbose:
        print("  [REPORT] NOTE no TrueType font found; falling back to Pillow's "
              "bitmap font (labels will look coarse)")
    builder.set_camera_aspects(tracks)
    builder.add_summary(state, tracks, events)
    for ev in events:
        builder.add_event_page(ev)
    result["pages"] = len(builder.pages)
    result["labels"] = list(dict.fromkeys(builder.labels_drawn))

    builder.save(pdf_path)

    ok, detail = verify_pdf(pdf_path)
    result["verified"] = ok
    result["detail"] = detail

    # ---- clean up ONLY after a verified PDF ----
    if ok and not keep_evidence_frames:
        result["frames_cleaned"] = cleanup_temp_evidence(temp_dir, verbose=verbose)
    elif keep_evidence_frames and verbose:
        print(f"  [CLEANUP] skipped (--keep-evidence-frames): {temp_dir}")
    elif not ok and verbose:
        print(f"  [CLEANUP] skipped because PDF verification failed ({detail}); "
              f"temporary frames left in {temp_dir} for inspection")

    if verbose:
        print(f"  [REPORT] {PDF_FILENAME}: {result['pages']} page(s), "
              f"{result['slots_available']}/{result['slots_total']} evidence frame(s), "
              f"{detail}, {time.time() - t0:.1f}s")

    return result
