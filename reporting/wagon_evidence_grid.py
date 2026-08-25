"""Select 4 frames per (canonical wagon, camera) and record why. Report-side only.

This is the data layer behind the wagon-by-wagon combined PDF: for every
canonical GW_n it produces, per camera, exactly four slots -- each either a real
image path or an explicit "unavailable" marker with a reason.

It creates NO second wagon-mapping algorithm. Both things it needs already
exist:

  * the canonical roster -- `state.wagons`, the RIGHT_UP master timeline;
  * per-(wagon, camera) frame selection -- `_evidence_lookup.quartile_cache_paths`,
    which takes the 12.5 / 37.5 / 62.5 / 87.5% frames of that wagon's span in
    THAT camera's own local clock.

Camera isolation is structural, not checked afterwards: `_cache_frame_path` keys
the directory on `C.CAMERA_FOLDER[camera_id]`, so a RIGHT_UP_TOP slot can only
ever resolve to a RIGHT_UP_TOP file. There is no path by which a master frame
lands in a support slot.

Three things this module refuses to do
--------------------------------------
**Fabricate.** A camera with fewer than four cached frames for a wagon gets
`available=False` slots carrying a reason. It never repeats an image to fill
four, because four copies of one frame looks like four pieces of evidence.

**Guess the identity.** `gw_id` comes from `state.wagons` and is passed straight
through to the path, so a frame selected for GW_25 cannot be filed under GW_26.

**Run inference.** It reads `wagon_cache` and the Stage-3 damage evidence that
already exist. No model, no video decode.

The camera clock offset
-----------------------
The four cameras are not synchronised, and the cache filenames are
offset-corrected local indices (`wagon_cache_builder.py:92`). So the offset is
passed through to the lookup. Omitting it -- which the reporting path did until
now -- makes every support-camera slot resolve to None while the frames sit on
disk: silent, and indistinguishable from missing evidence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core import constants as C
from core.logging_setup import get_logger

from . import _evidence_lookup as ev

log = get_logger("reporting.wagon_evidence_grid")

#: Slots per camera per wagon. Four, matching `quartile_cache_paths`.
SLOTS_PER_CAMERA = 4

#: Rendering order: TOP cameras first, then SIDE. The PDF's page order follows
#: this, so the two are defined once.
TOP_ORDER = (C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP)
SIDE_ORDER = (C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP)
CAMERA_ORDER = TOP_ORDER + SIDE_ORDER

#: Why a slot has no image. Distinct reasons, because "no cache root" and "the
#: frame was never written" call for different fixes.
NO_CACHE_ROOT = "no_wagon_cache_root"
NO_SPAN = "wagon_has_no_frames_on_this_camera"
NOT_ON_DISK = "frame_not_written_to_wagon_cache"

UNAVAILABLE_LABEL = "NO VALID FRAME"


@dataclass
class Slot:
    """One image position for one (wagon, camera)."""

    index: int
    camera_id: str
    global_id: str
    available: bool = False
    path: Optional[str] = None
    frame: Optional[int] = None
    timestamp_sec: Optional[float] = None
    selection_reason: str = ""
    unavailable_reason: str = ""

    @property
    def label(self) -> str:
        if not self.available:
            return UNAVAILABLE_LABEL
        t = f"{self.timestamp_sec:.2f}s" if self.timestamp_sec is not None else "-"
        return f"f{self.frame} @ {t}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "camera_id": self.camera_id,
            "global_id": self.global_id, "available": self.available,
            "image_path": self.path, "source_frame": self.frame,
            "timestamp_sec": (round(self.timestamp_sec, 4)
                              if self.timestamp_sec is not None else None),
            "selection_reason": self.selection_reason,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass
class WagonGrid:
    """One canonical wagon's 4x4 evidence, plus its identity from the state."""

    global_id: str
    wagon_index: int = 0
    classification: str = ""
    total_wagons: int = 0
    by_camera: Dict[str, List[Slot]] = field(default_factory=dict)

    def slots(self, camera_id: str) -> List[Slot]:
        return self.by_camera.get(camera_id, [])

    @property
    def available_count(self) -> int:
        return sum(1 for c in self.by_camera.values() for s in c if s.available)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_id": self.global_id,
            "wagon_index": self.wagon_index,
            "classification": self.classification,
            "total_wagons": self.total_wagons,
            "slots_per_camera": SLOTS_PER_CAMERA,
            "cameras": {cam: [s.to_dict() for s in self.by_camera.get(cam, [])]
                        for cam in CAMERA_ORDER},
            "available_images": self.available_count,
            "expected_images": SLOTS_PER_CAMERA * len(CAMERA_ORDER),
        }


def _camera_meta(state: Any, camera_id: str,
                 camera_meta: Optional[Dict[str, Dict[str, Any]]]) -> tuple:
    """`(fps, total_frames, offset)` for one camera, from existing state."""
    meta = (camera_meta or {}).get(camera_id) or {}
    fps = float(meta.get("fps") or 0.0)
    total = int(meta.get("total_frames") or 0)
    if fps <= 0:
        fps = float(getattr(state, "master_fps", 0.0) or 0.0)
    if total <= 0:
        total = int(getattr(state, "master_total_frames", 0) or 0)
    offsets = {}
    try:
        offsets = state.camera_time_offsets() or {}
    except Exception:                                            # noqa: BLE001
        offsets = {}
    return fps, total, float(offsets.get(camera_id, 0.0) or 0.0)


def build_wagon_grid(
    wagon: Any, *, cache_root: Optional[str], state: Any,
    camera_meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> WagonGrid:
    """The 4x4 grid for ONE canonical wagon. Never raises."""
    gw_id = str(getattr(wagon, "global_id", "") or "")
    grid = WagonGrid(
        global_id=gw_id,
        wagon_index=int(getattr(wagon, "wagon_index", 0) or 0),
        classification=str(getattr(wagon, "classification", "") or ""),
        total_wagons=len(getattr(state, "wagons", None) or []),
    )
    st = float(getattr(wagon, "start_time", 0.0) or 0.0)
    en = float(getattr(wagon, "end_time", 0.0) or 0.0)

    for cam in CAMERA_ORDER:
        fps, total, offset = _camera_meta(state, cam, camera_meta)
        paths: Sequence[Optional[str]] = [None] * SLOTS_PER_CAMERA
        sf = ef = None
        if cache_root:
            try:
                paths = ev.quartile_cache_paths(
                    cache_root=cache_root, gw_id=gw_id, camera_id=cam,
                    wagon_start_time=st, wagon_end_time=en,
                    local_fps=fps, local_total_frames=total,
                    time_offset=offset)
                sf, ef = ev.wagon_local_frames(st, en, fps, total, offset)
            except Exception as e:                               # noqa: BLE001
                log.warning("[GRID] %s %s selection failed: %s", gw_id, cam, e)
                paths = [None] * SLOTS_PER_CAMERA

        slots: List[Slot] = []
        for i, p in enumerate(list(paths)[:SLOTS_PER_CAMERA]):
            slot = Slot(index=i, camera_id=cam, global_id=gw_id)
            if p:
                slot.available = True
                slot.path = p
                # The frame index is IN the filename the materializer wrote, so
                # it is read back rather than recomputed -- the number shown on
                # the page is then provably the file being shown.
                base = os.path.splitext(os.path.basename(p))[0]
                try:
                    slot.frame = int(base.rsplit("_", 1)[-1])
                    if fps > 0:
                        slot.timestamp_sec = slot.frame / fps + offset
                except (TypeError, ValueError):
                    slot.frame = None
                slot.selection_reason = (
                    f"quartile {(0.125, 0.375, 0.625, 0.875)[i]:.3f} of "
                    f"{gw_id} span on {cam}"
                    + (f" (local frames {sf}..{ef})" if sf is not None else ""))
            else:
                slot.unavailable_reason = (
                    NO_CACHE_ROOT if not cache_root
                    else NO_SPAN if (ef is None or sf is None or ef <= sf)
                    else NOT_ON_DISK)
            slots.append(slot)
        grid.by_camera[cam] = slots
    return grid


def damage_from_evidence(
    *, evidence_root: Optional[str], state: Any, verbose: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """`{GW_n: [damage rows]}` read from the Stage-3 evidence already on disk.

    No damage inference runs here. Each row is one confirmed track as the damage
    processor recorded it in `evidence/<GW>/damage/metadata.json`, paired with
    the image it wrote.

    The image name is CAMERA-SCOPED (`track_2__RIGHT_UP_TOP.jpg`, via
    `core.evidence_identity.damage_track_slot`) because both top cameras write
    into one directory and the index alone collides -- the two photograph the
    same roof from opposite sides, so a mix-up renders as a plausible photo of
    the wrong camera. The legacy unscoped name is accepted only as a fallback
    for evidence written before that rename.

    Iterates the canonical roster, so a row can only ever be filed under the
    GW_n whose directory it came out of.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not evidence_root:
        return out
    from core.evidence_identity import (damage_track_slot,
                                        legacy_damage_track_slot)
    for w in (getattr(state, "wagons", None) or []):
        gw = str(getattr(w, "global_id", "") or "")
        d = os.path.join(evidence_root, gw, "damage")
        meta = ev.evidence_metadata(evidence_root, gw, "damage") or {}
        rows: List[Dict[str, Any]] = []
        for t in (meta.get("tracks") or []):
            if not isinstance(t, dict):
                continue
            cam = str(t.get("camera_id") or "")
            img = None
            try:
                idx = int(t.get("track_idx"))
            except (TypeError, ValueError):
                idx = None
            for cand in ([f"{damage_track_slot(idx, cam)}.jpg"]
                         if idx is not None and cam else []) + (
                         [f"{legacy_damage_track_slot(idx)}.jpg"]
                         if idx is not None else []):
                p = os.path.join(d, cand)
                if os.path.isfile(p):
                    img = p
                    break
            rows.append({
                "global_id": gw,
                "camera_id": cam,
                "track_idx": idx,
                "class_name": t.get("class_name"),
                "confidence": t.get("best_confidence", t.get("confidence")),
                "frame": t.get("best_frame_idx", t.get("frame_idx")),
                "bbox": t.get("bbox"),
                "image_path": img,
                "image_available": bool(img),
                "unavailable_reason": ("" if img else
                                       "damage snapshot not on disk"),
            })
        if rows:
            out[gw] = rows
    if verbose and out:
        log.info("[REPORT-GRID] damage evidence: %d wagon(s), %d finding(s)",
                 len(out), sum(len(v) for v in out.values()))
    return out


def build_manifest(
    *, state: Any, cache_root: Optional[str],
    camera_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    damage_by_wagon: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """The report manifest: every canonical wagon, its 4x4 grid, and damage.

    Driven by `state.wagons` -- the canonical timeline -- so a wagon with no
    feature finding still gets its sixteen slots and cannot drop out of the PDF.
    """
    grids = [build_wagon_grid(w, cache_root=cache_root, state=state,
                              camera_meta=camera_meta)
             for w in (getattr(state, "wagons", None) or [])]
    dmg = dict(damage_by_wagon or {})
    manifest = {
        "canonical_wagons": len(grids),
        "slots_per_camera": SLOTS_PER_CAMERA,
        "camera_order": list(CAMERA_ORDER),
        "top_cameras": list(TOP_ORDER),
        "side_cameras": list(SIDE_ORDER),
        "wagons": [g.to_dict() for g in grids],
        "damage_by_wagon": {gw: list(rows) for gw, rows in dmg.items()
                            if rows},
        "wagons_with_damage": sorted(
            (gw for gw, rows in dmg.items() if rows),
            key=lambda g: int(str(g).split("_")[-1])
            if str(g).split("_")[-1].isdigit() else 0),
        "images_expected": len(grids) * SLOTS_PER_CAMERA * len(CAMERA_ORDER),
        "images_available": sum(g.available_count for g in grids),
    }
    manifest["images_unavailable"] = (manifest["images_expected"]
                                     - manifest["images_available"])
    if verbose:
        log.info("[REPORT-GRID] %d wagon(s) x %d cameras x %d slots = %d "
                 "images expected, %d available, %d marked unavailable",
                 manifest["canonical_wagons"], len(CAMERA_ORDER),
                 SLOTS_PER_CAMERA, manifest["images_expected"],
                 manifest["images_available"], manifest["images_unavailable"])
        if manifest["wagons_with_damage"]:
            log.info("[REPORT-GRID] damage evidence for %d wagon(s): %s",
                     len(manifest["wagons_with_damage"]),
                     manifest["wagons_with_damage"])
    return manifest
