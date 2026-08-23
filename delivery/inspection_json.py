"""Exact per-camera ``inspection_data.json`` in the V4 Train-Inspection-Engine
schema, for all four camera angles.

This module is the SINGLE owner of that contract.  It is a faithful port of
`reporting/json_builder.py` from
``s2pl/CCTV-TrainVideo-ML-V2-wagon-Rithish@V4/Train-Inspection-Engine``: the same
two flavours, the same key names, the same key ORDER, the same derivations.  The
only thing that changes is where the numbers come from -- the V4 engine reads its
own per-camera pandas frames, while here every value is derived from finalized
global_train artifacts (the sealed GlobalTrainState, the fused UnifiedWagonStates,
and each camera's own per-feature JSON + evidence).

Two flavours (V4 vocabulary, verbatim)
--------------------------------------
Top cameras (``RIGHT_UP_TOP``, ``LEFT_UP_TOP``) -- ``flavour="top"``
  * ``segment_type``: ``wagon_empty`` / ``wagon_loaded`` / ``engine`` / ``brakevan``
  * ``segment_type_map[str(seg_id)]`` = ``{type, number, wagon_count}``
  * per-wagon: 3 damage booleans (``floor_dmg`` / ``inner_wall_dmg`` /
    ``floor_dmg_probable``) + ``load_status``
  * top-level: ``floor_dmg_wagons``, ``inner_wall_dmg_wagons``,
    ``floor_dmg_probable_wagons``, ``probable_damage_wagons``
  * ``rake_status`` = majority vote of this camera's loaded vs empty wagons
  * ``loco_number_results`` = ``{}``

Side cameras (``RIGHT_UP``, ``LEFT_UP``) -- ``flavour="side"``
  * ``segment_type``: ``wagon`` / ``engine`` / ``brakevan`` (no load split)
  * ``segment_type_map[str(seg_id)]`` = ``{type, number}``
  * per-wagon: ``door_status`` + ``door_close_detected`` + ``door_partial_detected``
    + ``damage_detected``
  * top-level: ``doors_open``, ``doors_partially_closed``, ``doors_closed``,
    ``damaged_wagons``
  * ``rake_status`` inferred from travel ``direction`` (left-to-right = Loaded)
  * ``wagon_number_results`` keyed by ``str(wagon_count)``

Camera authority is respected (this is what makes the four files differ):
``RIGHT_UP`` reports the right door + OCR, ``LEFT_UP`` the left door,
``RIGHT_UP_TOP`` / ``LEFT_UP_TOP`` each report their OWN load + damage reads --
never the other camera's.  A field this camera has no authority for is reported
from its own per-camera state or left at the schema's neutral value; it is never
back-filled from a different camera.

Everything here is PURE: it reads already-written JSON + checks for the existence
of evidence JPEGs, and returns a dict.  No model, no video, no network, no writes.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

from core import constants as C
from core.evidence_identity import (
    damage_track_slot, legacy_damage_track_slot, load_best_frame_slot,
    legacy_load_best_frame_slot)
from core.logging_setup import get_logger

log = get_logger("delivery.inspection_json")

FLAVOUR_TOP = "top"
FLAVOUR_SIDE = "side"

# -----------------------------------------------------------------------------
# Schema dialect.
#
# The V4 engine and the older V1 dashboard feed agree on the top-level key list
# (V4 side == V1 side plus `doors_partially_closed` and `damage_model_active`),
# but they disagree on FIVE nested details.  Emitting V4 shapes at a V1 consumer
# breaks it, so the dialect follows the document's `version`:
#
#   detail                     v1 (old dashboard)              v4 (V4 engine)
#   ------------------------------------------------------------------------
#   bounding_box               {bounding_box_coordinates,      [x1,y1,x2,y2]
#                               confidence, class_name}
#   open-door problem_type     "door_open"                     "open_door"
#   problem_frames_by_type     {damage, door_open}             {damage, open_door,
#                                                               closed_door,
#                                                               partially_closed}
#   problem segment_number     the wagon_count                 null (side)
#   side rake_status           right-to-left == Loaded          left-to-right == Loaded
#
# The last one is a genuine SEMANTIC disagreement between the two sources, not a
# formatting choice -- the old RIGHT_UP pipeline treats right-to-left as loaded,
# while V4's right_up.yaml sets loaded_direction: left-to-right.  Each dialect
# reproduces its own source so neither consumer silently flips.
# -----------------------------------------------------------------------------

SCHEMA_V1 = "v1"
SCHEMA_V4 = "v4"


def schema_for_version(version: str) -> str:
    """Map a document `version` to the schema dialect to emit."""
    return SCHEMA_V1 if (version or "").strip().lower() == "v1" else SCHEMA_V4


def flavour_for(camera: str) -> str:
    """``"top"`` for the two top cameras, ``"side"`` for the two side cameras."""
    return FLAVOUR_TOP if camera in C.TOP_CAMERAS else FLAVOUR_SIDE


# ---------------------------------------------------------------------------
# Segment-type vocabulary (V4 artifacts.display_segment_type, verbatim)
# ---------------------------------------------------------------------------

# global_train classification -> V4 internal segment_type
_INTERNAL_FROM_CLASSIFICATION = {
    C.CLASS_ENGINE:    "engine",
    C.CLASS_BRAKE_VAN: "brakevan",
    C.CLASS_WAGON:     "wagon",
    C.CLASS_UNKNOWN:   "wagon",
}

TOP_DISPLAY = {
    "wagon": "wagon_empty",
    "wagon_loaded": "wagon_loaded",
    "engine": "engine",
    "brakevan": "brakevan",
}
SIDE_DISPLAY = {
    "wagon": "wagon",
    "wagon_loaded": "wagon",       # side cameras don't differentiate
    "engine": "engine",
    "brakevan": "brakevan",
}


def display_segment_type(seg_type: str, flavour: str) -> str:
    """Map internal segment_type to the JSON-emitted label (V4 rule)."""
    if flavour == FLAVOUR_TOP:
        return TOP_DISPLAY.get(seg_type, seg_type)
    return SIDE_DISPLAY.get(seg_type, seg_type)


def _is_wagon(seg_type: str) -> bool:
    return seg_type in {"wagon", "wagon_empty", "wagon_loaded"}


def _load_status(seg_type: str) -> Optional[str]:
    if seg_type == "wagon_loaded":
        return "loaded"
    if seg_type in {"wagon", "wagon_empty"}:
        return "empty"
    return None


def _rake_status_from_wagon_counts(wagons_loaded: int, wagons_empty: int) -> str:
    """Top flavour: majority vote across this camera's segment load classes."""
    return "Loaded" if wagons_loaded >= wagons_empty else "Empty"


def _rake_status_from_direction(direction: str,
                                schema: str = SCHEMA_V4) -> str:
    """Side flavour: no per-wagon load class exists, so infer the rake's overall
    status from travel direction.

    The two sources disagree on the polarity, so each dialect keeps its own:
      * v4  -- ``left-to-right`` == Loaded (V4 json_builder + right_up.yaml's
               ``loaded_direction: left-to-right``)
      * v1  -- ``right-to-left`` == Loaded (the old RIGHT_UP pipeline's
               ``"Loaded" if "right-to-left" in detected_direction``)
    """
    if direction not in ("left-to-right", "right-to-left"):
        return "Unknown"
    loaded = "right-to-left" if schema == SCHEMA_V1 else "left-to-right"
    return "Loaded" if direction == loaded else "Empty"


def strip_camera_prefix(folder: str) -> str:
    """``camera_CCTV_HZBN_DHN_6_LEFT_TOP`` -> ``CCTV_HZBN_DHN_6_LEFT_TOP``."""
    if folder and folder.lower().startswith("camera_"):
        return folder[7:]
    return folder


# ---------------------------------------------------------------------------
# Per-camera source reads
# ---------------------------------------------------------------------------

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


#: Which camera each feature is AUTHORITATIVE for in this package's fusion
#: (fusion/wagon_state_builder.py + README "Authority rules").  Consulted only
#: when reading the FLAT layout, to decide whether a fused file may be attributed
#: to the camera asking for it.  `None` means "any camera may read it".
_FEATURE_AUTHORITY: Dict[str, Optional[tuple]] = {
    # OCR is RIGHT_UP-only; attributing a wagon number to any other camera would
    # publish a number that camera never read.
    "ocr":    (C.CAMERA_RIGHT_UP,),
    # Load is a TOP-camera concern (RIGHT_UP_TOP primary, LEFT_UP_TOP fallback).
    "load":   C.TOP_CAMERAS,
    # Doors and damage carry their own per-camera/per-side breakdown, resolved
    # below, so both side / both top cameras may read them.
    "door":   None,
    "damage": None,
}


def _project_camera_view(payload: Dict[str, Any], feature: str,
                         camera: str) -> Optional[Dict[str, Any]]:
    """Narrow one FUSED per-wagon feature file to what `camera` observed.

    This package fuses each feature into a single ``wagon_states/<feature>/
    <GW_n>.json`` covering every contributing camera, whereas the V4 schema is
    per-camera: each document must state what THAT camera saw and nothing else.
    Handing the fused file over unnarrowed would republish (say) RIGHT_UP_TOP's
    damage as LEFT_UP_TOP's finding.

    Narrowing uses what the processors already record -- no new inference:
      * ``per_camera[camera]``   the damage processor's own per-camera block,
                                 overlaid on the fused base.
      * ``camera_id`` on each    detail/track entry, so evidence lists keep only
                                 this camera's rows.
      * ``_FEATURE_AUTHORITY``   features that are one camera's alone.

    Returns ``None`` when this camera contributed nothing to the feature, which
    the callers already treat as "no result from me" (falling back to the fused
    value where the schema allows it).
    """
    allowed = _FEATURE_AUTHORITY.get(feature, None)
    if allowed is not None and camera not in allowed:
        return None

    view = dict(payload)

    # 1) An explicit per-camera block wins: it IS this camera's own result.
    per_camera = payload.get("per_camera")
    if isinstance(per_camera, dict):
        own = per_camera.get(camera)
        if not isinstance(own, dict):
            return None          # this camera contributed nothing
        view.update(own)

    # 2) Keep only this camera's rows in every per-detection list.
    for key in ("top_damage_details", "side_damage_details", "details", "tracks"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        tagged = [r for r in rows if isinstance(r, dict) and "camera_id" in r]
        if tagged:
            view[key] = [r for r in tagged if r.get("camera_id") == camera]

    # 3) A camera named in `supporting_cameras` but absent from every detail row
    #    still legitimately reported "nothing found"; one not named there did not
    #    report at all.
    supporting = payload.get("supporting_cameras")
    if isinstance(supporting, list) and supporting and camera not in supporting:
        return None
    return view


def _camera_feature_json(states_root: str, feature: str, camera: str,
                         gw_id: str) -> Optional[Dict[str, Any]]:
    """This camera's OWN per-wagon feature result (never another camera's).

    Two on-disk layouts are supported, in order:

      1. ``wagon_states/<feature>/<CAMERA>/<GW_n>.json`` -- a genuinely
         per-camera file; returned verbatim, nothing to narrow.
      2. ``wagon_states/<feature>/<GW_n>.json`` -- this package's FUSED layout,
         narrowed to `camera` by :func:`_project_camera_view`.
    """
    nested = _read_json(os.path.join(states_root, feature, camera, f"{gw_id}.json"))
    if nested is not None:
        return nested
    flat = _read_json(os.path.join(states_root, feature, f"{gw_id}.json"))
    if flat is None:
        return None
    return _project_camera_view(flat, feature, camera)


def _evidence_meta(evidence_root: str, gw_id: str, feature: str,
                   camera: str) -> Dict[str, Any]:
    """Evidence metadata for one (wagon, feature, camera).

    Same two layouts as :func:`_camera_feature_json`: a per-camera evidence
    subtree if one exists, else this package's flat ``evidence/<GW_n>/<feature>/``
    directory narrowed by the ``per_camera`` block the processors write there.
    """
    nested = _read_json(os.path.join(evidence_root, gw_id, feature, camera,
                                     "metadata.json"))
    if nested is not None:
        return nested
    flat = _read_json(os.path.join(evidence_root, gw_id, feature,
                                   "metadata.json")) or {}
    if not flat:
        return {}
    narrowed = _project_camera_view(flat, feature, camera)
    return narrowed if narrowed is not None else {}


# Damage class name (lowercased, as the processors record it) -> the V4
# top-flavour boolean it drives.  `damage.pt` emits `Floor_damage`,
# `Inner_wall_damage` and `Floor__probable_damage`; the last one carries a DOUBLE
# underscore, which is the model's real class name.  Getting that key wrong is
# what previously left `floor_dmg_probable_*` structurally zero.
_DAMAGE_CLASS_TO_FLAG = {
    # confirmed
    "floor_damage":            "floor_dmg_detected",
    "floor_dmg":               "floor_dmg_detected",
    "inner_wall_damage":       "inner_wall_dmg_detected",
    "inner_wall_dmg":          "inner_wall_dmg_detected",
    # probable (NOT confirmed damage)
    "floor__probable_damage":  "floor_dmg_probable_detected",   # damage.pt
    "floor_probable_damage":   "floor_dmg_probable_detected",
    "floor_damage_probable":   "floor_dmg_probable_detected",
    "floor_dmg_probable":      "floor_dmg_probable_detected",
    "probable_floor_damage":   "floor_dmg_probable_detected",
}


def _top_damage_flags(states_root: str, camera: str, gw_id: str) -> Dict[str, bool]:
    """Per-class top-damage booleans from THIS camera's own damage result.

    global_train's damage processor records each confirmed track with its class
    name; those map onto V4's three top-damage booleans.  A class this model
    doesn't emit (notably ``floor_dmg_probable``, which the current
    ``damage.pt`` has no equivalent for) stays ``False`` -- reported, never
    invented.
    """
    flags = {"floor_dmg_detected": False,
             "inner_wall_dmg_detected": False,
             "floor_dmg_probable_detected": False}
    payload = _camera_feature_json(states_root, "damage", camera, gw_id)
    if not payload or payload.get("status") != C.STATUS_OK:
        return flags
    if payload.get("damage_status") != C.DAMAGE_PRESENT:
        return flags
    details = payload.get("top_damage_details") or []
    for det in details:
        cls = str(det.get("class_name") or det.get("class") or "").lower()
        flag = _DAMAGE_CLASS_TO_FLAG.get(cls)
        if flag:
            flags[flag] = True
    if not any(flags.values()):
        # A track survived filtering but its class label is outside the known
        # vocabulary.  Attribute it to confirmed floor damage (the model's
        # dominant class) rather than losing the finding -- but only if the label
        # isn't a probable variant, which must never become confirmed damage.
        unknown_probable = any(
            C.is_probable_damage(str(d.get("class_name") or d.get("class") or ""))
            for d in details)
        if unknown_probable:
            flags["floor_dmg_probable_detected"] = True
        else:
            flags["floor_dmg_detected"] = True
    return flags


# global_train door state -> (V4 door_status, door_partial, damage_detected)
_DOOR_STATE_MAP = {
    C.DOOR_OPEN:    ("open", False, False),
    C.DOOR_CLOSED:  ("closed", False, False),
    C.DOOR_PARTIAL: ("partially_closed", True, False),
    # A DAMAGED door is a damage finding, not a door position; V4's side flavour
    # carries those on `damage_detected` and leaves door_status at closed.
    C.DOOR_DAMAGED: ("closed", False, True),
}


def _side_door_fields(states_root: str, camera: str, gw_id: str,
                      unified_door: Optional[str]) -> Dict[str, Any]:
    """Door fields for a side camera, from THIS camera's own door result."""
    payload = _camera_feature_json(states_root, "door", camera, gw_id)
    side = "right" if camera == C.CAMERA_RIGHT_UP else "left"
    state: Optional[str] = None
    if payload and payload.get("status") == C.STATUS_OK:
        state = payload.get(f"{side}_door") or payload.get("door_state")
    if state in (None, C.NO_DATA):
        state = unified_door if unified_door not in (None, C.NO_DATA) else None

    status, partial, damaged = _DOOR_STATE_MAP.get(
        state or "", ("closed", False, False))
    return {
        "door_status": status,
        # V4's `door_close_detected` is "the closed-door class fired for this
        # wagon" -- i.e. the door was positively observed shut, as opposed to
        # defaulting to closed because nothing was detected.
        "door_close_detected": bool(state == C.DOOR_CLOSED),
        "door_partial_detected": bool(partial),
        "_damage_from_door": bool(damaged),
    }


# ---------------------------------------------------------------------------
# Wagon-frame gallery (references only files that exist)
# ---------------------------------------------------------------------------

# Ordered gallery candidates per camera role, as "<feature>/<filename>".
_SIDE_GALLERY = ("door/{side}_best.jpg", "door/{side}_crop.jpg",
                 "ocr/best_frame.jpg", "ocr/number_crop.jpg")
#: `{cam}` is substituted with the asking camera. Both entries MUST carry it:
#: `load/best_frame.jpg` and `damage/track_1.jpg` are camera-ambiguous names,
#: and handing either to whichever camera asked is what published one top
#: camera's photo as the other's. The camera-scoped names are what the writers
#: actually produce now (features/load/processor.py, features/damage/processor.py
#: via core.evidence_identity).
_TOP_GALLERY = ("load/best_frame__{cam}.jpg",
                "damage/track_1__{cam}.jpg",
                "damage/track_2__{cam}.jpg",
                "damage/track_3__{cam}.jpg")

#: V4 side cameras sample 4 representative frames named start/mid1/mid2/end
#: (configs/cameras/right_up.yaml: representative_position_names).
POSITION_NAMES = ("start", "mid1", "mid2", "end")


def _wagon_frames(evidence_root: str, gw_id: str, camera: str,
                  flavour: str, url_for: Callable[..., Optional[str]]) -> List[Dict[str, Any]]:
    """``wagon_frames`` for one wagon: ``[{position, s3_url}, ...]``.

    Only evidence JPEGs that ACTUALLY exist are referenced -- a URL is never
    fabricated for a frame the pipeline did not produce.
    """
    side = "right" if camera == C.CAMERA_RIGHT_UP else "left"
    templates = (tuple(t.format(cam=camera) for t in _TOP_GALLERY)
                 if flavour == FLAVOUR_TOP
                 else tuple(t.format(side=side) for t in _SIDE_GALLERY))
    frames: List[Dict[str, Any]] = []
    for rel in templates:
        feature, filename = rel.split("/", 1)
        url = url_for(gw_id=gw_id, feature=feature, camera=camera,
                      filename=filename)
        if not url and feature == "load":
            # An evidence tree written before load frames carried the camera has
            # only the fused `best_frame.jpg`. That file belongs to ONE camera,
            # named in metadata.json's `source_camera`, and may be published by
            # that camera alone -- borrowing it is the bug, not the fallback.
            # Unproven ownership is not ownership: no attribution means no URL.
            meta = _read_json(os.path.join(evidence_root, gw_id, "load",
                                           "metadata.json")) or {}
            if (meta.get("source_camera") or meta.get("camera_id")) == camera:
                url = url_for(gw_id=gw_id, feature=feature, camera=camera,
                              filename=f"{legacy_load_best_frame_slot()}.jpg")
        if not url:
            continue
        frames.append({
            "position": POSITION_NAMES[min(len(frames), len(POSITION_NAMES) - 1)],
            "s3_url": url,
        })
        if len(frames) >= len(POSITION_NAMES):
            break
    return frames


# ---------------------------------------------------------------------------
# Problem frames
# ---------------------------------------------------------------------------

#: Open-door problem_type label per dialect (the old dashboard says "door_open",
#: V4 says "open_door" -- one is not a typo for the other, both are real).
PROBLEM_TYPE_OPEN_DOOR = {SCHEMA_V1: "door_open", SCHEMA_V4: "open_door"}


def _bounding_box(bbox: Optional[Sequence[float]], *, schema: str,
                  confidence: Optional[float],
                  class_name: Optional[str]) -> Any:
    """``bounding_box`` in the dialect's shape.

    v1 wraps the coordinates with the detection's confidence + class name; v4
    emits the bare ``[x1, y1, x2, y2]`` list.
    """
    if not bbox:
        return None
    coords = [float(v) for v in list(bbox)[:4]]
    if schema != SCHEMA_V1:
        return coords
    return {
        "bounding_box_coordinates": coords,
        "confidence": round(float(confidence or 0.0), 3),
        "class_name": class_name or "",
    }


#: How far to search when recovering a damage snapshot positionally. The damage
#: processor numbers tracks from 1 within a wagon and a wagon with more than a
#: handful of distinct damage tracks does not occur in practice.
_MAX_DAMAGE_TRACK_PROBE = 12


def _damage_track_url(url_for: Callable[..., Optional[str]], gw_id: str,
                      camera: str, track: Dict[str, Any],
                      claimed: Optional[set] = None) -> Optional[str]:
    """URL for one damage track's snapshot, or None -- never an exception.

    Resolution is by DISCOVERY, not by arithmetic, because the index is the one
    field that has proved unreliable and the picture is worth more than the
    bookkeeping. In production this wagon showed the failure exactly:

        wagon_frames   -> .../damage/track_1__LEFT_UP_TOP.jpg     (found)
        problem_frames -> null                                     (lost)

    Same wagon, same camera, same file. `wagon_frames` brute-forces the fixed
    names `track_1..3__<CAM>`; `problem_frames` computed the name from
    `track_idx`, which had been nulled upstream. One strategy survived a bad
    field and the other did not, so this now uses the surviving one.

    Order: the track's own index first, since that is the correct answer when
    the field is sound; then each index in turn, skipping any file an earlier
    track in this wagon already claimed, so two tracks never publish one photo;
    then the legacy unscoped name, which is safe because `_project_camera_view`
    has already dropped every track belonging to another camera.

    `track_idx` may be absent, null, or non-numeric -- `int(None)` raising here
    is what failed four trains' entire top-camera documents. One unusable track
    must cost that track's picture and nothing else.
    """
    claimed = claimed if claimed is not None else set()

    def _try(idx: int) -> Optional[str]:
        slot = damage_track_slot(idx, camera)
        if slot in claimed:
            return None
        u = url_for(gw_id=gw_id, feature="damage", camera=camera,
                    filename=f"{slot}.jpg")
        if u:
            claimed.add(slot)
        return u

    try:
        own = int(track.get("track_idx"))
    except (TypeError, ValueError):
        own = None
    if own is not None:
        got = _try(own)
        if got:
            return got

    for idx in range(1, _MAX_DAMAGE_TRACK_PROBE + 1):
        if idx == own:
            continue
        got = _try(idx)
        if got:
            return got

    if own is not None:
        legacy = legacy_damage_track_slot(own)
        if legacy not in claimed:
            u = url_for(gw_id=gw_id, feature="damage", camera=camera,
                        filename=f"{legacy}.jpg")
            if u:
                claimed.add(legacy)
                return u
    return None


def _damage_frame_number(track: Dict[str, Any]) -> Optional[int]:
    """The frame this damage was best seen in.

    The damage processor records it as `best_frame_idx`
    (features/damage/processor.py's `track_meta`); there is no `frame_idx` key,
    so reading that one alone published `frame_number: null` for every damage
    problem frame. Both names are accepted -- the state record's
    `top_damage_details` does use `frame_idx`.
    """
    for key in ("best_frame_idx", "frame_idx"):
        try:
            v = track.get(key)
            if v is not None:
                return int(v)
        except (TypeError, ValueError):
            continue
    return None


def _problem_frame(*, wagon_count: Optional[int], segment_type: str,
                   segment_number: Optional[int], problem_type: str,
                   frame_number: Optional[int], url: Optional[str],
                   bbox: Optional[Sequence[float]],
                   extra: Dict[str, Any],
                   schema: str = SCHEMA_V4,
                   confidence: Optional[float] = None,
                   class_name: Optional[str] = None) -> Dict[str, Any]:
    """One ``problem_frames[]`` entry (shared shape; flavour extras appended)."""
    filename = os.path.basename(url) if url else None
    entry: Dict[str, Any] = {
        "wagon_count": wagon_count,
        "segment_type": segment_type,
        # v1 carries the wagon number here; V4's side flavour leaves it null.
        "segment_number": (wagon_count if schema == SCHEMA_V1 else segment_number),
        "problem_type": problem_type,
        "frame_number": frame_number,
        "filename": filename,
        "s3_key": None,
        "s3_url": url,
        "is_annotated": bool(url),
        "annotated_image_url": url,
        "bounding_box": _bounding_box(bbox, schema=schema, confidence=confidence,
                                     class_name=class_name or problem_type),
    }
    entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def build_inspection_json(
    *,
    camera: str,
    camera_folder: str,
    raw_video_name: str,
    upload_timestamp: Optional[datetime],
    direction: str,
    state,
    unified: Dict[str, Any],
    states_root: str,
    evidence_root: str,
    url_for: Callable[..., Optional[str]],
    trimmed_video_url: Optional[str] = None,
    pdf_report_url: Optional[str] = None,
    detected_video_url: Optional[str] = None,
    raw_video_urls: Optional[List[str]] = None,
    damage_model_active: bool = True,
    version: str = "v4",
    identified_by: str = "model-v3",
    schema: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the ``{camera_id, version, inspection_data}`` document for `camera`.

    Parameters
    ----------
    state:
        The sealed ``GlobalTrainState`` -- supplies the canonical wagon sequence
        (``GW_1..GW_N``), so all four camera files describe the SAME wagons in the
        same order with the same ``wagon_count`` numbering.
    unified:
        ``{gw_id -> UnifiedWagonState-or-dict}`` from Stage-4 fusion.  Used only
        for values this camera has authority over, plus the load class that
        selects ``wagon_empty`` vs ``wagon_loaded`` on a top camera.
    url_for:
        ``url_for(gw_id=, feature=, camera=, filename=) -> url | None``.  Must
        return ``None`` when the underlying evidence file does not exist, so the
        gallery never references a missing image.
    """
    flavour = flavour_for(camera)
    # Dialect follows the document version unless the caller pins it.
    schema = schema or schema_for_version(version)
    wagons = list(getattr(state, "wagons", []) or [])

    def _u(gw_id: str) -> Dict[str, Any]:
        """Unified state for a wagon as a plain dict."""
        u = unified.get(gw_id)
        if u is None:
            return {}
        if isinstance(u, dict):
            return u
        to_dict = getattr(u, "to_dict", None)
        return to_dict() if callable(to_dict) else {}

    # ---- internal segment types (drives every downstream label) ----
    internal_types: Dict[str, str] = {}
    for gw in wagons:
        base = _INTERNAL_FROM_CLASSIFICATION.get(
            getattr(gw, "classification", "") or "", "wagon")
        if base == "wagon" and flavour == FLAVOUR_TOP:
            # Top flavour splits wagons by load class.  Prefer THIS camera's own
            # load read; fall back to the fused value only when this camera has
            # no result of its own (a report must still name a load class).
            own = _camera_feature_json(states_root, "load", camera, gw.global_id)
            load_status = None
            if own and own.get("status") == C.STATUS_OK:
                load_status = own.get("load_status")
            if load_status in (None, C.NO_DATA):
                load_status = _u(gw.global_id).get("load_status")
            if load_status == C.LOAD_LOADED:
                base = "wagon_loaded"
        internal_types[gw.global_id] = base

    # ---- segment_type_map + wagon_count numbering (V4 helper, verbatim) ----
    type_counters: Dict[str, int] = {}
    segment_type_map: Dict[str, Dict[str, Any]] = {}
    wagon_count_map: Dict[str, Optional[int]] = {}
    segment_ids: Dict[str, int] = {}
    wagon_counter = 0
    for gw in wagons:
        gw_id = gw.global_id
        seg_id = int(getattr(gw, "wagon_index", 0) or 0)
        segment_ids[gw_id] = seg_id
        internal = internal_types[gw_id]
        display = display_segment_type(internal, flavour)
        type_counters[display] = type_counters.get(display, 0) + 1
        number = type_counters[display]
        if _is_wagon(internal):
            wagon_counter += 1
            wagon_count_map[gw_id] = wagon_counter
        else:
            wagon_count_map[gw_id] = None
        entry: Dict[str, Any] = {"type": display, "number": number}
        if flavour == FLAVOUR_TOP:
            entry["wagon_count"] = wagon_count_map[gw_id]
        segment_type_map[str(seg_id)] = entry

    # ---- header block (identical key order to V4) ----
    inspection_data: Dict[str, Any] = {
        "raw_video_name": raw_video_name,
        "identified_by": identified_by,
        "upload_timestamp": (upload_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
                             if upload_timestamp else None),
        "upload_timestamp_readable": (
            upload_timestamp.strftime("%d-%m-%Y %H:%M:%S IST")
            if upload_timestamp else None),
        "direction": direction,
        "rake_status": (None if flavour == FLAVOUR_TOP
                        else _rake_status_from_direction(direction, schema)),
        "pdf_report_url": pdf_report_url,
        "trimmed_video_url": trimmed_video_url,
        "detected_video_url": detected_video_url,
        "raw_video_urls": list(raw_video_urls or []),
    }

    n_engines = 0
    n_brakevans = 0
    problem_frames: List[Dict[str, Any]] = []
    problem_type_counts: Dict[str, int] = {}

    def _bump(problem_type: str) -> None:
        problem_type_counts[problem_type] = problem_type_counts.get(problem_type, 0) + 1

    # =====================================================================
    # TOP flavour
    # =====================================================================
    if flavour == FLAVOUR_TOP:
        wagons_loaded = wagons_empty = 0
        damaged_wagons = probable_damage_wagons = 0
        floor_dmg_wagons = inner_wall_dmg_wagons = floor_dmg_probable_wagons = 0
        wagon_segments: List[Dict[str, Any]] = []
        wagon_number_block: Dict[str, Dict[str, Any]] = {}

        for gw in wagons:
            gw_id = gw.global_id
            seg_id = segment_ids[gw_id]
            internal = internal_types[gw_id]
            display = display_segment_type(internal, FLAVOUR_TOP)

            if display == "engine":
                n_engines += 1
                continue
            if display == "brakevan":
                n_brakevans += 1
                continue

            flags = _top_damage_flags(states_root, camera, gw_id)
            floor_dmg = flags["floor_dmg_detected"]
            inner_wall_dmg = flags["inner_wall_dmg_detected"]
            floor_prob = flags["floor_dmg_probable_detected"]
            damage_any = floor_dmg or inner_wall_dmg
            probable_any = floor_prob

            if internal == "wagon_loaded":
                wagons_loaded += 1
            else:
                wagons_empty += 1
            if damage_any:
                damaged_wagons += 1
            if probable_any:
                probable_damage_wagons += 1
            if floor_dmg:
                floor_dmg_wagons += 1
            if inner_wall_dmg:
                inner_wall_dmg_wagons += 1
            if floor_prob:
                floor_dmg_probable_wagons += 1

            wagon_count = wagon_count_map[gw_id]
            seg: Dict[str, Any] = {
                "segment_id": seg_id,
                "segment_type": display,
                "wagon_count": wagon_count,
                "load_status": _load_status(internal),
                "load_condition": None,
                "damage_detected": damage_any,
                "probable_damage_detected": probable_any,
                "floor_dmg_detected": floor_dmg,
                "inner_wall_dmg_detected": inner_wall_dmg,
                "floor_dmg_probable_detected": floor_prob,
                "wagon_frames": _wagon_frames(evidence_root, gw_id, camera,
                                              FLAVOUR_TOP, url_for),
            }
            # A top camera has no OCR authority (OCR is RIGHT_UP only), so the
            # number is reported from the fused value when it is valid.
            ocr = _ocr_fields(states_root, evidence_root, gw_id, _u(gw_id), url_for)
            if ocr["is_valid_11_digit"]:
                seg["wagon_number"] = ocr["display_number"]
                seg["is_valid_wagon_id"] = True
            else:
                seg["is_valid_wagon_id"] = False
            wagon_segments.append(seg)

            if wagon_count is not None and ocr["has_result"]:
                wagon_number_block[str(wagon_count)] = {
                    "is_valid_11_digit": ocr["is_valid_11_digit"],
                    "display_number": ocr["display_number"],
                    "is_manipulated": ocr["is_manipulated"],
                    "original_number": ocr["original_number"],
                }

            # ---- problem frames: one per confirmed damage track ----
            dmeta = _evidence_meta(evidence_root, gw_id, "damage", camera)
            # Per WAGON: two tracks on one wagon must not publish the same photo.
            claimed_damage_slots: set = set()
            for track in (dmeta.get("tracks") or []):
                cls = str(track.get("class_name") or "damage").lower()
                ptype = _V4_TOP_PROBLEM_TYPE.get(cls, cls)
                _bump(ptype)
                dmg_url = _damage_track_url(url_for, gw_id, camera, track,
                                            claimed_damage_slots)
                problem_frames.append(_problem_frame(
                    wagon_count=wagon_count, segment_type=display,
                    segment_number=segment_type_map[str(seg_id)].get("number"),
                    problem_type=ptype,
                    frame_number=_damage_frame_number(track),
                    url=dmg_url,
                    bbox=track.get("bbox"), schema=schema,
                    confidence=track.get("best_confidence", track.get("confidence")),
                    class_name=cls,
                    extra={
                        "load_status": _load_status(internal),
                        "load_condition": None,
                        "damage_detected": ptype in {"floor_dmg", "inner_wall_dmg"},
                    }))

        total_wagons = wagons_loaded + wagons_empty
        inspection_data["rake_status"] = _rake_status_from_wagon_counts(
            wagons_loaded, wagons_empty)
        inspection_data.update({
            "total_wagons": total_wagons,
            "wagons_loaded": wagons_loaded,
            "wagons_empty": wagons_empty,
            "damaged_wagons": damaged_wagons,
            "probable_damage_wagons": probable_damage_wagons,
            "floor_dmg_wagons": floor_dmg_wagons,
            "inner_wall_dmg_wagons": inner_wall_dmg_wagons,
            "floor_dmg_probable_wagons": floor_dmg_probable_wagons,
            "num_engines": n_engines,
            "num_brakevans": n_brakevans,
            "total_loco_frames": 0,
            "total_problem_frames": len(problem_frames),
            "problem_frames_by_type": {
                "floor_dmg": problem_type_counts.get("floor_dmg", 0),
                "inner_wall_dmg": problem_type_counts.get("inner_wall_dmg", 0),
                "floor_dmg_probable": problem_type_counts.get("floor_dmg_probable", 0),
            },
            "wagon_number_results": wagon_number_block,
            "loco_number_results": {},
            "segment_type_map": segment_type_map,
            "wagon_segments": wagon_segments,
            "loco_frames": [],
            "problem_frames": problem_frames,
        })
        # v1's key list has no damage_model_active; V4 does.
        if schema != SCHEMA_V1:
            inspection_data["damage_model_active"] = damage_model_active


    # =====================================================================
    # SIDE flavour
    # =====================================================================
    else:
        damaged_wagons = 0
        doors_open = doors_partially_closed = doors_closed = 0
        wagon_segments = []
        wagon_number_block = {}
        is_ocr_camera = (camera == C.CAMERA_RIGHT_UP)

        for gw in wagons:
            gw_id = gw.global_id
            seg_id = segment_ids[gw_id]
            internal = internal_types[gw_id]

            # Engines and brake vans are NOT wagons: they never enter
            # `wagon_segments`, never get a wagon_count, and never move the
            # wagon statistics -- that is V4's wagon list and it stays exact.
            #
            # They ARE still physical vehicles that can carry a defect, though.
            # Skipping them outright meant a damaged brake-van door was counted
            # nowhere and emitted nowhere, so the PDF (whose generator has no
            # such exclusion) showed the finding while the dashboard showed
            # "NO DAMAGE FOUND" for the same wagon.  Their door state is now
            # evaluated so a real defect still reaches `problem_frames[]`,
            # tagged with its own segment_type rather than masquerading as a
            # wagon.
            non_wagon = internal in ("engine", "brakevan")
            if internal == "engine":
                n_engines += 1
            elif internal == "brakevan":
                n_brakevans += 1

            u = _u(gw_id)
            side = "right" if camera == C.CAMERA_RIGHT_UP else "left"
            door = _side_door_fields(states_root, camera, gw_id,
                                     u.get(f"{side}_door"))
            door_status = door["door_status"]
            damage_any = door["_damage_from_door"]
            wagon_count = wagon_count_map[gw_id]      # None for non-wagons

            if not non_wagon:
                if damage_any:
                    damaged_wagons += 1
                if door_status == "open":
                    doors_open += 1
                elif door_status == "partially_closed":
                    doors_partially_closed += 1
                else:
                    doors_closed += 1

                seg = {
                    "segment_id": seg_id,
                    "segment_type": "wagon",
                    "wagon_count": wagon_count,
                    "door_status": door_status,
                    "door_close_detected": door["door_close_detected"],
                    "door_partial_detected": door["door_partial_detected"],
                    "damage_detected": damage_any,
                    "wagon_frames": _wagon_frames(evidence_root, gw_id, camera,
                                                  FLAVOUR_SIDE, url_for),
                }
                # OCR authority is RIGHT_UP only; LEFT_UP never claims a number.
                ocr = (_ocr_fields(states_root, evidence_root, gw_id, u, url_for)
                       if is_ocr_camera
                       else {"has_result": False, "is_valid_11_digit": False,
                             "display_number": "-", "is_manipulated": False,
                             "original_number": "-", "ocr_frame_s3_url": None})
                if ocr["is_valid_11_digit"]:
                    seg["wagon_number"] = ocr["display_number"]
                    seg["is_valid_wagon_id"] = True
                else:
                    seg["is_valid_wagon_id"] = False
                wagon_segments.append(seg)
            else:
                ocr = {"has_result": False, "is_valid_11_digit": False,
                       "display_number": "-", "is_manipulated": False,
                       "original_number": "-", "ocr_frame_s3_url": None}

            # Every read is emitted (even invalid ones) so a bad OCR attempt is
            # still auditable -- only is_valid_11_digit marks a usable number.
            if wagon_count is not None and ocr["has_result"]:
                wagon_number_block[str(wagon_count)] = {
                    "is_valid_11_digit": ocr["is_valid_11_digit"],
                    "display_number": ocr["display_number"],
                    "is_manipulated": ocr["is_manipulated"],
                    "original_number": ocr["original_number"],
                    "ocr_frame_s3_url": ocr["ocr_frame_s3_url"],
                }

            # ---- problem frames: open / partially-closed door, side damage ----
            dmeta = _evidence_meta(evidence_root, gw_id, "door", camera)
            side_meta = (dmeta.get("sides") or {}).get(side, {}) or {}
            ptype = None
            if door_status == "open":
                ptype = PROBLEM_TYPE_OPEN_DOOR[schema]
            elif door_status == "partially_closed":
                ptype = "partially_closed"
            elif damage_any:
                ptype = "damage"
            if ptype is not None:
                _bump(ptype)
                problem_frames.append(_problem_frame(
                    wagon_count=wagon_count,
                    # ALWAYS "wagon" in the problem-frame feed.
                    #
                    # A defect is reported against the vehicle that carries it,
                    # and the dashboard's problem list has only ever shown
                    # "wagon".  Sending "brakevan"/"engine" here would be more
                    # precise but risks the receiver ignoring a value it has
                    # never seen -- and the classification behind it is not
                    # trustworthy anyway: a wagon whose segment runs into the
                    # empty track after the rake is confidently mislabelled
                    # BRAKE_VAN (batch 20260808_125052, GW_59).
                    #
                    # The true type is still carried by segment_type_map, so
                    # nothing is lost -- only the problem feed is normalised.
                    segment_type="wagon",
                    segment_number=None,        # V4 side leaves this null
                    problem_type=ptype,
                    frame_number=side_meta.get("frame_idx"),
                    url=url_for(gw_id=gw_id, feature="door", camera=camera,
                                filename=f"{side}_best.jpg"),
                    bbox=side_meta.get("bbox"), schema=schema,
                    confidence=side_meta.get("confidence"),
                    class_name=ptype,
                    extra={
                        "door_status": door_status,
                        "door_close_detected": door["door_close_detected"],
                        "door_partial_detected": door["door_partial_detected"],
                        "damage_detected": damage_any,
                    }))

        # Loco bands: one per ENGINE wagon, numbers read via the loco OCR path.
        # Only the OCR-authority camera can claim them.
        if is_ocr_camera:
            loco_frames_block, loco_results_block = _loco_blocks(
                state, states_root, url_for)
        else:
            loco_frames_block, loco_results_block = [], {}

        inspection_data.update({
            "total_wagons": len(wagon_segments),
            "doors_open": doors_open,
        })
        # V4 adds a partially-closed bucket; the v1 feed has only open/closed, and
        # its consumer's key list is fixed -- so it is omitted there rather than
        # sent as an extra key.
        if schema != SCHEMA_V1:
            inspection_data["doors_partially_closed"] = doors_partially_closed
        inspection_data.update({
            "doors_closed": doors_closed,
            "damaged_wagons": damaged_wagons,
            "num_engines": n_engines,
            "total_loco_frames": sum(len(b["frames"]) for b in loco_frames_block),
            "total_problem_frames": len(problem_frames),
            # v1 exposes only {damage, door_open}; V4 adds closed/partial buckets.
            "problem_frames_by_type": (
                {"damage": problem_type_counts.get("damage", 0),
                 "door_open": problem_type_counts.get("door_open", 0)}
                if schema == SCHEMA_V1 else
                {"damage": problem_type_counts.get("damage", 0),
                 "open_door": problem_type_counts.get("open_door", 0),
                 "closed_door": problem_type_counts.get("closed_door", 0),
                 "partially_closed": problem_type_counts.get("partially_closed", 0)}
            ),
            "wagon_number_results": wagon_number_block,
            "loco_number_results": loco_results_block,
            "segment_type_map": segment_type_map,
            "wagon_segments": wagon_segments,
            "loco_frames": loco_frames_block,
            "problem_frames": problem_frames,
        })
        # v1's key list has no damage_model_active; V4 does.
        if schema != SCHEMA_V1:
            inspection_data["damage_model_active"] = damage_model_active


    return {
        "camera_id": strip_camera_prefix(camera_folder),
        "version": version,
        "inspection_data": inspection_data,
    }


# global_train damage class -> V4 top problem_type label
_V4_TOP_PROBLEM_TYPE = {
    "floor_damage": "floor_dmg",
    "floor_dmg": "floor_dmg",
    "inner_wall_damage": "inner_wall_dmg",
    "inner_wall_dmg": "inner_wall_dmg",
    "floor__probable_damage": "floor_dmg_probable",     # damage.pt's real name
    "floor_probable_damage": "floor_dmg_probable",
    "floor_damage_probable": "floor_dmg_probable",
    "floor_dmg_probable": "floor_dmg_probable",
    "probable_floor_damage": "floor_dmg_probable",
}


def _loco_blocks(state, states_root: str,
                 url_for: Callable[..., Optional[str]]):
    """``(loco_frames, loco_number_results)`` for the V4 side flavour.

    Every ENGINE-classified Global Wagon is one loco band, numbered from 1 in
    train order (V4 keys ``loco_number_results`` by ``str(loco_id)``).  The
    5-digit number comes from RIGHT_UP's own ocr result for that wagon, which the
    OCR processor produced through the loco path (`loco_no` plate class + 5-digit
    validator).  Every read is emitted, valid or not, so a bad read stays
    auditable -- only ``is_valid_5_digit`` marks a usable number.
    """
    frames: List[Dict[str, Any]] = []
    results: Dict[str, Dict[str, Any]] = {}
    loco_id = 0
    for gw in getattr(state, "wagons", []) or []:
        if getattr(gw, "classification", "") != C.CLASS_ENGINE:
            continue
        loco_id += 1
        gw_id = gw.global_id
        payload = _camera_feature_json(states_root, "ocr", C.CAMERA_RIGHT_UP,
                                       gw_id) or {}
        display = payload.get("display_number") or "-"
        is_valid = bool(payload.get("is_valid_5_digit"))
        sheet_url = (url_for(gw_id=gw_id, feature="ocr",
                             camera=C.CAMERA_RIGHT_UP, filename="ocr_sheet.jpg")
                     or url_for(gw_id=gw_id, feature="ocr",
                                camera=C.CAMERA_RIGHT_UP,
                                filename="best_frame.jpg"))
        if payload:
            results[str(loco_id)] = {
                "is_valid_5_digit": is_valid,
                "display_number": display,
                "raw_number": payload.get("raw_number", ""),
                "confidence": payload.get("confidence", 0.0),
                "ocr_confidence": payload.get("ocr_confidence", 0.0),
                "ocr_frame_s3_url": sheet_url,
            }
        # One frames[] entry per loco, mirroring wagon_frames' shape.
        loco_frames = []
        for filename, position in (("best_frame.jpg", "start"),
                                   ("number_crop.jpg", "mid1"),
                                   ("ocr_sheet.jpg", "end")):
            u = url_for(gw_id=gw_id, feature="ocr", camera=C.CAMERA_RIGHT_UP,
                        filename=filename)
            if not u:
                continue
            loco_frames.append({
                "position": position,
                "filename": os.path.basename(u),
                "s3_key": None,
                "s3_url": u,
                "frame_number": payload.get("best_frame"),
                "timestamp_sec": round(float(getattr(gw, "start_time", 0.0) or 0.0), 2),
            })
        # V4 builds this block FROM the loco frame entries, so a loco with no
        # usable frame produces no block at all rather than an empty shell.
        if loco_frames:
            frames.append({
                "loco_id": loco_id,
                "loco_number": display if is_valid else None,
                "frames": loco_frames,
            })
    return frames, results


def _ocr_fields(states_root: str, evidence_root: str, gw_id: str,
                unified: Dict[str, Any],
                url_for: Callable[..., Optional[str]]) -> Dict[str, Any]:
    """Normalise the OCR result for one wagon into the V4 reporting fields.

    Reads RIGHT_UP's own ocr JSON (the OCR authority) and falls back to the fused
    ``wagon_identifier``.  Works for both engines: the Rekognition path already
    writes ``display_number`` / ``is_valid_11_digit`` / ``fallback_triggered``,
    and the easyocr path now writes the same keys.
    """
    payload = _camera_feature_json(states_root, "ocr", C.CAMERA_RIGHT_UP, gw_id)
    has_result = bool(payload) and payload.get("status") in (
        C.STATUS_OK, C.STATUS_NO_FRAMES, C.NO_DATA, C.STATUS_FAILED)

    display = "-"
    original = "-"
    is_valid = False
    is_manipulated = False
    if payload:
        raw_ident = payload.get("wagon_identifier")
        display = payload.get("display_number") or (
            raw_ident if raw_ident and raw_ident != C.NO_DATA else "-")
        original = payload.get("raw_number") or display
        is_valid = bool(payload.get("is_valid_11_digit"))
        is_manipulated = bool(payload.get("fallback_triggered"))
    if not is_valid:
        # Fused value is authoritative for the report when it carries a number.
        fused = unified.get("wagon_identifier")
        if fused and fused != C.NO_DATA:
            digits = re.sub(r"\D", "", str(fused))
            if len(digits) == C.WAGON_NUMBER_LENGTH:
                display = digits
                original = original if original != "-" else digits
                is_valid = True
                has_result = True

    return {
        "has_result": has_result,
        "is_valid_11_digit": is_valid,
        "display_number": display if display else "-",
        "is_manipulated": is_manipulated,
        "original_number": original if original else "-",
        # The exact image sent to Rekognition, when the OCR engine saved one.
        "ocr_frame_s3_url": (
            url_for(gw_id=gw_id, feature="ocr", camera=C.CAMERA_RIGHT_UP,
                    filename="ocr_sheet.jpg")
            or url_for(gw_id=gw_id, feature="ocr", camera=C.CAMERA_RIGHT_UP,
                       filename="best_frame.jpg")),
    }
