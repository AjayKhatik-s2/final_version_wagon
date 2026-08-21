"""Evidence identity: what makes one snapshot distinguishable from another.

An evidence image is identified by

    camera_id + feature + wagon/segment id + observation index

The global wagon id says WHICH wagon; it is metadata, never a substitute for
the camera. Two cameras observing the same wagon produce two records, and the
report must be able to ask for one of them specifically.

This matters most for damage. A wagon's `evidence/<gw>/damage/` directory holds
the tracks of BOTH top cameras, and the index alone is unique only within the
single processor invocation that produced it. Two invocations writing the same
directory -- one per camera, as a per-camera pipeline naturally does -- would
both start at track 1 and overwrite each other. RIGHT_UP_TOP and LEFT_UP_TOP
photograph the same roof from opposite sides, so the substitution is invisible
on inspection: the report shows a real damage photo, of the wrong camera.

Putting the camera in the slot name makes that collision impossible instead of
merely unlikely. Kept here, in core, so the writer (features/damage) and the
readers (reporting/*) cannot drift apart.
"""

from __future__ import annotations

from typing import Optional

#: Separator between the observation index and the camera. Chosen because no
#: camera id contains it, so a slot can be parsed back apart unambiguously.
_SEP = "__"


def damage_track_slot(track_idx: int, camera_id: str) -> str:
    """Slot name for one damage observation, e.g. ``track_2__RIGHT_UP_TOP``.

    `track_idx` orders observations within a wagon; `camera_id` says who saw it.
    Both are required -- neither alone identifies the image.
    """
    return f"track_{int(track_idx)}{_SEP}{camera_id}"


def legacy_damage_track_slot(track_idx: int) -> str:
    """The pre-camera-scoped slot name, ``track_2``.

    Evidence trees written before the rename still resolve through this, but
    ONLY once the caller has confirmed from metadata that the track belongs to
    the camera being asked about. That check is what keeps the legacy path from
    reintroducing cross-camera substitution.
    """
    return f"track_{int(track_idx)}"


def parse_damage_track_slot(slot: str) -> tuple:
    """``track_2__RIGHT_UP_TOP`` -> ``(2, "RIGHT_UP_TOP")``.

    Returns ``(idx, None)`` for a legacy slot, whose camera is unknowable from
    the name alone -- which is the whole reason the name changed.
    """
    if not slot.startswith("track_"):
        return (None, None)
    body = slot[len("track_"):]
    idx_s, sep, cam = body.partition(_SEP)
    try:
        idx = int(idx_s)
    except (TypeError, ValueError):
        return (None, None)
    return (idx, cam if sep else None)
