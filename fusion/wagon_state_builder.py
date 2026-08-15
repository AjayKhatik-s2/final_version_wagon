"""Stage 4 -- merge per-feature, per-wagon JSONs into UnifiedWagonStates.

Reads:
    wagon_states/door/<GW_n>.json
    wagon_states/ocr/<GW_n>.json
    wagon_states/load/<GW_n>.json
    wagon_states/damage/<GW_n>.json

Writes:
    wagon_states/unified/<GW_n>.json

Authority rules (per spec):
    * classification          ← GlobalTrainState only (RIGHT_UP)
    * wagon_identifier        ← OCR (RIGHT_UP)
    * right_door              ← door processor (RIGHT_UP side)
    * left_door               ← door processor (LEFT_UP side)
    * load_status             ← load processor (RIGHT_UP_TOP authoritative)
    * top_damage              ← damage processor (any top camera DAMAGE wins)
    * side_damage             ← not implemented in this pass (no model
                                wired for side-camera damage); defaults
                                to NO_DATA.  Future: add a side-damage
                                feature processor.

The combined `confidence` is the unweighted mean of the non-NO_DATA
per-feature confidences, in [0, 1].

Any feature whose JSON is FAILED / missing → that field is NO_DATA.
A wagon with NO feature data still produces a UnifiedWagonState (Stage-0
classification + provenance).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from core import constants as C
from core.global_state_loader import GlobalTrainState, GlobalWagon
from core.unified_wagon_state import UnifiedWagonState
from core.feature_config import get_spec


# -----------------------------------------------------------------------------
# Per-feature JSON loaders (forgiving -- missing file -> None)
# -----------------------------------------------------------------------------

def _load(feature_dir: str, gw_id: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(feature_dir, f"{gw_id}.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _is_ok(payload: Optional[Dict[str, Any]]) -> bool:
    return bool(payload) and payload.get("status") == C.STATUS_OK


def _is_disabled(payload: Optional[Dict[str, Any]]) -> bool:
    return bool(payload) and payload.get("status") == C.STATUS_DISABLED


# -----------------------------------------------------------------------------
# Per-wagon fuser
# -----------------------------------------------------------------------------

def _fuse_one(
    gw: GlobalWagon,
    *,
    door:   Optional[Dict[str, Any]],
    ocr:    Optional[Dict[str, Any]],
    load:   Optional[Dict[str, Any]],
    damage: Optional[Dict[str, Any]],
) -> UnifiedWagonState:
    u = UnifiedWagonState(
        global_id=gw.global_id,
        wagon_index=gw.wagon_index,
        classification=gw.classification,
        classification_confidence=gw.classification_confidence,
    )

    # ---- OCR ----
    if _is_ok(ocr):
        if ocr.get("wagon_identifier") and ocr["wagon_identifier"] != C.NO_DATA:
            u.wagon_identifier = str(ocr["wagon_identifier"])
            u.wagon_identifier_confidence = float(
                ocr.get("wagon_identifier_confidence", 0.0) or 0.0
            )

    # ---- Doors ----
    if _is_ok(door):
        if door.get("left_door") and door["left_door"] != C.NO_DATA:
            u.left_door = str(door["left_door"])
            u.left_door_confidence = float(door.get("left_door_confidence", 0.0) or 0.0)
        if door.get("right_door") and door["right_door"] != C.NO_DATA:
            u.right_door = str(door["right_door"])
            u.right_door_confidence = float(door.get("right_door_confidence", 0.0) or 0.0)

    # ---- Load ----
    if _is_ok(load):
        if load.get("load_status") and load["load_status"] != C.NO_DATA:
            u.load_status = str(load["load_status"])
            u.load_confidence = float(load.get("load_confidence", 0.0) or 0.0)

    # ---- Top damage ----
    if _is_ok(damage):
        if damage.get("top_damage") and damage["top_damage"] != C.NO_DATA:
            u.top_damage = str(damage["top_damage"])
            u.top_damage_details = list(damage.get("top_damage_details") or [])

    # ---- Disabled-by-user features: stamp their owned fields with the
    #      DISABLED_DISPLAY sentinel so reports show "DISABLED BY USER" rather
    #      than NO_DATA / OK.  Done BEFORE the anomaly summary so a disabled
    #      feature never raises an anomaly (the sentinel matches no anomaly
    #      equality, incl. the OCR-missing NO_DATA check). ----
    for _key, _payload in (("door", door), ("ocr", ocr),
                           ("load", load), ("damage", damage)):
        if not _is_disabled(_payload):
            continue
        spec = get_spec(_key)
        if spec is None:
            continue
        for _fld in spec.owned_fields:
            if hasattr(u, _fld):
                setattr(u, _fld, C.DISABLED_DISPLAY)

    # ---- Provenance: union of supporting cameras across features ----
    supporting: set = set()
    for payload in (door, ocr, load, damage):
        if _is_ok(payload):
            for c in payload.get("supporting_cameras") or []:
                supporting.add(c)
    u.supporting_cameras = sorted(supporting)
    u.missing_cameras = sorted(set(C.ALL_CAMERAS) - supporting)

    # ---- Anomaly summary (used in PDF) ----
    if u.left_door == C.DOOR_OPEN:
        u.anomalies.append("LEFT_DOOR_OPEN")
    if u.right_door == C.DOOR_OPEN:
        u.anomalies.append("RIGHT_DOOR_OPEN")
    if u.top_damage == C.DAMAGE_PRESENT:
        u.anomalies.append("TOP_DAMAGE")
    if u.side_damage == C.DAMAGE_PRESENT:
        u.anomalies.append("SIDE_DAMAGE")
    if u.wagon_identifier == C.NO_DATA and u.classification == C.CLASS_WAGON:
        u.anomalies.append("OCR_MISSING")

    # ---- Combined confidence ----
    confs = [c for c in (u.wagon_identifier_confidence,
                          u.left_door_confidence,
                          u.right_door_confidence,
                          u.load_confidence) if c > 0]
    u.confidence = round(sum(confs) / len(confs), 4) if confs else 0.0

    # ---- Notes ----
    for cam in u.missing_cameras:
        u.notes.append(f"missing:{cam}")
    return u


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def build(
    *,
    state: GlobalTrainState,
    wagon_states_root: str,
    write_per_wagon_json: bool = True,
    verbose: bool = True,
) -> Dict[str, UnifiedWagonState]:
    """For each GlobalWagon, fuse all feature JSONs into UnifiedWagonState."""
    door_dir   = os.path.join(wagon_states_root, "door")
    ocr_dir    = os.path.join(wagon_states_root, "ocr")
    load_dir   = os.path.join(wagon_states_root, "load")
    damage_dir = os.path.join(wagon_states_root, "damage")
    unified_dir = os.path.join(wagon_states_root, "unified")
    if write_per_wagon_json:
        os.makedirs(unified_dir, exist_ok=True)

    unified: Dict[str, UnifiedWagonState] = {}
    t_start = time.time()
    if verbose:
        print(f"[STAGE4] fusing {len(state.wagons)} wagons "
              f"(reading from {wagon_states_root})")

    for gw in state.wagons:
        gw_id = gw.global_id
        u = _fuse_one(
            gw,
            door=_load(door_dir, gw_id),
            ocr=_load(ocr_dir, gw_id),
            load=_load(load_dir, gw_id),
            damage=_load(damage_dir, gw_id),
        )
        unified[gw_id] = u
        if write_per_wagon_json:
            p = os.path.join(unified_dir, f"{gw_id}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(u.to_dict(), f, indent=2)
        if verbose:
            print(
                f"  {gw_id:<6} cls={u.classification:<10} "
                f"ocr={(u.wagon_identifier or C.NO_DATA):<14} "
                f"L={u.left_door:<8} R={u.right_door:<8} "
                f"load={u.load_status:<8} "
                f"top_dmg={u.top_damage:<7} "
                f"anomalies={','.join(u.anomalies) or '—'}"
            )

    if verbose:
        elapsed = time.time() - t_start
        print(f"[STAGE4] done in {elapsed:.1f}s  -> {unified_dir}")

    return unified
