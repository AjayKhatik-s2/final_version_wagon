"""Load `global_train_state.json` into a lightweight in-memory record.

This is the ONE adapter between the wagon-counting engine (`wagon_count/`,
Stage 1) and every downstream consumer (materializer, features, fusion,
rendering, reporting).  Nothing downstream parses the counting engine's JSON
directly, so schema evolution in the counter is absorbed here.

This deliberately does NOT import from `wagon_count.global_train_state`:
we want `wagon_eye_v4/` to remain importable even when the wagon_count
subpackage is missing (e.g. for inspecting an already-computed state).

Two guarantees this module provides to the rest of the pipeline:

1. **The roster is immutable.**  `GlobalWagon` is a frozen dataclass and
   `GlobalTrainState.wagons` is a tuple, so inspection cannot append, remove,
   renumber, reorder or edit a wagon.  `roster_fingerprint()` lets the
   orchestrator prove the roster was untouched across the inspection stages.

2. **Wagon-only rosters still report train structure.**  The counting engine
   emits a WAGON-only roster (engines and brake vans are real train objects
   but are not wagons, so they never receive a GW id).  Their counts are still
   surfaced via `engine_count` / `brake_van_count`, read from the engine's
   `wagon_window` structure metadata, so the existing report KPIs keep
   working without any change to the reporting layer.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# Camera-offset statuses emitted by wagon_count/global_fusion.py that mean
# "this delta is trustworthy".  Anything else contributes no shift at all.
_USABLE_OFFSET_STATUSES = ("REFERENCE", "RESOLVED")


class RosterImmutabilityError(RuntimeError):
    """Raised when the finalized global wagon roster changed after Stage 1."""


@dataclass(frozen=True)
class GlobalWagon:
    """One finalized global wagon.  Immutable by construction."""
    global_id: str
    wagon_index: int
    start_frame_master: int
    end_frame_master: int
    start_time: float
    end_time: float
    classification: str
    classification_confidence: float = 0.0
    supporting_cameras: Tuple[str, ...] = ()
    split_from_global_id: Optional[str] = None
    leading_gap: Optional[Dict[str, Any]] = None
    trailing_gap: Optional[Dict[str, Any]] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class GlobalTrainState:
    total_wagons: int
    wagons: Tuple[GlobalWagon, ...] = ()
    master_camera: str = "RIGHT_UP"
    master_fps: float = 0.0
    master_total_frames: int = 0

    per_camera_local_counts: Dict[str, int] = field(default_factory=dict)
    per_camera_gap_counts:   Dict[str, int] = field(default_factory=dict)
    per_camera_status:       Dict[str, str] = field(default_factory=dict)

    corrections_applied: List[Dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str = ""
    notes: List[str] = field(default_factory=list)

    # ---- additive fields emitted by the master-fixed counting engine --------
    # All default to empty so a state produced by any earlier/legacy counter
    # still loads and behaves exactly as before.
    fusion_mode: str = ""
    master_wagon_count: int = 0
    wagon_window: Dict[str, Any] = field(default_factory=dict)
    camera_offsets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    support_alignment_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    invariant_checks: Dict[str, Any] = field(default_factory=dict)
    global_gap_count: int = 0

    def __post_init__(self) -> None:
        # Coerce the roster to a tuple no matter how it was constructed, so the
        # immutability guarantee holds for hand-built states too (tests).
        if not isinstance(self.wagons, tuple):
            object.__setattr__(self, "wagons", tuple(self.wagons))

    # ------------------------------------------------------------------
    # Train-structure counts
    # ------------------------------------------------------------------

    @property
    def regular_wagon_count(self) -> int:
        return sum(1 for w in self.wagons if w.classification == "WAGON")

    def _non_wagon_class_count(self, wanted: str) -> int:
        """Count a non-wagon class from the engine's wagon_window metadata.

        The master-fixed counter keeps ENGINE / BRAKE_VAN out of the roster,
        recording them under `wagon_window` instead.  Reading them back here
        keeps the existing report KPIs populated without the reporting layer
        needing to know the roster became wagon-only.
        """
        total = 0
        for key in ("leading_non_wagon_classes",
                    "interior_non_wagon_classes",
                    "trailing_non_wagon_classes"):
            bucket = self.wagon_window.get(key) or {}
            if isinstance(bucket, Mapping):
                try:
                    total += int(bucket.get(wanted, 0) or 0)
                except (TypeError, ValueError):
                    pass
        return total

    def _class_count(self, wanted: str) -> int:
        in_roster = sum(1 for w in self.wagons if w.classification == wanted)
        # A legacy roster carries engines/brake vans as GW entries; the
        # master-fixed roster does not.  Never double-count.
        return in_roster if in_roster else self._non_wagon_class_count(wanted)

    @property
    def engine_count(self) -> int:
        return self._class_count("ENGINE")

    @property
    def brake_van_count(self) -> int:
        return self._class_count("BRAKE_VAN")

    # ------------------------------------------------------------------
    # Camera synchronization
    # ------------------------------------------------------------------

    def camera_time_offsets(self) -> Dict[str, float]:
        """`{camera_id -> delta_seconds}` where synchronization was decisive.

        `t_local = t_global - delta`.  A camera whose offset the engine could
        not resolve (or a state from a counter that estimated no offsets at
        all) maps to 0.0 -- i.e. the historical shared-`t=0` assumption, never
        a guessed shift.
        """
        out: Dict[str, float] = {}
        for cam, meta in (self.camera_offsets or {}).items():
            if not isinstance(meta, Mapping):
                continue
            if meta.get("status") not in _USABLE_OFFSET_STATUSES:
                continue
            try:
                out[str(cam)] = float(meta.get("delta", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
        return out

    def camera_time_offset(self, camera_id: str) -> float:
        """Offset for one camera; 0.0 when unresolved/absent."""
        return self.camera_time_offsets().get(camera_id, 0.0)


# -----------------------------------------------------------------------------
# Roster immutability guard
# -----------------------------------------------------------------------------

def roster_fingerprint(state: GlobalTrainState) -> str:
    """Stable digest of the finalized roster: identity, order and boundaries.

    Covers everything inspection must not touch -- the GW id set, their order,
    their index numbering, their master-clock boundaries and their
    classification.
    """
    h = hashlib.sha256()
    h.update(f"n={len(state.wagons)};total={state.total_wagons}|".encode())
    for w in state.wagons:
        h.update(
            f"{w.global_id}|{w.wagon_index}|"
            f"{w.start_frame_master}|{w.end_frame_master}|"
            f"{w.start_time:.6f}|{w.end_time:.6f}|{w.classification}||".encode()
        )
    return h.hexdigest()


def assert_roster_unchanged(
    state: GlobalTrainState, expected_fingerprint: str, *, stage: str,
) -> None:
    """Raise if the roster changed since `expected_fingerprint` was taken."""
    actual = roster_fingerprint(state)
    if actual != expected_fingerprint:
        raise RosterImmutabilityError(
            f"global wagon roster was modified during {stage}: "
            f"expected fingerprint {expected_fingerprint[:16]}..., "
            f"got {actual[:16]}... -- the roster produced by Stage 1 is "
            f"immutable and must never be appended to, renumbered, reordered "
            f"or edited by inspection."
        )


def verify_roster_integrity(state: GlobalTrainState) -> List[str]:
    """Structural checks on the finalized roster.  Returns a list of problems.

    Verifies the contract downstream relies on: contiguous `GW_1..GW_N` ids,
    no duplicates, no gaps, positional index agreeing with the id, and
    `total_wagons` agreeing with the roster length.
    """
    problems: List[str] = []
    n = len(state.wagons)
    if state.total_wagons != n:
        problems.append(
            f"total_wagons={state.total_wagons} but roster holds {n} wagons")

    seen: Dict[str, int] = {}
    for pos, w in enumerate(state.wagons, start=1):
        if w.global_id in seen:
            problems.append(
                f"duplicate global_id {w.global_id} at positions "
                f"{seen[w.global_id]} and {pos}")
        seen[w.global_id] = pos
        if w.global_id != f"GW_{pos}":
            problems.append(
                f"non-contiguous id at position {pos}: expected GW_{pos}, "
                f"got {w.global_id}")
        if w.wagon_index != pos:
            problems.append(
                f"{w.global_id} has wagon_index={w.wagon_index}, expected {pos}")
        if w.end_time < w.start_time:
            problems.append(
                f"{w.global_id} has end_time {w.end_time} < start_time "
                f"{w.start_time}")
    return problems


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

def load_global_train_state(path: str) -> GlobalTrainState:
    """Parse `global_train_state.json` (as emitted by wagon_count)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"global_train_state.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return parse_global_train_state(doc)


def parse_global_train_state(doc: Dict[str, Any]) -> GlobalTrainState:
    """Build a GlobalTrainState from an already-decoded JSON document."""
    wagons: List[GlobalWagon] = []
    for w in doc.get("wagons", []):
        wagons.append(GlobalWagon(
            global_id=w["global_id"],
            wagon_index=int(w.get("wagon_index", 0)),
            start_frame_master=int(w.get("start_frame_master", 0)),
            end_frame_master=int(w.get("end_frame_master", 0)),
            start_time=float(w.get("start_time", 0.0)),
            end_time=float(w.get("end_time", 0.0)),
            classification=w.get("classification", "UNKNOWN"),
            classification_confidence=float(w.get("classification_confidence", 0.0)),
            supporting_cameras=tuple(w.get("supporting_cameras") or ()),
            split_from_global_id=w.get("split_from_global_id"),
            leading_gap=w.get("leading_gap"),
            trailing_gap=w.get("trailing_gap"),
        ))

    invariants = dict(doc.get("invariant_checks") or {})
    try:
        gap_count = int(doc.get("global_gap_count")
                        or invariants.get("global_gap_count")
                        or len(doc.get("global_gaps") or ()))
    except (TypeError, ValueError):
        gap_count = 0

    return GlobalTrainState(
        total_wagons=int(doc.get("total_wagons", 0)),
        wagons=tuple(wagons),
        master_camera=doc.get("master_camera", "RIGHT_UP"),
        master_fps=float(doc.get("master_fps", 0.0)),
        master_total_frames=int(doc.get("master_total_frames", 0)),
        per_camera_local_counts=dict(doc.get("per_camera_local_counts") or {}),
        per_camera_gap_counts=dict(doc.get("per_camera_gap_counts") or {}),
        per_camera_status=dict(doc.get("per_camera_status") or {}),
        corrections_applied=list(doc.get("corrections_applied") or []),
        fallback_used=bool(doc.get("fallback_used", False)),
        fallback_reason=doc.get("fallback_reason", "") or "",
        notes=list(doc.get("notes") or []),
        fusion_mode=str(doc.get("fusion_mode", "") or ""),
        master_wagon_count=int(doc.get("master_wagon_count", 0) or 0),
        wagon_window=dict(doc.get("wagon_window") or {}),
        camera_offsets=dict(doc.get("camera_offsets") or {}),
        support_alignment_summary=dict(doc.get("support_alignment_summary") or {}),
        invariant_checks=invariants,
        global_gap_count=gap_count,
    )


def load_per_camera_fps(per_camera_tracking_json: str) -> Dict[str, float]:
    """Read each camera's source fps from per_camera_tracking.json."""
    out: Dict[str, float] = {}
    try:
        with open(per_camera_tracking_json, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return out
    for cam, meta in doc.items():
        if isinstance(meta, dict):
            try:
                out[cam] = float(meta.get("fps") or 0.0) or 0.0
            except (TypeError, ValueError):
                pass
    return out
