"""UnifiedWagonState -- one physical wagon, fully fused across cameras.

This is the canonical record consumed by reporting/.  It carries:
    - identity         (global_id, classification, OCR)
    - per-side door state
    - load status
    - damage status
    - provenance (which cameras contributed) + an overall confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

from . import constants as C


@dataclass
class UnifiedWagonState:
    global_id: str
    wagon_index: int

    # Stage-0 authoritative
    classification: str = C.CLASS_UNKNOWN
    classification_confidence: float = 0.0

    # Identity
    wagon_identifier: str = C.NO_DATA
    wagon_identifier_confidence: float = 0.0

    # Doors (side cameras)
    left_door: str = C.NO_DATA
    left_door_confidence: float = 0.0
    right_door: str = C.NO_DATA
    right_door_confidence: float = 0.0

    # Load (top cameras)
    load_status: str = C.NO_DATA
    load_confidence: float = 0.0

    # Damage
    top_damage: str = C.NO_DATA
    top_damage_details: List[Dict[str, Any]] = field(default_factory=list)
    side_damage: str = C.NO_DATA
    side_damage_details: List[Dict[str, Any]] = field(default_factory=list)

    # Provenance
    supporting_cameras: List[str] = field(default_factory=list)
    missing_cameras: List[str] = field(default_factory=list)
    confidence: float = 0.0          # 0..1 combined
    anomalies: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ----------------------------------------------------------------
    # convenience predicates
    # ----------------------------------------------------------------

    @property
    def has_open_door(self) -> bool:
        return self.left_door == C.DOOR_OPEN or self.right_door == C.DOOR_OPEN

    @property
    def has_damage(self) -> bool:
        return (self.top_damage == C.DAMAGE_PRESENT
                or self.side_damage == C.DAMAGE_PRESENT)

    def damage_observations_by_camera(self) -> Dict[str, List[Dict[str, Any]]]:
        """This wagon's damage detections grouped by the camera that saw them.

        Damage is a property of the WAGON, assembled from per-camera
        observations. One camera detecting it is sufficient -- the other camera
        reporting nothing is not evidence of absence, because the two views
        differ in angle, timing, occlusion and detection quality. That asymmetry
        is already how the damage processor works (`any_damage` across the top
        cameras), and this does not change it.

        What this adds is a camera-keyed VIEW of provenance that is otherwise
        buried in a flat list, so a report can show each camera's own snapshot
        instead of one picture standing in for both. It is pure grouping over
        `camera_id`, which the damage processor already stamps on every track --
        no wagon matching happens here, and none should: the observation is
        already attached to this wagon by the existing global mapping.

        Returns {camera_id: [observation, ...]}, each list ordered by
        descending best_confidence. Cameras with no detection are ABSENT from
        the mapping rather than present with an empty list, so a caller cannot
        mistake "saw nothing" for "was not consulted".
        """
        out: Dict[str, List[Dict[str, Any]]] = {}
        for obs in list(self.top_damage_details or []) + \
                list(self.side_damage_details or []):
            if not isinstance(obs, dict):
                continue
            cam = obs.get("camera_id")
            if not cam:
                continue            # provenance-less: never guess a camera
            out.setdefault(str(cam), []).append(obs)
        for cam in out:
            out[cam].sort(key=lambda o: float(o.get("best_confidence") or 0.0),
                          reverse=True)
        return out

    @property
    def damage_cameras(self) -> List[str]:
        """Cameras that actually reported damage on this wagon, sorted."""
        return sorted(self.damage_observations_by_camera())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_wagons(wagons: List[UnifiedWagonState]) -> Dict[str, Any]:
    """Train-level summary: counts of every flagged condition."""
    return {
        "total_wagons":   len(wagons),
        "engine_count":   sum(1 for w in wagons if w.classification == C.CLASS_ENGINE),
        "wagon_count":    sum(1 for w in wagons if w.classification == C.CLASS_WAGON),
        "brake_van_count":sum(1 for w in wagons if w.classification == C.CLASS_BRAKE_VAN),
        "left_doors_open":  sum(1 for w in wagons if w.left_door == C.DOOR_OPEN),
        "right_doors_open": sum(1 for w in wagons if w.right_door == C.DOOR_OPEN),
        "loaded":           sum(1 for w in wagons if w.load_status == C.LOAD_LOADED),
        "empty":            sum(1 for w in wagons if w.load_status == C.LOAD_EMPTY),
        "top_damaged":      sum(1 for w in wagons if w.top_damage == C.DAMAGE_PRESENT),
        "side_damaged":     sum(1 for w in wagons if w.side_damage == C.DAMAGE_PRESENT),
        "ocr_captured":     sum(1 for w in wagons if w.wagon_identifier != C.NO_DATA),
    }
