"""Stage 3c -- associate existing damage evidence with the canonical wagon.

This consumes Stage-3 damage evidence and changes only WHICH `GW_n` owns it.
No detector runs here, no threshold moves, and the Stage-3 sampling defaults
(`damage = sampled / stride 3`) are neither read nor altered: by the time this
runs, inference is long finished and all that is left on disk is a list of
confirmed tracks, each carrying the camera and the camera-local frame it was
best seen in.

Why the wagon has to be recomputed at all
-----------------------------------------
The damage processor never decided a wagon number and never should.  It read
`wagon_cache/GW_n/<camera>/*.jpg` and inherited `GW_n` from the directory name.
Those directories are cut by the materializer at
`round((GW.time - delta) * local_fps)`, where `delta` is 0.0 for any camera
whose clock offset the counter could not resolve -- so on a displaced camera the
frames of one physical wagon are filed under its neighbour, and every damage
seen in them is reported against the wrong wagon.

`core.canonical_association` answers the question properly: normalize the
detection's local frame to the master clock, then look up which canonical
gap-delimited interval that time falls in.  This module is the adapter -- it
finds the damage evidence, feeds it to that resolver, and writes the answer
back.

Precedence against Stage 3b
---------------------------
`orchestrator.damage_boundary` (Stage 3b) resolves ownership from same-frame gap
GEOMETRY, and can only act when the boundary gap is actually visible in the
detection's frame -- i.e. right at a boundary.  That is precisely the band where
this module returns BOUNDARY_AMBIGUOUS and defers, so the two are complementary
rather than competing: 3b decides inside the ambiguity band, 3c decides outside
it.  Where they genuinely disagree -- a detection that 3b moved but which
normalizes to a time well clear of the gap -- the canonical timeline wins, and
the override is logged.  Stage 3b is not modified by this module in any way.

Stage 3b also cannot run in batch mode at all: it needs the full-fidelity
`tracking_full.json` (hit_frames / bbox_history), which batch mode does not
retain.  This module needs only `master_time` per canonical gap plus per-camera
fps and offsets, all of which live in `global_train_state.json`, so the same
resolver runs identically in both modes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core import constants as C
from core.logging_setup import get_logger
from core.canonical_association import (
    CanonicalTimeline, Detection, Assignment, AssociationConfig,
    DEFAULT_CONFIG, RESOLVED_STATUSES,
    STATUS_BOUNDARY_AMBIGUOUS, STATUS_RESOLVED, STATUS_RESOLVED_ASSUMED_OFFSET,
    METHOD_BEFORE_GAP, METHOD_AFTER_GAP,
)

log = get_logger("orchestrator.damage_association")

FEATURE_NAME = "damage"

#: Fields this module OWNS on a damage record.  Everything already there is
#: preserved; this is written alongside it, never over it.
PROVENANCE_KEY = "canonical_association"


# ---------------------------------------------------------------------------
# Reading the existing evidence
# ---------------------------------------------------------------------------

def _record_identity(rec: Mapping[str, Any]) -> Tuple[str, Any, int, str]:
    """Identity of one damage track within a wagon's record list."""
    return (
        str(rec.get("camera_id") or ""),
        rec.get("track_id"),
        int(rec.get("best_frame_idx") or rec.get("frame_idx") or 0),
        str(rec.get("class_name") or ""),
    )


def detection_for(rec: Mapping[str, Any]) -> Detection:
    """The camera-local coordinates of one damage track, as the processor wrote
    them.  `best_frame_idx` is the camera's own absolute frame number: the
    materializer names cached frames `frame_<local index>.jpg` and the damage
    processor parses that name back out, so no conversion is needed here -- only
    normalization, which the resolver does."""
    cam, tid, frame, cls = _record_identity(rec)
    return Detection(
        camera_id=cam,
        local_frame=frame,
        feature=FEATURE_NAME,
        detection_id=f"{cam}:t{tid}:f{frame}:{cls}",
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class DamageAssociationResult:
    """What the association decided, per detection, plus roll-ups."""

    assignments: List[Assignment] = field(default_factory=list)
    #: assignment index -> the wagon whose file the record was found in
    bucketed_owner: List[str] = field(default_factory=list)
    timeline_summary: Dict[str, Any] = field(default_factory=dict)

    moved: int = 0
    confirmed: int = 0
    ambiguous: int = 0
    unresolved: int = 0
    overrode_stage3b: int = 0
    wagons_touched: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.assignments)

    def examples(self, method: str, limit: int = 5) -> List[Assignment]:
        """Resolved assignments made by one method -- the audit's worked
        examples."""
        return [a for a in self.assignments
                if a.method == method and a.resolved][:limit]

    def render(self) -> str:
        return (f"[DAMAGE-ASSOC] {self.total} detection(s): "
                f"{self.confirmed} confirmed in place, {self.moved} reassigned, "
                f"{self.ambiguous} boundary-ambiguous, "
                f"{self.unresolved} unresolved")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "wagon_eye.damage_association.v1",
            "detections": self.total,
            "confirmed_in_place": self.confirmed,
            "reassigned": self.moved,
            "boundary_ambiguous": self.ambiguous,
            "unresolved": self.unresolved,
            "overrode_stage3b": self.overrode_stage3b,
            "wagons_touched": sorted(set(self.wagons_touched)),
            "canonical_timeline": dict(self.timeline_summary),
            "assignments": [
                {**a.to_dict(), "bucketed_global_wagon_id": src}
                for a, src in zip(self.assignments, self.bucketed_owner)
            ],
            "examples": {
                METHOD_BEFORE_GAP: [a.to_dict()
                                    for a in self.examples(METHOD_BEFORE_GAP)],
                METHOD_AFTER_GAP: [a.to_dict()
                                   for a in self.examples(METHOD_AFTER_GAP)],
            },
        }


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------

def resolve_train(
    *,
    state: Any,
    global_gaps: Sequence[Mapping[str, Any]],
    per_camera_fps: Mapping[str, float],
    damage_by_wagon: Mapping[str, Sequence[Mapping[str, Any]]],
    cfg: AssociationConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> DamageAssociationResult:
    """Associate every damage detection in one train.  Writes nothing.

    Kept pure so the decision can be tested without a filesystem, and so the
    caller stays the only thing that touches disk.
    """
    timeline = CanonicalTimeline.build(
        state=state, global_gaps=global_gaps,
        per_camera_fps=per_camera_fps, cfg=cfg)
    res = DamageAssociationResult(timeline_summary=timeline.summary())

    if verbose:
        s = res.timeline_summary
        log.info("[DAMAGE-ASSOC] canonical timeline: %d wagon(s), "
                 "%d usable gap boundary/ies of %d, gaps without timing: %s",
                 s.get("canonical_wagons"), s.get("usable_gap_boundaries"),
                 s.get("canonical_gaps"),
                 s.get("gaps_without_timing") or "none")

    for gw_id in sorted(damage_by_wagon or {}):
        for rec in (damage_by_wagon[gw_id] or ()):
            if not isinstance(rec, Mapping):
                continue
            a = timeline.assign(detection_for(rec))
            res.assignments.append(a)
            res.bucketed_owner.append(gw_id)

            if a.resolved:
                if a.global_wagon_id != gw_id:
                    res.moved += 1
                    res.wagons_touched.extend([gw_id, a.global_wagon_id])
                    if rec.get("moved_from_global_id"):
                        # Stage 3b had already moved this record; the canonical
                        # timeline disagrees from outside the ambiguity band.
                        res.overrode_stage3b += 1
                else:
                    res.confirmed += 1
                    res.wagons_touched.append(gw_id)
            elif a.status == STATUS_BOUNDARY_AMBIGUOUS:
                res.ambiguous += 1
                res.wagons_touched.append(gw_id)
            else:
                res.unresolved += 1
                res.wagons_touched.append(gw_id)

            if verbose:
                log.info("%s", a.render())

    if verbose:
        log.info("%s", res.render())
    return res


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_assignments(
    *,
    result: DamageAssociationResult,
    states_root: str,
    evidence_root: str,
    wagons: Sequence[Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Write the association back onto the per-wagon damage records.

    Two things change and nothing else:

    * a detection whose canonical wagon differs from the wagon whose file it was
      found in moves there, snapshot included;
    * every detection -- moved, confirmed, ambiguous or unresolved -- gets a
      `canonical_association` provenance block, so a reader can always tell how
      a placement was arrived at and how confident it was.

    `top_damage` is recomputed for any wagon whose track list changed: a wagon
    that lost its only detection is no longer damaged, and one that gained the
    first is.

    The evidence move and the evidence-metadata merge are `damage_boundary`'s
    (`_move_evidence`, `_merged_evidence_tracks`), called rather than
    reimplemented.  Those two functions carry specific production fixes -- the
    camera-scoped slot name that stops the two top cameras overwriting each
    other's JPEGs, and the `track_idx` merge that stopped a rebuild nulling the
    only handle on a snapshot file.  A second copy of that logic here would be a
    second place for those bugs to come back.
    """
    from orchestrator import damage_boundary as DBND

    dmg_dir = os.path.join(states_root, FEATURE_NAME)
    docs: Dict[str, Dict[str, Any]] = {}
    for w in wagons:
        gw = str(getattr(w, "global_id", "") or "")
        doc = DBND._load(os.path.join(dmg_dir, f"{gw}.json"))
        if doc is not None:
            docs[gw] = doc

    def _records(gw: str) -> List[Dict[str, Any]]:
        return (docs.get(gw, {}).get("top_damage_details") or [])

    touched: set = set()
    moved = 0

    for a, src in zip(result.assignments, result.bucketed_owner):
        if src not in docs:
            continue
        recs = _records(src)
        rec = next((r for r in recs
                    if detection_for(r).detection_id == a.detection_id), None)
        if rec is None:
            continue

        # Provenance goes on every record, resolved or not: "we could not tell"
        # is a finding, and a reader must be able to see it.
        rec[PROVENANCE_KEY] = {**a.to_dict(),
                               "damage_camera_id": a.camera_id,
                               "bucketed_global_wagon_id": src}
        touched.add(src)

        dst = a.global_wagon_id
        if not a.resolved or not dst or dst == src:
            continue
        if dst not in docs:
            docs[dst] = {"global_id": dst, "feature": FEATURE_NAME,
                         "status": C.STATUS_OK, "top_damage": C.DAMAGE_OK,
                         "top_damage_details": [], "per_camera": {},
                         "supporting_cameras": [], "frame_count": 0,
                         "evidence": {}}
        dst_recs = _records(dst)
        next_idx = 1 + max([int(r.get("track_idx") or 0) for r in dst_recs] or [0])
        moved_rec = dict(rec)
        new_slot = DBND._move_evidence(evidence_root, src, dst, a.camera_id,
                                       next_idx, rec)
        moved_rec["track_idx"] = next_idx
        moved_rec["moved_from_global_id"] = src
        moved_rec["moved_by"] = "canonical_association"
        if new_slot:
            moved_rec["evidence_slot"] = new_slot
            docs[dst].setdefault("evidence", {})[new_slot] = os.path.join(
                evidence_root, dst, FEATURE_NAME, f"{new_slot}.jpg")

        recs.remove(rec)
        docs[src]["top_damage_details"] = recs
        dst_recs.append(moved_rec)
        docs[dst]["top_damage_details"] = dst_recs
        cams = docs[dst].setdefault("supporting_cameras", [])
        if a.camera_id and a.camera_id not in cams:
            cams.append(a.camera_id)
        touched.update((src, dst))
        moved += 1
        if verbose:
            log.info("[DAMAGE-ASSOC] %s -> %s  %s (%s)",
                     src, dst, a.detection_id, a.method)

    for gw in sorted(touched):
        doc = docs.get(gw)
        if doc is None:
            continue
        has = bool(doc.get("top_damage_details"))
        before = doc.get("top_damage")
        if doc.get("status") == C.STATUS_OK:
            doc["top_damage"] = C.DAMAGE_PRESENT if has else C.DAMAGE_OK
        if verbose and doc.get("top_damage") != before:
            log.info("[DAMAGE-ASSOC] %s top_damage %s -> %s (%d track(s))",
                     gw, before, doc.get("top_damage"),
                     len(doc.get("top_damage_details") or []))
        DBND._save(os.path.join(dmg_dir, f"{gw}.json"), doc)

        # The combined PDF reads `evidence/<gw>/damage/metadata.json` and the
        # processed video reads `evidence/<gw>/damage/overlay.json`; both are
        # keyed by the wagon DIRECTORY. Keeping the metadata in step is what
        # makes the report and the video show the resolved GW_n rather than the
        # bucketed one.
        ev_meta_path = os.path.join(evidence_root, gw, FEATURE_NAME,
                                    "metadata.json")
        ev_meta = DBND._load(ev_meta_path)
        if ev_meta is None:
            # A wagon that had no damage before this move has no evidence
            # metadata to update -- and `damage_from_evidence` reads ONLY that
            # file, so without creating it the moved detection would vanish from
            # the combined PDF entirely. Losing a finding while "correcting" its
            # wagon is worse than the misattribution being corrected.
            if not doc.get("top_damage_details"):
                continue
            ev_meta = {"global_id": gw, "feature": FEATURE_NAME, "tracks": []}
            os.makedirs(os.path.dirname(ev_meta_path), exist_ok=True)
        ev_meta["top_damage"] = doc.get("top_damage")
        ev_meta["tracks"] = DBND._merged_evidence_tracks(
            ev_meta.get("tracks"), doc.get("top_damage_details"))
        DBND._save(ev_meta_path, ev_meta)

    out = {"wagons_rewritten": sorted(touched), "moved": moved}
    if verbose:
        log.info("[DAMAGE-ASSOC] rewrote %d wagon(s), moved %d detection(s)",
                 len(touched), moved)
    return out


# ---------------------------------------------------------------------------
# One call for both pipeline modes
# ---------------------------------------------------------------------------

def load_global_gaps(state_json_path: str) -> List[Dict[str, Any]]:
    """`global_gaps` out of the counting engine's own state file.

    Batch mode holds a v4 `GlobalTrainState`, which keeps only
    `global_gap_count`; the authoritative list stays in the JSON on disk. This
    is how batch mode reaches the same canonical gaps sequential mode has in
    memory, so both modes run the identical resolver on identical input.
    """
    try:
        with open(state_json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("[DAMAGE-ASSOC] could not read %s: %s", state_json_path, e)
        return []
    gaps = doc.get("global_gaps")
    return list(gaps) if isinstance(gaps, list) else []


def run(
    *,
    state: Any,
    global_gaps: Sequence[Mapping[str, Any]],
    per_camera_fps: Mapping[str, float],
    states_root: str,
    evidence_root: str,
    diagnostics_dir: Optional[str] = None,
    cfg: AssociationConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> Optional[DamageAssociationResult]:
    """Read the damage evidence, associate it, write it back, dump the audit.

    Returns None when there is nothing to associate.  Never raises: a wagon
    number being hard to decide must not fail a train that already has all its
    inspection results.
    """
    from orchestrator import damage_boundary as DBND

    wagons = list(getattr(state, "wagons", None) or ())
    damage_by_wagon = DBND.read_damage_by_wagon(states_root, wagons)
    if not damage_by_wagon:
        if verbose:
            log.info("[DAMAGE-ASSOC] no damage detections to associate")
        return None

    res = resolve_train(
        state=state, global_gaps=global_gaps, per_camera_fps=per_camera_fps,
        damage_by_wagon=damage_by_wagon, cfg=cfg, verbose=verbose)
    apply_assignments(result=res, states_root=states_root,
                      evidence_root=evidence_root, wagons=wagons,
                      verbose=verbose)

    if diagnostics_dir:
        try:
            os.makedirs(diagnostics_dir, exist_ok=True)
            with open(os.path.join(diagnostics_dir, "damage_association.json"),
                      "w", encoding="utf-8") as f:
                json.dump(res.to_dict(), f, indent=2)
        except OSError as e:
            log.warning("[DAMAGE-ASSOC] could not write diagnostics: %s", e)

    if verbose:
        for method in (METHOD_BEFORE_GAP, METHOD_AFTER_GAP):
            for a in res.examples(method, limit=3):
                log.info("[DAMAGE-ASSOC] example %s: %s", method, a.render())
    return res
