"""Decide which global wagon owns a damage seen near a wagon boundary.

The problem
-----------
A damage feature near a coupling is visible in frames on BOTH sides of a
reconstructed wagon boundary.  The materializer buckets each frame into exactly
one global wagon, so the two halves of one physical defect become two
independent damage tracks -- one on wagon N, one on wagon N+1 -- and the report
shows the same dent twice, on two wagons.

The rule, and why it is spatial
-------------------------------
Ownership is decided by WHERE the damage sits relative to the visible gap in the
SAME camera frame, never by which side of a time window the frame fell on.  For a
damage observation on camera X at absolute frame F:

    1. find the global boundary whose gap this camera is actually seeing at F
    2. read that gap's x position IN FRAME F
    3. compare the damage bbox centre against it
    4. assign to the wagon on that side of the gap

Every input already exists; this module invents no geometry.

Where the data comes from
-------------------------
    GlobalWagon.leading_gap / .trailing_gap   -> master track id of the boundary
    state.global_gaps[i]["master_track_id"]   -> the GlobalGap for that boundary
    ...["support_observations"][cam]           -> that camera's local_track_id
    tracks[cam].gaps  (tracking_full.json)     -> hit_frames / center_x_trajectory
                                                  / bbox_history for that track

`GapObservation.center_x` is deliberately NOT used: it is
`center_x_trajectory[-1]`, the gap's position at its LAST hit
(global_fusion.to_gap_observations), which says nothing about frame F.

Two conventions are REUSED rather than reinvented
-------------------------------------------------
*Interpolation* -- `wagon_count.video_segmenter._interp_gap_bbox`, the function
the overlay renderer already uses: clamp to the first bbox before the first hit,
clamp to the last after the last hit, component-wise linear interpolation
between the bracketing pair.  Called directly, so there is one interpolation
algorithm in the repository rather than two.

*Direction* -- per camera, from that camera's own gap tracks, exactly as
`gap_validation`'s "pass 2a: dominant direction, derived per camera (never
assumed)" does it.  This is not a detail: `gap_validation` records that
"RIGHT_UP_TOP gaps move in -x, LEFT_UP_TOP gaps move in +x", so a hardcoded
`x < gap_x` rule would be right on one top camera and backwards on the other.

Deriving before/after from the direction
---------------------------------------
Wagon N precedes wagon N+1 in the train, so N passes the camera FIRST and is
therefore further along the direction of travel at any instant.

    dominant = +1 (gaps sweep towards +x):  wagon N sits at LARGER x
    dominant = -1 (gaps sweep towards -x):  wagon N sits at SMALLER x

so with `delta = damage_centre_x - gap_x`:

    sign(delta) * dominant == +1  ->  the PRECEDING wagon  (N)
    sign(delta) * dominant == -1  ->  the FOLLOWING wagon  (N+1)

which is one expression covering all four camera orientations, with the sign
measured from the data rather than assumed.

What this module will not do
----------------------------
It never fabricates a gap position, never assigns one damage to two wagons, and
never invents a second wagon segmentation -- the only wagon identities used are
the existing RIGHT_UP-mastered `GW_n`.  When the geometry cannot be recovered
the existing bucketed owner is kept and the observation is marked
BOUNDARY_AMBIGUOUS with a reason, because "we could not tell" is a different
statement from "it belongs here".
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PKG = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PKG)
_WC = os.path.join(_ROOT, "wagon_count")
for _p in (_ROOT, _WC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("orchestrator.damage_boundary")

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SIDE_BEFORE = "BEFORE"        # the preceding wagon, N
SIDE_AFTER = "AFTER"          # the following wagon, N+1
SIDE_AMBIGUOUS = "AMBIGUOUS"

#: Why an observation could not be resolved spatially.  Each is a distinct
#: failure, and the report should be able to tell them apart.
REASON_RESOLVED = "RESOLVED"
REASON_NO_BOUNDARY = "NO_BOUNDARY"          # wagon has no adjacent boundary here
REASON_NO_GLOBAL_GAP = "NO_GLOBAL_GAP"      # boundary not found in global_gaps
REASON_NO_SUPPORT_GAP = "NO_SUPPORT_GAP"    # this camera never observed it
REASON_NO_TRACK = "NO_TRACK"                # support track id not in tracking_full
REASON_NO_GAP_POSITION = "NO_GAP_POSITION"  # no usable bbox history at frame F
REASON_NO_DAMAGE_BOX = "NO_DAMAGE_BOX"      # observation has no bbox
REASON_NO_DIRECTION = "NO_DIRECTION"        # camera direction indeterminate
REASON_WITHIN_TOLERANCE = "WITHIN_TOLERANCE"


@dataclass
class BoundaryConfig:
    """Tunables.  Defaults chosen to be conservative: prefer AMBIGUOUS over a
    confident wrong answer, since a mislabelled wagon is worse than a flagged
    one."""

    tolerance_px: float = 40.0
    """Half-width of the band around the gap in which the side cannot be called.

    A gap bbox is typically tens of pixels wide, and the damage centre can sit
    inside it. 40 px on a 960-wide frame is ~4% of the width.
    """

    min_tracks_for_direction: int = 3
    """Below this many usable gap tracks the camera's dominant direction is not
    established.  `gap_validation` uses 5 for REJECTING tracks; this is a
    read-only classification, so a lower bar is acceptable -- but 0 or 1 tracks
    genuinely cannot establish a direction and must yield NO_DIRECTION.
    """

    dedup_frame_window: int = 90
    """Two observations of the same class from the same camera whose best frames
    fall within this many frames are treated as ONE physical defect seen across
    a boundary, not two defects.  At 15 fps that is 6 s -- longer than a wagon
    boundary takes to cross the frame, short enough not to merge genuinely
    separate defects on the same wagon.
    """


DEFAULT_CONFIG = BoundaryConfig()


# ---------------------------------------------------------------------------
# Camera direction -- reused convention, per camera, measured not assumed
# ---------------------------------------------------------------------------

def track_direction(gap) -> int:
    """+1, -1 or 0 for one gap track, by majority sign of inter-hit velocity.

    The same statistic `gap_validation._motion_features` computes: sign the
    per-step displacement between consecutive hits and take the majority.
    Reimplemented here on purpose rather than imported, because that function
    returns a 20-field `GapMotionFeatures` and requires config objects; this
    needs one integer and must not depend on validation internals.
    """
    hits = list(getattr(gap, "hit_frames", None) or [])
    traj = list(getattr(gap, "center_x_trajectory", None) or [])
    n = min(len(hits), len(traj))
    if n < 2:
        return 0
    n_pos = n_neg = 0
    for i in range(n - 1):
        if hits[i + 1] <= hits[i]:
            continue
        d = traj[i + 1] - traj[i]
        if d > 0:
            n_pos += 1
        elif d < 0:
            n_neg += 1
    if n_pos > n_neg:
        return 1
    if n_neg > n_pos:
        return -1
    return 0


def camera_direction(tracks, cfg: BoundaryConfig = DEFAULT_CONFIG) -> int:
    """This camera's dominant gap direction: +1 (towards +x), -1, or 0.

    Mirrors `gap_validation`'s pass 2a -- a majority vote over the camera's own
    surviving gap tracks, never a global assumption. Returns 0 when too few
    tracks carry a direction, which callers must treat as "unavailable", not as
    "no movement".
    """
    if tracks is None:
        return 0
    dirs = [track_direction(g) for g in (getattr(tracks, "gaps", None) or [])]
    dirs = [d for d in dirs if d != 0]
    if len(dirs) < max(1, int(cfg.min_tracks_for_direction)):
        return 0
    n_pos = sum(1 for d in dirs if d > 0)
    n_neg = len(dirs) - n_pos
    if n_pos > n_neg:
        return 1
    if n_neg > n_pos:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Same-frame gap position -- reusing the renderer's interpolation
# ---------------------------------------------------------------------------

def _interp_bbox(gap, frame_idx: int) -> Optional[List[float]]:
    """The overlay renderer's own interpolator, called not copied.

    Falls back to a local no-op only if `wagon_count` is unavailable (it is on
    `sys.path` above, so this is defensive rather than expected).
    """
    try:
        from video_segmenter import _interp_gap_bbox
    except Exception:  # pragma: no cover - wagon_count always importable here
        return None
    return _interp_gap_bbox(gap, int(frame_idx))


def gap_x_at_frame(gap, frame_idx: int) -> Tuple[Optional[float], Optional[List[float]]]:
    """The gap's horizontal centre in THIS frame, plus the bbox for diagnostics.

    `(None, None)` when the frame is outside the track's span or the track has
    no usable bbox history -- the caller must then report NO_GAP_POSITION rather
    than guess.

    An exact hit needs no interpolation and gets none: `_interp_gap_bbox` clamps
    to the stored bbox when `frame_idx` equals a hit frame, so an exact match
    returns the recorded geometry unchanged.
    """
    bb = _interp_bbox(gap, frame_idx)
    if not bb or len(bb) < 4:
        return None, None
    return (float(bb[0]) + float(bb[2])) / 2.0, [float(v) for v in bb[:4]]


def _bbox_center_x(bbox: Optional[Sequence[float]]) -> Optional[float]:
    if not bbox or len(bbox) < 4:
        return None
    return (float(bbox[0]) + float(bbox[2])) / 2.0


# ---------------------------------------------------------------------------
# Boundary lookup: wagon pair -> this camera's gap track
# ---------------------------------------------------------------------------

def _master_track_id(boundary: Optional[Dict[str, Any]]) -> Optional[int]:
    """The master gap track id a wagon's leading/trailing_gap points at.

    `global_alignment` writes `{source, camera_id, track_id, center_time}`, with
    `{'source': 'video_start'}` at the edges -- those carry no track and are not
    a boundary between two wagons.
    """
    if not isinstance(boundary, dict):
        return None
    if boundary.get("source") in (None, "", "video_start", "video_end", "edge"):
        return None
    tid = boundary.get("track_id")
    try:
        tid = int(tid)
    except (TypeError, ValueError):
        return None
    return tid if tid > 0 else None


def _find_global_gap(global_gaps: Sequence[Dict[str, Any]],
                     master_track_id: int) -> Optional[Dict[str, Any]]:
    for g in global_gaps or ():
        if not isinstance(g, dict):
            continue
        try:
            if int(g.get("master_track_id")) == int(master_track_id):
                return g
        except (TypeError, ValueError):
            continue
    return None


def _support_track_id(global_gap: Dict[str, Any],
                      camera_id: str) -> Optional[int]:
    """This camera's own gap track id for that global boundary.

    A camera legitimately may not have observed a boundary -- `GlobalGap`
    records that in `missing_cameras` / `unavailable_cameras`. Absence is not an
    error and must never be read as "the gap is not there".
    """
    obs = (global_gap.get("support_observations") or {}).get(camera_id)
    if not isinstance(obs, dict):
        return None
    try:
        return int(obs.get("local_track_id"))
    except (TypeError, ValueError):
        return None


def _gap_event(tracks, track_id: int):
    for g in (getattr(tracks, "gaps", None) or []):
        if int(getattr(g, "track_id", -1)) == int(track_id):
            return g
    return None


def _covers(gap, frame_idx: int) -> bool:
    try:
        return (int(gap.start_frame) <= int(frame_idx) <= int(gap.end_frame))
    except (AttributeError, TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# One observation
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    """What was decided about one damage observation, and why."""
    gw_id: str
    camera_id: str
    frame_idx: int
    class_name: str = ""
    damage_center_x: Optional[float] = None
    gap_x: Optional[float] = None
    gap_bbox: Optional[List[float]] = None
    direction: int = 0
    side: str = SIDE_AMBIGUOUS
    reason: str = REASON_NO_BOUNDARY
    owner: str = ""                  # resolved global wagon id
    previous_owner: str = ""         # where it was bucketed
    boundary: Tuple[str, str] = ("", "")   # (preceding gw, following gw)
    ambiguous: bool = True
    moved: bool = False

    def render(self) -> str:
        d = ("-" if self.damage_center_x is None
             else f"{self.damage_center_x:.1f}")
        g = "-" if self.gap_x is None else f"{self.gap_x:.1f}"
        return (f"[DAMAGE-BOUNDARY] {self.camera_id} f={self.frame_idx} "
                f"pair={self.boundary[0] or '-'}|{self.boundary[1] or '-'} "
                f"gap_x={g} dmg_x={d} dir={self.direction:+d} "
                f"side={self.side} owner={self.owner or '-'} "
                f"(was {self.previous_owner or '-'}) reason={self.reason}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_id": self.gw_id, "camera_id": self.camera_id,
            "frame_idx": self.frame_idx, "class_name": self.class_name,
            "damage_center_x": self.damage_center_x, "gap_x": self.gap_x,
            "gap_bbox": self.gap_bbox, "direction": self.direction,
            "side": self.side, "reason": self.reason, "owner": self.owner,
            "previous_owner": self.previous_owner,
            "boundary": list(self.boundary),
            "boundary_ambiguous": self.ambiguous, "moved": self.moved,
        }


def _neighbours(wagons: Sequence[Any], gw_id: str
                ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """`(previous, this, next)` in roster order."""
    ids = [w.global_id for w in wagons]
    if gw_id not in ids:
        return None, None, None
    i = ids.index(gw_id)
    return (wagons[i - 1] if i > 0 else None,
            wagons[i],
            wagons[i + 1] if i + 1 < len(wagons) else None)


def resolve_observation(
    *,
    gw_id: str,
    observation: Dict[str, Any],
    wagons: Sequence[Any],
    global_gaps: Sequence[Dict[str, Any]],
    tracks_by_camera: Dict[str, Any],
    directions: Dict[str, int],
    cfg: BoundaryConfig = DEFAULT_CONFIG,
) -> Verdict:
    """Decide the owning wagon for ONE damage observation.

    Candidate boundaries are the two adjacent to `gw_id`. The one used is
    whichever this camera is ACTUALLY SEEING at the damage frame -- a gap track
    whose span covers that frame. That selection is spatial/observational, not
    temporal: it asks "is the boundary in shot", never "which time window did
    the frame fall in".
    """
    cam = str(observation.get("camera_id") or "")
    frame_idx = int(observation.get("best_frame_idx")
                    or observation.get("frame_idx") or 0)
    v = Verdict(gw_id=gw_id, camera_id=cam, frame_idx=frame_idx,
                class_name=str(observation.get("class_name") or ""),
                owner=gw_id, previous_owner=gw_id)

    v.damage_center_x = _bbox_center_x(observation.get("bbox"))
    if v.damage_center_x is None:
        v.reason = REASON_NO_DAMAGE_BOX
        return v

    prev_w, this_w, next_w = _neighbours(wagons, gw_id)
    if this_w is None:
        v.reason = REASON_NO_BOUNDARY
        return v

    tracks = tracks_by_camera.get(cam)
    if tracks is None:
        v.reason = REASON_NO_TRACK
        return v

    # Candidate boundaries: (preceding wagon, following wagon, master track id)
    candidates: List[Tuple[Optional[Any], Optional[Any], int]] = []
    lead = _master_track_id(getattr(this_w, "leading_gap", None))
    if lead is not None and prev_w is not None:
        candidates.append((prev_w, this_w, lead))
    trail = _master_track_id(getattr(this_w, "trailing_gap", None))
    if trail is not None and next_w is not None:
        candidates.append((this_w, next_w, trail))
    if not candidates:
        v.reason = REASON_NO_BOUNDARY
        return v

    # Keep only boundaries this camera can actually see in THIS frame.
    seen: List[Tuple[Any, Any, Any, float, List[float]]] = []
    reason = REASON_NO_GLOBAL_GAP
    for before_w, after_w, mtid in candidates:
        gg = _find_global_gap(global_gaps, mtid)
        if gg is None:
            continue
        stid = _support_track_id(gg, cam)
        if stid is None:
            reason = REASON_NO_SUPPORT_GAP
            continue
        ev = _gap_event(tracks, stid)
        if ev is None:
            reason = REASON_NO_TRACK
            continue
        if not _covers(ev, frame_idx):
            # The boundary exists but is not in shot at this frame; that is not
            # a failure, it just means this boundary cannot own the decision.
            continue
        gx, gbb = gap_x_at_frame(ev, frame_idx)
        if gx is None:
            reason = REASON_NO_GAP_POSITION
            continue
        seen.append((before_w, after_w, ev, gx, gbb))

    if not seen:
        v.reason = reason
        return v

    # More than one boundary in shot: the nearer gap owns the decision, which is
    # again spatial -- the damage is being placed relative to the boundary it is
    # actually beside.
    before_w, after_w, ev, gap_x, gap_bbox = min(
        seen, key=lambda s: abs(v.damage_center_x - s[3]))
    v.gap_x, v.gap_bbox = gap_x, gap_bbox
    v.boundary = (before_w.global_id, after_w.global_id)

    dominant = directions.get(cam, 0)
    v.direction = dominant
    if dominant == 0:
        v.reason = REASON_NO_DIRECTION
        return v

    delta = v.damage_center_x - gap_x
    if abs(delta) <= float(cfg.tolerance_px):
        v.reason = REASON_WITHIN_TOLERANCE
        return v

    product = (1 if delta > 0 else -1) * dominant
    if product > 0:
        v.side, v.owner = SIDE_BEFORE, before_w.global_id
    else:
        v.side, v.owner = SIDE_AFTER, after_w.global_id
    v.reason = REASON_RESOLVED
    v.ambiguous = False
    v.moved = (v.owner != v.previous_owner)
    return v


# ---------------------------------------------------------------------------
# Whole-train pass
# ---------------------------------------------------------------------------

@dataclass
class BoundaryResult:
    verdicts: List[Verdict] = field(default_factory=list)
    moved: int = 0
    ambiguous: int = 0
    deduplicated: int = 0
    wagons_touched: List[str] = field(default_factory=list)

    def render(self) -> str:
        return (f"[DAMAGE-BOUNDARY] {len(self.verdicts)} observation(s): "
                f"{self.moved} reassigned, {self.deduplicated} deduplicated, "
                f"{self.ambiguous} ambiguous")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "wagon_eye.damage_boundary.v1",
            "observations": len(self.verdicts),
            "moved": self.moved,
            "deduplicated": self.deduplicated,
            "ambiguous": self.ambiguous,
            "wagons_touched": sorted(set(self.wagons_touched)),
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def _dedup_key(v: Verdict) -> Tuple[str, str, str]:
    return (v.owner, v.camera_id, v.class_name)


def resolve_train(
    *,
    state,
    engine_global_gaps: Sequence[Dict[str, Any]],
    tracks_by_camera: Dict[str, Any],
    damage_by_wagon: Dict[str, List[Dict[str, Any]]],
    cfg: BoundaryConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> BoundaryResult:
    """Resolve every damage observation in one train.

    `damage_by_wagon` is `{gw_id: [observation, ...]}` -- the per-wagon damage
    track records as the damage processor wrote them, each carrying camera_id,
    best_frame_idx, bbox, confidence and class_name. Those fields are preserved
    verbatim on the way through; this only decides WHICH wagon they belong to.

    Returns the verdicts. It does NOT write anything: the caller applies the
    outcome, so this function stays pure and testable.
    """
    wagons = list(getattr(state, "wagons", None) or ())
    res = BoundaryResult()

    directions = {cam: camera_direction(tr, cfg)
                  for cam, tr in (tracks_by_camera or {}).items()}
    if verbose:
        log.info("[DAMAGE-BOUNDARY] camera gap directions: %s",
                 {c: f"{d:+d}" for c, d in sorted(directions.items())})

    for gw_id, observations in sorted((damage_by_wagon or {}).items()):
        for obs in observations or ():
            v = resolve_observation(
                gw_id=gw_id, observation=obs, wagons=wagons,
                global_gaps=engine_global_gaps,
                tracks_by_camera=tracks_by_camera,
                directions=directions, cfg=cfg)
            res.verdicts.append(v)
            if v.moved:
                res.moved += 1
                res.wagons_touched.extend([v.previous_owner, v.owner])
            if v.ambiguous:
                res.ambiguous += 1
            if verbose:
                log.info("%s", v.render())

    # ---- deduplicate one physical defect seen on both sides -------------
    # After reassignment, two observations that resolved to the SAME wagon from
    # the same camera, same class, within `dedup_frame_window` frames are the
    # two halves of one defect. Keep the one whose damage centre sits further
    # from the gap -- that is the view where the defect was most fully inside
    # the wagon, so it is the better evidence, and it is a spatial criterion
    # rather than a confidence one.
    by_key: Dict[Tuple[str, str, str], List[Verdict]] = {}
    for v in res.verdicts:
        if v.reason == REASON_RESOLVED:
            by_key.setdefault(_dedup_key(v), []).append(v)
    for group in by_key.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda x: x.frame_idx)
        keep: List[Verdict] = []
        for v in group:
            near = next((k for k in keep
                         if abs(k.frame_idx - v.frame_idx)
                         <= cfg.dedup_frame_window), None)
            if near is None:
                keep.append(v)
                continue
            res.deduplicated += 1
            better = max((near, v),
                         key=lambda x: abs((x.damage_center_x or 0.0)
                                           - (x.gap_x or 0.0)))
            if better is v:
                keep[keep.index(near)] = v
                near.owner = ""          # dropped: superseded by `v`
                near.reason = "DEDUPLICATED"
            else:
                v.owner = ""
                v.reason = "DEDUPLICATED"

    if verbose:
        log.info("%s", res.render())
    return res


# ---------------------------------------------------------------------------
# Applying the outcome
# ---------------------------------------------------------------------------

def _load(path: str) -> Optional[Dict[str, Any]]:
    import json
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _save(path: str, doc: Dict[str, Any]) -> bool:
    import json
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                    exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, default=str)
        os.replace(tmp, path)
        return True
    except OSError as e:
        log.error("[DAMAGE-BOUNDARY] could not write %s: %s", path, e)
        return False


def read_damage_by_wagon(states_root: str, wagons: Sequence[Any]
                         ) -> Dict[str, List[Dict[str, Any]]]:
    """`{gw_id: [track record, ...]}` from `wagon_states/damage/<gw>.json`.

    Reads `top_damage_details`, which is the damage processor's own list of
    per-track records -- camera_id, track_id, class_name, confidence,
    best_confidence, best_frame_idx, bbox. Nothing is transformed.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    d = os.path.join(states_root, "damage")
    for w in wagons:
        doc = _load(os.path.join(d, f"{w.global_id}.json"))
        if not doc or doc.get("status") != C.STATUS_OK:
            continue
        recs = [r for r in (doc.get("top_damage_details") or [])
                if isinstance(r, dict)]
        if recs:
            out[w.global_id] = recs
    return out


def _move_evidence(evidence_root: str, src_gw: str, dst_gw: str,
                   camera_id: str, next_idx: int,
                   record: Dict[str, Any]) -> Optional[str]:
    """Copy one observation's snapshot into its new owner's evidence directory.

    The snapshot must remain reachable from the wagon that now owns the damage,
    because every reader resolves `evidence/<gw>/damage/<slot>.jpg`. The file is
    COPIED, not moved: the source wagon's directory is also the record of what
    that camera saw there, and the pipeline never rewrites history to make a
    later decision look inevitable.

    The new slot keeps the camera in its name (`core.evidence_identity`), so two
    cameras' observations of the same wagon still cannot collide.
    """
    import shutil
    from core.evidence_identity import damage_track_slot

    old_idx = record.get("track_idx")
    if old_idx is None:
        return None
    old_slot = damage_track_slot(int(old_idx), camera_id)
    new_slot = damage_track_slot(int(next_idx), camera_id)
    src_dir = os.path.join(evidence_root, src_gw, "damage")
    dst_dir = os.path.join(evidence_root, dst_gw, "damage")
    src = os.path.join(src_dir, f"{old_slot}.jpg")
    if not os.path.isfile(src):
        return None
    try:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, f"{new_slot}.jpg"))
        crop = os.path.join(src_dir, f"{old_slot}_crop.jpg")
        if os.path.isfile(crop):
            shutil.copy2(crop, os.path.join(dst_dir, f"{new_slot}_crop.jpg"))
    except OSError as e:
        log.warning("[DAMAGE-BOUNDARY] evidence copy %s -> %s failed: %s",
                    src, dst_dir, e)
        return None
    return new_slot


def apply_verdicts(
    *,
    result: BoundaryResult,
    states_root: str,
    evidence_root: str,
    wagons: Sequence[Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Rewrite the per-wagon damage records to match the verdicts.

    Three things change and nothing else:

    * a reassigned observation moves to its resolved wagon, snapshot included;
    * an observation that stayed put but could not be resolved spatially is
      annotated `boundary_ambiguous` with its reason, so a reader can tell a
      confident placement from a fallback;
    * `top_damage` is recomputed for any wagon whose track list changed -- a
      wagon that lost its only observation is no longer damaged, which is the
      whole point of the exercise.

    Detection, confidences and thresholds are untouched; this is bookkeeping
    over records the damage processor already produced.
    """
    import json

    dmg_dir = os.path.join(states_root, "damage")
    docs: Dict[str, Dict[str, Any]] = {}
    for w in wagons:
        doc = _load(os.path.join(dmg_dir, f"{w.global_id}.json"))
        if doc is not None:
            docs[w.global_id] = doc

    def _records(gw: str) -> List[Dict[str, Any]]:
        return (docs.get(gw, {}).get("top_damage_details") or [])

    def _matches(rec: Dict[str, Any], v: Verdict) -> bool:
        return (str(rec.get("camera_id") or "") == v.camera_id
                and int(rec.get("best_frame_idx") or 0) == v.frame_idx
                and str(rec.get("class_name") or "") == v.class_name)

    touched: set = set()
    dropped = 0

    for v in result.verdicts:
        src = v.previous_owner
        if src not in docs:
            continue
        recs = _records(src)
        rec = next((r for r in recs if _matches(r, v)), None)
        if rec is None:
            continue

        # Every verdict carries its reasoning onto the record, resolved or not.
        rec["boundary_side"] = v.side
        rec["boundary_reason"] = v.reason
        rec["boundary_ambiguous"] = bool(v.ambiguous)
        rec["boundary_gap_x"] = v.gap_x
        rec["boundary_gap_bbox"] = v.gap_bbox
        rec["boundary_damage_center_x"] = v.damage_center_x
        rec["boundary_camera_direction"] = v.direction
        rec["boundary_pair"] = list(v.boundary)
        # Annotating IS a change: mark the wagon so the record is written back.
        # Without this, an observation that stayed put but could not be resolved
        # spatially kept its verdict only in memory, and the report had no way
        # to tell a confident placement from a fallback.
        touched.add(src)

        if v.reason == "DEDUPLICATED":
            recs.remove(rec)
            docs[src]["top_damage_details"] = recs
            touched.add(src)
            dropped += 1
            continue

        if not v.moved or not v.owner or v.owner == src:
            continue

        dst = v.owner
        if dst not in docs:
            docs[dst] = {"global_id": dst, "feature": "damage",
                         "status": C.STATUS_OK, "top_damage": C.DAMAGE_OK,
                         "top_damage_details": [], "per_camera": {},
                         "supporting_cameras": [], "frame_count": 0,
                         "evidence": {}}
        dst_recs = _records(dst)
        next_idx = 1 + max([int(r.get("track_idx") or 0)
                            for r in dst_recs] or [0])
        moved_rec = dict(rec)
        new_slot = _move_evidence(evidence_root, src, dst, v.camera_id,
                                  next_idx, rec)
        moved_rec["track_idx"] = next_idx
        moved_rec["moved_from_global_id"] = src
        if new_slot:
            moved_rec["evidence_slot"] = new_slot
            docs[dst].setdefault("evidence", {})[new_slot] = os.path.join(
                evidence_root, dst, "damage", f"{new_slot}.jpg")

        recs.remove(rec)
        docs[src]["top_damage_details"] = recs
        dst_recs.append(moved_rec)
        docs[dst]["top_damage_details"] = dst_recs
        cams = docs[dst].setdefault("supporting_cameras", [])
        if v.camera_id not in cams:
            cams.append(v.camera_id)
        touched.update((src, dst))

    # Recompute the verdict for every wagon whose list changed.
    for gw in sorted(touched):
        doc = docs.get(gw)
        if doc is None:
            continue
        has = bool(doc.get("top_damage_details"))
        before = doc.get("top_damage")
        if doc.get("status") == C.STATUS_OK:
            doc["top_damage"] = C.DAMAGE_PRESENT if has else C.DAMAGE_OK
        if verbose and doc.get("top_damage") != before:
            log.info("[DAMAGE-BOUNDARY] %s top_damage %s -> %s (%d track(s))",
                     gw, before, doc.get("top_damage"),
                     len(doc.get("top_damage_details") or []))
        _save(os.path.join(dmg_dir, f"{gw}.json"), doc)

        # Keep the evidence metadata in step, so damage-track resolvers in the
        # reporting layer see the same set.
        ev_meta_path = os.path.join(evidence_root, gw, "damage",
                                    "metadata.json")
        ev_meta = _load(ev_meta_path)
        if ev_meta is not None:
            ev_meta["top_damage"] = doc.get("top_damage")
            ev_meta["tracks"] = [
                {k: r.get(k) for k in
                 ("track_idx", "camera_id", "track_id", "class_name",
                  "confidence", "best_confidence", "best_frame_idx", "bbox",
                  "boundary_side", "boundary_reason", "boundary_ambiguous")}
                for r in (doc.get("top_damage_details") or [])
            ]
            _save(ev_meta_path, ev_meta)

    out = {"wagons_rewritten": sorted(touched), "deduplicated_dropped": dropped}
    if verbose:
        log.info("[DAMAGE-BOUNDARY] rewrote %d wagon(s), dropped %d duplicate "
                 "observation(s)", len(touched), dropped)
    return out
