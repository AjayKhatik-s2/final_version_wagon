"""The wagon-active region from FOUR cameras, on one common timeline.

The problem
-----------
The four cameras do not see the same wagon at the same local time. They sit at
different points along the track and their clocks are not aligned, so the same
physical first wagon can legitimately appear at 18.6 s on RIGHT_UP, 21.2 s on
LEFT_UP, 16.8 s on RIGHT_UP_TOP and 20.1 s on LEFT_UP_TOP. Those four numbers
are not four contradictory boundaries; they are one boundary seen through four
clocks. Comparing them raw -- or taking the earliest, or the latest -- reads a
viewpoint difference as a disagreement.

What this module reuses rather than reinvents
---------------------------------------------
The pipeline already has exactly one clock-alignment mechanism, and this module
consumes it read-only:

    `wagon_count.global_fusion` estimates a per-camera offset against the master
    gap sequence and records it as `state.camera_offsets[cam] =
    {"delta": ..., "status": REFERENCE | RESOLVED | UNRESOLVED, ...}`, with
    `t_global = t_local + delta`.

    `train_structure.build_local_wagon_region` already gives each support camera
    its own wagon region in its OWN local clock.

    `train_structure.get_master_wagon_window` already selects the counted region
    out of the master's gap-delimited segments and renumbers GW_1..GW_N.

No second clock, no second gap detector, no second counter. This module
normalizes boundaries that already exist and decides between them.

Where the previous behaviour fell short
--------------------------------------
1.  Only the two TOP cameras were consulted for boundary evidence. LEFT_UP -- a
    SIDE camera, and structurally the strongest corroboration available -- was
    not consulted at all.
2.  A camera whose offset is UNRESOLVED was normalized with `delta = 0.0`, i.e.
    compared as though its clock agreed with the master's. That manufactures
    both false agreement and false dissent out of a missing measurement. Here an
    unresolved camera's evidence is retained and reported but takes no part in
    the vote, because "we do not know when this camera's 18.6 s happened" is not
    the same as "it happened at 18.6 s global".
3.  Top cameras could only corroborate or dissent; they could never help recover
    a boundary the master's own classifier missed.

The rule
--------
Weighted consensus, not unanimity. Cameras disagree for real reasons --
viewpoint, occlusion, classifier error, timing -- so requiring all four to agree
would mean never deciding.

    SIDE cameras (RIGHT_UP, LEFT_UP)   weight 2   structural evidence
    TOP cameras                        weight 1   supporting evidence

The master's own boundary is the incumbent and stays unless the evidence to move
it is both strong and physically possible:

  * the weighted support for the proposed boundary reaches
    `min_move_weight`, and that support cannot come from top cameras alone --
    a top camera on its own may confirm a boundary, never relocate one;
  * the proposed boundary coincides, within `gap_tolerance_sec`, with a
    VALIDATED MASTER GAP. This is what stops a camera manufacturing a wagon:
    the new boundary has to be a place the master's own physical gap detection
    already found a coupling. A top camera calling WAGON over the locomotive has
    no master gap under it and therefore cannot move anything.

Both halves are required. The first alone would let two cameras agree a boundary
into existence with no physical evidence; the second alone would let a single
noisy camera relocate a boundary just because a gap happened to be nearby.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core import constants as C

#: Offset statuses `global_fusion` treats as decisive.
USABLE_OFFSET_STATUSES = ("REFERENCE", "RESOLVED")

#: Structural evidence. RIGHT_UP is the master; LEFT_UP is the other side view.
SIDE_WEIGHT = 2
#: Supporting evidence.
TOP_WEIGHT = 1

DECISION_MASTER_HELD = "MASTER_BOUNDARY_HELD"
DECISION_MOVED = "MOVED_ON_CONSENSUS"
DECISION_NO_MASTER = "NO_MASTER_BOUNDARY"
DECISION_NO_EVIDENCE = "NO_COMPARABLE_EVIDENCE"

REASON_NOT_COMPARABLE = "OFFSET_UNRESOLVED_NOT_COMPARABLE"
REASON_NO_REGION = "NO_WAGON_REGION_ON_THIS_CAMERA"
REASON_NO_MASTER_GAP = "NO_VALIDATED_MASTER_GAP_AT_PROPOSED_BOUNDARY"
REASON_INSUFFICIENT_WEIGHT = "INSUFFICIENT_WEIGHTED_SUPPORT"
REASON_TOP_ONLY = "TOP_CAMERAS_ALONE_CANNOT_RELOCATE_A_BOUNDARY"


@dataclass(frozen=True)
class ConsensusConfig:
    """Tunables. Defaults chosen to prefer the master's boundary: a wrong count
    is worse than a boundary that stays where the master put it."""

    agree_tolerance_sec: float = 1.0
    """How close a camera's normalized boundary must be to count as agreeing.
    Roughly a quarter of a wagon at line speed -- tight enough that a wagon's
    worth of disagreement is never called agreement."""

    min_move_weight: int = 3
    """Weighted support needed to move a boundary. 3 cannot be reached by the two
    top cameras alone (1 + 1), so relocating always requires a side camera."""

    gap_tolerance_sec: float = 0.75
    """How close the proposed boundary must sit to a validated master gap. A gap
    is visible for roughly half a second, so this admits the gap it is actually
    on and nothing further."""


DEFAULT_CONFIG = ConsensusConfig()


@dataclass
class CameraBoundary:
    """One camera's wagon-region boundary, local and normalized."""

    camera_id: str
    local_start: Optional[float] = None
    local_end: Optional[float] = None
    offset: float = 0.0
    offset_status: str = "UNKNOWN"
    found: bool = False
    reason: str = ""

    @property
    def is_side(self) -> bool:
        return self.camera_id in C.SIDE_CAMERAS

    @property
    def weight(self) -> int:
        return SIDE_WEIGHT if self.is_side else TOP_WEIGHT

    @property
    def comparable(self) -> bool:
        """Whether this camera's times may be compared with the master's.

        False when the offset was never resolved: normalizing with 0.0 would
        assert a synchronization nobody measured.
        """
        return (self.found
                and self.offset_status in USABLE_OFFSET_STATUSES
                and self.local_start is not None
                and self.local_end is not None)

    @property
    def global_start(self) -> Optional[float]:
        if self.local_start is None or not self.comparable:
            return None
        return self.local_start + self.offset

    @property
    def global_end(self) -> Optional[float]:
        if self.local_end is None or not self.comparable:
            return None
        return self.local_end + self.offset

    def to_dict(self) -> Dict[str, Any]:
        def _r(v):
            return round(v, 3) if isinstance(v, float) else v
        return {
            "camera_id": self.camera_id,
            "role": ("SIDE" if self.is_side else "TOP"),
            "weight": self.weight,
            "found": self.found,
            "local_start": _r(self.local_start),
            "local_end": _r(self.local_end),
            "offset_sec": _r(self.offset),
            "offset_status": self.offset_status,
            "comparable": self.comparable,
            "global_start": _r(self.global_start),
            "global_end": _r(self.global_end),
            "reason": self.reason or ("" if self.comparable
                                      else (REASON_NOT_COMPARABLE
                                            if self.found
                                            else REASON_NO_REGION)),
        }

    def render(self) -> str:
        if not self.found:
            return (f"[ACTIVE-REGION] {self.camera_id:<13} "
                    f"no wagon region on this camera ({REASON_NO_REGION})")
        ls = "?" if self.local_start is None else f"{self.local_start:7.2f}"
        le = "?" if self.local_end is None else f"{self.local_end:7.2f}"
        if not self.comparable:
            return (f"[ACTIVE-REGION] {self.camera_id:<13} "
                    f"local {ls}..{le}s  offset={self.offset:+.2f}s "
                    f"[{self.offset_status}] -> NOT COMPARABLE "
                    f"({REASON_NOT_COMPARABLE})")
        return (f"[ACTIVE-REGION] {self.camera_id:<13} "
                f"local {ls}..{le}s  offset={self.offset:+.2f}s "
                f"[{self.offset_status}] -> global "
                f"{self.global_start:7.2f}..{self.global_end:7.2f}s  "
                f"weight={self.weight} ({'SIDE' if self.is_side else 'TOP'})")


@dataclass
class BoundaryDecision:
    """One end of the region: what the master said, what the cameras said, and
    what was decided."""

    kind: str                          # "start" | "end"
    master_time: Optional[float] = None
    canonical_time: Optional[float] = None
    decision: str = DECISION_MASTER_HELD
    reason: str = ""
    supported_by: List[str] = field(default_factory=list)
    disagreed_by: List[str] = field(default_factory=list)
    not_comparable: List[str] = field(default_factory=list)
    support_weight: int = 0
    proposed_time: Optional[float] = None
    proposed_weight: int = 0
    proposed_by: List[str] = field(default_factory=list)
    master_gap_time: Optional[float] = None

    @property
    def moved(self) -> bool:
        return self.decision == DECISION_MOVED

    def to_dict(self) -> Dict[str, Any]:
        def _r(v):
            return round(v, 3) if isinstance(v, float) else v
        return {
            "kind": self.kind,
            "master_time": _r(self.master_time),
            "canonical_time": _r(self.canonical_time),
            "decision": self.decision,
            "reason": self.reason,
            "supported_by": list(self.supported_by),
            "support_weight": self.support_weight,
            "disagreed_by": list(self.disagreed_by),
            "not_comparable": list(self.not_comparable),
            "proposed_time": _r(self.proposed_time),
            "proposed_weight": self.proposed_weight,
            "proposed_by": list(self.proposed_by),
            "master_gap_time": _r(self.master_gap_time),
            "moved": self.moved,
        }

    def render(self) -> str:
        ct = "?" if self.canonical_time is None else f"{self.canonical_time:.2f}"
        mt = "?" if self.master_time is None else f"{self.master_time:.2f}"
        return (f"[ACTIVE-REGION] {self.kind.upper():<5} canonical={ct}s "
                f"(master said {mt}s)  {self.decision}  "
                f"supported_by={','.join(self.supported_by) or 'none'} "
                f"(weight {self.support_weight})  "
                f"disagreed={','.join(self.disagreed_by) or 'none'}  "
                f"not_comparable={','.join(self.not_comparable) or 'none'}  "
                f"reason={self.reason}")


@dataclass
class RegionConsensus:
    """The whole decision, per camera and per boundary."""

    boundaries: List[CameraBoundary] = field(default_factory=list)
    start: BoundaryDecision = field(
        default_factory=lambda: BoundaryDecision(kind="start"))
    end: BoundaryDecision = field(
        default_factory=lambda: BoundaryDecision(kind="end"))
    master_camera: str = C.CAMERA_RIGHT_UP

    @property
    def moved_any(self) -> bool:
        return self.start.moved or self.end.moved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "wagon_eye.region_consensus.v1",
            "master_camera": self.master_camera,
            "side_cameras": list(C.SIDE_CAMERAS),
            "top_cameras": list(C.TOP_CAMERAS),
            "side_weight": SIDE_WEIGHT,
            "top_weight": TOP_WEIGHT,
            "per_camera": [b.to_dict() for b in self.boundaries],
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "moved_any_boundary": self.moved_any,
        }

    def render_lines(self) -> List[str]:
        out = [f"[ACTIVE-REGION] normalizing 4 cameras onto the global timeline "
               f"(master={self.master_camera}, t_global = t_local + offset)"]
        out += [b.render() for b in self.boundaries]
        out += [self.start.render(), self.end.render()]
        return out


# ---------------------------------------------------------------------------
# Building the per-camera evidence
# ---------------------------------------------------------------------------

def _finite(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def camera_boundaries(
    *,
    master_camera: str,
    master_window: Mapping[str, Any],
    support_regions: Mapping[str, Any],
    camera_offsets: Mapping[str, Mapping[str, Any]],
) -> List[CameraBoundary]:
    """One `CameraBoundary` per camera, from evidence that already exists.

    The master's boundary comes from its own `wagon_window`, which is already in
    global time -- RIGHT_UP defines the clock, so its offset is 0 by definition
    and its status is REFERENCE. Every support camera's comes from the
    `LocalWagonRegion` its own classification pass produced, in its own clock.
    """
    out: List[CameraBoundary] = []

    m_start = _finite((master_window or {}).get("wagon_start_time"))
    m_end = _finite((master_window or {}).get("wagon_end_time"))
    out.append(CameraBoundary(
        camera_id=master_camera, local_start=m_start, local_end=m_end,
        offset=0.0, offset_status="REFERENCE",
        found=bool((master_window or {}).get("found")) and m_start is not None,
        reason="master timeline; defines global time"))

    for cam in C.ALL_CAMERAS:
        if cam == master_camera:
            continue
        meta = (camera_offsets or {}).get(cam) or {}
        status = str(meta.get("status") or "UNKNOWN")
        delta = _finite(meta.get("delta"))
        reg = (support_regions or {}).get(cam)
        found = bool(getattr(reg, "found", False)) if reg is not None else False
        out.append(CameraBoundary(
            camera_id=cam,
            local_start=_finite(getattr(reg, "start_time", None)),
            local_end=_finite(getattr(reg, "end_time", None)),
            offset=(delta if delta is not None else 0.0),
            offset_status=status, found=found,
            reason=str(getattr(reg, "reason", "") or "") if reg is not None
            else REASON_NO_REGION))
    return out


# ---------------------------------------------------------------------------
# Deciding
# ---------------------------------------------------------------------------

def _nearest_gap(gap_times: Sequence[float], t: float
                 ) -> Tuple[Optional[float], float]:
    best, best_d = None, math.inf
    for g in gap_times or ():
        gf = _finite(g)
        if gf is None:
            continue
        d = abs(gf - t)
        if d < best_d:
            best, best_d = gf, d
    return best, best_d


def _decide(
    kind: str,
    boundaries: Sequence[CameraBoundary],
    master_time: Optional[float],
    master_gap_times: Sequence[float],
    cfg: ConsensusConfig,
) -> BoundaryDecision:
    d = BoundaryDecision(kind=kind, master_time=master_time,
                         canonical_time=master_time)

    def _t(b: CameraBoundary) -> Optional[float]:
        return b.global_start if kind == "start" else b.global_end

    for b in boundaries:
        if not b.found:
            continue
        if not b.comparable:
            d.not_comparable.append(b.camera_id)

    if master_time is None:
        d.decision = DECISION_NO_MASTER
        d.reason = ("the master produced no wagon region, and no other camera "
                    "may invent one")
        return d

    usable = [b for b in boundaries if b.comparable and _t(b) is not None]
    if not usable:
        d.decision = DECISION_NO_EVIDENCE
        d.reason = ("no camera's boundary could be normalized onto the global "
                    "timeline; the master's own boundary stands")
        return d

    # Who agrees with the incumbent, and who does not.
    for b in usable:
        if abs(_t(b) - master_time) <= cfg.agree_tolerance_sec:
            d.supported_by.append(b.camera_id)
            d.support_weight += b.weight
        else:
            d.disagreed_by.append(b.camera_id)

    dissent = [b for b in usable if b.camera_id in d.disagreed_by]
    if not dissent:
        d.reason = (f"all comparable cameras agree within "
                    f"{cfg.agree_tolerance_sec:.2f}s")
        return d

    # The dissenters' proposal: for START the earliest, for END the latest --
    # i.e. the boundary that would make the wagon region LARGER. A proposal that
    # would shrink the region is never acted on, because dropping a wagon the
    # master already counted needs stronger justification than a classifier
    # boundary, and this module must not reduce the count.
    if kind == "start":
        cand = min(dissent, key=lambda b: _t(b))
        if _t(cand) >= master_time:
            d.reason = ("dissent would shrink the region; the master's boundary "
                        "stands and no wagon is dropped")
            return d
    else:
        cand = max(dissent, key=lambda b: _t(b))
        if _t(cand) <= master_time:
            d.reason = ("dissent would shrink the region; the master's boundary "
                        "stands and no wagon is dropped")
            return d

    target = _t(cand)
    agreeing = [b for b in usable
                if abs(_t(b) - target) <= cfg.agree_tolerance_sec]
    d.proposed_time = target
    d.proposed_by = [b.camera_id for b in agreeing]
    d.proposed_weight = sum(b.weight for b in agreeing)

    if d.proposed_weight < cfg.min_move_weight:
        d.reason = (f"{REASON_INSUFFICIENT_WEIGHT}: weight "
                    f"{d.proposed_weight} < {cfg.min_move_weight}")
        return d
    if not any(b.is_side for b in agreeing):
        # A top camera may confirm a boundary; it may not relocate one. This is
        # the guard against a top classifier calling WAGON over the locomotive.
        d.reason = REASON_TOP_ONLY
        return d

    gap_t, gap_d = _nearest_gap(master_gap_times, target)
    d.master_gap_time = gap_t
    if gap_t is None or gap_d > cfg.gap_tolerance_sec:
        # No physical coupling there on the master's own validated sequence, so
        # moving here would manufacture a wagon out of classification alone.
        d.reason = (f"{REASON_NO_MASTER_GAP}: nearest validated master gap is "
                    + ("none" if gap_t is None else f"{gap_d:.2f}s away")
                    + f" (tolerance {cfg.gap_tolerance_sec:.2f}s)")
        return d

    d.canonical_time = target
    d.decision = DECISION_MOVED
    d.reason = (f"moved to a validated master gap at {gap_t:.2f}s on weighted "
                f"support {d.proposed_weight} from "
                f"{','.join(d.proposed_by)} (side camera included)")
    return d


def resolve(
    *,
    master_camera: str,
    master_window: Mapping[str, Any],
    support_regions: Mapping[str, Any],
    camera_offsets: Mapping[str, Mapping[str, Any]],
    master_gap_times: Sequence[float] = (),
    cfg: ConsensusConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> RegionConsensus:
    """Normalize all four cameras and decide the canonical START / END.

    Pure: reads the state, returns a decision. It never touches the roster --
    the caller applies the outcome, and the wagon units themselves stay the
    master's gap-delimited segments.
    """
    res = RegionConsensus(master_camera=master_camera)
    res.boundaries = camera_boundaries(
        master_camera=master_camera, master_window=master_window,
        support_regions=support_regions, camera_offsets=camera_offsets)

    m_start = _finite((master_window or {}).get("wagon_start_time"))
    m_end = _finite((master_window or {}).get("wagon_end_time"))
    res.start = _decide("start", res.boundaries, m_start, master_gap_times, cfg)
    res.end = _decide("end", res.boundaries, m_end, master_gap_times, cfg)

    if verbose:
        from core.logging_setup import get_logger
        log = get_logger("core.region_consensus")
        for line in res.render_lines():
            log.info("%s", line)
    return res
