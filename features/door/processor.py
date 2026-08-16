"""Door feature processor (v4, train-state-native, ALL legacy intelligence
ported).

Per wagon, for each side camera (RIGHT_UP / LEFT_UP):

    1. Iterate cached JPEGs in wagon_cache/<GW_n>/<camera>/.
    2. Run YOLO door_state.pt on the raw frame (half=False -- CPU build).
    3. Apply the model's own confidence gate.
    4. Feed surviving detections into the legacy DoorTracker (Kalman +
       Hungarian + per-track 30-frame quality-weighted majority vote +
       state machine with 2x hysteresis on OPEN -> CLOSED transitions +
       sticky DAMAGE state).
    5. After all frames, finalize the tracker -> per-track {state,
       confidence, snapshot}.
    6. Run DoorIdentityMerger to collapse fragmented tracks of the same
       physical door (spatial + temporal + context + structural).
    7. Pick the dominant door state per CAMERA SIDE, and persist ONE best
       representative snapshot per side.

REMOVED from this path (both modules remain on disk, untouched; Door was
their only consumer):

  * IlluminationProcessor -- was inert.  The call passed a `frame_idx=`
    kwarg the method does not accept, raising TypeError on every frame;
    the bare `except` pinned quality to 1.0.  Now hard-coded to 1.0, so
    the tracker receives exactly the same value as before.
  * GeometricShapePrior -- calibrated for PORTRAIT doors, while these
    side cameras see LANDSCAPE doors.  It rejected 100% of real
    detections on aspect and vertical-edge grounds, starved the tracker
    below n_init, and on LEFT_UP discarded an OPEN_DOOR detection,
    yielding a false CLOSED.  See the inline note at the call site for
    the measured before/after.

The per-CAMERA dominant state IS the per-side door state (RIGHT_UP -> right
door, LEFT_UP -> left door).  Same convention the legacy combined report
used.

Output JSON shape (per wagon):
    {
        "global_id":   "GW_7",
        "feature":     "door",
        "status":      "OK" | "NO_FRAMES" | "FAILED" | "NO_DATA",
        "left_door":   "CLOSED" | "OPEN" | "PARTIAL" | "DAMAGED" | "NO_DATA",
        "left_door_confidence":  0.91,
        "right_door":  "...",
        "right_door_confidence": 0.83,
        "tracks": [
            {camera_id, track_id, state, confidence, first_frame,
             last_frame, total_hits},
            ...
        ],
        "supporting_cameras": ["LEFT_UP", "RIGHT_UP"],
        "frame_count": ...,
    }
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.frame_quality import (
    detection_quality, snapshot_score, expand_bbox, _DOOR_BBOX_EXPAND_FRAC,
)

from features._common import (
    load_yolo, iter_wagon_frames, list_wagon_frames,
    write_per_wagon_json, empty_payload, FeatureTimer,
)

# Mature intelligence ported from legacy
from features.inference_lib.door_tracker import (
    DoorTracker, TrackerConfig, DoorState, yolo_to_detections,
)
from features.inference_lib.door_identity_merger import (
    DoorIdentityMerger, MergeConfig,
)
# NOTE: IlluminationProcessor and GeometricShapePrior are deliberately NOT
# imported here any more -- see the module docstring.  Both modules are left
# untouched on disk; only Door's use of them is removed.
from features._evidence import (
    BestFrameTracker, wagon_evidence_dir,
    save_jpeg, safe_crop, write_metadata, draw_annotated_bbox,
)
# EXPERIMENTAL sampled path only; the legacy tracker path never touches this.
from features.evidence_aggregator import EvidenceAggregator, Observation


FEATURE_NAME = "door"


# -----------------------------------------------------------------------------
# Canonicalization of legacy state strings
# -----------------------------------------------------------------------------

# DoorTracker.DoorState values are: UNKNOWN / CLOSED / OPEN / PARTIAL_CLOSED /
# DAMAGE / OTHER.  Map to the v4 canonical vocabulary.
_STATE_TO_CANONICAL = {
    "OPEN":            C.DOOR_OPEN,
    "CLOSED":          C.DOOR_CLOSED,
    "PARTIAL_CLOSED":  C.DOOR_PARTIAL,
    "PARTIAL":         C.DOOR_PARTIAL,
    "DAMAGE":          C.DOOR_DAMAGED,
    "DAMAGED":         C.DOOR_DAMAGED,
    "OTHER":           C.NO_DATA,
    "UNKNOWN":         C.NO_DATA,
}


def _canonical(state_value: str) -> str:
    s = str(state_value or "").strip().upper()
    return _STATE_TO_CANONICAL.get(s, s if s else C.NO_DATA)


# -----------------------------------------------------------------------------
# Per-camera tracker run
# -----------------------------------------------------------------------------

def _run_tracker_one_camera(
    yolo_model,
    tracker_config: TrackerConfig,
    merger_config: MergeConfig,
    cache_root: str,
    gw_id: str,
    camera_id: str,
) -> Tuple[List[Dict[str, Any]], int, int, int, Dict[str, "BestFrameTracker"]]:
    """Run the full per-camera door pipeline on one wagon.

    Returns:
        (track_decisions, n_frames, width, height, evidence_candidates)

    ``evidence_candidates`` is a ``{canonical_state -> BestFrameTracker}`` map.
    For each door state observed on this camera we keep the single highest
    snapshot-quality frame (legacy ``_score_detection``: area + horizontal
    centre + confidence + crop quality, with an edge-hugging penalty).  The
    caller picks the bucket matching the wagon's reported side-state so the
    persisted snapshot actually shows that (often anomalous) state, falling
    back to the globally best-scored frame when no such frame exists.
    """
    paths = list_wagon_frames(cache_root, gw_id, camera_id, trim_stable=True)
    if not paths:
        return [], 0, 0, 0, {}, {"tracks": [], "events": []}

    # Fresh tracker per (gw, camera).  Wagons are independent in the new
    # train-state-native world, so each one resets the tracker.
    tracker = DoorTracker(config=tracker_config)
    tracker.reset()

    frame_w, frame_h = 0, 0
    used = 0
    cands: Dict[str, BestFrameTracker] = {}

    # Per-frame confirmed-track positions for the Stage-4b overlay.  We record
    # the Kalman-smoothed tlbr + FSM state of every CONFIRMED track after each
    # tracker step -- including frames where the door was only predicted (no
    # detection) -- exactly mirroring how the legacy door_processor wrote its
    # `_tracked.mp4` (draws `track.tlbr` for confirmed tracks every frame, so
    # boxes glide smoothly through detection-less frames).  Persisting it lets
    # the visualization-only renderer replay the motion WITHOUT re-running any
    # detector.  Keyed by track_id.
    trajectory: Dict[int, Dict[str, Any]] = {}
    # Ordered absolute cache frame indices, one entry per tracker.update() step.
    # DoorTracker numbers its events with an INTERNAL 1-based step counter
    # (self.frame_idx, ++ per update), but the renderer keys the event banner by
    # ABSOLUTE cache frame index.  _snapshot_confirmed runs exactly once per
    # tracker.update(), so recording `fi` here builds the step->absolute map used
    # to translate event frames at finalize -- otherwise the banner fires on the
    # wrong frame / a neighbouring wagon's span.
    step_to_abs: List[int] = []

    def _snapshot_confirmed(frame_index: int) -> None:
        step_to_abs.append(int(frame_index))
        for t in tracker.tracks:
            if not t.is_confirmed():
                continue
            try:
                bb = [float(v) for v in t.tlbr]
            except Exception:
                continue
            # ITEM 4: expand the persisted overlay box so the processed-video
            # rectangle visually contains the WHOLE door (matches the expanded
            # evidence crop below).  Clipped to the frame; still a clean rect.
            bb = expand_bbox(bb, _DOOR_BBOX_EXPAND_FRAC, frame_w, frame_h)
            try:
                vel = [float(t.velocity[0]), float(t.velocity[1])]
            except Exception:
                vel = [0.0, 0.0]
            # Persist the RAW legacy fields the overlay needs to reproduce the
            # exact legacy door annotation: the raw DoorState value (for colour
            # + label), last_class (UNKNOWN colour/label fallback), the raw
            # last-frame confidence, and the velocity vector (arrow).
            entry = trajectory.setdefault(int(t.track_id), {
                "camera_id": camera_id,
                "track_id":  int(t.track_id),
                "frames":    [],
            })
            entry["frames"].append({
                "frame_idx":  int(frame_index),
                "bbox":       bb,
                "state_raw":  str(t.state_machine.get_state().value),
                "last_class": str(getattr(t, "last_class", "") or ""),
                "confidence": float(getattr(t, "last_confidence", 0.0) or 0.0),
                "velocity":   vel,
            })

    # ------- frame loop (stable interior only) -------
    # EVERY frame of the stable interior is inspected.  Frame sub-sampling was
    # measured and REJECTED: at stride 2 the per-track hit count halves
    # (GW_21 6->2, GW_22 8->2), dropping below TrackerConfig.n_init=3 /
    # min_hits_for_decision=3, so marginal tracks stop confirming.  On both
    # wagons LEFT_UP lost its track entirely and fell back to CLOSED/0.000 --
    # re-breaking exactly what removing the geometric prior had recovered.
    for fi, frame in iter_wagon_frames(cache_root, gw_id, camera_id,
                                       trim_stable=True):
        if frame_w == 0:
            frame_h, frame_w = frame.shape[:2]
        used += 1

        # 1) Detection quality weight handed to the tracker.
        #    The legacy IlluminationProcessor is REMOVED from this path.  It was
        #    already inert: the call site passed a `frame_idx=` kwarg that
        #    `process_frame(self, frame)` does not accept, so it raised
        #    TypeError on EVERY frame and the bare `except` pinned quality to
        #    1.0.  Hard-coding 1.0 therefore feeds the tracker exactly the same
        #    value it has always received -- bit-identical behaviour, minus a
        #    per-frame exception.
        quality = 1.0

        # 2) YOLO detection on raw frame.
        #    half=False is REQUIRED: production runs on a CPU-only torch build,
        #    which has no native fp16 kernels.  Measured on door_state.pt with
        #    identical frames and parameters (torch 2.12.0+cpu):
        #        half=True   112,637 ms/frame   0 detections
        #        half=False       675 ms/frame  1 detection @ conf 0.93
        #    fp16 is emulated (167x slower) AND degrades the numerics enough
        #    that every box falls below threshold, so the door state silently
        #    collapsed to CLOSED/0.00 for every wagon.
        try:
            results = yolo_model(frame, verbose=False, half=False)[0]
        except Exception:
            continue
        if results.boxes is None or len(results.boxes) == 0:
            tracker.update([], frame=frame,
                           frame_width=frame_w, frame_height=frame_h)
            _snapshot_confirmed(fi)
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss  = results.boxes.cls.cpu().numpy().astype(int)

        # 3) confidence floor (the tracker has its own per-class thresholds
        #    too; this gate just discards obviously-noisy detections early).
        min_conf = float(tracker_config.closed_confidence_threshold)
        keep = confs >= min_conf
        boxes, confs, clss = boxes[keep], confs[keep], clss[keep]
        if len(boxes) == 0:
            tracker.update([], frame=frame,
                           frame_width=frame_w, frame_height=frame_h)
            _snapshot_confirmed(fi)
            continue

        # 4) The legacy GeometricShapePrior filter is REMOVED from this path.
        #    Measured on wagon GW_21 of the validation set (real door_state.pt,
        #    51 frames per side, identical inputs):
        #
        #      prior ON   RIGHT_UP 1 track hits=1  | LEFT_UP 0 tracks -> NO_DATA
        #      prior OFF  RIGHT_UP 1 track hits=3  | LEFT_UP 1 track  hits=3
        #
        #    It is calibrated for a PORTRAIT door (config documents "taller than
        #    wide", preferred aspect 0.5, vertical edges dominant).  These side
        #    cameras see LANDSCAPE doors: measured aspect 1.49-2.18 with
        #    HORIZONTAL edge energy dominant, so the aspect and vertical-edge
        #    gates fail on 100% of real detections.  Worse, on LEFT_UP it
        #    discarded an OPEN_DOOR detection outright, leaving the no-track
        #    fallback to report CLOSED -- a safety-critical false negative.
        #    Suppressing detections also starved the tracker below n_init=3, so
        #    tracks never confirmed.  Detection reliability now rests on the
        #    model's own confidence gate plus temporal confirmation.
        #
        #    `features/inference_lib/geometric_shape_prior.py` is left intact;
        #    Door was its only consumer.

        # 5) convert to Detection objects + feed tracker
        names = getattr(yolo_model, "names", {}) or {}
        detections = yolo_to_detections(
            boxes=boxes, confidences=confs, class_ids=clss,
            class_names=names, illumination_quality=quality,
        )
        tracker.update(
            detections=detections,
            frame=frame,
            frame_width=frame_w,
            frame_height=frame_h,
        )
        _snapshot_confirmed(fi)

        # 6) evidence:  bucket each surviving detection by its canonical state
        # and keep the single highest snapshot-QUALITY frame per state (legacy
        # _score_detection -- area + horizontal centre + confidence + crop
        # quality, edge-hugging penalty).  This is what makes the persisted
        # snapshot sharp / centred / non-edge instead of merely high-confidence.
        for bbox, conf, cls_id in zip(boxes, confs, clss):
            cls_name = str(names.get(int(cls_id), "")).lower()
            canon = _canonical(cls_name)
            bbox_list = [float(bbox[0]), float(bbox[1]),
                         float(bbox[2]), float(bbox[3])]
            # Score on the RAW detection box (true area / centre / quality).
            crop_q = detection_quality(frame, bbox_list)
            sc = snapshot_score(bbox_list, float(conf), crop_q,
                                frame_w, frame_h)
            # ITEM 4: persist an EXPANDED box so the evidence crop + annotated
            # frame + metadata bbox visually contain the WHOLE door, consistent
            # with the expanded overlay box.
            bbox_store = expand_bbox(bbox_list, _DOOR_BBOX_EXPAND_FRAC,
                                     frame_w, frame_h)
            bucket = cands.setdefault(canon, BestFrameTracker())
            if sc > bucket.score:
                bucket.update(
                    score=sc, frame=frame, bbox=bbox_store, frame_idx=fi,
                    state=canon, confidence=float(conf),
                    raw_class=cls_name, quality=float(crop_q),
                )

    # ------- finalize -------
    # Bundle the per-frame trajectory + door-level events for the overlay.
    def _abs_event_frame(rel: Any) -> int:
        # DoorTracker numbers events with a 1-based internal step counter; step k
        # corresponds to step_to_abs[k-1] (the absolute cache frame index).
        try:
            k = int(rel)
        except (TypeError, ValueError):
            return -1
        if 1 <= k <= len(step_to_abs):
            return step_to_abs[k - 1]
        return -1

    overlay = {
        "tracks": list(trajectory.values()),
        "events": [
            {"frame_idx": _abs_event_frame(e.get("frame_idx", -1)),
             "event":     str(e.get("event", "")),
             "track_id":  int(e.get("track_id", -1)),
             "camera_id": camera_id}
            for e in (tracker.get_events() or [])
        ],
    }
    final_states = tracker.get_final_door_states()
    if not final_states:
        return [], used, frame_w, frame_h, cands, overlay

    # Run identity merger on the final track set (collapses fragmented IDs
    # of the same physical door).  Operates on the live + deleted track
    # objects exposed by the tracker.
    try:
        merger = DoorIdentityMerger(config=merger_config)
        all_tracks_objs = list(tracker.tracks) + list(tracker.deleted_tracks)
        merged_groups = merger.merge_all_tracks(all_tracks_objs)
        # merge_all_tracks returns mapping {canonical_id: [member_ids]};
        # we keep the canonical id for each group as the surviving track.
        if isinstance(merged_groups, dict) and merged_groups:
            merged_ids = set(merged_groups.keys())
        elif isinstance(merged_groups, list) and merged_groups:
            merged_ids = set(merged_groups)
        else:
            merged_ids = set(final_states.keys())
    except Exception:
        merged_ids = set(final_states.keys())   # fallback: keep everything

    decisions: List[Dict[str, Any]] = []
    all_tracks = list(tracker.tracks) + list(tracker.deleted_tracks)
    by_id = {t.track_id: t for t in all_tracks}

    for tid, state_dict in final_states.items():
        if merged_ids and tid not in merged_ids:
            continue
        tr = by_id.get(tid)
        mean_cx = float(np.mean([d['bbox'][[0,2]].mean()
                                 for d in (tr.detections if tr else [])])) \
            if (tr and getattr(tr, "detections", None)) else 0.0
        decisions.append({
            "camera_id":   camera_id,
            "track_id":    tid,
            "state":       _canonical(state_dict.get("state")),
            "confidence":  float(state_dict.get("confidence", 0.0) or 0.0),
            "first_frame": int(state_dict.get("first_frame", 0)),
            "last_frame":  int(state_dict.get("last_frame", 0)),
            "total_hits":  int(state_dict.get("total_hits", 0)),
            "mean_center_x": mean_cx,
        })
    return decisions, used, frame_w, frame_h, cands, overlay


# -----------------------------------------------------------------------------
# EXPERIMENTAL sampled path (inference_mode="sampled")
# -----------------------------------------------------------------------------

def _run_sampled_one_camera(
    yolo_model,
    tracker_config: TrackerConfig,
    cache_root: str,
    gw_id: str,
    camera_id: str,
    sample_stride: int = 2,
) -> Tuple[List[Dict[str, Any]], int, int, int, Dict[str, "BestFrameTracker"],
           Dict[str, Any]]:
    """Sampled-frame Door inference -- EXPERIMENTAL, not the default path.

    Returns the SAME 6-tuple as `_run_tracker_one_camera`, so `run()` below is
    identical for both modes: same `_pick_side_state`, same `_resolve_evidence`,
    same evidence files, same JSON.  Only how detections are gathered differs.

    Why this exists: Door spends ~97% of its wall clock in YOLO, so the only
    real lever is fewer calls.  Plain stride-2 against the legacy tracker
    failed because `n_init`/`min_hits_for_decision` are ABSOLUTE hit counts
    that halve with the sample rate (GW_21 6->2, GW_22 8->2, both losing the
    LEFT_UP track).  EvidenceAggregator replaces that with a support FRACTION,
    which is stride-invariant.  The tracker is NOT used here and its thresholds
    are therefore neither read nor modified.

    Deliberately unchanged from legacy: the model, the confidence gate, the
    class mapping, and the evidence-bucket keying -- including the known
    `_canonical()` bucket-key quirk.  Replicating that quirk keeps the A/B
    comparison clean; fixing it here would conflate two changes.
    """
    paths = list_wagon_frames(cache_root, gw_id, camera_id, trim_stable=True)
    if not paths:
        return [], 0, 0, 0, {}, {"tracks": [], "events": []}

    stride = max(1, int(sample_stride))
    frame_w, frame_h = 0, 0
    used = 0
    cands: Dict[str, BestFrameTracker] = {}
    names = getattr(yolo_model, "names", {}) or {}
    min_conf = float(tracker_config.closed_confidence_threshold)

    agg: Optional[EvidenceAggregator] = None
    trajectory: Dict[int, Dict[str, Any]] = {}

    for fi, frame in iter_wagon_frames(cache_root, gw_id, camera_id,
                                       every_nth=stride, trim_stable=True):
        if frame_w == 0:
            frame_h, frame_w = frame.shape[:2]
            agg = EvidenceAggregator(frame_width=frame_w, frame_height=frame_h,
                                     stride=stride)
        used += 1

        try:
            results = yolo_model(frame, verbose=False, half=False)[0]
        except Exception:
            continue
        if results.boxes is None or len(results.boxes) == 0:
            agg.add_frame(fi, [])
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)
        keep = confs >= min_conf
        boxes, confs, clss = boxes[keep], confs[keep], clss[keep]
        if len(boxes) == 0:
            agg.add_frame(fi, [])
            continue

        observations: List[Observation] = []
        for bbox, conf, cls_id in zip(boxes, confs, clss):
            raw = str(names.get(int(cls_id), "")).lower()
            # Raw YOLO class -> canonical door state, via the SAME table the
            # rest of the pipeline uses.  _canonical() is the fallback for any
            # label the table does not cover.
            canon_state = C.DOOR_LABEL_TO_STATE.get(raw, _canonical(raw))
            bl = [float(v) for v in bbox]
            crop_q = detection_quality(frame, bl)
            sc = snapshot_score(bl, float(conf), crop_q, frame_w, frame_h)
            observations.append(Observation(
                frame_idx=int(fi), state=canon_state, confidence=float(conf),
                bbox=(bl[0], bl[1], bl[2], bl[3]), score=float(sc),
            ))

            # Evidence buckets -- keyed exactly as the legacy path keys them.
            bucket_key = _canonical(raw)
            bbox_store = expand_bbox(bl, _DOOR_BBOX_EXPAND_FRAC, frame_w, frame_h)
            bucket = cands.setdefault(bucket_key, BestFrameTracker())
            if sc > bucket.score:
                bucket.update(score=sc, frame=frame, bbox=bbox_store,
                              frame_idx=fi, state=bucket_key,
                              confidence=float(conf), raw_class=raw,
                              quality=float(crop_q))

            entry = trajectory.setdefault(len(trajectory) + 1, {
                "camera_id": camera_id, "track_id": len(trajectory) + 1,
                "frames": [],
            })
            entry["frames"].append({
                "frame_idx": int(fi), "bbox": bbox_store,
                "state_raw": raw, "last_class": raw,
                "confidence": float(conf), "velocity": [0.0, 0.0],
            })

        agg.add_frame(fi, observations)

    if agg is None:
        return [], used, frame_w, frame_h, cands, {"tracks": [], "events": []}

    result = agg.finalize()
    decisions: List[Dict[str, Any]] = []
    for g in result["accepted"]:
        best = g.get("best")
        decisions.append({
            "camera_id":   camera_id,
            "track_id":    int(g["candidate_id"]),
            "state":       str(g["state"]),
            "confidence":  float(g["confidence"]),
            "first_frame": int(g["first_frame"]),
            "last_frame":  int(g["last_frame"]),
            # Frame support is the sampled-mode analogue of tracker hits.
            "total_hits":  int(g["frame_support"]),
            "mean_center_x": float(best.center[0]) if best else 0.0,
        })

    overlay = {"tracks": list(trajectory.values()), "events": []}
    return decisions, used, frame_w, frame_h, cands, overlay


# -----------------------------------------------------------------------------
# Per-side decision picker
# -----------------------------------------------------------------------------

def _pick_side_state(track_decisions: List[Dict[str, Any]]) -> Tuple[str, float]:
    """Pick the dominant door state for one camera/side.

    Priority order:
        1. Any DAMAGED track  -> DAMAGED (terminal in the FSM)
        2. Any OPEN track     -> OPEN  (safety-critical; legacy code biases here)
        3. Any PARTIAL track  -> PARTIAL
        4. Most-frequent CLOSED-class result by total_hits, confidence weighted
        5. NO_DATA
    """
    if not track_decisions:
        return C.NO_DATA, 0.0

    def _max_conf(items):
        return max(items, key=lambda d: (d["total_hits"], d["confidence"]))

    damaged = [d for d in track_decisions if d["state"] == C.DOOR_DAMAGED]
    if damaged:
        best = _max_conf(damaged)
        return C.DOOR_DAMAGED, best["confidence"]

    opens = [d for d in track_decisions if d["state"] == C.DOOR_OPEN]
    if opens:
        best = _max_conf(opens)
        return C.DOOR_OPEN, best["confidence"]

    partials = [d for d in track_decisions if d["state"] == C.DOOR_PARTIAL]
    if partials:
        best = _max_conf(partials)
        return C.DOOR_PARTIAL, best["confidence"]

    closeds = [d for d in track_decisions if d["state"] == C.DOOR_CLOSED]
    if closeds:
        best = _max_conf(closeds)
        return C.DOOR_CLOSED, best["confidence"]

    return C.NO_DATA, 0.0


def _resolve_evidence(
    cands: Dict[str, "BestFrameTracker"], reported_state: str,
) -> "BestFrameTracker":
    """Pick the evidence frame for one side.

    Prefer the highest snapshot-quality frame that actually shows the wagon's
    reported side-state (anomaly-central: an OPEN/DAMAGED snapshot for an
    OPEN/DAMAGED door).  If no frame of that state was captured, fall back to
    the globally best-scored frame on the camera.
    """
    bucket = cands.get(reported_state)
    if bucket is not None and bucket.has_data():
        return bucket
    best = BestFrameTracker()
    for b in cands.values():
        if b.has_data() and b.score > best.score:
            best = b
    return best


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def run(
    *,
    state: GlobalTrainState,
    cache_root: str,
    feature_models_dir: str,
    output_dir: str,
    evidence_root: Optional[str] = None,   # NEW: enables evidence persistence
    confidence: float = C.CONF_DOOR,
    every_nth: int = 1,
    max_frames: int = 0,           # 0 = unbounded (legacy used the whole wagon)
    verbose: bool = True,
    inference_mode: str = "legacy",
    sample_stride: int = 2,
) -> Dict[str, str]:
    """Run the door feature on every wagon.

    `inference_mode`:
        "legacy"  (DEFAULT) every frame + DoorTracker.  The known-good path;
                  byte-for-byte the behaviour benchmarked on EC2.
        "sampled" EXPERIMENTAL: every `sample_stride`-th frame +
                  EvidenceAggregator.  Never selected unless a caller asks for
                  it explicitly -- the orchestrator does not pass this
                  argument, so production is unaffected by construction.

    Both modes share the same model, confidence gate, class mapping, per-side
    resolution, evidence persistence and JSON schema.
    """
    del every_nth, max_frames  # kept for API symmetry; we iterate every frame
    mode = str(inference_mode or "legacy").strip().lower()
    if mode not in ("legacy", "sampled"):
        raise ValueError(
            f"door inference_mode must be 'legacy' or 'sampled', got {mode!r}")

    model_path = os.path.join(feature_models_dir, C.MODEL_DOOR_STATE)
    yolo_model = load_yolo(model_path)

    feature_out = os.path.join(output_dir, FEATURE_NAME)
    os.makedirs(feature_out, exist_ok=True)
    timer = FeatureTimer("door")
    summary: Dict[str, str] = {}

    # Pre-construct shared per-process helpers (loaded once across wagons)
    tracker_cfg  = TrackerConfig()
    merger_cfg   = MergeConfig()

    if yolo_model is None and verbose:
        print(f"[FEAT/door] WARNING: {model_path} missing; emitting NO_DATA "
              f"for every wagon.")

    if verbose:
        print(f"[FEAT/door] running on {len(state.wagons)} wagons "
              f"(conf>={confidence}, legacy DoorTracker + IdentityMerger + "
              f"GeometricPrior + IlluminationQuality)")

    for gw in state.wagons:
        gw_id = gw.global_id
        t0 = time.time()
        try:
            if yolo_model is None:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.NO_DATA,
                    left_door=C.NO_DATA, left_door_confidence=0.0,
                    right_door=C.NO_DATA, right_door_confidence=0.0,
                    tracks=[], supporting_cameras=[],
                    error="door_state.pt not present",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.NO_DATA
                continue

            def _one_camera(cam):
                if mode == "sampled":
                    return _run_sampled_one_camera(
                        yolo_model, tracker_cfg, cache_root, gw_id, cam,
                        sample_stride=sample_stride,
                    )
                return _run_tracker_one_camera(
                    yolo_model, tracker_cfg, merger_cfg, cache_root, gw_id, cam,
                )

            l_decisions, l_used, _, _, l_cands, l_overlay = _one_camera(C.CAMERA_LEFT_UP)
            r_decisions, r_used, _, _, r_cands, r_overlay = _one_camera(C.CAMERA_RIGHT_UP)

            supporting: List[str] = []
            if l_used > 0: supporting.append(C.CAMERA_LEFT_UP)
            if r_used > 0: supporting.append(C.CAMERA_RIGHT_UP)

            if l_used == 0 and r_used == 0:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
                    left_door=C.NO_DATA, left_door_confidence=0.0,
                    right_door=C.NO_DATA, right_door_confidence=0.0,
                    tracks=[], supporting_cameras=[],
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_NO_FRAMES
                continue

            l_state, l_conf = _pick_side_state(l_decisions)
            r_state, r_conf = _pick_side_state(r_decisions)

            # If frames existed but tracker produced no confirmed tracks,
            # treat as CLOSED with low confidence -- typical pattern when
            # the wagon's doors are uniformly closed and the model doesn't
            # bother firing.  The conservative legacy default.
            if l_used > 0 and l_state == C.NO_DATA:
                l_state, l_conf = C.DOOR_CLOSED, 0.0
            if r_used > 0 and r_state == C.NO_DATA:
                r_state, r_conf = C.DOOR_CLOSED, 0.0

            # Resolve the evidence frame per side now that the reported state
            # is known: prefer the best-quality frame that shows that state
            # (anomaly-central), else the globally best-scored frame.
            l_best = _resolve_evidence(l_cands, l_state)
            r_best = _resolve_evidence(r_cands, r_state)

            # Persist evidence: best left + right door snapshot (full
            # frame + bbox crop) into evidence/<gw>/door/.
            evidence_paths: Dict[str, str] = {}
            if evidence_root and (l_best.has_data() or r_best.has_data()):
                ev_dir = wagon_evidence_dir(evidence_root, gw_id, FEATURE_NAME)
                meta: Dict[str, Any] = {"global_id": gw_id,
                                        "feature": FEATURE_NAME,
                                        "sides": {}}
                for side_key, side_best, cam in (
                    ("left",  l_best, C.CAMERA_LEFT_UP),
                    ("right", r_best, C.CAMERA_RIGHT_UP),
                ):
                    if not side_best.has_data():
                        continue
                    full_p = os.path.join(ev_dir, f"{side_key}_best.jpg")
                    crop_p = os.path.join(ev_dir, f"{side_key}_crop.jpg")
                    # full frame with annotation drawn for the report
                    annotated = draw_annotated_bbox(
                        side_best.frame, side_best.bbox,
                        label=f"{side_best.meta.get('state','?')} "
                              f"{side_best.meta.get('confidence',0.0):.2f}",
                        color=(0, 255, 255),
                    )
                    save_jpeg(full_p, annotated)
                    crop_img = safe_crop(side_best.frame, side_best.bbox, pad=12)
                    if crop_img is not None:
                        save_jpeg(crop_p, crop_img)
                    evidence_paths[f"{side_key}_best"] = full_p
                    if crop_img is not None:
                        evidence_paths[f"{side_key}_crop"] = crop_p
                    meta["sides"][side_key] = {
                        "camera_id":  cam,
                        "frame_idx":  side_best.frame_idx,
                        "bbox":       side_best.bbox,
                        "state":      side_best.meta.get("state"),
                        "confidence": side_best.meta.get("confidence"),
                        "raw_class":  side_best.meta.get("raw_class"),
                        "quality":    side_best.meta.get("quality"),
                    }
                write_metadata(os.path.join(ev_dir, "metadata.json"), meta)

            # Persist the per-frame track trajectory for the Stage-4b overlay.
            # SEPARATE from metadata.json (which carries only the single best
            # frame for the PDF).  Includes EVERY confirmed door track -- closed
            # doors too -- so the renderer draws all tracked boxes per frame,
            # matching the legacy `_tracked.mp4` behaviour.
            if evidence_root:
                door_tracks = l_overlay.get("tracks", []) + r_overlay.get("tracks", [])
                door_events = l_overlay.get("events", []) + r_overlay.get("events", [])
                if door_tracks or door_events:
                    ov_dir = wagon_evidence_dir(evidence_root, gw_id, FEATURE_NAME)
                    write_metadata(
                        os.path.join(ov_dir, "overlay.json"),
                        {"global_id": gw_id, "feature": FEATURE_NAME,
                         "tracks": door_tracks, "events": door_events},
                    )

            payload: Dict[str, Any] = {
                "global_id":   gw_id,
                "feature":     FEATURE_NAME,
                "status":      C.STATUS_OK,
                "left_door":   l_state,
                "left_door_confidence":  round(float(l_conf), 4),
                "right_door":  r_state,
                "right_door_confidence": round(float(r_conf), 4),
                "tracks":      l_decisions + r_decisions,
                "supporting_cameras": supporting,
                "frame_count": l_used + r_used,
                "frames_left":  l_used,
                "frames_right": r_used,
                "evidence":    evidence_paths,
            }
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_OK
            if verbose:
                print(f"  [door/{gw_id}]  L={l_state} ({l_conf:.2f})  "
                      f"R={r_state} ({r_conf:.2f})  "
                      f"tracks={len(l_decisions)+len(r_decisions)}  "
                      f"frames={l_used + r_used}")
        except Exception as e:
            payload = empty_payload(
                gw_id, FEATURE_NAME, C.STATUS_FAILED,
                left_door=C.NO_DATA, right_door=C.NO_DATA,
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(limit=2),
            )
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_FAILED
            if verbose:
                print(f"  [door/{gw_id}] FAILED: {e}")
        finally:
            timer.stamp(gw_id, t0)

    if verbose:
        n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
        print(f"[FEAT/door] done in {timer.total():.1f}s  ok={n_ok}/{len(summary)}")
    return summary
