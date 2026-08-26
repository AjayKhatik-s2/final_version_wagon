"""Sequential mode: global assembly from SEALED camera bundles.

This stage IS the old batch pipeline from Stage 2 onward, run late. Once every
camera has arrived and persisted its Stage-1 output, assembly performs exactly
what `run_global_count.py` STEP 3 and `master_runner.process_batch` Stages 2-5
did, in the same order, with the same functions and arguments:

    STEP 3   gf.assemble_global_train_state_master_fixed(..., wagon_regions=)
    Stage 2  materializer.wagon_cache_builder.build(...)
    Stage 3  features load -> {door, damage}      (load first, as in batch)
    Stage 4  fusion.wagon_state_builder.build(...)
    Stage 5  reporting.combined_train_report.build(...)

Two things follow from that, and they are the whole point of this module:

  * Support-camera evidence is bucketed by the materializer's arithmetic --
    `local_frame = round((GW.time - delta) * local_fps)` -- NOT by matching a
    camera's own local segments against the global wagons. The old pipeline has
    no local->global segment matcher, so neither does this one. The overlap
    mapper is retained ONLY to write a diagnostic audit file; nothing reads it.

  * Feature inference therefore runs HERE, not at camera arrival: a support
    camera's clock offset is unknowable until the master and that camera have
    both been seen, so its feature windows cannot be known earlier. This is not
    a Stage-1 re-run -- no gap model, tracker, stitching, validation or
    classification executes in this module. Those results are read back from
    the bundles.

Nothing under wagon_count/, reconstruction/, fusion/, materializer/ or
reporting/ is modified; every one of them is called exactly as batch calls it.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_PKG = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PKG)
_WC = os.path.join(_ROOT, "wagon_count")
for _p in (_ROOT, _WC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core import constants as C
from core.camera_evidence import (
    CameraEvidenceBundle, MAP_UNRESOLVED, map_segments_to_global,
    mapping_summary, ready_for_global_assembly,
)
from core.camera_tracks_io import read_tracks
from core.global_state_loader import parse_global_train_state


@dataclass
class AssemblyResult:
    ready: bool = False
    reason: str = ""
    total_wagons: int = 0
    sealed_cameras: List[str] = field(default_factory=list)
    failed_cameras: List[str] = field(default_factory=list)
    mapping_by_camera: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    wagon_regions_applied: List[str] = field(default_factory=list)
    cache_summary: Any = None
    train_window: Any = None
    train_window_filter: str = ""
    engine_frames: Any = None
    feature_summary: Dict[str, Any] = field(default_factory=dict)
    missing_cameras: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    state_json_path: str = ""
    report_pdf_path: str = ""
    report_json_path: str = ""
    yolo_calls_during_assembly: int = 0     # must stay 0


def _load_master_classifications(bundle: CameraEvidenceBundle) -> List[Any]:
    from global_train_state import _MasterClassification
    out: List[Any] = []
    for c in (bundle.read_json("classification.json") or []):
        try:
            out.append(_MasterClassification(
                segment_index=int(c["segment_index"]),
                start_frame=int(c["start_frame"]),
                end_frame=int(c["end_frame"]),
                label=str(c["label"]),
                confidence=float(c.get("confidence", 0.0)),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _load_wagon_region(bundle: CameraEvidenceBundle):
    """Rebuild this camera's `LocalWagonRegion` from its bundle.

    STEP 2b of `run_global_count.py` classifies every SUPPORT camera and builds
    a `train_structure.LocalWagonRegion`, which STEP 3 passes to fusion as
    `wagon_regions=`. Fusion uses it to keep engine / brake-van observations out
    of wagon alignment -- `global_fusion.filter_observations_to_wagon_region`,
    whose output "IS the canonical sequence given to the DP".

    Sequential mode already computes the identical region at camera time
    (`camera_pipeline.run_support_camera` -> `ts.build_local_wagon_region`) and
    persists it. This reads it back verbatim; the dataclass has no from_dict, so
    the fields are restored explicitly and any unknown key is ignored rather
    than silently dropping the whole region.
    """
    from train_structure import LocalWagonRegion

    d = bundle.read_json("wagon_region.json")
    if not isinstance(d, dict) or not d.get("camera_id"):
        return None
    return LocalWagonRegion(
        camera_id=str(d.get("camera_id") or bundle.camera_id),
        classifier_model=str(d.get("classifier_model") or ""),
        found=bool(d.get("found", False)),
        reason=str(d.get("reason") or ""),
        start_time=(None if d.get("start_time") is None
                    else float(d["start_time"])),
        end_time=(None if d.get("end_time") is None
                  else float(d["end_time"])),
        start_frame=(None if d.get("start_frame") is None
                     else int(d["start_frame"])),
        end_frame=(None if d.get("end_frame") is None
                   else int(d["end_frame"])),
        class_counts=dict(d.get("class_counts") or {}),
        segment_labels=list(d.get("segment_labels") or []),
        unmapped_classes=list(d.get("unmapped_classes") or []),
    )


#: Stage-3 order and per-feature arguments, identical to master_runner's.
#: LOAD first -- the damage processor reads the sibling load JSON.
DOOR_STRIDE = 3
DAMAGE_STRIDE = 3
LOAD_STRIDE = 2

_FEATURE_ORDER = (
    ("load",   dict(inference_mode="sampled", sample_stride=LOAD_STRIDE)),
    ("door",   dict(inference_mode="sampled", sample_stride=DOOR_STRIDE)),
    ("damage", dict(inference_mode="sampled", sample_stride=DAMAGE_STRIDE)),
)


def _feature_module(name: str):
    """Import a feature processor lazily, so assembly costs nothing if unused."""
    if name == "load":
        from features.load import processor as m
    elif name == "door":
        from features.door import processor as m
    elif name == "damage":
        from features.damage import processor as m
    else:
        raise ValueError(f"unknown feature {name!r}")
    return m


def assemble(
    *,
    evidence_root: str,
    output_root: str,
    batch_key: str,
    feat_models_dir: str = "",
    use_train_window: bool = True,
    master_camera: str = C.MASTER_CAMERA,
    all_cameras: Tuple[str, ...] = C.ALL_CAMERAS,
    verbose: bool = True,
) -> AssemblyResult:
    """Fuse sealed camera bundles into the global train + combined report."""
    import global_fusion as gf

    res = AssemblyResult()
    t_all = time.perf_counter()
    feat_models_dir = feat_models_dir or os.path.join(_ROOT, "models",
                                                      "features")

    ok, why = ready_for_global_assembly(evidence_root, master_camera,
                                        all_cameras)
    res.ready, res.reason = ok, why
    if not ok:
        if verbose:
            print(f"[ASSEMBLY] not ready: {why}")
        return res

    bundles = {c: CameraEvidenceBundle(evidence_root, c) for c in all_cameras}
    for c, b in bundles.items():
        st = b.load_manifest().state
        (res.sealed_cameras if st == "SEALED" else res.failed_cameras).append(c)

    # ---- reconstruct tracks (no Stage-1 re-run) ------------------------
    t0 = time.perf_counter()
    tracks = {}
    for c in res.sealed_cameras:
        t = read_tracks(os.path.join(bundles[c].dir, "tracking_full.json"))
        if t is not None:
            tracks[c] = t
    if master_camera not in tracks:
        res.ready, res.reason = False, "master tracking_full.json unreadable"
        return res
    res.timings["load_bundles"] = round(time.perf_counter() - t0, 3)

    # ---- STEP 3: existing fixed-master fusion, called as batch calls it ----
    # Support-camera wagon regions are restored from the bundles and passed
    # through, exactly as run_global_count.py STEP 3 does. Without them the
    # engine / brake-van observations of a support camera stay in the DP
    # alignment and displace correct matches -- the count is unaffected
    # (offsets are estimated pre-filter, by design) but the per-wagon evidence
    # association is not.
    support_regions: Dict[str, Any] = {}
    for c in res.sealed_cameras:
        if c == master_camera:
            continue                       # master has no support region
        region = _load_wagon_region(bundles[c])
        if region is not None:
            support_regions[c] = region
    res.wagon_regions_applied = sorted(support_regions)
    if verbose:
        missing = [c for c in res.sealed_cameras
                   if c != master_camera and c not in support_regions]
        print(f"[ASSEMBLY] wagon regions restored: {res.wagon_regions_applied}"
              + (f"  MISSING: {missing}" if missing else ""))

    # Support order follows ALL_CAMERAS, as batch does, not dict insertion.
    support = [tracks[c] for c in all_cameras
               if c != master_camera and c in tracks]

    # ---- canonical TRAIN WINDOW, before the master gaps are frozen --------
    # Classification decides where the physical train begins and ends; gaps
    # never do. The window then removes any RIGHT_UP gap lying outside it --
    # a detection on empty track ahead of the rake, or across the ENGINE's
    # leading face, which would otherwise become an inter-wagon boundary and
    # add a phantom wagon at the head of the train.
    #
    # Subtractive only. Fusion still receives RIGHT_UP's gaps and is still the
    # sole minter of the global sequence; nothing here touches that invariant.
    from core import train_window as TW

    master_tracks = tracks[master_camera]
    tw_window = None
    if use_train_window:
        segs_by_cam = {c: bundles[c].read_segments() for c in res.sealed_cameras}
        master_cls = _load_master_classifications(bundles[master_camera])
        tw_window = TW.detect_train_window(
            master_spans=TW.spans_from_master_classifications(
                master_cls, float(master_tracks.fps or 0.0), master_camera),
            # Support spans are projected at offset 0. Clock offsets are an
            # OUTPUT of fusion, which has not run yet, so they are not
            # available at this point -- and waiting for them would invert the
            # dependency, since the window is meant to constrain the gaps that
            # fusion consumes. The master's own classification sets the
            # boundary whenever it exists (the normal case) and needs no
            # offset, being the reference clock; support cameras only
            # corroborate, so a sub-second misalignment cannot move the edge.
            support_spans={
                c: TW.spans_from_local_segments(segs_by_cam.get(c) or [], c,
                                                0.0)
                for c in res.sealed_cameras if c != master_camera},
            master_gap_times=[g.center_time for g in master_tracks.gaps],
            master_camera=master_camera)
        res.train_window = tw_window
        if verbose:
            for line in tw_window.summary_lines():
                print(f"[TRAINWIN] {line}")
        filt = TW.filter_gaps_to_window(master_tracks.gaps, tw_window,
                                        fps=float(master_tracks.fps or 0.0))
        res.train_window_filter = filt.summary()
        if verbose:
            print(f"[TRAINWIN] {filt.summary()}")
        if filt.applied and filt.dropped:
            master_tracks.gaps = list(filt.kept)

    t0 = time.perf_counter()
    engine_state = gf.assemble_global_train_state_master_fixed(
        master_tracks=master_tracks,
        support_tracks=support,
        initial_classifications=_load_master_classifications(
            bundles[master_camera]),
        config=gf.FusionConfig(),
        verbose=verbose,
        wagon_regions=support_regions,
        wagon_only=True,
    )
    res.timings["fusion_alignment"] = round(time.perf_counter() - t0, 3)

    batch_root = os.path.join(output_root, batch_key)
    gs_dir = os.path.join(batch_root, "global_state")
    states_root = os.path.join(batch_root, "wagon_states")
    reports_root = os.path.join(batch_root, "reports")
    # Stage 2/3 write here, under GW_n names, exactly as batch does. These are
    # NOT the camera-local trees: each bundle keeps its own cache and evidence
    # for its own report, and assembly never reads them.
    global_evidence = os.path.join(batch_root, "evidence")
    global_cache = os.path.join(batch_root, "wagon_cache")
    for d in (gs_dir, states_root, reports_root):
        os.makedirs(d, exist_ok=True)
    if tw_window is not None:
        TW.write_artifact(tw_window, gs_dir)
    res.state_json_path = os.path.join(gs_dir, "global_train_state.json")
    with open(res.state_json_path, "w", encoding="utf-8") as f:
        f.write(engine_state.to_json())

    state = parse_global_train_state(engine_state.to_dict())
    res.total_wagons = state.total_wagons
    offsets_meta = state.camera_offsets or {}
    resolved = state.camera_time_offsets()

    # ---- DIAGNOSTIC ONLY: local segment -> global wagon audit -----------
    # This mapping is NOT how evidence is assigned. The old pipeline has no
    # local->global segment matcher; frames are bucketed by the materializer's
    # arithmetic below. The audit is written so a run can be compared against
    # the batch reference, and nothing downstream reads it.
    t0 = time.perf_counter()
    audit: Dict[str, Any] = {"_note": ("diagnostic only -- evidence assignment "
                                       "is done by the materializer, not by "
                                       "this mapping")}
    for c in res.sealed_cameras:
        segs = bundles[c].read_segments()
        is_resolved = (offsets_meta.get(c, {}) or {}).get(
            "status") in ("REFERENCE", "RESOLVED")
        maps = map_segments_to_global(
            segs, state.wagons, camera_id=c,
            offset=resolved.get(c, 0.0), offset_resolved=bool(is_resolved))
        summary = mapping_summary(maps)
        res.mapping_by_camera[c] = summary
        audit[c] = {"summary": summary, "mappings": [m.to_dict() for m in maps]}
    with open(os.path.join(gs_dir, "local_to_global_mapping.json"), "w",
              encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)
    res.timings["mapping_audit"] = round(time.perf_counter() - t0, 3)

    # ---- Stage 2: materializer, byte-for-byte the batch call ------------
    # `local_frame = round((GW.time - delta) * local_fps)` is the ONLY rule the
    # old pipeline uses to decide which of a camera's frames belong to a global
    # wagon. Source videos come from each bundle's manifest, so no Stage-1 work
    # is repeated -- this decodes frames, nothing else.
    from materializer import wagon_cache_builder

    video_paths: Dict[str, str] = {}
    per_camera_fps: Dict[str, float] = {}
    for c in res.sealed_cameras:
        mf = bundles[c].load_manifest()
        if mf.video_path and os.path.exists(mf.video_path):
            video_paths[c] = mf.video_path
            per_camera_fps[c] = float(mf.fps or 0.0)
        else:
            res.missing_cameras.append(c)
            print(f"[ASSEMBLY] {c}: source video unavailable "
                  f"({mf.video_path!r}) -- no cache for this camera")

    t0 = time.perf_counter()
    res.cache_summary = wagon_cache_builder.build(
        state=state,
        video_paths=video_paths,
        per_camera_fps=per_camera_fps,
        cache_root=global_cache,
        camera_offsets=resolved,
        verbose=verbose,
    )
    res.timings["stage2_materializer"] = round(time.perf_counter() - t0, 3)

    # Engine frames: the SAME shared extractor batch mode calls. Train-level
    # evidence, never a wagon -- see orchestrator/engine_frames.py.
    t0 = time.perf_counter()
    try:
        from orchestrator import engine_frames
        res.engine_frames = engine_frames.extract(
            state=state, video_paths=video_paths,
            per_camera_fps=per_camera_fps, output_root=batch_root,
            camera_offsets=resolved, verbose=verbose)
    except Exception as e:
        print(f"[ENGINE] extraction failed (non-fatal): "
              f"{type(e).__name__}: {e}")
    res.timings["engine_frames"] = round(time.perf_counter() - t0, 3)

    # ---- Stage 3: feature inference over the GLOBAL wagons --------------
    # Same processors, same strides, same order as master_runner: LOAD runs to
    # completion first so the damage processor's loaded-wagon floor-damage
    # filter always reads a fully-written wagon_states/load/<gw>.json.
    feature_kwargs = dict(state=state, cache_root=global_cache,
                          feature_models_dir=feat_models_dir,
                          output_dir=states_root,
                          evidence_root=global_evidence, verbose=verbose)
    t0 = time.perf_counter()
    for name, extra in _FEATURE_ORDER:
        try:
            mod = _feature_module(name)
        except Exception as e:
            print(f"[ASSEMBLY/{name}] unavailable: {e}")
            continue
        t1 = time.perf_counter()
        try:
            res.feature_summary[name] = mod.run(**feature_kwargs, **extra) or {}
        except Exception as e:
            print(f"[ASSEMBLY/{name}] CRASHED: {e}")
            traceback.print_exc(limit=3)
            res.feature_summary[name] = {}
        res.timings[f"stage3_{name}"] = round(time.perf_counter() - t1, 3)
    res.timings["stage3_features"] = round(time.perf_counter() - t0, 3)

    # ---- Stage 4: fuse (existing builder, unchanged) --------------------
    t0 = time.perf_counter()
    from fusion import wagon_state_builder
    unified = wagon_state_builder.build(state=state,
                                        wagon_states_root=states_root,
                                        verbose=verbose)
    res.timings["fusion_state"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    try:
        from reporting import combined_train_report
        out = combined_train_report.build(
            state=state, unified=unified, output_dir=reports_root,
            batch_key=batch_key, source_video_urls={},
            processed_video_urls={},
            evidence_root=global_evidence,
            wagon_states_root=states_root, cache_root=global_cache,
            missing_cameras=sorted(set(res.failed_cameras)
                                   | set(res.missing_cameras)),
            camera_pdf_urls={},
            logo_path=os.path.join(_ROOT, "reporting", "assets", "Logo.jpeg"),
            verbose=verbose)
        res.report_json_path = out.get("json_path") or ""
        res.report_pdf_path = out.get("pdf_path") or ""
    except Exception as e:
        print(f"[ASSEMBLY] combined report failed: {e}")
    res.timings["combined_report"] = round(time.perf_counter() - t0, 3)

    res.timings["total"] = round(time.perf_counter() - t_all, 3)
    if verbose:
        print(f"[ASSEMBLY] wagons={res.total_wagons} "
              f"sealed={res.sealed_cameras} failed={res.failed_cameras}")
        for c, s in res.mapping_by_camera.items():
            print(f"  {c:<13} {s['by_kind']}   (diagnostic)")
        print(f"  wagon regions applied : {res.wagon_regions_applied}")
        for k, v in sorted(res.feature_summary.items()):
            print(f"  feature {k:<8} wagons={len(v)}")
    return res
