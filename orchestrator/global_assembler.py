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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    #: `core.vehicle_type.resolve_train` summary: per-wagon resolved type,
    #: source camera, confidence, corroboration/dissent, and what the top
    #: cameras predicted (recorded as ignored).
    type_resolution: Dict[str, Any] = field(default_factory=dict)
    #: `core.active_region.resolve` summary: the BEFORE/ACTIVE/AFTER lifecycle,
    #: both boundaries with provenance, excluded leading/trailing non-wagons,
    #: ignored out-of-region predictions, and whether the gate held.
    active_region: Dict[str, Any] = field(default_factory=dict)
    cache_summary: Any = None
    feature_summary: Dict[str, Any] = field(default_factory=dict)
    missing_cameras: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    state_json_path: str = ""
    report_pdf_path: str = ""
    report_json_path: str = ""
    yolo_calls_during_assembly: int = 0     # must stay 0
    delivery: Any = None                   # delivery.finalize.DeliveryResult
    # Stage 4b + the URL maps the published documents embed.
    processed_video_paths: Dict[str, str] = field(default_factory=dict)
    processed_video_urls: Dict[str, str] = field(default_factory=dict)
    source_video_urls: Dict[str, str] = field(default_factory=dict)
    per_camera_tracking_path: str = ""
    damage_boundary: Any = None      # damage_boundary.BoundaryResult


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


def _load_classifications(bundle: CameraEvidenceBundle):
    """This camera's persisted segment classifications, as plain records.

    Sequential cameras are processed and sealed independently, so a side
    camera's classification is on disk long before the others arrive. Reading it
    back here is what lets final assembly resolve type against the canonical
    timeline without needing all four feeds at camera time.

    Returns [] for a camera that has not been processed -- absence must mean
    "no evidence", never "not a wagon".
    """
    rows = bundle.read_json("classification.json") or []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        out.append(_Cls(segment_index=r.get("segment_index"),
                        start_frame=r.get("start_frame"),
                        end_frame=r.get("end_frame"),
                        label=r.get("label") or "",
                        confidence=float(r.get("confidence") or 0.0)))
    return out


@dataclass
class _Cls:
    """The four fields the type resolver reads. Deliberately not the engine's
    own class: this module must not depend on wagon_count internals to read a
    JSON file it wrote itself."""
    segment_index: Any = None
    start_frame: Any = None
    end_frame: Any = None
    label: str = ""
    confidence: float = 0.0


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
#:
#: NO FRAME IS SKIPPED. Every feature runs on every frame of every wagon.
#:
#: This is `inference_mode="legacy"` -- which each processor documents as "every
#: frame + <feature>Tracker; the known-good path, byte-for-byte the behaviour
#: benchmarked on EC2". It is also each processor's DEFAULT, so the way to ask
#: for it is to pass no inference arguments at all, exactly as OCR already does.
#: Reaching it via `sampled` with sample_stride=1 would NOT be equivalent: that
#: path uses EvidenceAggregator instead of the tracker, so it is a different
#: algorithm rather than the same one at full rate.
#:
#: `every_nth=1` is passed to LOAD explicitly because load samples even in
#: legacy mode -- its `every_nth` defaults to 2, and (since the max_frames=0 ->
#: None fix) that default actually takes effect. Legacy mode alone would still
#: have skipped every other frame there.
#:
#: The strides that were here (door 3, damage 3, load 2) are gone rather than set
#: to 1, so there is no dormant value to be quietly re-enabled.
def stage3_order(inference_opts=None):
    """Stage-3 order + per-feature args, from the ONE shared builder.

    Order is fixed -- LOAD first, because the damage processor reads the sibling
    load JSON -- while the arguments come from
    `camera_runner.stage3_extras`, the same builder the per-camera plan uses. A
    second copy of these literals here is what let assembly and camera
    processing sample differently.
    """
    from orchestrator.camera_runner import stage3_extras
    e = stage3_extras(inference_opts)
    return (("load", e["load"]), ("door", e["door"]),
            ("damage", e["damage"]), ("ocr", e["ocr"]))


_FEATURE_ORDER = (
    ("load",   dict(every_nth=1)),
    ("door",   {}),
    ("damage", {}),
    # OCR takes NO stride arguments -- it discards `every_nth`/`max_frames`
    # because the Rekognition and EasyOCR readers each pick their own frames
    # (banding -> 3-frame vertical sheet), so an empty extras dict is correct
    # rather than an oversight. `master_runner`'s `_feature_extra` omits it for
    # the same reason.
    #
    # It runs LAST because it is the most expensive per wagon (a Rekognition
    # DetectText call each) and nothing downstream in Stage 3 reads its output;
    # a crash here therefore costs the least.
    ("ocr",    {}),
)


def _processed_video_url(batch_key: str, local_path: str,
                         will_upload: bool) -> str:
    """Deterministic S3 URL for an overlay video, before it is uploaded.

    Uses the SAME key construction `delivery.s3_upload.upload_tree` will apply
    (`<S3_TRAIN_BATCH_PREFIX>/<batch_key>/processed_videos/<file>`), which is how
    batch mode lets Stage 5 embed the link before Stage 6 runs. Identical to
    `master_runner`'s local helper of the same name.

    When nothing will be uploaded the LOCAL PATH is returned rather than a URL
    that would 404 -- a report from a non-delivering run should point at the file
    that actually exists.
    """
    if not local_path:
        return ""
    if not will_upload:
        return local_path
    key = (f"{C.S3_TRAIN_BATCH_PREFIX}/{batch_key}/"
           f"processed_videos/{os.path.basename(local_path)}")
    return f"https://{C.S3_OUTPUT_BUCKET}.s3.{C.S3_REGION}.amazonaws.com/{key}"


def _feature_module(name: str):
    """Import a feature processor lazily, so assembly costs nothing if unused."""
    if name == "load":
        from features.load import processor as m
    elif name == "door":
        from features.door import processor as m
    elif name == "damage":
        from features.damage import processor as m
    elif name == "ocr":
        from features.ocr import processor as m
    else:
        raise ValueError(f"unknown feature {name!r}")
    return m


def assemble(
    *,
    inference_opts: Optional[Dict[str, Any]] = None,
    evidence_root: str,
    output_root: str,
    batch_key: str,
    feat_models_dir: str = "",
    master_camera: str = C.MASTER_CAMERA,
    all_cameras: Tuple[str, ...] = C.ALL_CAMERAS,
    # Delivery is OFF by default so an assembly under validation cannot publish
    # to the live dashboard.  Existing callers keep their exact behaviour.
    deliver: bool = False,
    send_email: bool = False,
    s3_client=None,
    # {camera_id -> the S3 URL the clip was staged FROM}. Assembly only ever
    # sees the local copy, so a caller that downloaded the clips must pass this
    # for `trimmed_video_url` / `raw_video_urls` to be populated. Omitted, those
    # fields stay empty rather than being guessed from a local path.
    source_video_urls: Optional[Dict[str, str]] = None,
    # Which Stage-3 features to run. None = all of them, which is what every
    # existing caller got implicitly. Passing it matters because assembly used
    # to run its whole feature list unconditionally, so `--disable-features ocr`
    # was honoured for the camera-local pass and then silently ignored here --
    # and OCR is a billed Rekognition call per wagon.
    enabled_features: Optional[Sequence[str]] = None,
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

    t0 = time.perf_counter()
    engine_state = gf.assemble_global_train_state_master_fixed(
        master_tracks=tracks[master_camera],
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
    res.state_json_path = os.path.join(gs_dir, "global_train_state.json")
    with open(res.state_json_path, "w", encoding="utf-8") as f:
        f.write(engine_state.to_json())

    state = parse_global_train_state(engine_state.to_dict())

    # ---- Wagon-ACTIVE-REGION audit + gate -------------------------------
    # The region itself is the master's `wagon_window`, already computed and
    # already denying a GW id to anything outside it. This narrates it as
    # BEFORE -> ACTIVE -> AFTER, records top-camera corroboration (read-only,
    # so a top camera can neither move a boundary nor reopen a closed region),
    # lists every out-of-region prediction it ignored, and ASSERTS the gate
    # held. Same function batch mode calls.
    try:
        from core import active_region as AR
        _top_cls = {c: _load_classifications(bundles[c])
                    for c in C.TOP_CAMERAS if c in bundles}
        _fps = {c: float(getattr(tracks.get(c, None), "fps", 0.0) or 0.0)
                for c in C.ALL_CAMERAS}
        res.active_region = AR.resolve(
            state, top_classifications=_top_cls, camera_fps=_fps,
            camera_offsets=state.camera_time_offsets(), verbose=verbose).to_dict()
    except Exception as e:  # noqa: BLE001 -- an audit must not fail a train
        print(f"[ACTIVE-REGION] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        res.active_region = {}

    # ---- Vehicle TYPE resolution (identity is already fixed) -------------
    # Global Wagon IDENTITY is settled above by the RIGHT_UP master
    # reconstruction and is NOT revisited here. This resolves what each
    # already-established object IS, from the SIDE cameras only, and records
    # why. `core.vehicle_type` is the same function batch mode calls, so the two
    # pipelines cannot answer differently for the same evidence.
    #
    # RIGHT_UP is primary and always wins; LEFT_UP corroborates, dissents, or --
    # only when RIGHT_UP has no classification for that wagon -- becomes the
    # source. Top-camera predictions are passed in solely to be RECORDED as
    # ignored, so an audit can show what they said and that it carried no
    # weight. They cannot change a type, and iterating `state.wagons` means the
    # resolver can neither add nor drop a wagon.
    try:
        from core import vehicle_type as VT
        _side_cls = {c: _load_classifications(bundles[c])
                     for c in C.SIDE_CAMERAS if c in bundles}
        _top_cls = {c: _load_classifications(bundles[c])
                    for c in C.TOP_CAMERAS if c in bundles}
        _fps = {c: float(getattr(tracks.get(c, None), "fps", 0.0) or 0.0)
                for c in C.ALL_CAMERAS}
        res.type_resolution = VT.resolve_train(
            state, side_classifications=_side_cls,
            camera_fps=_fps, camera_offsets=state.camera_time_offsets(),
            top_classifications=_top_cls, verbose=verbose)
    except Exception as e:  # noqa: BLE001 -- provenance must not fail a train
        print(f"[TYPE-RESOLUTION] FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        res.type_resolution = {}
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


    # ---- Stage 3: feature inference over the GLOBAL wagons --------------
    # Same processors, same strides, same order as master_runner: LOAD runs to
    # completion first so the damage processor's loaded-wagon floor-damage
    # filter always reads a fully-written wagon_states/load/<gw>.json.
    feature_kwargs = dict(state=state, cache_root=global_cache,
                          feature_models_dir=feat_models_dir,
                          output_dir=states_root,
                          evidence_root=global_evidence, verbose=verbose)
    t0 = time.perf_counter()
    wanted = (None if enabled_features is None else set(enabled_features))
    for name, extra in stage3_order(inference_opts):
        if wanted is not None and name not in wanted:
            print(f"[ASSEMBLY/{name}] DISABLED by feature config -- skipped")
            res.feature_summary[name] = {}
            continue
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

    # ---- Stage 3b: damage boundary ownership ---------------------------
    # HERE and nowhere else. This is the only point where BOTH halves of the
    # required data are live at once:
    #
    #   * `engine_state.global_gaps` -- the global boundaries with each
    #     camera's `support_observations[cam].local_track_id`. The v4
    #     `GlobalTrainState` the feature processors receive keeps only
    #     `global_gap_count`, so a resolver inside features/damage could not
    #     see them.
    #   * `tracks` -- full-fidelity LocalCameraTracks read from
    #     `tracking_full.json`, carrying hit_frames / center_x_trajectory /
    #     bbox_history. `per_camera_tracking.json` drops those (GapEvent.to_dict
    #     is a reporting view), which is why batch mode cannot run this yet.
    #
    # It rewrites only WHICH wagon owns an observation. No detector runs, no
    # threshold moves, and the RIGHT_UP-mastered roster is read, never altered.
    t0 = time.perf_counter()
    try:
        from orchestrator import damage_boundary as DBND
        dmg_by_wagon = DBND.read_damage_by_wagon(states_root, state.wagons)
        if dmg_by_wagon:
            res.damage_boundary = DBND.resolve_train(
                state=state,
                engine_global_gaps=list(getattr(engine_state, "global_gaps",
                                                None) or []),
                tracks_by_camera=tracks,
                damage_by_wagon=dmg_by_wagon,
                verbose=verbose)
            DBND.apply_verdicts(
                result=res.damage_boundary, states_root=states_root,
                evidence_root=global_evidence, wagons=state.wagons,
                verbose=verbose)
            try:
                with open(os.path.join(gs_dir, "damage_boundary.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(res.damage_boundary.to_dict(), f, indent=2)
            except OSError as e:
                print(f"[DAMAGE-BOUNDARY] could not write diagnostics: {e}")
        elif verbose:
            print("[DAMAGE-BOUNDARY] no damage observations to resolve")
    except Exception as e:  # noqa: BLE001 - ownership must not fail a train
        print(f"[DAMAGE-BOUNDARY] resolver FAILED (non-fatal): "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(limit=3)
    res.timings["stage3b_damage_boundary"] = round(time.perf_counter() - t0, 3)

    # ---- Stage 4: fuse (existing builder, unchanged) --------------------
    t0 = time.perf_counter()
    from fusion import wagon_state_builder
    unified = wagon_state_builder.build(state=state,
                                        wagon_states_root=states_root,
                                        verbose=verbose)
    res.timings["fusion_state"] = round(time.perf_counter() - t0, 3)

    # Fusion must produce exactly one UnifiedWagonState per global wagon -- no
    # invented ids, none dropped. Batch mode has enforced this since it was
    # written (master_runner, "fusion changed the wagon set"); the sequential
    # path had NO such check, so a wagon lost here would simply have been absent
    # downstream with nothing said. Same invariant, same failure, both modes.
    _roster_ids = [gw.global_id for gw in state.wagons]
    _fused_ids = set(unified)
    print(f"[REPORT-AUDIT] master timeline wagons={len(_roster_ids)}")
    print(f"[REPORT-AUDIT] fused/materialized wagons={len(_fused_ids)}")
    if _fused_ids != set(_roster_ids):
        _missing = sorted(set(_roster_ids) - _fused_ids)
        _extra = sorted(_fused_ids - set(_roster_ids))
        print(f"[REPORT-AUDIT] SEVERE: fusion changed the wagon set -- "
              f"missing={_missing} unexpected={_extra}", file=sys.stderr)
        raise RuntimeError(
            f"fusion changed the wagon set: missing={_missing[:5]} "
            f"unexpected={_extra[:5]}")

    # ---- Stage 4b: feature overlay rendering ----------------------------
    # The SAME renderer batch mode calls, with the same arguments -- sequential
    # used to skip this entirely, which left `detected_video_url` empty in every
    # published document even after assembly, because there was no overlay video
    # to point at. Visualization only: the renderer runs no detector and cannot
    # touch the roster.
    #
    # `per_camera_tracking.json` is written from the tracks already
    # reconstructed above, in the same shape run_global_count writes it
    # (`{camera: tracks.to_dict()}`), so the renderer and the camera reports
    # read the file they already expect rather than a sequential-only variant.
    processed_root = os.path.join(batch_root, "processed_videos")
    pct_path = os.path.join(gs_dir, "per_camera_tracking.json")
    try:
        os.makedirs(processed_root, exist_ok=True)
        with open(pct_path, "w", encoding="utf-8") as f:
            json.dump({c: t.to_dict(include_classifications=(c == master_camera))
                       for c, t in tracks.items()}, f, indent=2)
        res.per_camera_tracking_path = pct_path
    except Exception as e:  # noqa: BLE001
        print(f"[ASSEMBLY] per_camera_tracking.json failed: {e}")
        pct_path = ""

    t0 = time.perf_counter()
    try:
        from rendering import feature_overlay_renderer
        res.processed_video_paths = feature_overlay_renderer.render_all_cameras(
            state=state,
            unified=unified,
            evidence_root=global_evidence,
            video_paths=video_paths,
            per_camera_tracking_path=pct_path,
            output_dir=processed_root,
            camera_offsets=resolved,
            verbose=verbose,
        )
    except Exception as e:  # noqa: BLE001 - visualization must not fail a train
        print(f"[ASSEMBLY] overlay rendering FAILED: {e}", file=sys.stderr)
        traceback.print_exc(limit=3)
        res.processed_video_paths = {}
    res.timings["stage4b_overlay_render"] = round(time.perf_counter() - t0, 3)

    res.processed_video_urls = {
        c: _processed_video_url(batch_key, p, deliver)
        for c, p in res.processed_video_paths.items()
    }
    # Source clips: the caller knows which S3 object each camera was staged
    # from; assembly only sees the local copy. Absent, the fields stay empty
    # rather than being guessed from a local path.
    res.source_video_urls = {c: u for c, u in (source_video_urls or {}).items()
                             if c in res.sealed_cameras and u}

    t0 = time.perf_counter()
    try:
        from reporting import combined_train_report
        out = combined_train_report.build(
            state=state, unified=unified, output_dir=reports_root,
            batch_key=batch_key,
            source_video_urls=res.source_video_urls,
            processed_video_urls=res.processed_video_urls,
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

    # ---- Stage 6: deliver (S3 archive + dashboard feed + ML API) -----------
    # Sequential mode used to END here.  It uploaded nothing, posted nothing and
    # emailed nothing, so a sequential run produced ZERO dashboard entries --
    # which became a production hole the moment sequential became the default
    # foreground mode.  Delivery is the SAME disk-based implementation the
    # `--deliver-only` republish path uses, so there is one place that talks to
    # S3 and the receivers from a finished batch tree.
    #
    # Opt-in, and off by default: an assembly run that is being validated must
    # not publish to the live dashboard until asked.  `deliver=True` turns it on.
    if deliver:
        t0 = time.perf_counter()
        try:
            from delivery import finalize
            res.delivery = finalize.deliver(
                batch_root=batch_root,
                s3_client=s3_client,
                batch_key=batch_key,
                missing_cameras=sorted(set(res.failed_cameras)
                                       | set(res.missing_cameras)),
                send_email=send_email,
                verbose=verbose,
            )
        except Exception as e:  # noqa: BLE001 - delivery must not fail assembly
            print(f"[ASSEMBLY] delivery failed (non-fatal): {e}")
        res.timings["delivery"] = round(time.perf_counter() - t0, 3)
    elif verbose:
        print("[ASSEMBLY] delivery DISABLED (no S3 upload, no dashboard ingest, "
              "no email) -- pass deliver=True / --historical-deliver to enable")

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
