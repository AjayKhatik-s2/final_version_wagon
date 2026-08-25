
"""WagonEye v4 Master Orchestrator -- train-state-native.

Run modes:
    python -m orchestrator.master_runner --auto         # continuous S3 polling
    python -m orchestrator.master_runner --once         # one batch, exit
    python -m orchestrator.master_runner --batch <key>  # replay a specific batch
    python -m orchestrator.master_runner --local-only --local-inputs DIR

Pipeline (per batch):
    Stage 1  reconstruction.runner.run     -> GlobalTrainState
    Stage 2  materializer.wagon_cache_builder.build  -> wagon_cache/
    Stage 3  features.{door,load,damage,ocr}.processor.run (parallel)
    Stage 4  fusion.wagon_state_builder.build
    Stage 5  reporting.combined_train_report.build
    Stage 6  delivery.{s3_upload, notification}

There is NO legacy v3 fallback.  Stage-1 failure -> batch is marked
failed_no_global_state and abandoned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make sibling packages importable when running this file directly.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Local packages
from core import constants as C
from core.feature_config import (
    FeatureConfig, FEATURE_REGISTRY, FEATURE_KEYS, parse_disable_arg,
)
from core.batch import (
    CameraVideo, TrainBatch,
    build_local_batch, scan_local_video_dir,
)
from core.global_state_loader import (
    GlobalTrainState, assert_roster_unchanged, roster_fingerprint,
)
from core import config as CFG
from core import model_sync
from core.logging_setup import setup_logging
from core.pipeline_source import PipelineSource
from core.stage_timing import StageTimer
from core.unified_wagon_state import UnifiedWagonState, summarize_wagons

from reconstruction import runner as reconstruction_runner
from materializer import wagon_cache_builder
from features.door   import processor as door_proc
from features.load   import processor as load_proc
from features.damage import processor as damage_proc
from features.ocr    import processor as ocr_proc
from fusion import wagon_state_builder
from reporting import combined_train_report, camera_reports
from rendering import feature_overlay_renderer
from delivery import s3_upload, notification
from delivery import dashboard_ingest, ml_api


# Default per-batch paths (relative to a workspace root)
DEFAULT_WORKSPACE_PARENT = os.path.join(_REPO_ROOT, "batch_outputs")
DEFAULT_MODELS_DIR        = os.path.join(_REPO_ROOT, "models")
DEFAULT_RECON_MODELS_DIR  = os.path.join(DEFAULT_MODELS_DIR, "reconstruction")
DEFAULT_FEAT_MODELS_DIR   = os.path.join(DEFAULT_MODELS_DIR, "features")


# -----------------------------------------------------------------------------
# Outcome
# -----------------------------------------------------------------------------

@dataclass
class BatchOutcome:
    batch: TrainBatch
    state: Optional[GlobalTrainState] = None
    unified: Dict[str, UnifiedWagonState] = field(default_factory=dict)
    feature_summary: Dict[str, Dict[str, str]] = field(default_factory=dict)
    cache_summary: Optional[Any] = None
    report_pdf_path: Optional[str] = None
    report_pdf_url: Optional[str] = None
    report_json_path: Optional[str] = None
    report_json_url: Optional[str] = None
    camera_pdf_paths: Dict[str, str] = field(default_factory=dict)
    camera_pdf_urls:  Dict[str, str] = field(default_factory=dict)
    processed_video_paths: Dict[str, str] = field(default_factory=dict)
    processed_video_urls:  Dict[str, str] = field(default_factory=dict)
    # Stage-6 external delivery (dashboard per-camera feed + ML API callback).
    dashboard_result: Dict[str, Any] = field(default_factory=dict)
    ml_api_result: Dict[str, Any] = field(default_factory=dict)
    final_status: str = "unknown"
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    # Per-stage wall clock; also persisted to archive/timings.json.
    timings: Dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Per-batch flow
# -----------------------------------------------------------------------------

def _print_feature_config(cfg: FeatureConfig, *, header: str) -> None:
    print(header)
    for spec in FEATURE_REGISTRY:
        flag = "[ON] " if cfg.is_enabled(spec.key) else "[OFF]"
        print(f"  {flag} {spec.display_name}")


def resolve_feature_config(
    *,
    disable_features: str = "",
    interactive: Optional[bool] = None,
) -> FeatureConfig:
    """Decide which Stage-3 features run this session.

    Precedence:
        1. --disable-features CLI value (explicit, never prompts).
        2. Interactive TTY prompt (only when stdin is a real terminal AND
           the caller allows it).
        3. Default: every feature ON (auto / cron / piped runs).

    Safe for non-interactive/auto/cron: when stdin is not a TTY we NEVER block
    on input -- we return all-ON (or honour the CLI list).
    """
    cli_disabled = parse_disable_arg(disable_features)
    if cli_disabled:
        cfg = FeatureConfig.from_disabled(cli_disabled)
        _print_feature_config(
            cfg, header="Feature Configuration (from --disable-features):")
        return cfg

    try:
        is_tty = sys.stdin.isatty()
    except Exception:
        is_tty = False
    interactive = bool(is_tty if interactive is None else (interactive and is_tty))

    cfg = FeatureConfig.all_on()
    if not interactive:
        return cfg

    _print_feature_config(cfg, header="Current Feature Configuration:")
    try:
        ans = input("Turn OFF any feature? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return cfg
    if ans not in ("y", "yes"):
        return cfg

    print("\nSelect feature(s) to turn OFF (comma-separated numbers, e.g. 2,4):")
    for i, spec in enumerate(FEATURE_REGISTRY, start=1):
        print(f"  {i}. {spec.display_name}")
    try:
        sel = input("Disable: ").strip()
    except (EOFError, KeyboardInterrupt):
        return cfg
    for tok in sel.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            n = int(tok)
        except ValueError:
            continue
        if 1 <= n <= len(FEATURE_REGISTRY):
            cfg.disable(FEATURE_REGISTRY[n - 1].key)

    _print_feature_config(cfg, header="\nFinal Feature Configuration:")
    return cfg


def process_batch(
    *,
    batch: TrainBatch,
    workspace_root: str,
    recon_models_dir: str,
    feat_models_dir: str,
    s3_client=None,
    skip_upload: bool = False,
    skip_email: bool = False,
    verbose: bool = True,
    feature_config: Optional[FeatureConfig] = None,
    # Stage-3 sampling defaults come from `core.constants.STAGE3_*`, the SAME
    # source the sequential plan and the argparse defaults read, so the two modes
    # cannot disagree. They were previously duplicated in three places, which is
    # exactly how they drifted apart.
    door_inference_mode: str = C.STAGE3_DOOR_MODE,
    door_sample_stride: int = C.STAGE3_DOOR_STRIDE,
    damage_inference_mode: str = C.STAGE3_DAMAGE_MODE,
    damage_sample_stride: int = C.STAGE3_DAMAGE_STRIDE,
    load_inference_mode: str = C.STAGE3_LOAD_MODE,
    load_sample_stride: int = C.STAGE3_LOAD_STRIDE,
    load_every_nth: int = C.STAGE3_LOAD_STRIDE,
) -> BatchOutcome:
    if feature_config is None:
        feature_config = FeatureConfig.all_on()
    t_batch = time.time()
    out = BatchOutcome(batch=batch)
    timer = StageTimer()
    batch_root  = os.path.join(workspace_root, batch.batch_key)
    download_root  = os.path.join(batch_root, "downloads")
    stage0_root    = os.path.join(batch_root, "global_state")
    cache_root     = os.path.join(batch_root, "wagon_cache")
    states_root    = os.path.join(batch_root, "wagon_states")
    evidence_root  = os.path.join(batch_root, "evidence")
    processed_root = os.path.join(batch_root, "processed_videos")
    reports_root   = os.path.join(batch_root, "reports")
    archive_root   = os.path.join(batch_root, "archive")
    for d in (download_root, stage0_root, cache_root, states_root,
              evidence_root, processed_root, reports_root, archive_root):
        os.makedirs(d, exist_ok=True)

    print(f"\n{'=' * 78}\n  BATCH {batch.batch_key}\n{'=' * 78}")
    print(f"  cameras present : {batch.present_cameras()}")
    print(f"  cameras missing : {batch.missing_cameras() or '—'}")

    # ---- Download (or pass through local paths) ----
    video_paths: Dict[str, str] = {}
    try:
        for cam in C.ALL_CAMERAS:
            cv = batch.videos.get(cam)
            if cv is None:
                continue
            if cv.bucket == "__local__":
                video_paths[cam] = cv.s3_key
            else:
                local_path = os.path.join(download_root, f"{cam}_{cv.filename}")
                s3_client.download_file(cv.bucket, cv.s3_key, local_path)
                video_paths[cam] = local_path
    except Exception as e:
        out.error = f"download: {e}"
        out.final_status = C.BATCH_FAILED
        out.elapsed_seconds = time.time() - t_batch
        return out

    # ---- Stage 1: reconstruction ----
    print(f"\n--- STAGE 1  Global train reconstruction ---")
    try:
        with timer.stage("stage1_reconstruction"):
            recon = reconstruction_runner.run(
                video_paths=video_paths,
                reconstruction_models_dir=recon_models_dir,
                output_dir=stage0_root,
                repo_root=_REPO_ROOT,
                verbose=verbose,
            )
        out.state = recon.state
    except reconstruction_runner.ReconstructionError as e:
        out.error = f"stage1: {e}"
        out.final_status = C.BATCH_FAILED_NO_GLOBAL
        out.elapsed_seconds = time.time() - t_batch
        print(f"[BATCH] aborted: {e}", file=sys.stderr)
        return out

    # ---- The finalized roster is now immutable for the rest of the batch ----
    # Stage 1 (wagon_count) is the sole counting authority.  Every inspection
    # stage below is checked against this fingerprint, so nothing downstream can
    # append, remove, renumber, reorder or re-time a global wagon.
    roster_guard = recon.roster_fingerprint or roster_fingerprint(recon.state)
    print(f"  roster: {recon.state.total_wagons} wagons "
          f"(GW_1..GW_{recon.state.total_wagons}) "
          f"fingerprint={roster_guard[:16]}  [IMMUTABLE]")

    # ---- Stage 2: materializer ----
    print(f"\n--- STAGE 2  Wagon cache materialization ---")
    try:
        with timer.stage("stage2_materializer"):
            out.cache_summary = wagon_cache_builder.build(
                state=recon.state,
                video_paths=video_paths,
                per_camera_fps=recon.per_camera_fps,
                cache_root=cache_root,
                camera_offsets=recon.camera_offsets,
                verbose=verbose,
            )
        assert_roster_unchanged(recon.state, roster_guard, stage="Stage 2 (materializer)")

    except Exception as e:
        out.error = f"stage2: {e}"
        out.final_status = C.BATCH_FAILED
        out.elapsed_seconds = time.time() - t_batch
        traceback.print_exc()
        return out

    # ---- Stage 3: feature processors ----
    # The damage processor reads the sibling `load` JSON to drop floor_damage
    # tracks on LOADED wagons.  Under full 4-way parallelism that read raced the
    # load writer (handled fail-open, but nondeterministic).  We therefore run
    # the LOAD feature to completion FIRST, then door / ocr / damage in parallel
    # -- so the loaded-wagon floor-damage filter always sees a fully-written
    # wagon_states/load/<gw>.json.  Feature-wise execution + per-model reuse are
    # preserved (each YOLO/easyocr model still loads once and is reused across
    # all wagons within its processor).
    print(f"\n--- STAGE 3  Feature inference ---")
    _print_feature_config(feature_config, header="  feature config:")
    feature_kwargs = dict(
        state=recon.state,
        cache_root=cache_root,
        feature_models_dir=feat_models_dir,
        output_dir=states_root,
        evidence_root=evidence_root,
        verbose=verbose,
    )

    # Per-feature inference mode.  Defaults run EVERY frame.
    #
    # `load_every_nth` is separate and necessary: load samples even in legacy
    # mode, its own `every_nth` defaulting to 2, and that default does take
    # effect.  Passing the mode alone would still have skipped every other frame
    # there, which is the one case where "legacy" does not mean "every frame".
    _feature_extra: Dict[str, Dict[str, Any]] = {
        "door":   dict(inference_mode=door_inference_mode,
                       sample_stride=int(door_sample_stride)),
        "damage": dict(inference_mode=damage_inference_mode,
                       sample_stride=int(damage_sample_stride)),
        "load":   dict(inference_mode=load_inference_mode,
                       sample_stride=int(load_sample_stride),
                       every_nth=max(1, int(load_every_nth))),
    }
    print(f"  inference modes : door={door_inference_mode}/"
          f"stride={door_sample_stride}  "
          f"damage={damage_inference_mode}/stride={damage_sample_stride}  "
          f"load={load_inference_mode}/stride={load_sample_stride}/"
          f"every_nth={load_every_nth}")

    def _run_feature(name, fn):
        with timer.stage(f"stage3_{name}"):
            try:
                return fn(**feature_kwargs, **_feature_extra.get(name, {}))
            except Exception as e:
                print(f"[STAGE3/{name}] CRASHED: {e}", file=sys.stderr)
                traceback.print_exc(limit=3)
                return {}

    def _mark_disabled(name):
        """Write a DISABLED_BY_USER sentinel JSON for every wagon of a
        toggled-off feature so fusion + reports show 'DISABLED BY USER'
        instead of silently treating the field as NO_DATA."""
        from features._common import write_per_wagon_json, empty_payload
        feature_out = os.path.join(states_root, name)
        summary: Dict[str, str] = {}
        for gw in recon.state.wagons:
            payload = empty_payload(
                gw.global_id, name, C.STATUS_DISABLED,
                disabled_by_user=True,
            )
            write_per_wagon_json(feature_out, gw.global_id, payload)
            summary[gw.global_id] = C.STATUS_DISABLED
        print(f"[STAGE3/{name}] DISABLED BY USER -- wrote sentinel for "
              f"{len(summary)} wagons")
        return summary

    with timer.stage("stage3_total"):
        # 1) Load first (deterministic input for damage's load-aware filter).
        if feature_config.is_enabled("load"):
            out.feature_summary["load"] = _run_feature("load", load_proc.run)
        else:
            out.feature_summary["load"] = _mark_disabled("load")

        # 2) Then door / ocr / damage -- only the enabled ones run (in parallel).
        all_parallel = {
            "door":   door_proc.run,
            "ocr":    ocr_proc.run,
            "damage": damage_proc.run,
        }
        parallel_targets = {n: fn for n, fn in all_parallel.items()
                            if feature_config.is_enabled(n)}
        for name in all_parallel:
            if name not in parallel_targets:
                out.feature_summary[name] = _mark_disabled(name)

        if parallel_targets:
            with ThreadPoolExecutor(max_workers=len(parallel_targets)) as ex:
                futs = {ex.submit(_run_feature, name, fn): name
                        for name, fn in parallel_targets.items()}
                for f in as_completed(futs):
                    out.feature_summary[futs[f]] = f.result()

    assert_roster_unchanged(recon.state, roster_guard,
                            stage="Stage 3 (feature inference)")

    # ---- Stage 4: fusion ----
    print(f"\n--- STAGE 4  Wagon state fusion ---")
    try:
        with timer.stage("stage4_fusion"):
            out.unified = wagon_state_builder.build(
                state=recon.state,
                wagon_states_root=states_root,
                verbose=verbose,
            )
        assert_roster_unchanged(recon.state, roster_guard,
                                stage="Stage 4 (fusion)")
        # Fusion must produce exactly one UnifiedWagonState per global wagon --
        # no invented ids, none dropped.
        _roster_ids = [gw.global_id for gw in recon.state.wagons]
        _fused_ids = set(out.unified)
        print(f"[REPORT-AUDIT] master timeline wagons={len(_roster_ids)}")
        print(f"[REPORT-AUDIT] fused/materialized wagons={len(_fused_ids)}")
        if _fused_ids != set(_roster_ids):
            raise RuntimeError(
                f"fusion changed the wagon set: "
                f"missing={sorted(set(_roster_ids) - _fused_ids)[:5]} "
                f"unexpected={sorted(_fused_ids - set(_roster_ids))[:5]}"
            )
    except Exception as e:
        out.error = f"stage4: {e}"
        out.final_status = C.BATCH_FAILED
        out.elapsed_seconds = time.time() - t_batch
        traceback.print_exc()
        return out

    # ---- Stage 4b: feature overlay video rendering (visualization only) ----
    print(f"\n--- STAGE 4b  Feature overlay rendering ---")
    try:
        with timer.stage("stage4b_overlay_render"):
            out.processed_video_paths = feature_overlay_renderer.render_all_cameras(
                state=recon.state,
                unified=out.unified,
                evidence_root=evidence_root,
                video_paths=video_paths,
                per_camera_tracking_path=recon.per_camera_tracking_path,
                output_dir=processed_root,
                enabled_features=set(feature_config.enabled_keys()),
                camera_offsets=recon.camera_offsets,
                verbose=verbose,
            )
    except Exception as e:
        print(f"[STAGE4b] feature overlay rendering FAILED: {e}", file=sys.stderr)
        traceback.print_exc(limit=3)
        out.processed_video_paths = {}

    # Deterministic S3 URLs for processed videos so Stage 5 can embed them
    # before Stage 6 actually uploads (mirrors `s3_upload.upload_tree`'s key
    # construction: <S3_TRAIN_BATCH_PREFIX>/<batch_key>/processed_videos/<file>).
    def _processed_video_url(cam: str, local_path: str) -> str:
        if not local_path or skip_upload:
            return local_path or ""
        key = (f"{C.S3_TRAIN_BATCH_PREFIX}/{batch.batch_key}/"
               f"processed_videos/{os.path.basename(local_path)}")
        return f"https://{C.S3_OUTPUT_BUCKET}.s3.{C.S3_REGION}.amazonaws.com/{key}"

    out.processed_video_urls = {
        cam: _processed_video_url(cam, p)
        for cam, p in out.processed_video_paths.items()
    }

    # Resolve logo asset (copied from old_system into the package)
    _logo_path = os.path.join(_PKG_DIR, "reporting", "assets", "Logo.jpeg")
    _per_camera_tracking_path = recon.per_camera_tracking_path

    # ---- Stage 5a: camera-wise reports (legacy hierarchy; built first so
    # the combined report's DETAILED CAMERA REPORTS table can link them) ----
    print(f"\n--- STAGE 5a  Camera-wise reports ---")
    try:
        with timer.stage("stage5a_camera_reports"):
            _cam_reports = camera_reports.build_all(
                state=recon.state,
                unified=out.unified,
                evidence_root=evidence_root,
                wagon_states_root=states_root,
                cache_root=cache_root,
                per_camera_tracking_path=_per_camera_tracking_path,
                output_dir=reports_root,
                batch_key=batch.batch_key,
                logo_path=_logo_path,
                verbose=verbose,
            )
        out.camera_pdf_paths = {cam: v for cam, v in _cam_reports.items() if v}
    except Exception as e:
        print(f"[STAGE5a] camera reports FAILED: {e}", file=sys.stderr)
        traceback.print_exc(limit=3)
        out.camera_pdf_paths = {}

    # Relative basenames are linkable both locally (sibling file:// in the
    # reports/ dir) and on S3 (sibling object under reports/<batch>/).
    camera_pdf_urls: Dict[str, str] = {
        cam: os.path.basename(p) for cam, p in out.camera_pdf_paths.items()
    }

    # ---- Stage 5b: combined report (aggregates the 4 camera reports) ----
    print(f"\n--- STAGE 5b  Combined report ---")
    try:
        with timer.stage("stage5b_combined_report"):
            result = combined_train_report.build(
                state=recon.state,
                unified=out.unified,
                output_dir=reports_root,
                batch_key=batch.batch_key,
                source_video_urls={
                    cam: (batch.videos[cam].s3_url
                          if cam in batch.videos
                          and batch.videos[cam].bucket != "__local__"
                          else "")
                    for cam in C.ALL_CAMERAS if cam in batch.videos
                },
                processed_video_urls=out.processed_video_urls,
                evidence_root=evidence_root,
                wagon_states_root=states_root,
                cache_root=cache_root,
                missing_cameras=list(batch.missing_cameras()),
                camera_pdf_urls=camera_pdf_urls,
                logo_path=_logo_path,
                verbose=verbose,
            )
        out.report_json_path = result.get("json_path")
        out.report_pdf_path  = result.get("pdf_path")
        assert_roster_unchanged(recon.state, roster_guard,
                                stage="Stage 5 (reporting)")
    except Exception as e:
        out.error = f"stage5: {e}"
        out.final_status = C.BATCH_REPORT_FAILED
        out.elapsed_seconds = time.time() - t_batch
        traceback.print_exc()
        return out

    # ---- decide completion class ----
    partial = any(
        v in (C.STATUS_NO_FRAMES, C.STATUS_FAILED, C.NO_DATA)
        for d in out.feature_summary.values() for v in d.values()
    )
    if out.report_pdf_path is None:
        out.final_status = C.BATCH_REPORT_FAILED
    else:
        out.final_status = (
            C.BATCH_COMPLETED_PARTIAL if partial else C.BATCH_COMPLETED
        )

    # ---- Persist the stage timings (measurement only; no behaviour depends
    # on this file).  Written before delivery so a skip-upload benchmarking
    # run still produces it. ----
    def _feature_frame_counts() -> Dict[str, int]:
        """Frames inspected (== YOLO calls) per feature, from the per-wagon JSON.

        `frame_count` is what each processor already records, so this needs no
        new instrumentation inside the feature code.
        """
        out: Dict[str, int] = {}
        for key in FEATURE_KEYS:
            d = os.path.join(states_root, key)
            if not os.path.isdir(d):
                continue
            total = 0
            for fn in os.listdir(d):
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                        total += int(json.load(f).get("frame_count") or 0)
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            out[key] = total
        return out

    def _emit_timings() -> None:
        feature_spans = [f"stage3_{k}" for k in FEATURE_KEYS]
        frame_counts = _feature_frame_counts()
        extra = {
            "batch_key": batch.batch_key,
            "total_wagons": recon.state.total_wagons,
            "enabled_features": feature_config.enabled_keys(),
            "inference_modes": {
                "door":   {"mode": door_inference_mode,
                           "sample_stride": int(door_sample_stride)},
                "damage": {"mode": damage_inference_mode,
                           "sample_stride": int(damage_sample_stride)},
                "load":   {"mode": load_inference_mode,
                           "sample_stride": int(load_sample_stride)},
            },
            # frames inspected == YOLO calls for these detectors
            "yolo_calls": frame_counts,
            # >1.0 means the Stage-3 features genuinely overlapped;
            # ~1.0 means they serialized on the CPU.
            "stage3_overlap_factor": timer.overlap_factor(
                "stage3_total", feature_spans),
        }
        out.timings = dict(timer.to_dict()["wall_clock_seconds"])
        try:
            timer.write(os.path.join(archive_root, "timings.json"), extra=extra)
        except OSError as e:
            print(f"[TIMING] could not write timings.json: {e}", file=sys.stderr)
        print(timer.render_table(title=f"STAGE TIMINGS  {batch.batch_key}"))
        of = extra["stage3_overlap_factor"]
        if of is not None:
            verdict = ("features overlapped" if of > 1.15
                       else "features effectively SERIALIZED on CPU")
            print(f"  stage3 overlap factor: {of:.2f}x  ({verdict})")
        if frame_counts:
            print("  frames inspected (== YOLO calls):")
            for k in FEATURE_KEYS:
                if k not in frame_counts:
                    continue
                mode = ""
                if k == "door":
                    mode = f"  [{door_inference_mode}/stride={door_sample_stride}]"
                elif k == "damage":
                    mode = f"  [{damage_inference_mode}/stride={damage_sample_stride}]"
                elif k == "load":
                    mode = f"  [{load_inference_mode}/stride={load_sample_stride}]"
                print(f"    {k:<8} {frame_counts[k]:>7}{mode}")

    # ---- Stage 6: delivery ----
    if skip_upload:
        _emit_timings()
        out.elapsed_seconds = time.time() - t_batch
        return out

    print(f"\n--- STAGE 6  Delivery ---")
    _emit_timings()
    if out.report_pdf_path:
        out.report_pdf_url = s3_upload.upload_pdf(
            s3_client, out.report_pdf_path, batch.batch_key,
        )
    if out.report_json_path:
        out.report_json_url = s3_upload.upload_json(
            s3_client, out.report_json_path, batch.batch_key,
        )
    # Camera-wise PDFs go through the same microservice flow.
    for cam, path in out.camera_pdf_paths.items():
        url = s3_upload.upload_pdf(s3_client, path, batch.batch_key)
        if url:
            out.camera_pdf_urls[cam] = url

    # Archive everything per-feature + the cache (skip huge JPEGs in cache to S3
    # by default; keep wagon_states + global_state + reports which are small)
    n_state  = s3_upload.upload_tree(s3_client, stage0_root, batch.batch_key,
                                     sub_prefix="global_state",
                                     skip_extensions={".jpg", ".jpeg"})
    n_states = s3_upload.upload_tree(s3_client, states_root, batch.batch_key,
                                     sub_prefix="wagon_states")
    n_reports = s3_upload.upload_tree(s3_client, reports_root, batch.batch_key,
                                      sub_prefix="reports")
    n_evidence = s3_upload.upload_tree(s3_client, evidence_root, batch.batch_key,
                                       sub_prefix="evidence")
    n_videos = s3_upload.upload_tree(s3_client, processed_root, batch.batch_key,
                                     sub_prefix="processed_videos")
    print(f"[STAGE6] archived: global_state={n_state} files, "
          f"wagon_states={n_states} files, reports={n_reports} files, "
          f"evidence={n_evidence} files, processed_videos={n_videos} files")

    # ---- Stage 6b: the V4 dashboard feed + ML API callback ----
    # Four exact-V4 `{camera_id, version, inspection_data}` documents, uploaded
    # and POSTed to the V4 ingest receivers.  Runs AFTER the archive upload so
    # the evidence URLs each document references already resolve in S3.
    # Both calls are failure-isolated: a receiver outage cannot fail a batch
    # whose reports are already built and uploaded.
    print(f"\n--- STAGE 6b  Dashboard feed (V4 inspection JSON) ---")
    # Seed the finalization marker with the URLs Stage 6 just produced.
    #
    # `dashboard_ingest` reads each document's `pdf_report_url` out of the
    # marker's `upload_urls` (that is where the per-camera PDF links live in the
    # V4 contract).  Nothing else in this package writes that marker, so without
    # this every document reached the dashboard with `pdf_report_url` EMPTY --
    # the report was ingested but had no link back to the PDF.
    #
    # An existing marker is never overwritten: if some other step already
    # recorded what it delivered, that record wins.
    try:
        from delivery import finalization as _FIN
        _urls = {f"camera_{cam}": u
                 for cam, u in (out.camera_pdf_urls or {}).items() if u}
        if out.report_pdf_url:
            _urls["pdf"] = out.report_pdf_url
        if out.report_json_url:
            _urls["json"] = out.report_json_url
        if _urls and _FIN.load(batch_root) is None:
            _FIN.write(batch_root, {
                "batch_key": batch.batch_key,
                "terminal_status": out.final_status,
                "upload_urls": _urls,
                "uploaded": True,
            })
    except Exception as e:  # noqa: BLE001 - a marker failure must not fail delivery
        print(f"[STAGE6b] could not seed the finalization marker: {e}",
              file=sys.stderr)

    out.dashboard_result = dashboard_ingest.run(
        batch_root=batch_root, s3_client=s3_client, skip_upload=skip_upload,
    )
    if out.dashboard_result.get("enabled"):
        for cam, info in (out.dashboard_result.get("cameras") or {}).items():
            print(f"  [dashboard/{cam}] {info.get('status')}"
                  + (f"  run_id={info['run_id']}" if info.get("run_id") else ""))
    else:
        print("  dashboard ingest disabled "
              "(WAGONEYE_DASHBOARD_INGEST_ENABLED=false)")

    out.ml_api_result = ml_api.submit_batch(
        batch_key=batch.batch_key,
        cameras=list(batch.present_cameras()),
        source_video_urls={
            cam: (batch.videos[cam].s3_url
                  if cam in batch.videos and batch.videos[cam].bucket != "__local__"
                  else "")
            for cam in C.ALL_CAMERAS if cam in batch.videos
        },
        processed_video_urls=out.processed_video_urls,
        camera_pdf_urls=out.camera_pdf_urls,
        combined_pdf_url=out.report_pdf_url,
    )

    if not skip_email:
        summary = summarize_wagons(list(out.unified.values()))
        notification.send_email(
            batch_key=batch.batch_key,
            report_pdf_url=out.report_pdf_url,
            report_json_url=out.report_json_url,
            summary=summary,
            cameras_present=batch.present_cameras(),
            cameras_missing=batch.missing_cameras(),
            final_status=out.final_status,
        )

    out.elapsed_seconds = time.time() - t_batch
    print(f"\n[BATCH {batch.batch_key}] {out.final_status}  "
          f"({out.elapsed_seconds:.1f}s)")
    return out


# -----------------------------------------------------------------------------
# Continuous mode (S3 polling).  This is a minimal placeholder: the
# legacy `train_batch_manager.py` polling code can be plugged in here.
# -----------------------------------------------------------------------------

def run_auto(*args, **kwargs):
    """Continuous S3 polling loop -- the production auto pipeline.

    Two halves, decoupled through S3:

      * PRODUCER (only when the pipeline source is `raw`): an ExtractionManager
        thread discovers raw CCTV, detects a completed train pass, trims it and
        uploads the clip to the trimmed bucket.
      * CONSUMER (always): `orchestrator.train_batch_manager` discovers trimmed
        clips, clusters the four cameras into one TrainBatch, and hands each
        runnable batch to `process_batch` -- which is UNCHANGED.

    The producer writes to exactly the location the consumer reads, so one
    process can own both without the two being coupled in code.  With
    `--source trimmed` (the default) no producer is started and this is the pure
    consumer it always was.
    """
    workspace_root = kwargs.get("workspace") or CFG.WORKSPACE_ROOT
    recon_models_dir = kwargs.get("recon_models_dir") or DEFAULT_RECON_MODELS_DIR
    feat_models_dir  = kwargs.get("feat_models_dir")  or DEFAULT_FEAT_MODELS_DIR
    poll_interval    = kwargs.get("poll_interval", 60)
    partial_wait     = kwargs.get("partial_wait_minutes", 30.0)
    run_once         = kwargs.get("run_once", False)
    force_key        = kwargs.get("force_batch_key")
    skip_upload      = kwargs.get("skip_upload", False)
    skip_email       = kwargs.get("skip_email", False)
    feature_config   = kwargs.get("feature_config") or FeatureConfig.all_on()
    source           = kwargs.get("source") or CFG.PIPELINE_SOURCE
    skip_model_sync  = kwargs.get("skip_model_sync", False)
    mode             = "once" if run_once else "auto"

    # ---- fail fast on a misconfiguration instead of polling forever ----
    errors = CFG.validate_config(mode=mode, skip_upload=skip_upload,
                                 skip_email=skip_email, source=source)
    if errors:
        print("[ORCH] refusing to start -- configuration errors:", file=sys.stderr)
        for e in errors:
            print(f"  * {e}", file=sys.stderr)
        return 2
    print(CFG.startup_summary(mode=mode, source=source))

    # ---- every model this run needs must exist before the first batch ----
    if not skip_model_sync:
        # `include_extraction` follows the source: the extraction classifiers
        # are only required when THIS process produces its own trimmed clips.
        report = model_sync.ensure_models_or_report(
            enabled_features=feature_config.enabled_keys(),
            include_extraction=source.requires_extraction,
        )
        if not report.ok:
            print("[ORCH] refusing to start -- required model(s) unavailable",
                  file=sys.stderr)
            return 2

    from orchestrator import train_batch_manager as TBM

    try:
        import boto3
    except ImportError:
        print("[ORCH] boto3 is required for --auto (pip install boto3)",
              file=sys.stderr)
        return 2

    s3 = boto3.client("s3", region_name=C.S3_REGION)
    state_loc = f"{C.S3_OUTPUT_BUCKET}/{C.S3_STATE_KEY}"

    os.makedirs(workspace_root, exist_ok=True)

    # ---- PRODUCER: only when this deployment has raw CCTV, not clips ----
    extractor = None
    if source.requires_extraction:
        from orchestrator.extraction_manager import ExtractionManager
        extractor = ExtractionManager(poll_interval=CFG.EXTRACTION_POLL_INTERVAL)
        if run_once:
            counts = extractor.run_once()
            print(f"[ORCH] extraction sweep: {counts}")
        else:
            extractor.start()

    processed = TBM.load_batch_state(s3, state_loc)
    start = CFG.discovery_cutoff_utc()
    print(f"[ORCH] workspace: {workspace_root}")
    print(f"[ORCH] source   : {source.value}"
          + ("  (extraction owned by this process)"
             if source.requires_extraction else "  (pure consumer)"))
    print(f"[ORCH] discovery cutoff: {start.isoformat()}")
    print(f"[ORCH] processed batches so far: {len(processed)}")

    try:
        while True:
            try:
                batches = TBM.poll_for_batches(
                    s3_client=s3, processed_batches=processed,
                    start_time=start,
                    tolerance_sec=TBM.DEFAULT_BATCH_TOLERANCE_SEC,
                    # An explicit `--batch <key>` replay is allowed to reach
                    # past the discovery window; continuous polling is not.
                    apply_cutoff=not bool(force_key),
                )
                if force_key:
                    batch = next((b for b in batches
                                  if b.batch_key == force_key), None)
                    if batch is None and run_once:
                        print(f"[ORCH] batch {force_key} not found")
                        return 0
                else:
                    batch = TBM.select_runnable_batch(
                        batches, partial_wait_minutes=partial_wait)
                if batch is None:
                    if run_once:
                        return 0
                    print(f"[ORCH] no runnable batch; sleeping {poll_interval}s")
                    time.sleep(poll_interval)
                    continue

                outcome = process_batch(
                    batch=batch, workspace_root=workspace_root,
                    recon_models_dir=recon_models_dir,
                    feat_models_dir=feat_models_dir,
                    s3_client=s3, skip_upload=skip_upload,
                    skip_email=skip_email,
                    feature_config=feature_config,
                    **(kwargs.get("inference_opts") or {}),
                )
                processed[batch.batch_key] = outcome.final_status
                TBM.save_batch_state(s3, state_loc, processed)
                if run_once:
                    return 0
            except KeyboardInterrupt:
                raise
            except Exception as e:
                traceback.print_exc()
                print(f"[ORCH] unhandled error: {e}", file=sys.stderr)
                if run_once:
                    return 3
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[ORCH] interrupted")
        return 0
    finally:
        if extractor is not None and extractor.is_running():
            extractor.stop()


# -----------------------------------------------------------------------------
# Local mode
# -----------------------------------------------------------------------------

def run_local(
    *,
    local_inputs: str,
    batch_key: Optional[str],
    workspace: Optional[str],
    recon_models_dir: str,
    feat_models_dir: str,
    feature_config: Optional[FeatureConfig] = None,
    inference_opts: Optional[Dict[str, Any]] = None,
) -> int:
    if not os.path.isdir(local_inputs):
        print(f"ERROR: {local_inputs} does not exist", file=sys.stderr)
        return 2
    video_paths = scan_local_video_dir(local_inputs)
    missing = [c for c in C.ALL_CAMERAS if c not in video_paths]
    if missing:
        print(f"ERROR: missing videos for {missing} in {local_inputs}.",
              file=sys.stderr)
        return 2
    batch = build_local_batch(video_paths, batch_key=batch_key)
    workspace = workspace or DEFAULT_WORKSPACE_PARENT

    # No s3 client needed; skip_upload=True
    class _NoopS3:
        def download_file(self, *a, **kw):
            raise RuntimeError("s3 download invoked in --local-only mode")
        def upload_file(self, *a, **kw):
            return None

    outcome = process_batch(
        batch=batch, workspace_root=workspace,
        recon_models_dir=recon_models_dir,
        feat_models_dir=feat_models_dir,
        s3_client=_NoopS3(),
        skip_upload=True, skip_email=True,
        feature_config=feature_config or FeatureConfig.all_on(),
        **(inference_opts or {}),
    )
    if outcome.report_pdf_path:
        print(f"[LOCAL] PDF : {outcome.report_pdf_path}")
    if outcome.report_json_path:
        print(f"[LOCAL] JSON: {outcome.report_json_path}")
    for cam, path in outcome.camera_pdf_paths.items():
        print(f"[LOCAL] {cam:<13} PDF: {path}")
    for cam, path in outcome.processed_video_paths.items():
        print(f"[LOCAL] VIDEO  {cam}: {path}")
    return 0 if outcome.final_status in (C.BATCH_COMPLETED,
                                          C.BATCH_COMPLETED_PARTIAL) else 3


# -----------------------------------------------------------------------------
# SEQUENTIAL MODE (opt-in) -- one camera at a time, then global assembly
# -----------------------------------------------------------------------------

def run_sessions(
    *,
    sessions: List[Tuple[str, Dict[str, str]]],
    workspace: Optional[str] = None,
    recon_models_dir: str = DEFAULT_RECON_MODELS_DIR,
    feat_models_dir: str = DEFAULT_FEAT_MODELS_DIR,
    feature_config: Optional[FeatureConfig] = None,
    expected_cameras: Optional[Sequence[str]] = None,
    deliver: bool = False,
    deliver_per_camera: bool = True,
    send_email: bool = False,
    scheduler=None,
    verbose: bool = True,
) -> int:
    """Drive one or more trains through sequential mode under the scheduler.

    `sessions` is `[(train_id, {camera_id: video_path}), ...]` -- the caller has
    already decided which clip belongs to which train, using the pipeline's
    existing identification logic.  Train identity is never re-derived here.

    The scheduler owns ORDER ONLY.  Each camera is still run by the existing
    `camera_runner.run_camera` (which renders that camera's report and publishes
    it the moment it seals) and each train is still assembled by the existing
    `global_assembler.assemble`.  No stage is reimplemented.

    A train is assembled the instant its last camera finishes, before any newer
    train receives further work -- so an older train's combined report is never
    held behind a newer train's inference.

    Pass `scheduler=` to supply a pre-seeded `TrainScheduler` (used by tests and
    by callers that want to submit feeds as they arrive rather than up front).
    """
    from orchestrator import camera_runner, global_assembler
    from orchestrator.train_scheduler import TrainScheduler

    workspace = workspace or DEFAULT_WORKSPACE_PARENT
    os.makedirs(workspace, exist_ok=True)
    cams = tuple(expected_cameras or C.ALL_CAMERAS)
    enabled = (feature_config or FeatureConfig.all_on()).enabled_keys()

    sched = scheduler or TrainScheduler(
        expected_cameras=cams,
        state_path=os.path.join(workspace, "scheduler_state.json"),
        verbose=verbose)

    for train_id, videos in sessions:
        for cam in cams:
            vp = videos.get(cam)
            if vp:
                sched.submit_camera_video(
                    camera_id=cam, video_path=vp, train_id=train_id,
                    train_timestamp=train_id)

    def _evidence_root(train_id: str) -> str:
        p = os.path.join(workspace, train_id, "camera_evidence")
        os.makedirs(p, exist_ok=True)
        return p

    results: Dict[str, List[Any]] = {}
    assemblies: Dict[str, Any] = {}
    t0 = time.time()

    print("=" * 78)
    print(f"  SCHEDULED SEQUENTIAL MODE  ({len(sched.sessions)} session(s))")
    print("=" * 78)

    def _assemble(train_id: str) -> None:
        print(f"--- GLOBAL ASSEMBLY {train_id} ---")
        try:
            asm = global_assembler.assemble(
                evidence_root=_evidence_root(train_id),
                output_root=os.path.join(workspace, train_id),
                batch_key=train_id, feat_models_dir=feat_models_dir,
                deliver=deliver, send_email=send_email,
                enabled_features=enabled, verbose=verbose)
            assemblies[train_id] = asm
        except Exception as e:  # noqa: BLE001 - one train must not stop the rest
            print(f"[SCHED] assembly of {train_id} raised "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
        sched.mark_assembled(train_id)

    while True:
        job = sched.next_job()
        if job is None:
            pending = sched.sessions_ready_to_assemble()
            if not pending:
                break
            _assemble(pending[0].train_id)     # oldest first
            continue

        print(f"--- {job.train_id} / {job.camera_id} ---")
        res = camera_runner.run_camera(
            camera_id=job.camera_id, video_path=job.video_path,
            recon_models_dir=recon_models_dir,
            feat_models_dir=feat_models_dir,
            evidence_root=_evidence_root(job.train_id),
            enabled_features=enabled,
            # The camera's own report is rendered and published inside
            # run_camera, immediately after it SEALS -- not after the train.
            deliver_per_camera=deliver_per_camera,
            train_id=job.train_id,
            verbose=verbose)
        results.setdefault(job.train_id, []).append(res)

        if res.sealed:
            sess = sched.mark_camera_completed(job.train_id, job.camera_id)
        else:
            sess = sched.mark_camera_failed(job.train_id, job.camera_id,
                                            res.failure_reason)
        # Assemble the moment this train is done, before any newer train gets
        # another job.
        if sess.is_complete() and not sess.assembled:
            _assemble(sess.train_id)

    elapsed = time.time() - t0
    print("=" * 78)
    print("  SCHEDULED SEQUENTIAL SUMMARY")
    print("=" * 78)
    print(sched.render_status())
    for train_id, rs in results.items():
        for r in rs:
            print(f"  {train_id} {r.camera_id:<13} {r.state:<8} "
                  f"segments={r.local_segments:<4} engine_frames="
                  f"{r.engine_frames:<3} {r.timings.get('total', 0):.1f}s")
        asm = assemblies.get(train_id)
        if asm is not None:
            print(f"  {train_id} global wagons : {asm.total_wagons}")
            print(f"  {train_id} combined pdf  : "
                  f"{asm.report_pdf_path or '(none)'}")
    print(f"  TOTAL         : {elapsed:.1f}s")

    ok = bool(assemblies) and all(
        a.ready and a.total_wagons > 0 for a in assemblies.values())
    return 0 if ok else 3


def run_sequential(
    *,
    local_inputs: str,
    workspace: Optional[str] = None,
    recon_models_dir: str = DEFAULT_RECON_MODELS_DIR,
    feat_models_dir: str = DEFAULT_FEAT_MODELS_DIR,
    batch_key: Optional[str] = None,
    feature_config: Optional[FeatureConfig] = None,
    arrival_order: Optional[List[str]] = None,
    deliver: bool = False,
    deliver_per_camera: bool = False,
    send_email: bool = False,
    verbose: bool = True,
) -> int:
    """Process each camera independently, then assemble.

    Every camera is fully persisted and SEALED before the next starts, so the
    sequence is resumable and no camera waits on another. Global assembly runs
    only afterwards and re-runs no detector.
    """
    from datetime import datetime
    from orchestrator import camera_runner, global_assembler
    from orchestrator.train_scheduler import TrainScheduler

    if not os.path.isdir(local_inputs):
        print(f"ERROR: {local_inputs} does not exist", file=sys.stderr)
        return 2
    videos = scan_local_video_dir(local_inputs)
    order = arrival_order or list(C.ALL_CAMERAS)
    key = batch_key or ("sequential_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    workspace = workspace or DEFAULT_WORKSPACE_PARENT
    evidence_root = os.path.join(workspace, key, "camera_evidence")
    os.makedirs(evidence_root, exist_ok=True)
    enabled = (feature_config or FeatureConfig.all_on()).enabled_keys()

    print("=" * 78)
    print(f"  SEQUENTIAL MODE  {key}")
    print("=" * 78)
    print(f"  arrival order : {order}")
    print(f"  output        : {os.path.join(workspace, key)}")

    t0 = time.time()
    results = []

    # The scheduler is the single source of truth for camera-job ORDER, even
    # for this single-train path: one session, its feeds submitted in arrival
    # order, and `next_job()` asked what to run next. Behaviour for one train is
    # identical to the previous plain loop -- with all feeds present the oldest
    # (only) session is immediately processable and its cameras are handed out
    # in `expected_cameras` order -- but the ordering decision now lives in one
    # place instead of two, so multi-train scheduling cannot drift from it.
    sched = TrainScheduler(
        expected_cameras=tuple(order),
        state_path=os.path.join(workspace, key, "scheduler_state.json"),
        verbose=verbose)
    for cam in order:
        vp = videos.get(cam)
        if not vp:
            print(f"--- ARRIVAL {cam}: no video, skipping ---")
            continue
        sched.submit_camera_video(camera_id=cam, video_path=vp,
                                  train_id=key, train_timestamp=key)

    while True:
        job = sched.next_job()
        if job is None:
            break
        print(f"--- ARRIVAL {job.camera_id} ---")
        r = camera_runner.run_camera(
            camera_id=job.camera_id, video_path=job.video_path,
            recon_models_dir=recon_models_dir,
            feat_models_dir=feat_models_dir,
            evidence_root=evidence_root,
            enabled_features=enabled,
            deliver_per_camera=deliver_per_camera,
            train_id=key,
            verbose=verbose)
        results.append(r)
        if r.sealed:
            sched.mark_camera_completed(key, job.camera_id)
        else:
            sched.mark_camera_failed(key, job.camera_id, r.failure_reason)

    print("--- GLOBAL ASSEMBLY ---")
    asm = global_assembler.assemble(
        evidence_root=evidence_root, output_root=workspace,
        batch_key=key, feat_models_dir=feat_models_dir,
        deliver=deliver, send_email=send_email,
        enabled_features=enabled, verbose=verbose)

    elapsed = time.time() - t0
    print("=" * 78)
    print(f"  SEQUENTIAL SUMMARY  {key}")
    print("=" * 78)
    if deliver_per_camera:
        print("  per-camera dashboard posts (camera-local numbering, "
              "superseded by assembly):")
        for r in results:
            if getattr(r, "per_camera_ingest", None) is not None:
                print(f"    {r.per_camera_ingest.render()}")
    for r in results:
        print(f"  {r.camera_id:<13} {r.state:<8} segments={r.local_segments:<4} "
              f"gaps={r.accepted_gaps:<4} calls={r.feature_calls} "
              f"{r.timings.get('total', 0):.1f}s")
    print(f"  global wagons : {asm.total_wagons}")
    for cam, sm in asm.mapping_by_camera.items():
        print(f"  mapping {cam:<13} {sm['by_kind']}")
    print(f"  combined pdf  : {asm.report_pdf_path or '(none)'}")
    if getattr(asm, "delivery", None) is not None:
        print(asm.delivery.render())
    print(f"  TOTAL         : {elapsed:.1f}s")
    return 0 if asm.ready and asm.total_wagons > 0 else 3


# -----------------------------------------------------------------------------
# HISTORICAL MODE (opt-in) -- input selection only; reuses process_batch
# -----------------------------------------------------------------------------

def run_historical(args, *, feature_config=None, inference_opts=None,
                   mode_explicit: bool = False) -> int:
    """CLI adapter for `--historical`.

    Resolves the requested window, opens an S3 client, and delegates to
    `historical_runner.run`, which selects the matching already-trimmed clips and
    feeds each discovered train to the SAME `process_batch` the live path uses.

    Nothing in `run_auto` is entered: no polling loop, no live discovery cutoff,
    and `processed_batches.json` is neither read nor written -- so a historical
    run can neither be blocked by a live batch nor mark one terminal.
    """
    from orchestrator import historical_runner as HR
    from orchestrator import train_batch_manager as TBM

    # Only an explicit `--mode sequential` switches architecture (see the
    # dispatch comment): a bare `--historical` stays on the validated batch path
    # even though --mode now defaults to sequential for foreground runs.
    hist_mode = ("sequential" if (mode_explicit and args.mode == "sequential")
                 else "batch")
    if hist_mode == "sequential":
        print("[HISTORICAL] architecture: SEQUENTIAL (per-camera -> assembly)"
              + ("  + per-camera dashboard posts"
                 if args.deliver_per_camera else ""))

    # The requested window is parsed FIRST: it is pure and instant, so a typo in
    # --date / --start-time is reported immediately rather than after a
    # multi-second S3 model check.
    try:
        window = HR.resolve_window(
            date=args.date, start_time=args.start_time, end_time=args.end_time,
            timezone_name=args.timezone, start_iso=args.start, end_iso=args.end,
        )
    except ValueError as e:
        print(f"[HISTORICAL] {e}", file=sys.stderr)
        return 2

    # Same fail-fast discipline the live path gets, with the historical branch
    # (discovery config required, email endpoint not -- delivery is opt-in).
    errors = CFG.validate_config(mode="historical",
                                 skip_upload=not args.historical_deliver,
                                 skip_email=not args.historical_deliver,
                                 source=PipelineSource.TRIMMED)
    if errors:
        print("[HISTORICAL] refusing to start -- configuration errors:",
              file=sys.stderr)
        for e in errors:
            print(f"  * {e}", file=sys.stderr)
        return 2

    # A --dry-run only lists S3 and prints the manifest, so it must not require
    # weights.  A real run does: without them every batch would fail inside
    # Stage 1 with a per-batch error instead of one clear message up front.
    if not args.dry_run and not args.skip_model_sync:
        report = model_sync.ensure_models_or_report(
            enabled_features=(feature_config or FeatureConfig.all_on()).enabled_keys(),
            # Historical mode is ALWAYS a pure consumer of already-trimmed clips,
            # so the extraction classifiers are irrelevant to it.
            include_extraction=False,
        )
        if not report.ok:
            print("[HISTORICAL] refusing to start -- required model(s) "
                  "unavailable", file=sys.stderr)
            return 2

    try:
        import boto3
        s3 = boto3.client("s3", region_name=C.S3_REGION)
    except Exception as e:  # noqa: BLE001
        print(f"[HISTORICAL] could not create an S3 client: {e}", file=sys.stderr)
        return 2

    return HR.run(
        s3_client=s3,
        window=window,
        workspace_root=args.workspace or DEFAULT_WORKSPACE_PARENT,
        recon_models_dir=args.recon_models_dir or DEFAULT_RECON_MODELS_DIR,
        feat_models_dir=args.feat_models_dir or DEFAULT_FEAT_MODELS_DIR,
        feature_config=feature_config,
        pad_minutes=(HR.DEFAULT_PAD_MINUTES if args.pad_minutes is None
                     else args.pad_minutes),
        tolerance_sec=(TBM.DEFAULT_BATCH_TOLERANCE_SEC
                       if args.tolerance_sec is None else args.tolerance_sec),
        dry_run=args.dry_run,
        keep_inputs=args.keep_inputs,
        deliver=args.historical_deliver,
        # --historical-deliver turns on upload + dashboard ingest; email stays
        # separately suppressible with the existing --skip-email, so a bulk
        # re-run can reach the dashboard without mailing the operators N times.
        send_email=not args.skip_email,
        manifest_out=args.manifest_out,
        inference_opts=inference_opts,
        mode=hist_mode,
        deliver_per_camera=args.deliver_per_camera,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orchestrator.master_runner",
        description="WagonEye v4 train-state-native orchestrator.",
    )
    p.add_argument("--auto",  action="store_true", help="continuous S3 polling")
    p.add_argument("--once",  action="store_true", help="one batch then exit")
    p.add_argument("--batch", default=None,
                   help="force a specific batch_key (replay / debug)")
    p.add_argument("--local-only",   action="store_true",
                   help="skip S3 entirely; videos come from --local-inputs")
    p.add_argument("--local-inputs", default="local_inputs",
                   help="folder to scan in --local-only mode")
    p.add_argument("--workspace",    default=None,
                   help="workspace root (default: ./batch_outputs)")
    p.add_argument("--recon-models-dir", default=DEFAULT_RECON_MODELS_DIR)
    p.add_argument("--feat-models-dir",  default=DEFAULT_FEAT_MODELS_DIR)
    p.add_argument("--poll-interval",    type=int,   default=60)
    p.add_argument("--partial-wait",     type=float, default=30.0)
    p.add_argument("--skip-upload",      action="store_true")
    p.add_argument("--skip-email",       action="store_true")
    p.add_argument("--disable-features", default="",
                   help="comma-separated feature keys to turn OFF "
                        "(door,ocr,load,damage); skips the interactive prompt")
    p.add_argument("--no-interactive",   action="store_true",
                   help="never prompt for feature config (force all-ON unless "
                        "--disable-features given)")

    # ---- Stage-3 inference mode (Door / Damage only) ----------------------
    # 'sampled' inspects every Nth frame and resolves state with
    # EvidenceAggregator; 'legacy' inspects every frame and uses the original
    # Kalman/Hungarian trackers.  Legacy is fully retained -- pass
    # `--door-inference-mode legacy --damage-inference-mode legacy` to restore
    # the pre-optimization pipeline exactly.  Load and OCR are unaffected.
    p.add_argument("--door-inference-mode", choices=("sampled", "legacy"),
                   default=C.STAGE3_DOOR_MODE,
                   help="Door Stage-3 inference mode (default: sampled)")
    p.add_argument("--door-sample-stride", type=int, default=C.STAGE3_DOOR_STRIDE,
                   help="Door frame stride when sampled (default: 3)")
    p.add_argument("--damage-inference-mode", choices=("sampled", "legacy"),
                   default=C.STAGE3_DAMAGE_MODE,
                   help="Damage Stage-3 inference mode (default: sampled)")
    p.add_argument("--damage-sample-stride", type=int, default=C.STAGE3_DAMAGE_STRIDE,
                   help="Damage frame stride when sampled (default: 3)")
    p.add_argument("--load-inference-mode", choices=("sampled", "legacy"),
                   default=C.STAGE3_LOAD_MODE,
                   help="Load Stage-3 inference mode (default: sampled). NOTE: "
                        "Load already sampled at every_nth=2, so sampled/2 is "
                        "behaviourally identical to legacy -- the flag only "
                        "makes the stride explicit.")
    p.add_argument("--load-sample-stride", type=int, default=C.STAGE3_LOAD_STRIDE,
                   help="Load frame stride when sampled (default: 2)")
    p.add_argument("--mode", choices=("batch", "sequential"),
                   default="sequential",
                   help="Pipeline architecture for FOREGROUND runs. "
                        "'sequential' (DEFAULT) processes and seals each "
                        "camera independently, then assembles the persisted "
                        "evidence. 'batch' is the original process_batch() "
                        "path, unchanged and still fully supported. This flag "
                        "does NOT affect --auto/--once/--batch, which always "
                        "run the live S3 discovery path.")
    p.add_argument("--legacy-inference", action="store_true",
                   help="shorthand: force BOTH Door and Damage to legacy "
                        "every-frame tracking (pre-optimization behaviour)")

    # ---- republish a FINISHED batch, with no inference re-run --------------
    p.add_argument("--deliver-only", default=None, metavar="BATCH_DIR",
                   help="deliver an already-finished batch directory: upload its "
                        "reports/evidence/videos, post the 4 per-camera dashboard "
                        "documents and call the ML API, running NO inference.  "
                        "Use it when a delivery failed, or was left off, and "
                        "re-running the pipeline would cost ~30 min per train.")
    p.add_argument("--deliver", action="store_true",
                   help="in --mode sequential, publish after assembly (S3 upload "
                        "+ dashboard ingest + ML API).  OFF by default so a "
                        "sequential run under validation cannot reach the live "
                        "dashboard.")
    p.add_argument("--deliver-per-camera", action="store_true",
                   help="in --mode sequential, POST each camera's inspection to "
                        "the dashboard THE MOMENT that camera seals, without "
                        "waiting for the other three or for global assembly.  "
                        "These documents use the camera's OWN segment numbering "
                        "(wagon n is that camera's nth segment, NOT the fused "
                        "GW_n), exactly as V4's independent per-camera pipelines "
                        "do, and they are superseded by the canonical documents "
                        "assembly publishes afterwards.  Written to a separate "
                        "per_camera/ S3 prefix so the two never overwrite.")

    # ---- historical (time-range) mode --------------------------------------
    # Purely an INPUT-SELECTION layer: it resolves which already-trimmed S3 clips
    # fall in a requested time range and hands each resulting TrainBatch to the
    # SAME `process_batch` the live path uses.  None of these flags is read by
    # --auto / --once / --batch / --local-only; see historical_runner.py.
    hist = p.add_argument_group("historical mode (--historical)")
    hist.add_argument("--historical", action="store_true",
                      help="process already-trimmed S3 clips from a time range")
    hist.add_argument("--date", default=None, help="YYYY-MM-DD")
    hist.add_argument("--start-time", default=None, help="HH:MM[:SS]")
    hist.add_argument("--end-time", default=None, help="HH:MM[:SS]")
    hist.add_argument("--timezone", default=None,
                      help="IANA zone for --date/--start-time/--end-time "
                           "(default Asia/Kolkata)")
    hist.add_argument("--start", default=None,
                      help="ISO-8601 start, e.g. 2026-08-08T10:00:00+05:30 "
                           "(alternative to --date/--start-time)")
    hist.add_argument("--end", default=None, help="ISO-8601 end")
    hist.add_argument("--tolerance-sec", type=int, default=None,
                      help="seconds between two cameras' clips for them to be "
                           "the same train (default 120, the live value).  Some "
                           "days stamp the four cameras minutes apart -- check "
                           "--dry-run and widen if batches come out partial")
    hist.add_argument("--pad-minutes", type=float, default=None,
                      help="how far past its filename timestamp a clip may still "
                           "hold its train (default 15)")
    hist.add_argument("--dry-run", action="store_true",
                      help="discover + print the manifest; download nothing, "
                           "run no inference")
    hist.add_argument("--keep-inputs", action="store_true",
                      help="keep staged clips after a successful batch")
    hist.add_argument("--historical-deliver", action="store_true",
                      help="enable S3 upload + dashboard ingest + email for "
                           "historical batches (OFF by default so a re-run "
                           "cannot overwrite or re-notify the live delivery)")
    hist.add_argument("--manifest-out", default=None,
                      help="path for the JSON manifest (default: "
                           "<workspace>/historical/historical_manifest.json)")

    # ---- pipeline source: WHAT this deployment consumes --------------------
    p.add_argument("--source", dest="source", choices=("trimmed", "raw"),
                   default=None,
                   help="Input the pipeline consumes. 'trimmed' (DEFAULT) = the "
                        "input prefixes already hold trimmed train clips and "
                        "this process is a pure consumer. 'raw' = only raw CCTV "
                        "exists, so this process also runs train extraction "
                        "(raw -> trimmed) before consuming. Defaults to "
                        "WAGONEYE_PIPELINE_SOURCE, else 'trimmed'.")
    p.add_argument("--skip-model-sync", action="store_true",
                   help="skip the startup model availability check / S3 sync "
                        "(models must already be present locally)")
    p.add_argument("--ocr-engine", choices=("rekognition", "easyocr"),
                   default=None,
                   help="Wagon-number OCR engine. 'rekognition' (DEFAULT, V4 "
                        "parity: AWS DetectText on a 3-frame sheet) | 'easyocr' "
                        "(local, no network). Sets WAGONEYE_OCR_ENGINE.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Did the operator actually type --mode, or is this the default?
    # --mode now defaults to "sequential", so paths that must reject an
    # explicit sequential request (historical) cannot infer intent from
    # args.mode alone.
    _raw_argv = list(sys.argv[1:] if argv is None else argv)
    mode_explicit = any(a == "--mode" or a.startswith("--mode=")
                        for a in _raw_argv)

    # File + stdout logging for the long-running service (idempotent).
    setup_logging()

    # The OCR engine is read from the environment by the processor itself, so an
    # explicit flag is expressed as an env override -- one authority, no drift.
    if args.ocr_engine:
        os.environ["WAGONEYE_OCR_ENGINE"] = args.ocr_engine

    # The flag is the operator's intent; `camera_inspection.is_enabled()` is the
    # runtime authority and reads this variable, so one place decides.
    if getattr(args, "deliver_per_camera", False):
        os.environ["WAGONEYE_PER_CAMERA_INGEST"] = "true"

    # Pipeline source: explicit flag wins over WAGONEYE_PIPELINE_SOURCE.
    source = PipelineSource.resolve(args.source)

    # Continuous --auto polling is a daemon: never prompt there.  Interactive
    # toggling is only offered for --local-only / --once / --batch foreground
    # runs, and only when stdin is a real TTY (resolve_feature_config gates it).
    interactive = (not args.no_interactive) and (not args.auto)
    feature_config = resolve_feature_config(
        disable_features=args.disable_features,
        interactive=interactive,
    )

    # Stage-3 inference mode.  --legacy-inference is a shorthand that forces
    # BOTH detectors back to the original every-frame tracker path.
    _door_mode = "legacy" if args.legacy_inference else args.door_inference_mode
    _dmg_mode  = "legacy" if args.legacy_inference else args.damage_inference_mode
    _load_mode = "legacy" if args.legacy_inference else args.load_inference_mode
    inference_opts = {
        "door_inference_mode":   _door_mode,
        "door_sample_stride":    int(args.door_sample_stride),
        "damage_inference_mode": _dmg_mode,
        "damage_sample_stride":  int(args.damage_sample_stride),
        "load_inference_mode":   _load_mode,
        "load_sample_stride":    int(args.load_sample_stride),
    }
    print(f"Stage-3 inference: door={_door_mode}/stride={args.door_sample_stride}"
          f"  damage={_dmg_mode}/stride={args.damage_sample_stride}"
          f"  load={_load_mode}/stride={args.load_sample_stride}")

    # --deliver-only dispatches before everything: it runs no inference, touches
    # no discovery and returns unconditionally, so it can never fall through into
    # a processing mode.
    if args.deliver_only:
        from delivery import finalize
        res = finalize.deliver(
            batch_root=args.deliver_only,
            send_email=not args.skip_email,
            dry_run=args.dry_run,
            verbose=True,
        )
        print(res.render())
        return 0 if (res.ok or args.dry_run) else 3

    # Historical is opt-in and dispatches FIRST, so it can never fall through
    # into the live polling loop.  It returns unconditionally.
    if args.historical:
        # `--historical --mode sequential` is now SUPPORTED: historical stages
        # each discovered batch's clips out of S3 and then runs the same
        # per-camera -> assembly path the foreground sequential mode uses.  It
        # used to be rejected only because historical knew how to call nothing
        # but `process_batch`.
        #
        # An EXPLICIT --mode still decides.  Since --mode defaults to sequential,
        # a bare `--historical` must NOT silently switch architecture: it stays
        # on the batch path, which is the one validated end to end (2026-07-29:
        # 2 trains, 58 wagons, 4/4 cameras ingested).
        if args.auto:
            print("ERROR: --historical and --auto are mutually exclusive "
                  "(one processes a past window once, the other polls for live "
                  "batches forever).  Pick one.", file=sys.stderr)
            return 2
        return run_historical(args, feature_config=feature_config,
                              inference_opts=inference_opts,
                              mode_explicit=mode_explicit)

    # --mode selects the architecture for FOREGROUND runs only.
    #
    # The live S3 paths -- --auto (polling daemon), --once (one discovery
    # cycle) and a bare --batch <key> (one discovered batch) -- have always
    # been served by run_auto(), and they must keep reaching it. This branch
    # returns unconditionally, so without the guard below, defaulting --mode to
    # sequential would divert `--auto` into a local run and silently stop
    # production polling. Live dispatch therefore wins over --mode.
    #
    # `--batch <key>` with --local-only is NOT a live invocation: there the key
    # only names the output directory, exactly as it does for run_local().
    live_dispatch = bool(args.auto or args.once
                         or (args.batch and not args.local_only))

    if args.mode == "sequential" and not live_dispatch:
        return run_sequential(
            local_inputs=args.local_inputs, workspace=args.workspace,
            recon_models_dir=args.recon_models_dir,
            feat_models_dir=args.feat_models_dir,
            batch_key=args.batch, feature_config=feature_config,
            deliver=args.deliver,
            deliver_per_camera=args.deliver_per_camera,
            send_email=not args.skip_email)

    if args.local_only:
        return run_local(
            local_inputs=args.local_inputs,
            batch_key=args.batch,
            workspace=args.workspace,
            recon_models_dir=args.recon_models_dir,
            feat_models_dir=args.feat_models_dir,
            feature_config=feature_config,
            inference_opts=inference_opts,
        )

    if not (args.auto or args.once or args.batch):
        print("ERROR: pass --auto, --once, --batch <key>, or --local-only",
              file=sys.stderr)
        return 2

    return run_auto(
        workspace=args.workspace,
        recon_models_dir=args.recon_models_dir,
        feat_models_dir=args.feat_models_dir,
        poll_interval=args.poll_interval,
        partial_wait_minutes=args.partial_wait,
        run_once=(args.once or bool(args.batch)),
        force_batch_key=args.batch,
        skip_upload=args.skip_upload,
        skip_email=args.skip_email,
        feature_config=feature_config,
        inference_opts=inference_opts,
        source=source,
        skip_model_sync=args.skip_model_sync,
    )


if __name__ == "__main__":
    sys.exit(main())
