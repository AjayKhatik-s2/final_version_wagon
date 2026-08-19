"""Sequential mode: process ONE camera, end to end, independently.

A camera runs its complete proven Stage-1 chain, materializes its own local
segments, runs the features it is authoritative for, persists a
CameraEvidenceBundle plus a camera-local report, then SEALS.

It never waits for another camera and never sees a GW id -- global wagon ids do
not exist until global assembly. State lives on disk, so cameras may arrive
minutes or hours apart and the process may exit in between.

Nothing under wagon_count/, reconstruction/, fusion/ or reporting/ is touched;
the Stage-1 work is delegated to orchestrator/camera_pipeline.py, which is a
literal extraction of the proven per-camera order.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core import constants as C
from core.camera_evidence import (
    CameraEvidenceBundle, CameraManifest, as_feature_wagons,
)
from core.camera_tracks_io import write_tracks
from materializer import wagon_cache_builder
from orchestrator import camera_pipeline as cp

# Stage-3 configuration -- IDENTICAL to the tuned batch defaults.
DOOR_STRIDE = 3
DAMAGE_STRIDE = 3
LOAD_STRIDE = 2


@dataclass
class CameraRunResult:
    camera_id: str
    state: str = "PENDING"
    sealed: bool = False
    failure_reason: str = ""
    per_camera_ingest: Any = None       # delivery.camera_inspection result
    local_segments: int = 0
    accepted_gaps: int = 0
    rejected_gaps: int = 0
    recovered_gaps: int = 0
    raw_detections: int = 0
    frames_materialized: int = 0
    feature_calls: Dict[str, int] = field(default_factory=dict)
    feature_summary: Dict[str, Dict[str, str]] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    report_path: str = ""
    bundle_dir: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "camera_id": self.camera_id, "state": self.state,
            "sealed": self.sealed, "failure_reason": self.failure_reason,
            "local_segments": self.local_segments,
            "accepted_gaps": self.accepted_gaps,
            "rejected_gaps": self.rejected_gaps,
            "recovered_gaps": self.recovered_gaps,
            "raw_detections": self.raw_detections,
            "frames_materialized": self.frames_materialized,
            "feature_yolo_calls": dict(self.feature_calls),
            "timings": dict(self.timings),
            "report_path": self.report_path,
        }


def _feature_frame_count(states_dir: str, feature: str) -> int:
    """Frames inspected == YOLO calls, read from the per-segment JSON."""
    d = os.path.join(states_dir, feature)
    if not os.path.isdir(d):
        return 0
    n = 0
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                n += int(json.load(f).get("frame_count") or 0)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return n


def _feature_plan(camera_id: str, enabled: set) -> List:
    """Which features this camera is AUTHORITATIVE for.

    Door lives on the side cameras; Load and Damage on the top cameras. Running
    every feature on every camera would quadruple the work and produce
    meaningless results, so each camera runs only its own.
    """
    plan = []
    if camera_id in C.SIDE_CAMERAS and "door" in enabled:
        from features.door import processor as door_proc
        plan.append(("door", door_proc,
                     dict(inference_mode="sampled", sample_stride=DOOR_STRIDE)))
    if camera_id in C.TOP_CAMERAS and "load" in enabled:
        from features.load import processor as load_proc
        plan.append(("load", load_proc,
                     dict(inference_mode="sampled", sample_stride=LOAD_STRIDE)))
    if camera_id in C.TOP_CAMERAS and "damage" in enabled:
        from features.damage import processor as damage_proc
        plan.append(("damage", damage_proc,
                     dict(inference_mode="sampled",
                          sample_stride=DAMAGE_STRIDE)))
    return plan


def run_camera(
    *,
    camera_id: str,
    video_path: str,
    recon_models_dir: str,
    feat_models_dir: str,
    evidence_root: str,
    enabled_features: Optional[List[str]] = None,
    camera_local_features: bool = True,
    deliver_per_camera: bool = False,
    s3_client=None,
    verbose: bool = True,
) -> CameraRunResult:
    """Drive ONE camera PENDING -> SEALED.

    Never raises: any failure seals this camera FAILED and returns, so it
    cannot block another camera. Global assembly decides separately whether a
    failed support camera is tolerable (a failed MASTER is not).
    """
    enabled = set(enabled_features if enabled_features is not None
                  else ("door", "damage", "load"))
    bundle = CameraEvidenceBundle(evidence_root, camera_id)
    res = CameraRunResult(camera_id=camera_id, bundle_dir=bundle.dir)
    t_all = time.perf_counter()

    def _t(name: str, t0: float) -> None:
        res.timings[name] = round(time.perf_counter() - t0, 3)

    try:
        os.makedirs(bundle.dir, exist_ok=True)
        bundle.save_manifest(CameraManifest(camera_id=camera_id,
                                            video_path=video_path))
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"video not available: {video_path}")

        # ---- Stage 1: the proven camera-local chain --------------------
        t0 = time.perf_counter()
        if camera_id == C.MASTER_CAMERA:
            out = cp.run_master_camera(
                camera_id=camera_id, video_path=video_path,
                gap_model_path=os.path.join(recon_models_dir,
                                            "right_up_wagon_gap.pt"),
                side_cls_path=os.path.join(recon_models_dir,
                                           "side_classification.pt"),
                verbose=verbose)
        else:
            is_top = camera_id in C.TOP_CAMERAS
            gap_model = "top_gap.pt" if is_top else "left_up_wagon_gap.pt"
            cls_name = ("top_classification.pt" if is_top
                        else "side_classification.pt")
            cls_path = os.path.join(recon_models_dir, cls_name)
            out = cp.run_support_camera(
                camera_id=camera_id, video_path=video_path,
                gap_model_path=os.path.join(recon_models_dir, gap_model),
                classifier_path=cls_path if os.path.exists(cls_path) else None,
                is_top=is_top, verbose=verbose)
        _t("stage1", t0)

        bundle.advance("TRACKING", fps=out.tracks.fps,
                       total_frames=out.tracks.total_frames,
                       width=out.tracks.width, height=out.tracks.height)
        bundle.advance("VALIDATED")

        res.accepted_gaps = len(out.tracks.gaps)
        res.rejected_gaps = len(getattr(out.validation, "rejected", []) or [])
        res.recovered_gaps = (len(getattr(out.recovery, "recovered", []) or [])
                              if out.recovery is not None else 0)
        res.raw_detections = sum(
            len(v) for v in (out.tracks.raw_frame_detections or {}).values())

        # ---- persist Stage-1 evidence ----------------------------------
        # FULL-FIDELITY snapshot -- global assembly reconstructs
        # LocalCameraTracks from this without re-running Stage 1.
        # `tracking.json` keeps the human-readable reporting view alongside it.
        write_tracks(os.path.join(bundle.dir, "tracking_full.json"), out.tracks)
        bundle.write_json("tracking.json",
                          out.tracks.to_dict(include_classifications=True))
        if out.validation is not None:
            bundle.write_json("gap_validation.json",
                              out.validation.to_dict(include_rejections=True))
        if out.stitch is not None:
            bundle.write_json("fragments.json", out.stitch.to_dict())
        if out.recovery is not None:
            bundle.write_json("wagon_active_recovery.json",
                              out.recovery.to_dict())
        if out.wagon_region is not None and hasattr(out.wagon_region, "to_dict"):
            bundle.write_json("wagon_region.json", out.wagon_region.to_dict())
        bundle.write_json("classification.json",
                          [c.to_dict() for c in out.classifications
                           if hasattr(c, "to_dict")])
        bundle.write_segments(out.segments)
        res.local_segments = len(out.segments)
        bundle.advance("SEGMENTED")

        # ---- materialize LOCAL segments --------------------------------
        t0 = time.perf_counter()
        cache_root = os.path.join(bundle.dir, "camera_cache")
        counts = wagon_cache_builder.build_camera_local(
            camera_id=camera_id, video_path=video_path,
            segments=out.segments, cache_root=cache_root, verbose=verbose)
        res.frames_materialized = sum(counts.values())
        _t("materialize", t0)
        bundle.advance("MATERIALIZED")

        # ---- camera-local features, for the camera-local PDF ------------
        # These produce the ONLY evidence a camera-local report can embed:
        # reporting/camera_reports.py resolves every image through
        # `evidence_snapshot(evidence_root, <id>, <feature>, <slot>)`, so with
        # no local feature run its snapshot pages come out empty.
        #
        # They are NOT the global answer and are never promoted to one. The old
        # pipeline runs features AFTER fusion, over frames the materializer
        # bucketed with `round((GW.time - delta) * local_fps)`; a support
        # camera's `delta` is estimated from master + support observations
        # together, so it cannot be known while that camera is processed alone.
        # Global assembly therefore recomputes all three features over the
        # global wagons and ignores everything written here.
        #
        # The cost is one extra feature pass per camera. Pass
        # camera_local_features=False to skip it: the global result is
        # bit-for-bit unaffected, and only the camera PDFs lose their images.
        roster = as_feature_wagons(out.segments, camera_id)
        states_dir = os.path.join(bundle.dir, "features")
        common = dict(state=None, cache_root=cache_root,
                      feature_models_dir=feat_models_dir,
                      output_dir=states_dir,
                      evidence_root=os.path.join(bundle.dir, "evidence"),
                      segments=roster, verbose=verbose)
        plan = _feature_plan(camera_id, enabled) if camera_local_features else []
        if not plan and verbose:
            print(f"[SEQ/{camera_id}] camera-local features SKIPPED -- "
                  f"inference runs at global assembly, over the materialized "
                  f"global wagon windows (batch-equivalent)")
        for name, mod, extra in plan:
            t0 = time.perf_counter()
            try:
                res.feature_summary[name] = mod.run(**common, **extra) or {}
            except Exception as e:
                print(f"[SEQ/{camera_id}/{name}] CRASHED: {e}")
                traceback.print_exc(limit=3)
                res.feature_summary[name] = {}
            _t(f"feature_{name}", t0)
            res.feature_calls[name] = _feature_frame_count(states_dir, name)
        bundle.advance("FEATURES")

        # ---- camera-local report -----------------------------------
        # Rendered by the EXISTING proven renderer
        # (reporting/camera_reports.py::build_camera_report) via a thin state
        # adapter, so layout/styling/sections are identical to the batch camera
        # reports. Only the wagon ids differ: L_<CAM>_<n>, never GW_n.
        t0 = time.perf_counter()
        # Machine-readable audit of what this camera actually did. Kept
        # alongside the PDF: it carries the gap/recovery counters and per-
        # feature call counts that the rendered report does not show.
        bundle.write_json("camera_report.json", {
            "schema": "wagon_eye.camera_report.v1",
            "camera_id": camera_id,
            "is_master": camera_id == C.MASTER_CAMERA,
            "fps": out.tracks.fps,
            "total_frames": out.tracks.total_frames,
            "raw_detections": res.raw_detections,
            "accepted_gaps": res.accepted_gaps,
            "rejected_gaps": res.rejected_gaps,
            "recovered_gaps": res.recovered_gaps,
            "reclassified_after_recovery": out.reclassified_after_recovery,
            "local_segments": [s.to_dict() for s in out.segments],
            "feature_summary": res.feature_summary,
            "feature_yolo_calls": res.feature_calls,
            "frames_materialized": res.frames_materialized,
            "notes": list(out.notes),
        })
        pdf = None
        try:
            from orchestrator.camera_report_adapter import build_local_camera_pdf
            pdf = build_local_camera_pdf(
                bundle,
                output_pdf=os.path.join(bundle.dir, f"{camera_id}_report.pdf"),
                batch_key=f"{camera_id} (camera-local)",
                fps=out.tracks.fps, total_frames=out.tracks.total_frames,
                logo_path=os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "reporting", "assets", "Logo.jpeg"),
                verbose=verbose)
        except Exception as e:
            print(f"[SEQ/{camera_id}] camera PDF failed: "
                  f"{type(e).__name__}: {e}")
        res.report_path = pdf or os.path.join(bundle.dir, "camera_report.json")
        _t("report", t0)
        bundle.advance("REPORTED")

        bundle.advance("SEALED")
        res.state, res.sealed = "SEALED", True

        # ---- publish THIS camera to the dashboard, immediately -------------
        # Deliberately AFTER the seal: the camera's work is already persisted, so
        # a receiver outage cannot un-seal it or fail the run.  The document uses
        # this camera's OWN segment numbering -- see delivery.camera_inspection
        # for what that means and does not mean.
        if deliver_per_camera:
            try:
                from delivery import camera_inspection
                res.per_camera_ingest = camera_inspection.publish(
                    bundle,
                    s3_client=s3_client,
                    fps=out.tracks.fps,
                    total_frames=out.tracks.total_frames,
                    raw_video_name=os.path.basename(video_path),
                    verbose=verbose,
                )
                if verbose and res.per_camera_ingest is not None:
                    print(f"[SEQ/{camera_id}] {res.per_camera_ingest.render()}")
            except Exception as e:  # noqa: BLE001 - never un-seal a camera
                print(f"[SEQ/{camera_id}] per-camera ingest failed "
                      f"(non-fatal): {type(e).__name__}: {e}")

    except Exception as e:
        res.failure_reason = f"{type(e).__name__}: {e}"
        res.state = "FAILED"
        try:
            bundle.fail(res.failure_reason)
        except Exception:
            pass
        print(f"[SEQ/{camera_id}] FAILED: {res.failure_reason}")
        traceback.print_exc(limit=3)

    res.timings["total"] = round(time.perf_counter() - t_all, 3)
    try:
        bundle.write_json("run_result.json", res.to_dict())
    except Exception:
        pass
    if verbose:
        print(f"[SEQ/{camera_id}] {res.state}  segments={res.local_segments} "
              f"gaps={res.accepted_gaps} frames={res.frames_materialized} "
              f"calls={res.feature_calls} {res.timings['total']:.1f}s")
    return res
