"""Camera-local state adapter for the EXISTING camera report renderer.

`reporting/camera_reports.py::build_camera_report()` is the proven per-camera
renderer -- same `_brand` styles, `_pages.make_doc()` frame, summary page,
quartile wagon pages, anomaly summary and evidence grid. It needs a
`GlobalTrainState` plus a `{id -> UnifiedWagonState}` map, and it treats the id
purely as an OPAQUE key for lookups and labels.

So sequential mode does not need a second renderer: it needs the same two
dataclasses populated with CAMERA-LOCAL ids. `L_RIGHT_UP_1` is a genuine
identifier at camera-local time -- no `GW_n` is invented, and none can be,
because global ids do not exist until assembly.

This module builds those objects from a sealed CameraEvidenceBundle. It
renders nothing itself and modifies no reporting code.

Path contract, matching what camera_runner already writes:

    camera_evidence/<CAM>/camera_cache/<LOCAL_ID>/<camera_folder>/*.jpg
    camera_evidence/<CAM>/features/<feature>/<LOCAL_ID>.json
    camera_evidence/<CAM>/evidence/<LOCAL_ID>/<feature>/...

which are exactly the shapes `_wagon_covered()`,
`_evidence_lookup.read_wagon_feature_json()` and
`_evidence_lookup.evidence_snapshot()` already probe.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from core import constants as C
from core.camera_evidence import CameraEvidenceBundle, LocalSegment
from core.global_state_loader import GlobalTrainState, GlobalWagon


def local_wagons(segments: List[LocalSegment],
                 camera_id: str) -> Tuple[GlobalWagon, ...]:
    """Camera-local segments as `GlobalWagon` records keyed by LOCAL id.

    Frame numbers and times stay this camera's own absolute values -- they are
    not rebased onto a master clock, because no master clock exists yet.
    """
    return tuple(
        GlobalWagon(
            global_id=s.local_id,          # LOCAL id, never GW_n
            wagon_index=s.index,
            start_frame_master=s.start_frame,
            end_frame_master=s.end_frame,
            start_time=s.start_time,
            end_time=s.end_time,
            classification=s.label,
            classification_confidence=s.confidence,
            supporting_cameras=(camera_id,),
        )
        for s in segments
    )


def local_state(segments: List[LocalSegment], camera_id: str,
                *, fps: float = 0.0, total_frames: int = 0) -> GlobalTrainState:
    """A `GlobalTrainState` scoped to ONE camera, keyed by local ids.

    `total_wagons` is this camera's own segment count. The renderer uses it
    only for the summary line, so it reads as "segments this camera saw"
    rather than a global claim.
    """
    wagons = local_wagons(segments, camera_id)
    return GlobalTrainState(
        total_wagons=len(wagons),
        wagons=wagons,
        master_camera=camera_id,
        master_fps=float(fps or 0.0),
        master_total_frames=int(total_frames or 0),
        per_camera_status={camera_id: "camera-local (pre-assembly)"},
    )


def local_unified(bundle: CameraEvidenceBundle, segments: List[LocalSegment]):
    """Fuse this camera's persisted feature JSON into UnifiedWagonStates.

    Reuses the EXISTING `fusion.wagon_state_builder` so the authority rules,
    anomaly precedence and confidence maths are the proven ones -- it is
    pointed at the camera's own `features/` tree and keyed by local id.
    `write_per_wagon_json=False`: this is a read for rendering, and the
    camera-local tree must not gain a `unified/` directory that global
    assembly might later mistake for real fused output.
    """
    from fusion import wagon_state_builder

    state = local_state(segments, bundle.camera_id)
    return wagon_state_builder.build(
        state=state,
        wagon_states_root=os.path.join(bundle.dir, "features"),
        write_per_wagon_json=False,
        verbose=False,
    )


def adapt(bundle: CameraEvidenceBundle,
          *, fps: float = 0.0, total_frames: int = 0):
    """-> (state, unified, paths) ready for `build_camera_report()`."""
    segments = bundle.read_segments()
    state = local_state(segments, bundle.camera_id,
                        fps=fps, total_frames=total_frames)
    unified = local_unified(bundle, segments)
    paths = {
        "cache_root": os.path.join(bundle.dir, "camera_cache"),
        "wagon_states_root": os.path.join(bundle.dir, "features"),
        "evidence_root": os.path.join(bundle.dir, "evidence"),
    }
    return state, unified, paths


def build_local_camera_pdf(
    bundle: CameraEvidenceBundle,
    *,
    output_pdf: str,
    batch_key: str,
    fps: float = 0.0,
    total_frames: int = 0,
    per_camera_tracking_path: Optional[str] = None,
    logo_path: Optional[str] = None,
    verbose: bool = True,
) -> Optional[str]:
    """Render this camera's PDF with the EXISTING proven renderer.

    Delegates to `reporting.camera_reports.build_camera_report()` unchanged, so
    layout, styling, page structure, tables, ordering and the evidence grid are
    identical to the batch camera reports. The only difference is that the
    wagon ids read `L_<CAM>_<n>` instead of `GW_n`.

    Returns None on failure -- a report problem must never un-seal a camera
    whose inference succeeded.
    """
    from reporting import camera_reports

    state, unified, paths = adapt(bundle, fps=fps, total_frames=total_frames)
    return camera_reports.build_camera_report(
        camera_id=bundle.camera_id,
        state=state,
        unified=unified,
        evidence_root=paths["evidence_root"],
        wagon_states_root=paths["wagon_states_root"],
        cache_root=paths["cache_root"],
        per_camera_tracking_path=per_camera_tracking_path,
        output_pdf=output_pdf,
        batch_key=batch_key,
        logo_path=logo_path,
        verbose=verbose,
    )
