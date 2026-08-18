"""Sequential mode: global assembly from SEALED camera bundles.

Runs only once the required camera evidence exists. It reconstructs each
camera's tracks from its bundle, applies the EXISTING fixed-master fusion,
maps every camera-local segment onto a `GW_n`, relabels the ALREADY-COMPUTED
feature evidence, fuses, and emits the combined report.

HARD RULE: no detector runs here. Door/Damage/Load inference happened while
each camera was being processed; assembly only moves and fuses files. A test
asserts zero YOLO calls during this stage.

Nothing under wagon_count/, reconstruction/ or fusion/ is modified, and the
existing global report builder is used unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
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
    relabelled: Dict[str, int] = field(default_factory=dict)
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


def _relabel_feature_evidence(
    bundle: CameraEvidenceBundle, mappings, states_root: str,
) -> int:
    """Copy per-LOCAL-segment feature JSON onto its GLOBAL wagon id.

    Pure file movement -- the payload is rewritten only to carry the new
    `global_id` and an audit trail of where it came from. No inference.
    Several locals landing on one GW (MANY_TO_ONE) do not overwrite each
    other silently: the first wins and the rest are recorded in
    `merged_from`, so nothing is lost.
    """
    n = 0
    by_local = {m.local_id: m for m in mappings}
    feat_dir = os.path.join(bundle.dir, "features")
    if not os.path.isdir(feat_dir):
        return 0
    for feature in sorted(os.listdir(feat_dir)):
        src_dir = os.path.join(feat_dir, feature)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(states_root, feature)
        for fn in sorted(os.listdir(src_dir)):
            if not fn.endswith(".json"):
                continue
            local_id = fn[:-5]
            m = by_local.get(local_id)
            if m is None or not m.global_id:
                continue                      # UNMATCHED -> never invented
            try:
                with open(os.path.join(src_dir, fn), "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            payload["global_id"] = m.global_id
            payload.setdefault("_sequential_audit", {})
            payload["_sequential_audit"].update({
                "source_local_id": local_id,
                "source_camera": m.camera_id,
                "mapping_kind": m.kind,
                "overlap_fraction": m.overlap_fraction,
                "offset_applied": m.offset_applied,
            })
            # Created lazily: an all-UNMATCHED camera must leave no empty
            # feature directory behind to be mistaken for "ran, found nothing".
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, f"{m.global_id}.json")
            if os.path.exists(dst):
                try:
                    with open(dst, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    existing.setdefault("_sequential_audit", {}) \
                            .setdefault("merged_from", []).append(local_id)
                    with open(dst, "w", encoding="utf-8") as f:
                        json.dump(existing, f, indent=2, default=str)
                except (OSError, json.JSONDecodeError):
                    pass
                continue
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            n += 1
    return n


def assemble(
    *,
    evidence_root: str,
    output_root: str,
    batch_key: str,
    master_camera: str = C.MASTER_CAMERA,
    all_cameras: Tuple[str, ...] = C.ALL_CAMERAS,
    verbose: bool = True,
) -> AssemblyResult:
    """Fuse sealed camera bundles into the global train + combined report."""
    import global_fusion as gf

    res = AssemblyResult()
    t_all = time.perf_counter()

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

    # ---- existing fixed-master fusion, unchanged -----------------------
    t0 = time.perf_counter()
    engine_state = gf.assemble_global_train_state_master_fixed(
        master_tracks=tracks[master_camera],
        support_tracks=[t for c, t in tracks.items() if c != master_camera],
        initial_classifications=_load_master_classifications(
            bundles[master_camera]),
        config=gf.FusionConfig(),
        verbose=verbose,
        wagon_only=True,
    )
    res.timings["fusion_alignment"] = round(time.perf_counter() - t0, 3)

    batch_root = os.path.join(output_root, batch_key)
    gs_dir = os.path.join(batch_root, "global_state")
    states_root = os.path.join(batch_root, "wagon_states")
    reports_root = os.path.join(batch_root, "reports")
    for d in (gs_dir, states_root, reports_root):
        os.makedirs(d, exist_ok=True)
    res.state_json_path = os.path.join(gs_dir, "global_train_state.json")
    with open(res.state_json_path, "w", encoding="utf-8") as f:
        f.write(engine_state.to_json())

    state = parse_global_train_state(engine_state.to_dict())
    res.total_wagons = state.total_wagons
    offsets_meta = state.camera_offsets or {}
    resolved = state.camera_time_offsets()

    # ---- map local -> global, then relabel evidence --------------------
    t0 = time.perf_counter()
    audit: Dict[str, Any] = {}
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
        res.relabelled[c] = _relabel_feature_evidence(bundles[c], maps,
                                                      states_root)
    with open(os.path.join(gs_dir, "local_to_global_mapping.json"), "w",
              encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)
    res.timings["mapping_relabel"] = round(time.perf_counter() - t0, 3)

    # ---- fuse + combined report (existing builders, unchanged) ---------
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
            evidence_root=os.path.join(batch_root, "evidence"),
            wagon_states_root=states_root, cache_root=None,
            missing_cameras=list(res.failed_cameras), camera_pdf_urls={},
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
            print(f"  {c:<13} {s['by_kind']}  relabelled={res.relabelled.get(c, 0)}")
    return res
