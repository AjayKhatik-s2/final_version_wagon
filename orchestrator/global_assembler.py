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
    media_linked: Dict[str, Dict[str, int]] = field(default_factory=dict)
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
                # Another camera already reported this wagon. Each camera saw
                # only its own half, so MERGE rather than keep the first --
                # see _merge_payloads().
                try:
                    with open(dst, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                merged = _merge_payloads(feature, existing, payload)
                merged.setdefault("_sequential_audit", {}) \
                      .setdefault("merged_from", []).append({
                          "local_id": local_id, "camera": m.camera_id,
                          "mapping_kind": m.kind})
                try:
                    with open(dst, "w", encoding="utf-8") as f:
                        json.dump(merged, f, indent=2, default=str)
                except OSError:
                    pass
                continue
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            n += 1
    return n


def _is_real(v) -> bool:
    """A feature value that actually carries a reading."""
    return bool(v) and v != C.NO_DATA


def _union(a, b) -> List[str]:
    out = list(a or [])
    for x in (b or []):
        if x not in out:
            out.append(x)
    return out


def _merge_payloads(feature: str, base: Dict[str, Any],
                    new: Dict[str, Any]) -> Dict[str, Any]:
    """Combine two camera-local payloads describing the SAME global wagon.

    In batch mode ONE processor invocation read every relevant camera and
    combined them itself: `features/door/processor.py` calls
    `_one_camera(LEFT_UP)` and `_one_camera(RIGHT_UP)`, and the load/damage
    processors loop over `C.TOP_CAMERAS`.

    Sequential mode runs the same processors against ONE camera's cache at a
    time, so each camera emits only its own half: RIGHT_UP fills `right_door`
    and leaves `left_door` NO_DATA, LEFT_UP the reverse; each top camera
    reports only what it saw. Keeping the first writer would therefore DROP
    the second camera's readings -- every wagon's left door would come back
    NO_DATA in the global report.

    This reproduces the processors' OWN precedence rules and adds none:

      door    left_* comes from LEFT_UP, right_* from RIGHT_UP -- disjoint by
              construction, so the merge is per side. If both somehow carry a
              reading, the higher confidence wins.
      load    RIGHT_UP_TOP is authoritative, LEFT_UP_TOP is the fallback
              (features/load/processor.py: "RIGHT_UP_TOP authoritative when
              present; LEFT_UP_TOP supports").
      damage  any top camera reporting DAMAGE wins
              (features/damage/processor.py: `any_damage`).

    No inference, no thresholds -- only choosing between values already
    computed on disk.
    """
    m = dict(base)
    if C.STATUS_OK in (base.get("status"), new.get("status")):
        m["status"] = C.STATUS_OK
    m["supporting_cameras"] = _union(base.get("supporting_cameras"),
                                     new.get("supporting_cameras"))
    for k in ("frame_count", "frames_left", "frames_right"):
        if k in base or k in new:
            m[k] = int(base.get(k) or 0) + int(new.get(k) or 0)
    if base.get("tracks") is not None or new.get("tracks") is not None:
        m["tracks"] = list(base.get("tracks") or []) + list(new.get("tracks") or [])
    # Slot names are side/camera specific, so the first writer keeps its slot
    # and the other camera's slots are added alongside it.
    if base.get("evidence") or new.get("evidence"):
        m["evidence"] = {**(new.get("evidence") or {}),
                         **(base.get("evidence") or {})}
    if base.get("per_camera") or new.get("per_camera"):
        m["per_camera"] = {**(new.get("per_camera") or {}),
                           **(base.get("per_camera") or {})}

    if feature == "door":
        for side in ("left", "right"):
            key, ckey = f"{side}_door", f"{side}_door_confidence"
            b_ok, n_ok = _is_real(base.get(key)), _is_real(new.get(key))
            take_new = n_ok and (
                not b_ok
                or float(new.get(ckey) or 0.0) > float(base.get(ckey) or 0.0))
            if take_new:
                m[key] = new[key]
                m[ckey] = new.get(ckey, 0.0)
    elif feature == "load":
        b_ok, n_ok = (_is_real(base.get("load_status")),
                      _is_real(new.get("load_status")))
        auth = C.CAMERA_RIGHT_UP_TOP
        new_is_auth = auth in (new.get("supporting_cameras") or [])
        base_is_auth = auth in (base.get("supporting_cameras") or [])
        if n_ok and (not b_ok or (new_is_auth and not base_is_auth)):
            m["load_status"] = new["load_status"]
            m["load_confidence"] = new.get("load_confidence", 0.0)
    elif feature == "damage":
        b_dmg = base.get("top_damage") == C.DAMAGE_PRESENT
        n_dmg = new.get("top_damage") == C.DAMAGE_PRESENT
        if n_dmg and not b_dmg:
            m["top_damage"] = C.DAMAGE_PRESENT
            m["top_damage_confidence"] = new.get("top_damage_confidence", 0.0)
        elif not b_dmg and not _is_real(base.get("top_damage")) \
                and _is_real(new.get("top_damage")):
            m["top_damage"] = new["top_damage"]
            m["top_damage_confidence"] = new.get("top_damage_confidence", 0.0)
        if b_dmg or n_dmg:
            m["top_damage_details"] = _union(base.get("top_damage_details"),
                                             new.get("top_damage_details"))
    return m


def _link_or_copy(src: str, dst: str) -> bool:
    """Hardlink `src` to `dst`, copying only if the filesystem refuses.

    Frame crops and cache frames are the bulk of a run's disk footprint, and
    the previous experiment filled the root volume. A hardlink costs an inode,
    not a frame, so the global view of the evidence is free. Existing files
    are never overwritten.
    """
    if os.path.exists(dst):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
        return True
    except (OSError, AttributeError, NotImplementedError):
        try:
            shutil.copy2(src, dst)
            return True
        except OSError:
            return False


def _associate_media(
    bundle: CameraEvidenceBundle, mappings, *,
    evidence_dst: str, cache_dst: str,
) -> Dict[str, int]:
    """Give the ALREADY-GENERATED image evidence its global name.

    The combined report resolves evidence as
    `<evidence_root>/<GW_n>/<feature>/<slot>.jpg` and cache frames as
    `<cache_root>/<GW_n>/<camera_folder>/*.jpg`. Sequential mode wrote both
    under CAMERA-LOCAL ids inside each bundle, so without this step the global
    report renders with empty evidence pages -- a regression against the batch
    pipeline rather than a design choice.

    Pure file linking, mirroring `_relabel_feature_evidence`: UNMATCHED
    segments are never invented into a global id, first writer wins on a
    MANY_TO_ONE collision, and nothing in the bundle is moved or deleted --
    the camera-local PDFs keep resolving afterwards.
    """
    n = {"evidence": 0, "cache": 0}
    for m in mappings:
        if not m.global_id:
            continue                          # UNMATCHED -> never invented
        src_ev = os.path.join(bundle.dir, "evidence", m.local_id)
        if os.path.isdir(src_ev):
            for feature in sorted(os.listdir(src_ev)):
                fd = os.path.join(src_ev, feature)
                if not os.path.isdir(fd):
                    continue
                for fn in sorted(os.listdir(fd)):
                    if _link_or_copy(os.path.join(fd, fn),
                                     os.path.join(evidence_dst, m.global_id,
                                                  feature, fn)):
                        n["evidence"] += 1
        # Cache frames are already per camera folder, so two cameras landing
        # on the same GW cannot collide.
        src_ca = os.path.join(bundle.dir, "camera_cache", m.local_id)
        if os.path.isdir(src_ca):
            for folder in sorted(os.listdir(src_ca)):
                cd = os.path.join(src_ca, folder)
                if not os.path.isdir(cd):
                    continue
                for fn in sorted(os.listdir(cd)):
                    if _link_or_copy(os.path.join(cd, fn),
                                     os.path.join(cache_dst, m.global_id,
                                                  folder, fn)):
                        n["cache"] += 1
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
    # The global VIEW of evidence already produced per camera: hardlinks under
    # GW_n names, so the existing report lookups resolve without a second copy
    # of every crop.
    global_evidence = os.path.join(batch_root, "evidence")
    global_cache = os.path.join(batch_root, "camera_cache")
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
        res.media_linked[c] = _associate_media(
            bundles[c], maps, evidence_dst=global_evidence,
            cache_dst=global_cache)
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
            evidence_root=global_evidence,
            wagon_states_root=states_root, cache_root=global_cache,
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
            ml = res.media_linked.get(c, {})
            print(f"  {c:<13} {s['by_kind']}  relabelled={res.relabelled.get(c, 0)}"
                  f"  evidence={ml.get('evidence', 0)} cache={ml.get('cache', 0)}")
    return res
