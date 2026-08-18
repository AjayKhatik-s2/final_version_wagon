#!/usr/bin/env python3
"""Four-camera ARRIVAL simulation for sequential mode.

Simulates the real auto pipeline: each camera arrives on its own, is processed
in its OWN OS PROCESS, and is fully persisted to disk before the next one
starts. Between arrivals the orchestrator asserts filesystem state -- the
arrived cameras are SEALED with a PDF on disk, and the not-yet-arrived cameras
have no bundle directory at all.

This deliberately does NOT call `run_sequential()`. A single in-process loop
could pass while sharing state in memory; separate processes cannot. Global
assembly runs only after the fourth camera seals.

    # full simulation
    python benchmarks/run_sequential_arrivals.py --local-inputs local_inputs

    # one arrival (this is what the orchestrator subprocesses)
    python benchmarks/run_sequential_arrivals.py --camera RIGHT_UP \
        --evidence-root <dir> --local-inputs local_inputs

Exit 0 = all arrivals + assembly succeeded and every assertion held.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import constants as C                                    # noqa: E402
from core.batch import scan_local_video_dir                        # noqa: E402
from core.camera_evidence import CameraEvidenceBundle              # noqa: E402

ARRIVAL_ORDER = ["RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP", "LEFT_UP_TOP"]


def say(*a) -> None:
    print(*a, flush=True)


class AssertionFailed(RuntimeError):
    pass


def _check(cond: bool, msg: str) -> None:
    if cond:
        say(f"      PASS  {msg}")
    else:
        say(f"      FAIL  {msg}")
        raise AssertionFailed(msg)


def bundle_dir(evidence_root: str, cam: str) -> str:
    return os.path.join(evidence_root, cam)


def camera_pdf(evidence_root: str, cam: str) -> str:
    return os.path.join(bundle_dir(evidence_root, cam), f"{cam}_report.pdf")


def state_of(evidence_root: str, cam: str) -> str:
    return CameraEvidenceBundle(evidence_root, cam).load_manifest().state


def assert_after_arrival(evidence_root: str, arrived: List[str]) -> None:
    """Arrived cameras SEALED with a PDF; the rest must not exist at all."""
    for cam in arrived:
        _check(state_of(evidence_root, cam) == "SEALED", f"{cam} is SEALED")
        _check(os.path.isfile(os.path.join(bundle_dir(evidence_root, cam),
                                           "segments.json")),
               f"{cam} local segments persisted")
        _check(os.path.isfile(os.path.join(bundle_dir(evidence_root, cam),
                                           "tracking_full.json")),
               f"{cam} rich tracking persisted")
        _check(os.path.isfile(camera_pdf(evidence_root, cam)),
               f"{cam}_report.pdf exists")
    for cam in ARRIVAL_ORDER:
        if cam in arrived:
            continue
        _check(not os.path.exists(bundle_dir(evidence_root, cam)),
               f"{cam} bundle does NOT exist yet")


def run_one_arrival(evidence_root: str, cam: str, video: str,
                    models: Dict[str, str]) -> int:
    """Spawn a SEPARATE process for this camera."""
    cmd = [sys.executable, "-u", os.path.abspath(__file__),
           "--camera", cam, "--evidence-root", evidence_root,
           "--video", video,
           "--recon-models-dir", models["recon"],
           "--feat-models-dir", models["feat"]]
    say(f"      subprocess: {cam}")
    return subprocess.run(cmd, cwd=_ROOT).returncode


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-inputs", default=os.path.join(_ROOT, "local_inputs"))
    ap.add_argument("--workspace", default=os.path.join(_ROOT, "batch_outputs"))
    ap.add_argument("--recon-models-dir",
                    default=os.path.join(_ROOT, "models", "reconstruction"))
    ap.add_argument("--feat-models-dir",
                    default=os.path.join(_ROOT, "models", "features"))
    ap.add_argument("--batch-key", default=None)
    # single-arrival mode (used by the subprocess)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--evidence-root", default=None)
    ap.add_argument("--video", default=None)
    args = ap.parse_args(argv)

    # ---- single-arrival worker -------------------------------------
    if args.camera:
        from orchestrator import camera_runner
        r = camera_runner.run_camera(
            camera_id=args.camera, video_path=args.video,
            recon_models_dir=args.recon_models_dir,
            feat_models_dir=args.feat_models_dir,
            evidence_root=args.evidence_root,
            enabled_features=["door", "damage", "load"], verbose=True)
        return 0 if r.sealed else 3

    # ---- orchestrator ----------------------------------------------
    videos = scan_local_video_dir(args.local_inputs)
    missing = [c for c in ARRIVAL_ORDER if c not in videos]
    if missing:
        say(f"ERROR: no video for {missing} in {args.local_inputs}")
        return 2

    key = args.batch_key or ("sequential_" +
                             datetime.now().strftime("%Y%m%d_%H%M%S"))
    evidence_root = os.path.join(args.workspace, key, "camera_evidence")
    os.makedirs(evidence_root, exist_ok=True)
    models = {"recon": args.recon_models_dir, "feat": args.feat_models_dir}

    say("=" * 78)
    say(f"  SEQUENTIAL ARRIVAL SIMULATION   {key}")
    say("=" * 78)
    say(f"  evidence : {evidence_root}")
    say(f"  order    : {ARRIVAL_ORDER}")

    t_all = time.time()
    arrived: List[str] = []
    timings: Dict[str, float] = {}
    try:
        for i, cam in enumerate(ARRIVAL_ORDER, start=1):
            say(f"\n--- ARRIVAL {i}: {cam} ---")
            t0 = time.time()
            rc = run_one_arrival(evidence_root, cam, videos[cam], models)
            timings[cam] = round(time.time() - t0, 1)
            if rc != 0:
                say(f"      camera process exited {rc}")
            arrived.append(cam)
            say(f"    verifying persisted state after arrival {i}")
            assert_after_arrival(evidence_root, arrived)
            say(f"    {cam} done in {timings[cam]}s "
                f"(process exited; state on disk only)")

        say("\n--- GLOBAL ASSEMBLY (only now, after arrival 4) ---")
        from orchestrator import global_assembler
        t0 = time.time()
        asm = global_assembler.assemble(
            evidence_root=evidence_root, output_root=args.workspace,
            batch_key=key, verbose=True)
        timings["assembly"] = round(time.time() - t0, 1)

        reports = os.path.join(args.workspace, key, "reports")
        _check(asm.ready, "assembly reported ready")
        _check(os.path.isfile(os.path.join(reports,
                                           "combined_train_report.json")),
               "combined_train_report.json exists")
        _check(os.path.isfile(os.path.join(reports,
                                           "combined_train_report.pdf")),
               "combined_train_report.pdf exists")
    except AssertionFailed as e:
        say(f"\nSIMULATION FAILED: {e}")
        return 1

    say("\n" + "=" * 78)
    say("  RESULT")
    say("=" * 78)
    for cam in ARRIVAL_ORDER:
        b = CameraEvidenceBundle(evidence_root, cam)
        rr = b.read_json("run_result.json") or {}
        say(f"  {cam:<13} {state_of(evidence_root, cam):<8} "
            f"segments={rr.get('local_segments')} "
            f"gaps={rr.get('accepted_gaps')} "
            f"calls={rr.get('feature_yolo_calls')} {timings.get(cam)}s")
    say(f"  global wagons : {asm.total_wagons}")
    for cam, sm in asm.mapping_by_camera.items():
        say(f"  mapping {cam:<13} {sm['by_kind']}")
    say(f"  assembly      : {timings.get('assembly')}s")
    say(f"  TOTAL         : {time.time() - t_all:.1f}s")
    say(f"  output        : {os.path.join(args.workspace, key)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
