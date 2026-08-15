#!/usr/bin/env python3
"""
validate_ec2.py  --  pre-flight check for the standalone Global Wagon Count project
==================================================================================

Verifies that a machine (EC2, SageMaker, or a local dev box) is ready to run

    python run_global_count.py

It checks, and prints PASS / FAIL / WARN for, each of:

    1. Python version
    2. Required Python packages (numpy, opencv, torch, ultralytics) + versions
    3. CUDA / GPU availability                (WARN only -- CPU works, just slower)
    4. Required directories                   inputs/ models/ results/
    5. The 4 expected input videos            exists / size / OpenCV open /
                                              fps / frame count / w x h / duration /
                                              first frame decodes
    6. The 4 expected model weights           exists / size / loads through the
                                              project's own torch.load + ultralytics
                                              path / task / class names
    7. Filesystem write access to the output directory

This script deliberately does NOT run the wagon-count pipeline.  It never
writes to inputs/ or models/, and never modifies a video or a .pt file.

Usage
-----
    python validate_ec2.py                      # full check
    python validate_ec2.py --skip-models        # fast: skip .pt loading
    python validate_ec2.py --inputs-dir /data/videos --models-dir /opt/models

Exit status
-----------
    0  every check passed (warnings allowed)
    1  at least one check FAILED
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

# Same default conventions as run_global_count.py
REQUIRED_VIDEOS = [
    ("RIGHT_UP (master)", "right_up.mp4"),
    ("LEFT_UP",           "left_up.mp4"),
    ("RIGHT_UP_TOP",      "right_up_top.mp4"),
    ("LEFT_UP_TOP",       "left_up_top.mp4"),
]

# name -> (role, expected ultralytics task or None if either is acceptable)
REQUIRED_MODELS = [
    ("right_up_wagon_gap.pt",  "gap detection on RIGHT_UP (master)", "detect"),
    ("left_up_wagon_gap.pt",   "gap detection on LEFT_UP",           "detect"),
    ("top_gap.pt",             "gap detection on both TOP cameras",  "detect"),
    ("side_classification.pt", "ENGINE / WAGON / BRAKE_VAN on RIGHT_UP and LEFT_UP",
     None),
    ("top_classification.pt",  "ENGINE / WAGON / BRAKE_VAN on RIGHT_UP_TOP and "
                               "LEFT_UP_TOP", None),
]

#: Models whose absence disables a capability but does NOT stop a run. The wagon
#: count comes from RIGHT_UP alone, so top classification is not count-critical.
#: It is still reported as a FAIL of the *top classification capability* so a
#: missing file can never pass unnoticed, and no other model is ever substituted.
CAPABILITY_ONLY_MODELS = {"top_classification.pt"}

MIN_PYTHON = (3, 10)

_results: list[tuple[str, str, str]] = []   # (status, check name, detail)


def _record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "INFO": "[INFO]"}[status]
    line = f"  {icon}  {name}"
    if detail:
        line += f"\n           {detail}"
    print(line)


def _mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def _section(title: str) -> None:
    print()
    print("-" * 70)
    print(f"  {title}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# 1. Python
# ---------------------------------------------------------------------------

def check_python() -> None:
    _section("1. Python interpreter")
    v = sys.version_info
    detail = f"{sys.version.splitlines()[0]}  ({sys.executable})"
    if (v.major, v.minor) >= MIN_PYTHON:
        _record("PASS", f"Python {v.major}.{v.minor}.{v.micro}", detail)
    else:
        _record("FAIL", f"Python {v.major}.{v.minor}.{v.micro} is too old "
                        f"(need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})", detail)


# ---------------------------------------------------------------------------
# 2 + 3. Packages and GPU
# ---------------------------------------------------------------------------

def check_packages() -> None:
    _section("2. Python packages")

    try:
        import numpy
        _record("PASS", f"numpy {numpy.__version__}")
    except Exception as e:
        _record("FAIL", "numpy import failed", str(e))

    try:
        import cv2
        _record("PASS", f"opencv-python {cv2.__version__}")
    except Exception as e:
        _record("FAIL", "cv2 (opencv) import failed",
                f"{e}\n           On a headless Ubuntu EC2 install the system libs:\n"
                f"           sudo apt-get install -y libgl1 libglib2.0-0")

    try:
        import torch
        _record("PASS", f"torch {torch.__version__}")
    except Exception as e:
        _record("FAIL", "torch import failed", str(e))
        return

    try:
        import ultralytics
        _record("PASS", f"ultralytics {ultralytics.__version__}")
    except Exception as e:
        _record("FAIL", "ultralytics import failed", str(e))

    _section("3. Compute device")
    try:
        import torch
        if torch.cuda.is_available():
            names = ", ".join(torch.cuda.get_device_name(i)
                              for i in range(torch.cuda.device_count()))
            _record("PASS", f"CUDA available ({torch.cuda.device_count()} GPU)", names)
        else:
            _record("WARN", "No CUDA GPU detected -- inference will run on CPU",
                    "The pipeline is correct on CPU but markedly slower. "
                    "For GPU, install a CUDA build of torch (see README).")
    except Exception as e:
        _record("WARN", "Could not query CUDA", str(e))


# ---------------------------------------------------------------------------
# 4. Directories
# ---------------------------------------------------------------------------

def check_dirs(inputs_dir: str, models_dir: str, output_dir: str) -> None:
    _section("4. Directories")
    for label, path, must_exist in (
        ("inputs dir",  inputs_dir,  True),
        ("models dir",  models_dir,  True),
        ("output dir",  output_dir,  False),
    ):
        if os.path.isdir(path):
            _record("PASS", f"{label} present", path)
        elif must_exist:
            _record("FAIL", f"{label} missing", path)
        else:
            _record("INFO", f"{label} does not exist yet (created on first run)", path)


# ---------------------------------------------------------------------------
# 5. Videos
# ---------------------------------------------------------------------------

def check_videos(inputs_dir: str) -> None:
    _section("5. Input videos")
    try:
        import cv2
    except Exception:
        _record("FAIL", "Skipping video checks -- cv2 unavailable")
        return

    for role, filename in REQUIRED_VIDEOS:
        path = os.path.join(inputs_dir, filename)
        label = f"{filename}  ({role})"

        if not os.path.isfile(path):
            _record("FAIL", f"{label} -- not found", path)
            continue

        size = os.path.getsize(path)
        if size == 0:
            _record("FAIL", f"{label} -- file is empty (0 bytes)", path)
            continue

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            _record("FAIL", f"{label} -- OpenCV cannot open it ({_mb(size)})",
                    "File may be corrupt, or this OpenCV build lacks the codec.")
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
        cap.release()

        duration = (frames / fps) if fps > 0 else 0.0
        detail = (f"{_mb(size):>10}  fps={fps:.3f}  frames={frames}  "
                  f"{width}x{height}  duration={duration:.1f}s")

        if fps <= 0:
            # run_global_count.py raises on non-positive fps, so this is fatal.
            _record("FAIL", f"{label} -- reports non-positive fps", detail)
        elif not ok or frame is None:
            _record("FAIL", f"{label} -- opened but first frame did not decode", detail)
        elif frames <= 0:
            _record("WARN", f"{label} -- container misreports frame count", detail)
        else:
            _record("PASS", label, detail)


# ---------------------------------------------------------------------------
# 6. Models
# ---------------------------------------------------------------------------

def _patch_torch_load() -> None:
    """Mirror the torch.load monkey-patch used by tracker_engine.py.

    Ultralytics .pt checkpoints are pickled objects, so torch >= 2.6's
    default weights_only=True refuses to load them.  tracker_engine.py
    applies exactly this patch before importing YOLO; we do the same so
    the validation reflects real runtime behaviour.
    """
    import torch
    _orig_load = torch.load

    def _patched(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)

    torch.load = _patched


def check_models(models_dir: str) -> None:
    _section("6. Model weights")

    for filename, role, _expected_task in REQUIRED_MODELS:
        path = os.path.join(models_dir, filename)
        label = f"{filename}  ({role})"
        if not os.path.isfile(path):
            if filename in CAPABILITY_ONLY_MODELS:
                # Reported once, as a capability FAIL, in section 6b.
                _record("INFO", f"{label} -- not present (see 6b)", path)
            else:
                _record("FAIL", f"{label} -- not found", path)
        elif os.path.getsize(path) == 0:
            _record("FAIL", f"{label} -- file is empty (0 bytes)", path)
        else:
            _record("INFO", label, f"{_mb(os.path.getsize(path)):>10}  {path}")


def check_model_loading(models_dir: str) -> None:
    _section("6b. Model loading (ultralytics YOLO)")
    try:
        _patch_torch_load()
        from ultralytics import YOLO
    except Exception as e:
        _record("FAIL", "Cannot import ultralytics.YOLO -- skipping load test", str(e))
        return

    for filename, role, expected_task in REQUIRED_MODELS:
        path = os.path.join(models_dir, filename)
        label = f"{filename}"
        if not os.path.isfile(path):
            if filename in CAPABILITY_ONLY_MODELS:
                _record("FAIL",
                        f"{label} -- NOT FOUND: top-camera classification "
                        f"capability unavailable",
                        f"{path}\n           Needed for RIGHT_UP_TOP and "
                        f"LEFT_UP_TOP classification. No other model is "
                        f"substituted.\n           The wagon count is unaffected "
                        f"(RIGHT_UP is the only counting authority).\n"
                        f"           Place it with e.g.:\n"
                        f"             aws s3 cp s3://<bucket>/{filename} "
                        f"{os.path.join(models_dir, filename)}")
            else:
                _record("FAIL", f"{label} -- not found, cannot load", path)
            continue
        try:
            model = YOLO(path)
            task = getattr(model, "task", None)
            names = getattr(model, "names", {}) or {}
            n = len(names)
            detail = f"task={task}  classes={n}  names={dict(names)}"

            if expected_task is not None and task != expected_task:
                _record("WARN", f"{label} loaded, but task is '{task}' "
                                f"(expected '{expected_task}' for {role})", detail)
            else:
                _record("PASS", f"{label} loaded  ({role})", detail)

            # For classification models, report the semantic mapping derived from
            # the model's REAL class names. Indices are never assumed, and an
            # unrecognised class is never silently treated as a WAGON.
            if "classification" in filename:
                try:
                    from train_structure import build_label_mapping
                    lm = build_label_mapping(names, path)
                    _record("INFO", f"{label} semantic mapping",
                            "  ".join(f"{k!r}->{v}" for k, v in lm.mapping.items()))
                    if lm.unmapped:
                        _record("WARN",
                                f"{label} has {len(lm.unmapped)} unexpected class "
                                f"name(s); each maps to UNKNOWN, never to WAGON",
                                f"unexpected: {lm.unmapped}")
                except Exception as e:                  # pragma: no cover
                    _record("WARN", f"{label} mapping could not be derived", str(e))
        except Exception as e:
            _record("FAIL", f"{label} -- failed to load",
                    f"{type(e).__name__}: {e}")
            if os.environ.get("VALIDATE_VERBOSE"):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# 7. Write access
# ---------------------------------------------------------------------------

def check_write_access(output_dir: str) -> None:
    _section("7. Output filesystem write access")
    probe = os.path.join(output_dir, ".validate_write_probe")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        _record("PASS", "output directory is writable", os.path.abspath(output_dir))
    except Exception as e:
        _record("FAIL", "cannot write to output directory",
                f"{os.path.abspath(output_dir)}: {e}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(
        prog="validate_ec2.py",
        description="Pre-flight environment / asset validation for the "
                    "standalone Global Wagon Count project. Does NOT run the pipeline.",
    )
    p.add_argument("--inputs-dir", default=os.path.join(here, "inputs"),
                   help="Directory holding the 4 input videos (default: ./inputs)")
    p.add_argument("--models-dir", default=os.path.join(here, "models"),
                   help="Directory holding the 4 .pt models (default: ./models)")
    p.add_argument("--output", "-o", default=os.path.join(here, "results"),
                   help="Output directory to test for write access (default: ./results)")
    p.add_argument("--skip-models", action="store_true",
                   help="Skip the (slower) ultralytics model-loading test")
    args = p.parse_args(argv)

    print("=" * 70)
    print("  GLOBAL WAGON COUNT -- ENVIRONMENT VALIDATION")
    print("=" * 70)
    print(f"  project root : {here}")
    print(f"  inputs dir   : {args.inputs_dir}")
    print(f"  models dir   : {args.models_dir}")
    print(f"  output dir   : {args.output}")
    print(f"  platform     : {sys.platform}")

    check_python()
    check_packages()
    check_dirs(args.inputs_dir, args.models_dir, args.output)
    check_videos(args.inputs_dir)
    check_models(args.models_dir)
    if args.skip_models:
        _section("6b. Model loading (ultralytics YOLO)")
        _record("WARN", "Model loading test skipped (--skip-models)")
    else:
        check_model_loading(args.models_dir)
    check_write_access(args.output)

    failed = [r for r in _results if r[0] == "FAIL"]
    warned = [r for r in _results if r[0] == "WARN"]
    passed = [r for r in _results if r[0] == "PASS"]

    print()
    print("=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  PASS : {len(passed)}")
    print(f"  WARN : {len(warned)}")
    print(f"  FAIL : {len(failed)}")
    if warned:
        print()
        print("  Warnings:")
        for _, name, _detail in warned:
            print(f"    - {name}")
    if failed:
        print()
        print("  Failures:")
        for _, name, detail in failed:
            print(f"    - {name}")
            if detail:
                print(f"        {detail}")
        print()
        print("  RESULT: FAIL -- fix the items above before running "
              "run_global_count.py")
        print("=" * 70)
        return 1

    print()
    print("  RESULT: PASS -- this machine is ready. Run:")
    print("      python run_global_count.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
