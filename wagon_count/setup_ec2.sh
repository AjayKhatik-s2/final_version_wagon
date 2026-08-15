#!/usr/bin/env bash
# =============================================================================
# setup_ec2.sh -- prepare an Ubuntu/Linux EC2 instance to run the standalone
#                 Global Wagon Count project.
#
# WHAT THIS DOES
#   1. Verifies it is running on Linux
#   2. Verifies a suitable python3 (>= 3.10) is available
#   3. Creates a virtual environment at ./.venv  (reused if it already exists)
#   4. Installs the pinned dependencies from requirements.txt into it
#   5. Creates ./inputs ./models ./results if missing
#   6. Runs an import/version check on the installed stack
#   7. Reports whether the machine is ready
#
# WHAT THIS DELIBERATELY DOES NOT DO
#   - It never deletes or overwrites inputs/, models/, results/ or any file
#     inside them.
#   - It never removes or recreates an existing virtual environment.
#   - It never downloads model weights or videos from anywhere.
#   - It never touches the wagon-counting code or its algorithm.
#   - It does not run the pipeline. Use validate_ec2.py, then
#     run_global_count.py.
#
# USAGE
#   chmod +x setup_ec2.sh
#   ./setup_ec2.sh                 # CPU install (default)
#   ./setup_ec2.sh --with-apt-deps # also apt-get the OpenCV system libraries
#                                  # (needs sudo; see README for the GPU path)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${PROJECT_ROOT}/.venv"
INSTALL_APT_DEPS=0
MIN_PY_MAJOR=3
MIN_PY_MINOR=10

for arg in "$@"; do
    case "$arg" in
        --with-apt-deps) INSTALL_APT_DEPS=1 ;;
        -h|--help)
            sed -n '2,30p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg  (try --help)" >&2
            exit 2
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Pretty output helpers
# ---------------------------------------------------------------------------
step()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()    { printf '  \033[0;32m[OK]\033[0m   %s\n' "$*"; }
warn()  { printf '  \033[0;33m[WARN]\033[0m %s\n' "$*"; }
fail()  { printf '  \033[0;31m[FAIL]\033[0m %s\n' "$*" >&2; }
die()   { fail "$*"; exit 1; }

echo "======================================================================"
echo "  GLOBAL WAGON COUNT -- EC2 / LINUX SETUP"
echo "======================================================================"
echo "  project root : ${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# 1. Verify Linux
# ---------------------------------------------------------------------------
step "1. Verifying operating system"
UNAME_S="$(uname -s)"
if [ "$UNAME_S" != "Linux" ]; then
    die "This script targets Linux (EC2). Detected: ${UNAME_S}.
       On Windows/macOS use a normal virtualenv instead:
         python -m venv .venv && pip install -r requirements.txt"
fi
ok "Linux detected ($(uname -sr))"
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    ok "Distribution: ${PRETTY_NAME:-unknown}"
    case "${ID:-}" in
        ubuntu|debian) : ;;
        amzn)   warn "Amazon Linux detected. Use 'sudo yum install -y mesa-libGL glib2' instead of the apt packages." ;;
        *)      warn "Untested distribution '${ID:-unknown}'. Setup should still work if python3 >= ${MIN_PY_MAJOR}.${MIN_PY_MINOR} is present." ;;
    esac
fi

# ---------------------------------------------------------------------------
# 2. Verify Python
# ---------------------------------------------------------------------------
step "2. Verifying Python interpreter"
PY_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (${MIN_PY_MAJOR}, ${MIN_PY_MINOR}) else 1)" 2>/dev/null; then
            PY_BIN="$candidate"
            break
        fi
    fi
done
[ -n "$PY_BIN" ] || die "No python3 >= ${MIN_PY_MAJOR}.${MIN_PY_MINOR} found.
       On Ubuntu:  sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip"
ok "Using $(command -v "$PY_BIN")  ($("$PY_BIN" --version 2>&1))"

# ---------------------------------------------------------------------------
# 2b. Optional: OpenCV system libraries
# ---------------------------------------------------------------------------
if [ "$INSTALL_APT_DEPS" -eq 1 ]; then
    step "2b. Installing OpenCV system libraries (apt)"
    if ! command -v sudo >/dev/null 2>&1 && [ "$(id -u)" -ne 0 ]; then
        warn "sudo not available and not running as root -- skipping apt install."
    else
        SUDO=""
        [ "$(id -u)" -ne 0 ] && SUDO="sudo"
        $SUDO apt-get update -y
        # libgl1 + libglib2.0-0 are what opencv-python links against on a
        # headless server. ffmpeg libs come bundled inside the opencv wheel.
        $SUDO apt-get install -y --no-install-recommends \
            python3-venv python3-pip libgl1 libglib2.0-0
        ok "System libraries installed"
    fi
else
    step "2b. OpenCV system libraries"
    warn "Skipped (no --with-apt-deps). If 'import cv2' later fails with"
    warn "  'libGL.so.1: cannot open shared object file', run:"
    warn "    sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0"
fi

# ---------------------------------------------------------------------------
# 3 + 4. Virtual environment
# ---------------------------------------------------------------------------
step "3. Preparing virtual environment"
if [ -d "$VENV_DIR" ] && [ -x "${VENV_DIR}/bin/python" ]; then
    ok "Reusing existing virtual environment at ${VENV_DIR} (not recreated)"
else
    if [ -e "$VENV_DIR" ] && [ ! -x "${VENV_DIR}/bin/python" ]; then
        die "${VENV_DIR} exists but is not a usable virtualenv.
       Inspect it yourself and remove it manually if you are sure --
       this script will not delete anything."
    fi
    echo "  creating ${VENV_DIR} ..."
    "$PY_BIN" -m venv "$VENV_DIR" || die "venv creation failed.
       On Ubuntu install the venv module first:
         sudo apt-get install -y python3-venv"
    ok "Created ${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"
ok "Interpreter: $("$VENV_PY" --version 2>&1)"

# ---------------------------------------------------------------------------
# 5. Install dependencies
# ---------------------------------------------------------------------------
step "4. Installing dependencies from requirements.txt"
[ -f "${PROJECT_ROOT}/requirements.txt" ] || die "requirements.txt not found in ${PROJECT_ROOT}"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
"$VENV_PY" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
ok "Dependencies installed"

# ---------------------------------------------------------------------------
# 6. Runtime directories  (created only if absent; never cleared)
# ---------------------------------------------------------------------------
step "5. Ensuring runtime directories exist"
for d in inputs models results; do
    if [ -d "${PROJECT_ROOT}/${d}" ]; then
        ok "${d}/ already exists (left untouched)"
    else
        mkdir -p "${PROJECT_ROOT}/${d}"
        ok "${d}/ created"
    fi
done

# ---------------------------------------------------------------------------
# 7. Dependency check
# ---------------------------------------------------------------------------
step "6. Verifying the installed stack"
"$VENV_PY" - <<'PYCHECK'
import sys
print(f"  python       : {sys.version.split()[0]}")
failed = []
for mod, attr in (("numpy", "__version__"),
                  ("cv2", "__version__"),
                  ("torch", "__version__"),
                  ("ultralytics", "__version__")):
    try:
        m = __import__(mod)
        print(f"  {mod:<13}: {getattr(m, attr, '?')}")
    except Exception as e:
        failed.append(f"{mod}: {type(e).__name__}: {e}")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  cuda         : YES ({torch.cuda.device_count()} device(s), "
              f"{torch.cuda.get_device_name(0)})")
    else:
        print("  cuda         : no GPU visible -- inference will run on CPU")
except Exception:
    pass
if failed:
    print("\n  IMPORT FAILURES:")
    for f in failed:
        print(f"    - {f}")
    sys.exit(1)
PYCHECK
ok "All required packages import cleanly"

# ---------------------------------------------------------------------------
# 8. Report readiness
# ---------------------------------------------------------------------------
step "7. Readiness report"

missing=0
for f in right_up.mp4 left_up.mp4 right_up_top.mp4 left_up_top.mp4; do
    if [ -f "${PROJECT_ROOT}/inputs/${f}" ]; then
        ok "inputs/${f} present"
    else
        warn "inputs/${f} MISSING -- copy it onto this instance"
        missing=$((missing + 1))
    fi
done
for f in right_up_wagon_gap.pt left_up_wagon_gap.pt top_gap.pt side_classification.pt; do
    if [ -f "${PROJECT_ROOT}/models/${f}" ]; then
        ok "models/${f} present"
    else
        warn "models/${f} MISSING -- copy it onto this instance"
        missing=$((missing + 1))
    fi
done

echo
echo "======================================================================"
if [ "$missing" -eq 0 ]; then
    echo "  ENVIRONMENT READY -- all 8 runtime assets are in place."
else
    echo "  ENVIRONMENT READY, RUNTIME ASSETS INCOMPLETE (${missing} missing)."
    echo "  The videos and model weights are intentionally NOT stored in Git."
    echo "  Copy them in (see README -> 'Runtime assets') before running."
fi
echo "======================================================================"
echo "  Next steps:"
echo
echo "    source .venv/bin/activate"
echo "    python validate_ec2.py"
echo "    python run_global_count.py"
echo
echo "  This project requires NO environment variables."
echo "======================================================================"
