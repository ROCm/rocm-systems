#!/usr/bin/env bash
#
# Create a Python venv for running aorta workloads through an *emulated*
# MI350X via mirage. No physical GPU is required: rocjitsu emulates the
# device.
#
# What this does:
#   1. Create/reuse a venv.
#   2. Install torch + the ROCm SDK (which ships rocjitsu) from the
#      gfx950-dcgpu nightly index, plus numpy.
#   3. Install aorta from GitHub (pinned commit by default).
#   4. Run `rocm-sdk init` so the devel package's compressed contents
#      (headers, libs incl. librocjitsu_kmd.so) are unpacked.
#
# After this you can run, e.g.:
#   source <venv>/bin/activate
#   mirage profile create --num-nodes 1 --gpus-per-node 2 --agent MI350X double
#   mirage run --daemon --profile double -- \
#     torchrun --standalone --nproc_per_node=2 "$(which aorta)" \
#     triage run --recipe tests/aorta.yml
#
# Env knobs:
#   VENV          venv location (default: <mirage>/.venv)
#   INDEX_URL     pip index (default: gfx950-dcgpu nightly)
#   TORCH_VER     pinned torch version (default below)
#   ROCM_VER      pinned ROCm SDK version (default below)
#   NUMPY_VER     pinned numpy version (default below)
#   AORTA_REF     aorta git ref to install (default: pinned commit below)
#   SKIP_INSTALL  set to 1 to reuse an already-populated venv
#
# NOTE: torch/rocm versions are PINNED. Leaving them unpinned makes pip's
# dependency resolver backtrack through hundreds of multi-hundred-MB
# wheels (the meta `rocm` package and `torch` drift to different SDK
# builds) and effectively never finishes.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIRAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV="${VENV:-$MIRAGE_DIR/.venv}"
INDEX_URL="${INDEX_URL:-https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/}"
TORCH_VER="${TORCH_VER:-2.11.0+rocm7.13.0a20260426}"
ROCM_VER="${ROCM_VER:-7.13.0a20260426}"
NUMPY_VER="${NUMPY_VER:-2.4.4}"
AORTA_REF="${AORTA_REF:-2756fa149d5dacd63d48b05e7202e28951348c19}"

log() { printf '==> %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

# 1. Create the venv (idempotent).
if [[ ! -x "$VENV/bin/python3" ]]; then
  log "creating venv at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
VENV_PY="$VENV/bin/python3"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  # 2. Install the gfx950-dcgpu nightly (torch + ROCm SDK, which carries
  #    rocjitsu) plus numpy.
  log "upgrading pip"
  "$VENV_PY" -m pip install --upgrade pip

  log "installing torch + ROCm SDK + numpy from $INDEX_URL"
  "$VENV_PY" -m pip install --index-url "$INDEX_URL" \
    "torch==${TORCH_VER}" "numpy==${NUMPY_VER}" \
    "rocm[libraries,devel]==${ROCM_VER}"

  # 3. Install aorta from GitHub.
  log "installing aorta @ $AORTA_REF"
  "$VENV_PY" -m pip install "git+https://github.com/ROCm/aorta.git@${AORTA_REF}"
else
  log "SKIP_INSTALL=1: reusing existing venv packages"
fi

# 4. Expand the devel package. rocm[devel] ships its contents (headers,
#    libs incl. librocjitsu_kmd.so) compressed; `rocm-sdk init` unpacks
#    them into site-packages/_rocm_sdk_devel. Safe to re-run.
if [[ -x "$VENV/bin/rocm-sdk" ]]; then
  log "expanding ROCm devel contents (rocm-sdk init)"
  "$VENV/bin/rocm-sdk" init
fi

# Sanity checks.
log "verifying imports"
"$VENV_PY" -c "import torch, numpy; print('torch', torch.__version__); print('numpy', numpy.__version__)" \
  || fail "torch/numpy import failed"
"$VENV/bin/aorta" --help >/dev/null 2>&1 \
  || log "warning: 'aorta --help' did not succeed (check install)"

log "PASS: aorta venv ready at $VENV"
log "activate with: source $VENV/bin/activate"
