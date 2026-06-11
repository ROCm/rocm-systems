#!/usr/bin/env bash
#
# Run tests/fixtures/ml/tiny_torch.py on an *emulated* MI350X through
# mirage, using a Python venv populated from the gfx950-dcgpu (cdna4 /
# MI350) ROCm nightly. No physical GPU is required: rocjitsu emulates
# the device.
#
# What this does:
#   1. Create/reuse a venv.
#   2. Install torch + the ROCm SDK (which ships rocjitsu) from the
#      gfx950-dcgpu nightly index.
#   4. Run tiny_torch.py via `mirage run --profile rocjitsu-MI350X` with
#      the venv's python3, and assert it prints `tiny_torch_ok`.
#
# Env knobs:
#   VENV          venv location (default: <mirage>/.venv-mi350)
#   INDEX_URL     pip index (default: gfx950-dcgpu nightly)
#   SKIP_INSTALL  set to 1 to reuse an already-populated venv
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIRAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FIXTURE="$MIRAGE_DIR/tests/fixtures/ml/tiny_torch.py"
PROFILE="rocjitsu-MI350X"

VENV="${VENV:-$MIRAGE_DIR/.venv-mi350}"
INDEX_URL="${INDEX_URL:-https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/}"

log() { printf '==> %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

[[ -f "$FIXTURE" ]] || fail "fixture not found: $FIXTURE"

# 1. Create the venv (idempotent).
if [[ ! -x "$VENV/bin/python3" ]]; then
  log "creating venv at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
VENV_PY="$VENV/bin/python3"

# 2. Install the gfx950-dcgpu nightly (torch + ROCm SDK, which carries rocjitsu).
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  log "installing torch + ROCm SDK from $INDEX_URL"
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install --index-url "$INDEX_URL" "rocm[libraries,devel]" torch numpy
else
  log "SKIP_INSTALL=1: reusing existing venv packages"
fi

# 2b. Expand the devel package. rocm[devel] ships its contents (headers,
#     libs incl. librocjitsu_kmd.so) compressed; `rocm-sdk init` unpacks
#     them into site-packages/_rocm_sdk_devel. Safe to re-run.
if [[ -x "$VENV/bin/rocm-sdk" ]]; then
  log "expanding ROCm devel contents (rocm-sdk init)"
  "$VENV/bin/rocm-sdk" init
fi

# 3. Run the fixture through mirage on the emulated MI350X. Use the venv's
#    python3 explicitly so the workload imports the nightly torch. Build &
#    run mirage from the workspace (cargo run) so its own rocjitsu KMD
#    discovery and preload wiring work; the first `--` ends cargo's args.
cd "$MIRAGE_DIR"
log "running tiny_torch.py via mirage --profile $PROFILE"
set +e
OUTPUT="$(cargo run --quiet -- run --profile "$PROFILE" \
  -- "$VENV_PY" "$FIXTURE" 2>&1)"
STATUS=$?
set -e
printf '%s\n' "$OUTPUT"

# 5. Assert success.
[[ $STATUS -eq 0 ]] || fail "mirage run exited with status $STATUS"
grep -q 'tiny_torch_ok' <<<"$OUTPUT" || fail "tiny_torch.py did not report success"
log "PASS: tiny_torch ran on emulated $PROFILE using rocjitsu from the venv"
