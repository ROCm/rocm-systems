#!/usr/bin/env bash
#
# Train a tiny MLP with PyTorch DistributedDataParallel (DDP) on *emulated*
# MI350X GPUs through mirage, launched with `torchrun`. No physical GPU is
# required: rocjitsu emulates the devices.
#
# How it maps onto mirage:
#   * `mirage run --gpus-per-node N` makes the emulated node expose N GPUs
#     (rocjitsu synthesizes N KFD device nodes), so `torch.cuda.device_count()`
#     is N inside the workload.
#   * mirage exports `MASTER_ADDR`/`MASTER_PORT` on every node, so
#     `torchrun`'s rendezvous (and `torch.distributed`'s `env://`) work with no
#     extra wiring. For a single node, `torchrun --standalone` is enough.
#   * mirage runs the emulator as a separate daemon process by default, so the
#     rank processes share GPU memory through it. Pass `mirage run --in-process`
#     to give each process its own in-process emulator (no shared GPU memory).
#   * The fixture `ddp_mlp.py` is launched once per rank by torchrun; each rank
#     pins itself to GPU `LOCAL_RANK`, wraps an MLP in DDP, and trains. Gradients
#     are all-reduced with NCCL (RCCL on ROCm).
#
# What this does:
#   1. Create/reuse a venv populated from the gfx950-dcgpu (MI350) ROCm nightly.
#   2. `mirage run --gpus-per-node $NPROC -- torchrun --standalone
#      --nproc_per_node=$NPROC ddp_mlp.py`, asserting rank 0 prints
#      `ddp_mlp_ok`.
#
# Env knobs:
#   VENV          venv location (default: <mirage>/.venv-mi350)
#   INDEX_URL     pip index (default: gfx950-dcgpu nightly)
#   SKIP_INSTALL  set to 1 to reuse an already-populated venv
#   PROFILE       mirage profile (default: mi350x)
#   NPROC         GPUs / ranks per node, i.e. torchrun --nproc_per_node (default: 2)
#   STEPS         optimizer steps the MLP trains for (default: 50)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIRAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FIXTURE="$MIRAGE_DIR/tests/fixtures/ml/ddp_mlp.py"

PROFILE="${PROFILE:-mi350x}"
NPROC="${NPROC:-2}"
STEPS="${STEPS:-50}"
VENV="${VENV:-$MIRAGE_DIR/.venv-mi350}"
INDEX_URL="${INDEX_URL:-https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/}"

log()  { printf '==> %s\n' "$*"; }
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
TORCHRUN="$VENV/bin/torchrun"

# 2. Install the gfx950-dcgpu nightly (torch + ROCm SDK, which carries rocjitsu).
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  log "installing torch + ROCm SDK from $INDEX_URL"
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install --index-url "$INDEX_URL" "rocm[libraries,devel]" torch numpy
else
  log "SKIP_INSTALL=1: reusing existing venv packages"
fi

# 2b. Expand the devel package so librocjitsu.so + configs appear under
#     site-packages/_rocm_sdk_devel. Safe to re-run.
if [[ -x "$VENV/bin/rocm-sdk" ]]; then
  log "expanding ROCm devel contents (rocm-sdk init)"
  "$VENV/bin/rocm-sdk" init
fi

[[ -x "$TORCHRUN" ]] || fail "torchrun not found in venv: $TORCHRUN"

# 3. Run DDP training through mirage on the emulated node. Build & run mirage
#    from the workspace (cargo run) so its rocjitsu KMD discovery/preload work;
#    the first `--` ends cargo's args, the second ends mirage's.
cd "$MIRAGE_DIR"
log "training MLP with DDP via torchrun --nproc_per_node=$NPROC on emulated $PROFILE (nccl)"
set +e
OUTPUT="$(cargo run --quiet -- run \
  --profile "$PROFILE" \
  --gpus-per-node "$NPROC" \
  --env "DDP_STEPS=$STEPS" \
  -- "$TORCHRUN" --standalone --nproc_per_node="$NPROC" "$FIXTURE" 2>&1)"
STATUS=$?
set -e
printf '%s\n' "$OUTPUT"

# 4. Assert success.
[[ $STATUS -eq 0 ]] || fail "mirage run exited with status $STATUS"
grep -q 'ddp_mlp_ok' <<<"$OUTPUT" || fail "ddp_mlp.py did not report success"
log "PASS: DDP MLP trained on $NPROC emulated $PROFILE GPUs via torchrun"
