#!/usr/bin/env bash
#
# Fast single-token smoke test for OPT-125M on an *emulated* MI350X.
#
# Unlike `run_vllm_opt125m.sh` (which spins up the full vLLM OpenAI
# server and can take hours), this runs the lightweight
# `tests/fixtures/ml/opt125_onetoken.py` fixture: it loads
# `facebook/opt-125m` with eager attention and runs a *single* forward
# pass / one-token prediction on the rocjitsu functional emulator. In
# practice it completes in a couple of minutes.
#
# It runs the fixture inside the upstream vLLM container image (which
# already carries a ROCm torch + transformers stack) via `rocjitsu run`,
# so we don't have to provision torch on the host. `rocjitsu run` creates
# a transient session, runs the command, streams its output, and tears
# the session down afterwards.
#
# How this maps onto rocjitsu's CLI:
#   * `--image vllm/vllm-openai-rocm:latest` -> run the node inside that
#     container image.
#   * `--mount <fixtures>:/mnt/fixtures:ro` -> exposes the fixture file
#     inside the container.
#   * `--mount $HF_CACHE:/root/.cache/huggingface` -> caches the model
#     weights on the host so they survive across runs.
#   * `--env PYTORCH_*` -> surface the kernel dispatch trace.
#
# To watch the kernels being dispatched live from another terminal,
# list the transient session while it runs and attach to it:
#
#       rocjitsu session list
#       rocjitsu exec list <session>
#       rocjitsu attach <session> <exec>
#
# Like run_vllm_opt125m.sh, this launches rocjitsu through
# `scripts/rocjitsu.sh` rather than `cargo run`: that wrapper builds (on
# demand) and runs the glibc-portable, manylinux-built rocjitsu. rocjitsu
# bind-mounts its own binary (and the rocjitsu KMD interposer) into the
# workload container, and a host-built binary links a newer glibc than
# the vLLM image carries (`GLIBC_2.39 not found`). The manylinux build
# links an old, broadly-compatible glibc.
#
# Env knobs:
#   PROFILE      rocjitsu profile (default: mi350x)
#   IMAGE        container image (default: vllm/vllm-openai-rocm:latest)
#   HF_CACHE     host HuggingFace cache (default: ~/.cache/huggingface)
#   PYTHON       python interpreter inside the container (default: python3)
#   PROVIDER     container provider binary (default: autodetect docker/podman)
#   LOG_KERNELS  when 1, log every kernel execution from BOTH torch/HIP and
#                the rocjitsu emulator to the workload's stderr (default: 0)
#   (plus everything honoured by scripts/rocjitsu.sh, e.g. ROCJITSU_REBUILD)
#
# Kernel-execution logging (LOG_KERNELS=1) surfaces two independent traces:
#   * torch/HIP dispatch  -> AMD_LOG_LEVEL=3 makes the ROCm runtime print every
#     HIP kernel launch; PYTORCH_JIT_LOG_LEVEL / PYTORCH_SHOW_DISPATCH_TRACE add
#     the aten dispatch trace on instrumented torch builds.
#   * rocjitsu emulator   -> `--plugin logging` adds the logging plugin to the
#     resolved profile config; its default stderr sink streams each emulated
#     kernel to the workload's stderr.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCJITSU_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PROFILE="${PROFILE:-mi350x}"
IMAGE="${IMAGE:-docker.io/vllm/vllm-openai-rocm:latest}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PYTHON="${PYTHON:-python3}"
LOG_KERNELS="${LOG_KERNELS:-0}"

# Build the list of arguments that enable kernel-execution logging. Only
# populated when LOG_KERNELS=1 so the default run stays quiet.
KERNEL_LOG_ARGS=()
if [[ "$LOG_KERNELS" == "1" ]]; then
  KERNEL_LOG_ARGS+=(
    # torch / HIP kernel dispatch trace.
    --env AMD_LOG_LEVEL=3
    --env PYTORCH_JIT_LOG_LEVEL=kernels
    --env PYTORCH_SHOW_DISPATCH_TRACE=1
    # rocjitsu kernel logging, merged into the runtime config by rocjitsu.
    --plugin logging
  )
fi

# Launch rocjitsu through the convenience wrapper, which builds the
# glibc-portable manylinux prefix on demand and execs the installed
# binary.
ROCJITSU="$ROCJITSU_DIR/scripts/rocjitsu.sh"

# The fixture lives here on the host; we mount its directory into the
# container at this path and run it from there.
FIXTURE_DIR="$ROCJITSU_DIR/tests/fixtures/ml"
FIXTURE_NAME="opt125_onetoken.py"
CONTAINER_FIXTURE_DIR="/mnt/fixtures"
CONTAINER_FIXTURE="$CONTAINER_FIXTURE_DIR/$FIXTURE_NAME"

log()  { printf '==> %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

[[ -f "$FIXTURE_DIR/$FIXTURE_NAME" ]] || fail "fixture not found: $FIXTURE_DIR/$FIXTURE_NAME"
[[ -x "$ROCJITSU" ]] || fail "rocjitsu wrapper not found: $ROCJITSU"
log "using rocjitsu wrapper: $ROCJITSU"

# Make sure the HuggingFace cache directory exists on the host so the
# weights survive across runs.
mkdir -p "$HF_CACHE"
log "caching model weights in $HF_CACHE"

# ---------------------------------------------------------------------------
# Run the fixture through `rocjitsu run`: it spins up a transient session in
# the vLLM container, runs the one-token fixture, streams its output, and
# tears the session down when done. The fixture prints its own
# PASS/FAIL verdict; we surface rocjitsu's exit code.
# ---------------------------------------------------------------------------
log "running $FIXTURE_NAME via rocjitsu run (profile=$PROFILE, image=$IMAGE)"
if [[ "$LOG_KERNELS" == "1" ]]; then
  log "kernel-execution logging enabled (torch/HIP + rocjitsu)"
  export RJ_LOG_GROUPS="CP"  # enable all rocjitsu log groups for the fixture run
fi
cd "$ROCJITSU_DIR"
"$ROCJITSU" run \
  --in-process \
  --profile "$PROFILE" \
  --image "$IMAGE" \
  --mount "$FIXTURE_DIR:$CONTAINER_FIXTURE_DIR:ro" \
  --mount "$HF_CACHE:/root/.cache/huggingface" \
  "${KERNEL_LOG_ARGS[@]}" \
  -- "$PYTHON" "$CONTAINER_FIXTURE"
