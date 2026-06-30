#!/usr/bin/env bash
#
# Smoke-test serving vLLM through mirage on an *emulated* MI350X.
#
# It launches the upstream `vllm/vllm-openai-rocm:latest` image via
# `mirage run --daemon`, serves `facebook/opt-125m`, and then exercises
# the OpenAI-compatible API with a single text completion ("a
# generation").
#
# Notes on how this maps onto mirage's CLI:
#   * `--image vllm/vllm-openai-rocm:latest`  -> run the node inside that
#     container image.
#   * `--mount $HF_CACHE:/root/.cache/huggingface`  -> becomes the
#     container `-v $HF_CACHE:/root/.cache/huggingface` bind mount, so the
#     model weights are cached on the host and reused across runs.
#   * `--port $PORT:$PORT`  -> publishes the container port on the host
#     (docker `-p`), so the server is reachable at `localhost:$PORT`.
#   * `--daemon`  -> run the rocjitsu emulator in out-of-process daemon
#     mode (as requested).
#
# This uses the manylinux-built mirage from
# `scripts/mirage-docker-build.sh` rather than `cargo run`. mirage
# bind-mounts its own binary (and the rocjitsu KMD interposer) into the
# vLLM workload container; a binary built on a modern host links a newer
# glibc than the vLLM image carries and fails with `GLIBC_2.39 not
# found`. The manylinux build links an old, broadly-compatible glibc so
# it runs inside the vLLM container without the `--hack` workaround. The
# script builds it on demand when missing.
#
# Env knobs:
#   PROFILE     mirage profile (default: mi350x)
#   IMAGE       container image (default: vllm/vllm-openai-rocm:latest)
#   MODEL       model to serve (default: facebook/opt-125m)
#   HF_CACHE    host HuggingFace cache (default: ~/.cache/huggingface)
#   PORT        in-container server port (default: 8000)
#   PROVIDER    container provider binary (default: autodetect docker/podman)
#   READY_TIMEOUT  seconds to wait for the server (default: 600)
#   MIRAGE_BIN  manylinux mirage binary (default: <mirage>/build/manylinux/bin/mirage)
#   SKIP_BUILD  set to 1 to reuse an existing MIRAGE_BIN without rebuilding
#   GPU_MEM_UTIL    vLLM --gpu-memory-utilization (default: 0.2)
#   MAX_MODEL_LEN   vLLM --max-model-len (default: 2048)
#   VLLM_EXTRA_ARGS extra args appended verbatim to `vllm serve`
#   LOG_KERNELS     when 1, log every kernel execution from BOTH torch/HIP and
#                   the rocjitsu emulator to the server log (default: 0)
#
# Kernel-execution logging (LOG_KERNELS=1) surfaces two independent traces in
# the vLLM server log ($RUN_LOG):
#   * torch/HIP dispatch  -> AMD_LOG_LEVEL=3 makes the ROCm runtime print every
#     HIP kernel launch; PYTORCH_JIT_LOG_LEVEL / PYTORCH_SHOW_DISPATCH_TRACE add
#     the aten dispatch trace on instrumented torch builds.
#   * rocjitsu emulator   -> RJ_LOG=1 enables the interposer's kernel-logging
#     plugin and RJ_SINKS=stderr streams each emulated kernel to stderr.
#
# Why the vLLM init is constrained: the emulated MI350X advertises ~288
# GiB of VRAM, and vLLM's engine-core init runs a memory-profiling
# forward pass that by default grabs 90% of it and captures HIP graphs.
# That stresses the functional emulator into a GPU memory access fault
# ("Engine core initialization failed" / "Memory critical error by
# agent"). `--enforce-eager`, a small `--gpu-memory-utilization`, a
# short `--max-model-len`, and `--max-num-seqs 1` keep init light enough
# for the emulator to service.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIRAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PROFILE="${PROFILE:-mi350x}"
IMAGE="${IMAGE:-docker.io/vllm/vllm-openai-rocm:latest}"
MODEL="${MODEL:-facebook/opt-125m}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
MIRAGE_BIN="${MIRAGE_BIN:-$MIRAGE_DIR/build/manylinux/bin/mirage}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
LOG_KERNELS="${LOG_KERNELS:-0}"

# Build the list of --env flags that enable kernel-execution logging. Only
# populated when LOG_KERNELS=1 so the default run stays quiet.
KERNEL_LOG_ENV=()
if [[ "$LOG_KERNELS" == "1" ]]; then
  KERNEL_LOG_ENV+=(
    # torch / HIP kernel dispatch trace.
    --env AMD_LOG_LEVEL=3
    --env PYTORCH_JIT_LOG_LEVEL=kernels
    --env PYTORCH_SHOW_DISPATCH_TRACE=1
    --env RJ_LOG=1
  )
fi

log()  { printf '==> %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

# Build the glibc-compatible mirage binary in the manylinux container
# unless it already exists (or SKIP_BUILD=1 forces reuse).
if [[ "${SKIP_BUILD:-0}" != "1" && ! -x "$MIRAGE_BIN" ]]; then
  log "building manylinux mirage via scripts/mirage-docker-build.sh"
  "$MIRAGE_DIR/scripts/mirage-docker-build.sh" "$MIRAGE_DIR/build/manylinux"
fi
[[ -x "$MIRAGE_BIN" ]] || fail "mirage binary not found: $MIRAGE_BIN (run scripts/mirage-docker-build.sh)"
log "using mirage binary: $MIRAGE_BIN"

# Pick a container provider so we can inspect the launched node container.
if [[ -n "${PROVIDER:-}" ]]; then
  :
elif command -v docker >/dev/null 2>&1; then
  PROVIDER="docker"
elif command -v podman >/dev/null 2>&1; then
  PROVIDER="podman"
else
  fail "no container provider found (need docker or podman)"
fi
log "using container provider: $PROVIDER"

# Make sure the HuggingFace cache directory exists on the host so the
# weights survive across runs.
mkdir -p "$HF_CACHE"
log "caching model weights in $HF_CACHE"

# ---------------------------------------------------------------------------
# Launch the server through mirage in the background. mirage execs the
# given argv inside the node container, so we invoke vLLM's `serve` CLI
# directly (the image ENTRYPOINT does not apply to `exec`ed commands).
# Bind to 0.0.0.0 so the port is reachable over the session network.
#
# The serve flags keep engine-core init light enough for the functional
# emulator (see header): eager mode (no HIP graph capture), a small
# fraction of the emulated VRAM, a short context, and a single sequence.
# ---------------------------------------------------------------------------
RUN_LOG="$(mktemp -t mirage-vllm.XXXXXX.log)"
log "starting vLLM (model=$MODEL) via mirage run --daemon"
if [[ "$LOG_KERNELS" == "1" ]]; then
  log "kernel-execution logging enabled (torch/HIP + rocjitsu) -> $RUN_LOG"
  export RJ_LOG_GROUPS="CP"  # enable all rocjitsu log groups for the fixture run
fi
cd "$MIRAGE_DIR"
"$MIRAGE_BIN" run \
  --daemon \
  --profile "$PROFILE" \
  --image "$IMAGE" \
  --container-provider "$PROVIDER" \
  --port "$PORT:$PORT" \
  --mount "$HF_CACHE:/root/.cache/huggingface" \
  "${KERNEL_LOG_ENV[@]}" \
  -- vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
       --enforce-eager \
       --gpu-memory-utilization "$GPU_MEM_UTIL" \
       --max-model-len "$MAX_MODEL_LEN" \
       --max-num-seqs 1 \
       ${VLLM_EXTRA_ARGS:-} \
  >"$RUN_LOG" 2>&1 &
MIRAGE_PID=$!

CONTAINER=""
cleanup() {
  log "cleaning up"
  # Stop the backgrounded mirage run, which tears down the session and
  # its container/network.
  if kill -0 "$MIRAGE_PID" 2>/dev/null; then
    kill "$MIRAGE_PID" 2>/dev/null || true
    wait "$MIRAGE_PID" 2>/dev/null || true
  fi
  # Fall back to force-removing the node container if it lingers.
  if [[ -n "$CONTAINER" ]]; then
    "$PROVIDER" rm -f "$CONTAINER" >/dev/null 2>&1 || true
  fi
  rm -f "$RUN_LOG"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Wait for mirage to launch the node container (named
# `mirage-<session>-node-0`) and learn its IP on the session network.
# ---------------------------------------------------------------------------
log "waiting for the node container to appear"
DEADLINE=$(( $(date +%s) + READY_TIMEOUT ))
while :; do
  CONTAINER="$("$PROVIDER" ps --filter 'name=mirage-' --filter 'status=running' \
    --format '{{.Names}}' 2>/dev/null | grep -E 'mirage-.*-node-0$' | head -n1 || true)"
  [[ -n "$CONTAINER" ]] && break
  if ! kill -0 "$MIRAGE_PID" 2>/dev/null; then
    cat "$RUN_LOG" >&2 || true
    fail "mirage run exited before the container started"
  fi
  (( $(date +%s) < DEADLINE )) || { cat "$RUN_LOG" >&2; fail "timed out waiting for container"; }
  sleep 2
done
log "node container: $CONTAINER"

CONTAINER_IP="$("$PROVIDER" inspect -f \
  '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$CONTAINER" 2>/dev/null || true)"
log "container IP on session network: ${CONTAINER_IP:-<unknown>}"

# Helper: run curl against the server. The `--port` publish makes the
# server reachable at `localhost:$PORT` on the host; fall back to the
# container IP, then to running curl inside the container.
api_curl() {
  local path="$1"; shift
  if curl -fsS --max-time 5 "http://localhost:$PORT$path" "$@" 2>/dev/null; then
    return 0
  fi
  if [[ -n "$CONTAINER_IP" ]] && \
     curl -fsS --max-time 5 "http://$CONTAINER_IP:$PORT$path" "$@" 2>/dev/null; then
    return 0
  fi
  "$PROVIDER" exec "$CONTAINER" \
    curl -fsS --max-time 5 "http://localhost:$PORT$path" "$@"
}

# ---------------------------------------------------------------------------
# Wait for the OpenAI server to report healthy.
# ---------------------------------------------------------------------------
log "waiting for the vLLM server to become ready (model load can take a while)"
while :; do
  if api_curl /health >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$MIRAGE_PID" 2>/dev/null; then
    cat "$RUN_LOG" >&2 || true
    fail "mirage run exited before the server became ready"
  fi
  (( $(date +%s) < DEADLINE )) || { cat "$RUN_LOG" >&2; fail "timed out waiting for server"; }
  sleep 3
done
log "server is healthy"

# ---------------------------------------------------------------------------
# Test a generation through the OpenAI-compatible completions endpoint.
# ---------------------------------------------------------------------------
log "requesting a completion from $MODEL"
REQ="$(printf '{"model":"%s","prompt":"The capital of France is","max_tokens":6,"temperature":0}' "$MODEL")"
RESP="$(api_curl /v1/completions \
  -H 'Content-Type: application/json' \
  -d "$REQ")" || { cat "$RUN_LOG" >&2; fail "completion request failed"; }

printf 'response: %s\n' "$RESP"
grep -q '"choices"' <<<"$RESP" || fail "no choices in completion response"
log "PASS: vLLM served $MODEL through mirage and returned a generation"
