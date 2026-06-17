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
#   * `--daemon`  -> run the rocjitsu emulator in out-of-process daemon
#     mode (as requested).
#   * mirage has no host port-publish flag. Each node container joins a
#     per-session bridge network (`mirage-<session>`), so the server is
#     reached over that network via the container's IP. If the host
#     cannot route to the bridge (e.g. rootless podman), the script
#     falls back to running the request from inside the container.
#
# Env knobs:
#   PROFILE     mirage profile (default: mi350x)
#   IMAGE       container image (default: vllm/vllm-openai-rocm:latest)
#   MODEL       model to serve (default: facebook/opt-125m)
#   HF_CACHE    host HuggingFace cache (default: ~/.cache/huggingface)
#   PORT        in-container server port (default: 8000)
#   PROVIDER    container provider binary (default: autodetect docker/podman)
#   READY_TIMEOUT  seconds to wait for the server (default: 600)
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

log()  { printf '==> %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

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
# ---------------------------------------------------------------------------
RUN_LOG="$(mktemp -t mirage-vllm.XXXXXX.log)"
log "starting vLLM (model=$MODEL) via mirage run --daemon"
cd "$MIRAGE_DIR"
cargo run --quiet -- run \
  --daemon \
  --profile "$PROFILE" \
  --image "$IMAGE" \
  --container-provider "$PROVIDER" \
  --mount "$HF_CACHE:/root/.cache/huggingface" \
  -- vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
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

# Helper: run curl against the server. Prefer reaching the exposed port
# from the host via the container IP; fall back to running curl inside
# the container if the host cannot route to the session network.
api_curl() {
  local path="$1"; shift
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
REQ="$(printf '{"model":"%s","prompt":"Hello, my name is","max_tokens":16,"temperature":0}' "$MODEL")"
RESP="$(api_curl /v1/completions \
  -H 'Content-Type: application/json' \
  -d "$REQ")" || { cat "$RUN_LOG" >&2; fail "completion request failed"; }

printf 'response: %s\n' "$RESP"
grep -q '"choices"' <<<"$RESP" || fail "no choices in completion response"
log "PASS: vLLM served $MODEL through mirage and returned a generation"
