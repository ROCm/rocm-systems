#!/usr/bin/env bash
# Replay an HRR archive inside a Docker image (when capture used that image's ROCm stack).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCHIVE=""
LOG=""
GPU="${GPU:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) ARCHIVE="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    *) echo "error: unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$ARCHIVE" ]] || { echo "error: --archive required" >&2; exit 1; }
ARCHIVE="$(readlink -f "$ARCHIVE" 2>/dev/null || realpath "$ARCHIVE" 2>/dev/null || echo "$ARCHIVE")"
[[ -d "$ARCHIVE" ]] || { echo "error: archive not found: $ARCHIVE" >&2; exit 1; }

IMAGE="${HRR_DOCKER_IMAGE:-}"
[[ -n "$IMAGE" ]] || {
  echo "error: set HRR_DOCKER_IMAGE to the capture container image" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || { echo "error: docker not found" >&2; exit 1; }

PLAY="${HRR_PLAYBACK:-}"
if [[ -z "$PLAY" && -n "${CLR_BUILD:-}" ]]; then
  PLAY="${CLR_BUILD}/hipamd/src/hrr/playback/hrr-playback"
fi
[[ -x "$PLAY" ]] || { echo "error: hrr-playback not found (set HRR_PLAYBACK or CLR_BUILD)" >&2; exit 1; }

CLR_LIB="$(cd "$(dirname "$PLAY")/../../../lib" && pwd)"
ROCR_LIB="${HRR_DOCKER_ROCR_LIB:-${ROCR_LIB:-}}"
EXTRA_LD="${HRR_DOCKER_EXTRA_LD:-}"
if [[ -z "$EXTRA_LD" && "$IMAGE" == rocm/vllm:* ]]; then
  EXTRA_LD="/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib"
  echo "[replay_docker] default EXTRA_LD for vLLM image" >&2
fi

ARCH_ROOT="$(dirname "$ARCHIVE")"
PID_NAME="$(basename "$ARCHIVE")"

if [[ -z "$LOG" ]]; then
  LOG="$(pwd)/hrr-replay-${PID_NAME}-$(date -u +%Y%m%dT%H%M%SZ).log"
fi

LD_INSIDE="/opt/hrr/lib"
[[ -n "$ROCR_LIB" && -d "$ROCR_LIB" ]] && LD_INSIDE="${LD_INSIDE}:/opt/hrr/rocr"
if [[ -n "$EXTRA_LD" ]]; then
  LD_INSIDE="${LD_INSIDE}:${EXTRA_LD}:/opt/rocm/lib"
else
  LD_INSIDE="${LD_INSIDE}:/opt/rocm/lib"
fi

echo "[replay_docker] image=$IMAGE GPU=$GPU archive=$ARCHIVE" >&2
echo "[replay_docker] log=$LOG" >&2

MOUNTS=(
  -v "$ARCH_ROOT:/arch:ro"
  -v "$PLAY:/opt/hrr/bin/hrr-playback:ro"
  -v "$CLR_LIB:/opt/hrr/lib:ro"
)
[[ -n "$ROCR_LIB" && -d "$ROCR_LIB" ]] && MOUNTS+=(-v "$ROCR_LIB:/opt/hrr/rocr:ro")

set +e
if sudo -n true 2>/dev/null; then
  sudo -n docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    "${MOUNTS[@]}" \
    -e "HIP_VISIBLE_DEVICES=$GPU" \
    -e "LD_LIBRARY_PATH=$LD_INSIDE" \
    -e "HIP_HRR_REPLAY_PROGRESS_SECONDS=${HIP_HRR_REPLAY_PROGRESS_SECONDS:-30}" \
    "$IMAGE" \
    /opt/hrr/bin/hrr-playback "/arch/$PID_NAME" \
    2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
else
  docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    "${MOUNTS[@]}" \
    -e "HIP_VISIBLE_DEVICES=$GPU" \
    -e "LD_LIBRARY_PATH=$LD_INSIDE" \
    -e "HIP_HRR_REPLAY_PROGRESS_SECONDS=${HIP_HRR_REPLAY_PROGRESS_SECONDS:-30}" \
    "$IMAGE" \
    /opt/hrr/bin/hrr-playback "/arch/$PID_NAME" \
    2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
fi
set -e

echo "[replay_docker] exit=$RC log=$LOG" >&2
exit "$RC"
