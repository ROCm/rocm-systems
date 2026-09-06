#!/usr/bin/env bash
# Replay an HRR archive inside a Docker image (when capture used that image's ROCm stack).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCHIVE=""
LOG=""
GPU="${GPU:-0}"
MOUNT_CLR="${HRR_DOCKER_MOUNT_CLR:-0}"

usage() {
  cat <<'EOF' >&2
usage: replay_docker.sh --archive <pid-dir> [options]

Replay an HRR archive inside HRR_DOCKER_IMAGE. By default uses hrr-playback and
HIP libs from the image; set HRR_DOCKER_MOUNT_CLR=1 to overlay host CLR_BUILD.

Options:
  --archive PATH   pid-* archive directory (required)
  --log PATH       Replay log file (default: ./hrr-replay-<pid>-<ts>.log)
  --gpu N          HIP_VISIBLE_DEVICES inside container (default: 0 or GPU env)
  --mount-clr      Same as HRR_DOCKER_MOUNT_CLR=1
  -h, --help       Show this help

Environment:
  HRR_DOCKER_IMAGE      Capture/replay container image (required)
  HRR_DOCKER_MOUNT_CLR  1 = mount host hrr-playback + libamdhip64 from CLR_BUILD
  HRR_PLAYBACK / CLR_BUILD  Host playback binary / build tree (overlay mode)
  ROCR_LIB              Optional in-tree ROCR lib dir for overlay mode
  HRR_DOCKER_PLAYBACK   Path to hrr-playback inside image (default: auto-detect)
  HRR_DOCKER_EXTRA_LD   Extra LD_LIBRARY_PATH segment inside container
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --mount-clr) MOUNT_CLR=1; shift ;;
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

EXTRA_LD="${HRR_DOCKER_EXTRA_LD:-}"
if [[ -z "$EXTRA_LD" && "$IMAGE" == rocm/vllm:* ]]; then
  EXTRA_LD="/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib"
  echo "[replay_docker] default EXTRA_LD for vLLM image" >&2
fi

docker_run() {
  if sudo -n true 2>/dev/null; then
    sudo -n docker "$@"
  else
    docker "$@"
  fi
}

image_ld_path() {
  if [[ -n "$EXTRA_LD" ]]; then
    echo "${EXTRA_LD}:/opt/rocm/lib"
  else
    echo "/opt/rocm/lib"
  fi
}

find_image_playback() {
  local ld inside
  ld="$(image_ld_path)"
  inside="${HRR_DOCKER_PLAYBACK:-}"
  if [[ -n "$inside" ]]; then
    echo "$inside"
    return 0
  fi
  docker_run run --rm \
    -e "LD_LIBRARY_PATH=$ld" \
    "$IMAGE" \
    bash -lc 'for p in "$(command -v hrr-playback 2>/dev/null)" /opt/rocm/bin/hrr-playback; do [[ -n "$p" && -x "$p" ]] && { echo "$p"; exit 0; }; done; exit 1' \
    2>/dev/null | tail -1
}

ARCH_ROOT="$(dirname "$ARCHIVE")"
PID_NAME="$(basename "$ARCHIVE")"

if [[ -z "$LOG" ]]; then
  LOG="$(pwd)/hrr-replay-${PID_NAME}-$(date -u +%Y%m%dT%H%M%SZ).log"
fi

MOUNTS=(-v "$ARCH_ROOT:/arch:ro")
PLAY_INSIDE=""
LD_INSIDE=""

if [[ "$MOUNT_CLR" == "1" ]]; then
  PLAY="${HRR_PLAYBACK:-}"
  if [[ -z "$PLAY" && -n "${CLR_BUILD:-}" ]]; then
    PLAY="${CLR_BUILD}/hipamd/src/hrr/playback/hrr-playback"
  fi
  [[ -x "$PLAY" ]] || {
    echo "error: HRR_DOCKER_MOUNT_CLR=1 requires host hrr-playback (set HRR_PLAYBACK or CLR_BUILD)" >&2
    exit 1
  }
  CLR_LIB="$(cd "$(dirname "$PLAY")/../../../lib" && pwd)"
  ROCR_LIB="${HRR_DOCKER_ROCR_LIB:-${ROCR_LIB:-}}"
  LD_INSIDE="/opt/hrr/lib"
  [[ -n "$ROCR_LIB" && -d "$ROCR_LIB" ]] && LD_INSIDE="${LD_INSIDE}:/opt/hrr/rocr"
  if [[ -n "$EXTRA_LD" ]]; then
    LD_INSIDE="${LD_INSIDE}:${EXTRA_LD}:/opt/rocm/lib"
  else
    LD_INSIDE="${LD_INSIDE}:/opt/rocm/lib"
  fi
  MOUNTS+=(-v "$PLAY:/opt/hrr/bin/hrr-playback:ro" -v "$CLR_LIB:/opt/hrr/lib:ro")
  [[ -n "$ROCR_LIB" && -d "$ROCR_LIB" ]] && MOUNTS+=(-v "$ROCR_LIB:/opt/hrr/rocr:ro")
  PLAY_INSIDE="/opt/hrr/bin/hrr-playback"
  echo "[replay_docker] dev overlay: mounting host CLR from $CLR_LIB" >&2
else
  LD_INSIDE="$(image_ld_path)"
  PLAY_INSIDE="$(find_image_playback)" || true
  [[ -n "$PLAY_INSIDE" ]] || {
    echo "error: hrr-playback not found in image $IMAGE" >&2
    echo "error: set HRR_DOCKER_MOUNT_CLR=1 with CLR_BUILD to overlay a host dev build" >&2
    exit 1
  }
  echo "[replay_docker] using image HRR stack: playback=$PLAY_INSIDE" >&2
fi

echo "[replay_docker] image=$IMAGE GPU=$GPU archive=$ARCHIVE" >&2
echo "[replay_docker] log=$LOG" >&2

set +e
docker_run run --rm \
  --device=/dev/kfd --device=/dev/dri \
  "${MOUNTS[@]}" \
  -e "HIP_VISIBLE_DEVICES=$GPU" \
  -e "LD_LIBRARY_PATH=$LD_INSIDE" \
  -e "HIP_HRR_REPLAY_PROGRESS_SECONDS=${HIP_HRR_REPLAY_PROGRESS_SECONDS:-30}" \
  "$IMAGE" \
  "$PLAY_INSIDE" "/arch/$PID_NAME" \
  2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "[replay_docker] exit=$RC log=$LOG" >&2
exit "$RC"
