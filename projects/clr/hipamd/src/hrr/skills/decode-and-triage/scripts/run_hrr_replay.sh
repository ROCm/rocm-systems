#!/usr/bin/env bash
# Run full HRR GPU replay and write a timestamped log under the current directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
ENSURE="$SCRIPT_DIR/ensure_playback.sh"

ARCHIVE=""
LOG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) ARCHIVE="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "$ARCHIVE" ]] || { echo "error: --archive required" >&2; exit 1; }
ARCHIVE="$(readlink -f "$ARCHIVE" 2>/dev/null || realpath "$ARCHIVE" 2>/dev/null || echo "$ARCHIVE")"
[[ -d "$ARCHIVE" ]] || { echo "error: archive not found: $ARCHIVE" >&2; exit 1; }

resolve_playback() {
  local c candidates=()
  [[ -n "${HRR_PLAYBACK:-}" ]] && candidates+=("$HRR_PLAYBACK")
  if command -v hrr-playback >/dev/null 2>&1; then
    candidates+=("$(command -v hrr-playback)")
  fi
  if [[ -n "${CLR_BUILD:-}" ]]; then
    candidates+=("${CLR_BUILD}/hipamd/src/hrr/playback/hrr-playback")
  fi
  candidates+=("$ROCM_PATH/bin/hrr-playback")
  local p
  for p in "${candidates[@]}"; do
    [[ -n "$p" && -x "$p" ]] || continue
    echo "$p"
    return
  done
  echo ""
}

setup_library_path() {
  local play="$1"
  local bin_dir lib_dirs=() p seen=""
  bin_dir="$(cd "$(dirname "$play")" && pwd)"
  # CLR build: playback lives under build/hipamd/src/hrr/playback
  if [[ "$bin_dir" == *"/hipamd/src/hrr/playback" ]]; then
    lib_dirs+=("$(cd "$bin_dir/../../../lib" && pwd)")
  fi
  lib_dirs+=("$ROCM_PATH/lib")
  local sib="$(cd "$bin_dir/.." && pwd)/lib"
  [[ -d "$sib" ]] && lib_dirs+=("$sib")
  for p in "${lib_dirs[@]}"; do
    [[ -d "$p" ]] || continue
    [[ ":$seen:" == *":$p:"* ]] && continue
    seen="${seen:+$seen:}$p"
    LD_LIBRARY_PATH="${p}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  done
  export LD_LIBRARY_PATH
}

pick_gpu() {
  if [[ -n "${GPU:-}" ]]; then
    echo "$GPU"
    return
  fi
  if command -v rocm-smi >/dev/null 2>&1; then
    local best="" best_free=-1 idx free
    while read -r idx free; do
      [[ -n "$idx" ]] || continue
      if (( free > best_free )); then
        best_free=$free
        best=$idx
      fi
    done < <(rocm-smi --showmeminfo vram 2>/dev/null | awk '
      /GPU\[/ { gsub(/[^0-9]/,"",$1); idx=$1 }
      /Used Memory/ { used=$NF }
      /Total Memory/ { total=$NF; if (idx!="") { print idx, total-used; idx="" } }
    ')
    if [[ -n "$best" ]]; then
      echo "[run_hrr_replay] auto-selected GPU $best (most free VRAM)" >&2
      echo "$best"
      return
    fi
  fi
  echo "[run_hrr_replay] default GPU 0" >&2
  echo "0"
}

HRR_PLAY="$(resolve_playback)"
if [[ -z "$HRR_PLAY" && -x "$ENSURE" ]]; then
  HRR_PLAY="$("$ENSURE")" || HRR_PLAY=""
fi
[[ -n "$HRR_PLAY" ]] || {
  echo "error: hrr-playback not found (ensure_playback.sh failed). Set HRR_PLAYBACK or CLR_BUILD." >&2
  exit 1
}

setup_library_path "$HRR_PLAY"
GPU="$(pick_gpu)"

[[ -r /dev/kfd ]] || {
  echo "error: /dev/kfd not accessible — AMD GPU driver required" >&2
  exit 1
}

if [[ -z "$LOG" ]]; then
  LOG="$(pwd)/hrr-replay-$(basename "$ARCHIVE")-$(date -u +%Y%m%dT%H%M%SZ).log"
fi

echo "[run_hrr_replay] playback=$HRR_PLAY GPU=$GPU archive=$ARCHIVE" >&2
echo "[run_hrr_replay] log=$LOG" >&2

set +e
ROCR_VISIBLE_DEVICES="$GPU" HIP_HRR_REPLAY_PROGRESS_SECONDS="${HIP_HRR_REPLAY_PROGRESS_SECONDS:-30}" \
  "$HRR_PLAY" "$ARCHIVE" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "[run_hrr_replay] exit=$RC log=$LOG"
exit "$RC"
