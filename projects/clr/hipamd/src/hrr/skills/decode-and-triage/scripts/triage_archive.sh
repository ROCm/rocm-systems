#!/usr/bin/env bash
# HRR triage: optional GPU replay + structured finding. Sole agent entry point.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
ANALYZER="$SCRIPT_DIR/analyze_replay_finding.py"
ENSURE="$SCRIPT_DIR/ensure_playback.sh"
REPLAY_DOCKER="$SCRIPT_DIR/replay_docker.sh"
COMPAT="$SCRIPT_DIR/check_replay_compat.py"

ARCHIVE=""
REPLAY_MODE="auto"
OUTPUT=""
FORMAT="markdown"

usage() {
  cat <<'EOF' >&2
usage: triage_archive.sh --archive <pid-dir> [options]

Options:
  --archive PATH       pid-* HRR archive directory (required)
  --replay [MODE]      Replay mode: native, docker, auto (default), or bare --replay (= native)
  --no-replay          Metadata / --info only (no GPU replay or preflight block on replay)
  -o, --output PATH    Write finding to PATH (default: HRR_TRIAGE_WORKDIR/<pid>-<ts>.finding.md)
  --format FORMAT      markdown (default) or json
  -h, --help           Show this help

Environment (common):
  HRR_TRIAGE_WORKDIR   Output directory for findings and replay logs
  HRR_DOCKER_IMAGE     Docker image for --replay docker / auto
  HRR_DOCKER_MOUNT_CLR=1  Overlay host CLR for docker replay (dev builds)
  GPU                  Replay GPU ordinal (default: auto-pick)
  HRR_CONTINUE=1       Proceed after preflight HIP/comgr mismatch prompt
  HRR_SKIP_COMPAT=1    Skip manifest preflight
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --replay)
      if [[ $# -lt 2 || "$2" == --* ]]; then REPLAY_MODE="native"; shift
      else REPLAY_MODE="$2"; shift 2; fi ;;
    --no-replay) REPLAY_MODE="skip"; shift ;;
    -o|--output) OUTPUT="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    *) echo "error: unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$ARCHIVE" ]] || { echo "error: --archive required" >&2; exit 1; }
ARCHIVE="$(readlink -f "$ARCHIVE" 2>/dev/null || realpath "$ARCHIVE" 2>/dev/null || echo "$ARCHIVE")"
[[ -d "$ARCHIVE" ]] || { echo "error: archive not found: $ARCHIVE" >&2; exit 1; }

name="$(basename "$ARCHIVE")"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
WORKDIR="${HRR_TRIAGE_WORKDIR:-$(pwd)}"
mkdir -p "$WORKDIR"
LOG=""
ext=".finding.md"; [[ "$FORMAT" == "json" ]] && ext=".finding.json"
FINDING="${OUTPUT:-$WORKDIR/${name}-${ts}${ext}}"

pick_replay_mode() {
  if [[ "$REPLAY_MODE" != "auto" ]]; then echo "$REPLAY_MODE"; return; fi
  if [[ -n "${HRR_DOCKER_IMAGE:-}" ]]; then echo "docker"
  elif [[ -r /dev/kfd ]]; then echo "native"
  else echo "skip"; fi
}

setup_library_path() {
  local play="$1" bin_dir lib_dirs=() p seen="" clr built=""
  bin_dir="$(cd "$(dirname "$play")" && pwd)"
  if [[ "$bin_dir" == *"/hipamd/src/hrr/playback" ]]; then
    lib_dirs+=("$(cd "$bin_dir/../../../lib" && pwd)")
  fi
  clr="$(cd "$SCRIPT_DIR/../../../../../" 2>/dev/null && pwd || true)"
  for p in "${ROCR_LIB:-}" "${clr:+$clr/../rocr-runtime/build-local/rocr/lib}"; do
    [[ -n "$p" && -f "$p/libhsa-runtime64.so.1" ]] || continue
    lib_dirs+=("$p"); break
  done
  lib_dirs+=("$ROCM_PATH/lib")
  for p in "${lib_dirs[@]}"; do
    [[ -d "$p" ]] || continue
    [[ ":$seen:" == *":$p:"* ]] && continue
    seen="${seen:+$seen:}$p"
    built="${built:+$built:}$p"
  done
  export LD_LIBRARY_PATH="${built}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH:-}}"
}

pick_gpu() {
  [[ -n "${GPU:-}" ]] && { echo "$GPU"; return; }
  if command -v rocm-smi >/dev/null 2>&1; then
    local best="" best_free=-1 idx free
    while read -r idx free; do
      [[ -n "$idx" ]] || continue
      (( free > best_free )) && { best_free=$free; best=$idx; }
    done < <(rocm-smi --showmeminfo vram 2>/dev/null | awk '
      /GPU\[/ { gsub(/[^0-9]/,"",$1); idx=$1 }
      /Used Memory/ { used=$NF }
      /Total Memory/ { total=$NF; if (idx!="") { print idx, total-used; idx="" } }')
    [[ -n "$best" ]] && { echo "[triage] GPU $best (most free VRAM)" >&2; echo "$best"; return; }
  fi
  echo "0"
}

run_native_replay() {
  local play="$1" log="$2" gpu
  setup_library_path "$play"
  gpu="$(pick_gpu)"
  [[ -r /dev/kfd ]] || { echo "error: /dev/kfd not accessible" >&2; return 1; }
  echo "[triage] native replay playback=$play GPU=$gpu" >&2
  set +e
  ROCR_VISIBLE_DEVICES="$gpu" HIP_HRR_REPLAY_PROGRESS_SECONDS="${HIP_HRR_REPLAY_PROGRESS_SECONDS:-30}" \
    "$play" "$ARCHIVE" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "[triage] native replay exit=$rc" >&2
  return "$rc"
}

mode="$(pick_replay_mode)"
echo "[triage] archive=$ARCHIVE replay=$mode" >&2

run_replay_preflight() {
  local replay_mode="$1" gpu="$2"
  [[ -f "$COMPAT" ]] || return 0
  [[ "${HRR_SKIP_COMPAT:-}" == "1" ]] && {
    echo "[triage] skipping replay preflight (HRR_SKIP_COMPAT=1)" >&2
    return 0
  }
  local compat_args=(python3 "$COMPAT" --archive "$ARCHIVE" --gpu "$gpu")
  if [[ "$replay_mode" == "docker" ]]; then
    compat_args+=(--mode docker --docker-image "${HRR_DOCKER_IMAGE:?HRR_DOCKER_IMAGE required for docker preflight}")
  fi
  [[ "${HRR_STRICT_VERSION:-}" == "1" ]] && compat_args+=(--strict-version)
  [[ "${HRR_STRICT_ARCH:-}" == "1" ]] && compat_args+=(--strict-arch)
  echo "[triage] replay preflight (manifest metadata)" >&2
  set +e
  "${compat_args[@]}"
  local rc=$?
  set -e
  if [[ "$rc" -eq 2 ]]; then
    if [[ "${HRR_CONTINUE:-}" == "1" ]]; then
      echo "[triage] continuing despite version mismatch (HRR_CONTINUE=1)" >&2
      return 0
    fi
    if [[ -t 0 ]]; then
      echo ""
      read -r -p "Version mismatch detected. Do you want to continue? [y/N] " ans
      if [[ "$ans" =~ ^[Yy]$ ]]; then
        echo "[triage] continuing after confirmation" >&2
        return 0
      fi
      echo "[triage] aborted by user at version mismatch prompt" >&2
      exit 3
    fi
    echo "[triage] version mismatch requires confirmation (re-run with HRR_CONTINUE=1)" >&2
    exit 2
  fi
  if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
  fi
}

if [[ "$mode" != "skip" && "$mode" == "native" && -x "$ENSURE" ]]; then
  HRR_PLAYBACK="$("$ENSURE" --build)" || {
    echo "error: ensure_playback.sh --build failed (see SKILL.md; do not patch source)" >&2
    exit 1
  }
  export HRR_PLAYBACK
elif [[ "$mode" == "docker" && "${HRR_DOCKER_MOUNT_CLR:-0}" == "1" && -x "$ENSURE" ]]; then
  HRR_PLAYBACK="$("$ENSURE" --build)" || {
    echo "error: ensure_playback.sh --build failed for HRR_DOCKER_MOUNT_CLR=1" >&2
    exit 1
  }
  export HRR_PLAYBACK
fi

REPLAY_GPU="${GPU:-$(pick_gpu)}"
if [[ "$mode" != "skip" ]]; then
  run_replay_preflight "$mode" "$REPLAY_GPU"
fi

if [[ "$mode" == "docker" ]]; then
  LOG="$WORKDIR/hrr-replay-${name}-${ts}.log"
  set +e; "$REPLAY_DOCKER" --archive "$ARCHIVE" --log "$LOG" --gpu "$REPLAY_GPU"; set -e
elif [[ "$mode" == "native" ]]; then
  LOG="$WORKDIR/hrr-replay-${name}-${ts}.log"
  GPU="$REPLAY_GPU"
  set +e; run_native_replay "$HRR_PLAYBACK" "$LOG"; set -e
fi

CMD=(python3 "$ANALYZER" --format "$FORMAT" --archive "$ARCHIVE" -o "$FINDING")
[[ -n "${HRR_PLAYBACK:-}" ]] && CMD+=(--hrr-playback "$HRR_PLAYBACK")
[[ -n "$LOG" && -f "$LOG" ]] && CMD+=(--log "$LOG")
"${CMD[@]}"

echo "[triage] finding=$FINDING" >&2
cat "$FINDING"
