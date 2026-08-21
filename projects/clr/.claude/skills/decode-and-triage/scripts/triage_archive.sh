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
NO_SYNC=0
OUTPUT=""
FORMAT="markdown"

usage() {
  cat <<'EOF' >&2
usage: triage_archive.sh --archive <pid-dir> [options]

Options:
  --archive PATH       pid-* HRR archive directory (required)
  --replay [MODE]      Replay mode: native, docker, auto (default), or bare --replay (= native)
  --no-replay          Metadata / --info only (no GPU replay or preflight block on replay)
  --no-sync            Replay without --sync-after-launch (faster, but a fault is not attributed)
  -o, --output PATH    Write finding to PATH (default: HRR_TRIAGE_WORKDIR/<pid>-<ts>.finding.md)
  --format FORMAT      markdown (default) or json
  -h, --help           Show this help

Environment (common):
  HRR_TRIAGE_WORKDIR   Output directory for findings and replay logs
  HRR_DOCKER_IMAGE     Docker image for --replay docker / auto
  HRR_DOCKER_MOUNT_CLR=1  Overlay host CLR for docker replay (dev builds)
  GPU                  Replay GPU ordinal (default: auto-pick)
  HRR_REPLAY_TIMEOUT   Seconds before a stalled replay is stopped (default 1800, 0 disables)
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
    --no-sync) NO_SYNC=1; shift ;;
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
  # A packaged playback ships as bin/ and lib/ siblings rather than inside a
  # CLR build tree. Without this its libamdhip64 is never on the path, so the
  # binary loads the system one and fails on the symbols it was built against.
  if [[ -d "$bin_dir/../lib" ]]; then
    lib_dirs+=("$(cd "$bin_dir/../lib" && pwd)")
  fi
  clr="$(cd "$SCRIPT_DIR/../../../../" 2>/dev/null && pwd || true)"
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
      # The used line reads "VRAM Total Used Memory", so it also contains the
      # word Total and must be tested first. Total and used arrive on separate
      # lines whose order is not guaranteed, so collect both per device and
      # subtract at END rather than on whichever line happens to land last.
      /GPU\[/ {
        id = $1; gsub(/[^0-9]/, "", id)
        if ($0 ~ /Total Used Memory/) used[id] = $NF
        else if ($0 ~ /Total Memory/) total[id] = $NF
      }
      END {
        for (id in total)
          if (id in used) print id, total[id] - used[id]
      }')
    [[ -n "$best" ]] && { echo "[triage] GPU $best (most free VRAM)" >&2; echo "$best"; return; }
  fi
  echo "0"
}

replay_sync_args() {
  # Default replay serializes the GPU once at the end, so a fault is reported
  # but not attributed to a launch: the finding then has no failing event and
  # no kernel. Synchronizing after every launch is what makes the last launch
  # line before the fault the culprit. Opt out with --no-sync when throughput
  # matters more than attribution, for instance a long soak.
  [[ "$NO_SYNC" == "1" ]] && return 0
  echo "--sync-after-launch"
}

# A stop the replay does not explain itself has to be recorded here, because
# the log just ends where the process died. The analyzer would read that
# truncation as insufficient signal and then go looking for a kernel to blame
# for it, which is how a crash of the replay process ends up reported as a
# workload fault. 124 is what `timeout` reports; above 128 the shell is
# reporting death by signal 128+N. A signal is not on its own a verdict: a GPU
# memory fault also aborts, and there the log's own fault line is the better
# answer, so the classifier ranks these markers last.
record_replay_stop() {
  local rc="$1" log="$2"
  [[ -n "$log" ]] || return 0
  if [[ "$rc" == "124" ]]; then
    echo "[triage] replay timed out after ${HRR_REPLAY_TIMEOUT:-1800}s" | tee -a "$log"
  elif (( rc > 128 )); then
    echo "[triage] replay killed by signal $((rc - 128))" | tee -a "$log"
  fi
}

run_native_replay() {
  local play="$1" log="$2" gpu sync_args=()
  setup_library_path "$play"
  gpu="$(pick_gpu)"
  [[ -r /dev/kfd ]] || { echo "error: /dev/kfd not accessible" >&2; return 1; }
  read -r -a sync_args <<< "$(replay_sync_args)"
  # A hung workload replays as a hang, and without a bound the replay never
  # returns, so the analyzer never runs and the archive yields no finding at
  # all. The bound belongs here, around the playback process: wrapping the
  # container instead leaves the replay running inside it. Set
  # HRR_REPLAY_TIMEOUT=0 to disable.
  local timeout_s="${HRR_REPLAY_TIMEOUT:-1800}" runner=()
  if [[ "$timeout_s" != "0" ]] && command -v timeout >/dev/null 2>&1; then
    runner=(timeout "$timeout_s")
  fi
  echo "[triage] native replay playback=$play GPU=$gpu ${sync_args[*]}" >&2
  set +e
  # hrr-playback is a HIP program, so HIP_VISIBLE_DEVICES is the mask that
  # applies to it. Setting ROCR_VISIBLE_DEVICES re-indexes devices underneath a
  # HIP mask, which can land the replay on a device other than the one picked.
  HIP_VISIBLE_DEVICES="$gpu" HIP_HRR_REPLAY_PROGRESS_SECONDS="${HIP_HRR_REPLAY_PROGRESS_SECONDS:-30}" \
    ${runner[@]+"${runner[@]}"} "$play" "$ARCHIVE" ${sync_args[@]+"${sync_args[@]}"} 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  # Deliberately not re-enabling `set -e` here. The caller wraps this call in
  # `set +e` precisely so a failing replay is survivable, and restores `set -e`
  # afterwards. Re-arming it inside the function made the non-zero return fatal,
  # so a replay that faulted killed the script before the finding was written:
  # the skill produced nothing in exactly the case it exists for.
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
elif [[ "$mode" == "skip" && -x "$ENSURE" ]]; then
  # Metadata-only still wants `hrr-playback --info`, which needs no GPU. Locate
  # without --build so this never tries to compile; if there is no binary the
  # analyzer simply reports the archive path, as before.
  # Not a trailing `&&`: as the last statement of this branch it would become
  # the `if` statement's exit status and, under `set -e`, end the script.
  HRR_PLAYBACK="$("$ENSURE" 2>/dev/null || true)"
  if [[ -n "$HRR_PLAYBACK" ]]; then
    export HRR_PLAYBACK
    # The library path is otherwise only prepared inside the native replay, so
    # without this the analyzer finds the binary and then cannot load it.
    setup_library_path "$HRR_PLAYBACK"
  fi
elif [[ "$mode" == "docker" && "${HRR_DOCKER_MOUNT_CLR:-0}" == "1" && -x "$ENSURE" ]]; then
  HRR_PLAYBACK="$("$ENSURE" --build)" || {
    echo "error: ensure_playback.sh --build failed for HRR_DOCKER_MOUNT_CLR=1" >&2
    exit 1
  }
  export HRR_PLAYBACK
fi

REPLAY_GPU=""
if [[ "$mode" != "skip" ]]; then
  # Only when a device is actually going to be used: metadata-only needs no GPU
  # and should not announce a choice it never makes.
  REPLAY_GPU="${GPU:-$(pick_gpu)}"
  run_replay_preflight "$mode" "$REPLAY_GPU"
fi

if [[ "$mode" == "docker" ]]; then
  LOG="$WORKDIR/hrr-replay-${name}-${ts}.log"
  set +e; "$REPLAY_DOCKER" --archive "$ARCHIVE" --log "$LOG" --gpu "$REPLAY_GPU"; replay_rc=$?; set -e
  record_replay_stop "$replay_rc" "$LOG"
elif [[ "$mode" == "native" ]]; then
  LOG="$WORKDIR/hrr-replay-${name}-${ts}.log"
  GPU="$REPLAY_GPU"
  set +e; run_native_replay "$HRR_PLAYBACK" "$LOG"; replay_rc=$?; set -e
  record_replay_stop "$replay_rc" "$LOG"
fi

CMD=(python3 "$ANALYZER" --format "$FORMAT" --archive "$ARCHIVE" -o "$FINDING")
[[ -n "${HRR_PLAYBACK:-}" ]] && CMD+=(--hrr-playback "$HRR_PLAYBACK")
[[ -n "$LOG" && -f "$LOG" ]] && CMD+=(--log "$LOG")
"${CMD[@]}"

echo "[triage] finding=$FINDING" >&2
cat "$FINDING"
