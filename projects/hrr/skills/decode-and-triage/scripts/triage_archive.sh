#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
ANALYZER="$SCRIPT_DIR/analyze_replay_finding.py"
ENSURE="$SCRIPT_DIR/ensure_playback.sh"
REPLAY_DOCKER="$SCRIPT_DIR/replay_docker.sh"
COMPAT="$SCRIPT_DIR/check_replay_compat.py"
SANITIZER="$SCRIPT_DIR/archive_text.py"

safe() { python3 "$SANITIZER" --text "$1"; }

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
                       (default: $TMPDIR/hrr-triage-<uid>, mode 0700, never the
                       archive; /tmp if TMPDIR is inside the archive, and a
                       mktemp directory if that one is not ours)
  HRR_DOCKER_IMAGE     Docker image for --replay docker / auto
  HRR_DOCKER_MOUNT_CLR=1  Overlay host CLR for docker replay (dev builds)
  GPU                  Replay GPU ordinal (default: auto-pick)
  HRR_REPLAY_TIMEOUT   Seconds before a stalled replay is stopped (default 1800, 0 disables)
  HRR_CONTINUE=1       Proceed after preflight HIP/comgr mismatch prompt
  HRR_SKIP_COMPAT=1    Skip manifest preflight
EOF
}

needs_value() {
  (( $2 >= 2 )) || { echo "error: $1 needs a value" >&2; exit 1; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --archive) needs_value --archive $#; ARCHIVE="$2"; shift 2 ;;
    --replay)
      if [[ $# -lt 2 || "$2" == --* ]]; then REPLAY_MODE="native"; shift
      else REPLAY_MODE="$2"; shift 2; fi ;;
    --no-replay) REPLAY_MODE="skip"; shift ;;
    --no-sync) NO_SYNC=1; shift ;;
    -o|--output) needs_value --output $#; OUTPUT="$2"; shift 2 ;;
    --format) needs_value --format $#; FORMAT="$2"; shift 2 ;;
    *) echo "error: unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$ARCHIVE" ]] || { echo "error: --archive required" >&2; exit 1; }
ARCHIVE="$(readlink -f "$ARCHIVE" 2>/dev/null || realpath "$ARCHIVE" 2>/dev/null || echo "$ARCHIVE")"
[[ -d "$ARCHIVE" ]] || { echo "error: archive not found: $(safe "$ARCHIVE")" >&2; exit 1; }

name="$(basename "$ARCHIVE")"
ts="$(date -u +%Y%m%dT%H%M%SZ)-$$"
# Never default to the current directory: run inside a customer's archive, the finding and replay log would land in it.
ARCHIVE_ROOT="$ARCHIVE"
[[ "$name" == pid-* ]] && ARCHIVE_ROOT="$(dirname "$ARCHIVE")"
in_archive() {
  local p
  p="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1")"
  [[ "$p" == "$ARCHIVE_ROOT" || "$p" == "$ARCHIVE_ROOT"/* ]]
}
if [[ -n "${HRR_TRIAGE_WORKDIR:-}" ]]; then
  WORKDIR="$HRR_TRIAGE_WORKDIR"
  if in_archive "$WORKDIR"; then
    echo "error: HRR_TRIAGE_WORKDIR is inside the archive: $WORKDIR" >&2
    exit 1
  fi
  # Create the directory private. An existing one is the caller's, but the finding inside it is chmoded.
  mkdir -p -m 700 "$WORKDIR"
else
  tmp_base="${TMPDIR:-/tmp}"
  in_archive "$tmp_base" && tmp_base=/tmp
  if in_archive "$tmp_base"; then
    echo "error: the archive holds /tmp; set HRR_TRIAGE_WORKDIR outside it" >&2
    exit 1
  fi
  WORKDIR="$tmp_base/hrr-triage-$(id -u)"
  mkdir -p -m 700 "$WORKDIR" 2>/dev/null || true
  if [[ -L "$WORKDIR" || ! -d "$WORKDIR" || ! -O "$WORKDIR" || ! -w "$WORKDIR" ]] ||
     in_archive "$WORKDIR"; then
    WORKDIR="$(mktemp -d "$tmp_base/hrr-triage-XXXXXX")"
  else
    chmod 700 "$WORKDIR" 2>/dev/null || true
  fi
fi
LOG=""
ext=".finding.md"; [[ "$FORMAT" == "json" ]] && ext=".finding.json"
FINDING="${OUTPUT:-$WORKDIR/${name}-${ts}${ext}}"
# Check -o too: a finding named into the archive writes there as surely as a work directory does. /dev/stdout is the caller's stream.
if [[ -n "$OUTPUT" ]] && ! [[ "$FINDING" -ef /dev/stdout ]] && in_archive "$FINDING"; then
  echo "error: --output is inside the archive: $(safe "$OUTPUT")" >&2
  exit 1
fi

pick_replay_mode() {
  if [[ "$REPLAY_MODE" != "auto" ]]; then echo "$REPLAY_MODE"; return; fi
  if [[ -n "${HRR_DOCKER_IMAGE:-}" ]]; then echo "docker"
  elif [[ -r /dev/kfd ]]; then echo "native"
  else echo "skip"; fi
}

setup_library_path() {
  local play="$1" bin_dir lib_dirs=() p seen="" repo built=""
  bin_dir="$(cd "$(dirname "$play")" && pwd)"
  if [[ "$bin_dir" == */bin ]]; then
    lib_dirs+=("$(cd "$bin_dir/../lib" 2>/dev/null && pwd || true)")
  fi
  # A packaged playback ships bin/, lib/ and runtime-lib/ as siblings; without them its libamdhip64 and libhsa are not found.
  for p in lib runtime-lib; do
    [[ -d "$bin_dir/../$p" ]] && lib_dirs+=("$(cd "$bin_dir/../$p" && pwd)")
  done
  repo="$(cd "$SCRIPT_DIR/../../../../.." 2>/dev/null && pwd || true)"
  for p in "${ROCR_LIB:-}" "${repo:+$repo/projects/rocr-runtime/build-local/rocr/lib}"; do
    [[ -n "$p" && -f "$p/libhsa-runtime64.so.1" ]] || continue
    lib_dirs+=("$p"); break
  done
  for p in ${lib_dirs[@]+"${lib_dirs[@]}"}; do
    [[ -d "$p" ]] || continue
    [[ ":$seen:" == *":$p:"* ]] && continue
    seen="${seen:+$seen:}$p"
    built="${built:+$built:}$p"
  done
  local joined="" part parts=()
  for p in "$built" "${LD_LIBRARY_PATH:-}" "${ROCM_PATH:+$ROCM_PATH/lib}"; do
    IFS=: read -r -a parts <<< "$p"
    for part in ${parts[@]+"${parts[@]}"}; do
      [[ -n "$part" ]] || continue
      joined="${joined:+$joined:}$part"
    done
  done
  if [[ -n "$joined" ]]; then
    export LD_LIBRARY_PATH="$joined"
  else
    unset LD_LIBRARY_PATH
  fi
}

pick_gpu() {
  [[ -n "${GPU:-}" ]] && { echo "$GPU"; return; }
  if command -v rocm-smi >/dev/null 2>&1; then
    local best="" best_free=-1 idx free
    while read -r idx free; do
      [[ -n "$idx" ]] || continue
      (( free > best_free )) && { best_free=$free; best=$idx; }
    done < <(rocm-smi --showmeminfo vram 2>/dev/null | awk '
      # The used line reads "VRAM Total Used Memory", which also contains "Total", so test it first.
      /GPU\[/ {
        id = $1; gsub(/[^0-9]/, "", id)
        if ($0 ~ /Total Used Memory/) used[id] = $NF
        else if ($0 ~ /Total Memory/) total[id] = $NF
      }
      END {
        for (id in total)
          if (id in used) printf "%s %.0f\n", id, total[id] - used[id]
      }')
    [[ -n "$best" ]] && { echo "[triage] GPU $best (most free VRAM)" >&2; echo "$best"; return; }
  fi
  echo "0"
}

replay_sync_args() {
  # The default replay syncs the GPU once at the end, so a fault is reported but not attributed to a launch.
  [[ "$NO_SYNC" == "1" ]] && return 0
  echo "--sync-after-launch"
}

# Record a stop the replay does not explain: the log ends where the process died.
record_replay_stop() {
  local rc="$1" log="$2"
  [[ -n "$log" ]] || return 0
  if [[ "$rc" == "124" ]]; then
    echo "[triage] replay timed out after ${HRR_REPLAY_TIMEOUT:-1800}s" | tee -a "$log" >&2
  elif (( rc > 128 )); then
    echo "[triage] replay killed by signal $((rc - 128))" | tee -a "$log" >&2
  fi
}

run_native_replay() {
  local play="$1" log="$2" gpu sync_args=()
  setup_library_path "$play"
  gpu="$(pick_gpu)"
  [[ -r /dev/kfd ]] || { echo "error: /dev/kfd not accessible" >&2; return 1; }
  read -r -a sync_args <<< "$(replay_sync_args)"
  # Bound the replay: a hung workload would otherwise never return, so the analyzer never runs and no finding is produced.
  local timeout_s="${HRR_REPLAY_TIMEOUT:-1800}" runner=()
  if [[ "$timeout_s" != "0" ]] && command -v timeout >/dev/null 2>&1; then
    runner=(timeout "$timeout_s")
  fi
  echo "[triage] native replay playback=$play GPU=$gpu ${sync_args[*]}" >&2
  set +e
  HIP_VISIBLE_DEVICES="$gpu" HIP_HRR_REPLAY_PROGRESS_SECONDS="${HIP_HRR_REPLAY_PROGRESS_SECONDS:-30}" \
    ${runner[@]+"${runner[@]}"} "$play" "$ARCHIVE" --max-wall-seconds "$timeout_s" ${sync_args[@]+"${sync_args[@]}"} 2>&1 | tee "$log" | python3 "$SANITIZER"
  local rc=${PIPESTATUS[0]}
  # Do not re-enable `set -e`: the caller wraps this call in `set +e` so a failing replay is survivable.
  echo "[triage] native replay exit=$rc" >&2
  return "$rc"
}

mode="$(pick_replay_mode)"
echo "[triage] archive=$(safe "$ARCHIVE") replay=$mode" >&2

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
  # Metadata-only still wants `hrr-playback --info`, which needs no GPU. Locate without --build so nothing compiles.
  HRR_PLAYBACK="$("$ENSURE" 2>/dev/null || true)"
  if [[ -n "$HRR_PLAYBACK" ]]; then
    export HRR_PLAYBACK
    # The library path is otherwise set only for native replay; without it the analyzer cannot load the binary.
    setup_library_path "$HRR_PLAYBACK"
  fi
elif [[ "$mode" == "docker" && "${HRR_DOCKER_MOUNT_CLR:-0}" == "1" && -x "$ENSURE" ]]; then
  HRR_PLAYBACK="$("$ENSURE" --build)" || {
    echo "error: ensure_playback.sh --build failed for HRR_DOCKER_MOUNT_CLR=1" >&2
    exit 1
  }
  export HRR_PLAYBACK
fi

# The replay log and finding name a customer's kernels and addresses, so make them private from creation.
umask 077

REPLAY_GPU=""
if [[ "$mode" != "skip" ]]; then
  # Pick a GPU only when replaying: metadata-only runs need none.
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

# The analyzer prints the finding itself, so skip -o when it names stdout to avoid printing twice.
CMD=(python3 "$ANALYZER" --format "$FORMAT" --archive "$ARCHIVE")
[[ "$FINDING" -ef /dev/stdout ]] || CMD+=(-o "$FINDING")
[[ -n "${HRR_PLAYBACK:-}" ]] && CMD+=(--hrr-playback "$HRR_PLAYBACK")
[[ -n "$LOG" && -f "$LOG" ]] && CMD+=(--log "$LOG")
"${CMD[@]}"

# Only regular files: -o may name /dev/stdout or /dev/null, and -f follows links.
[[ -f "$FINDING" && ! -L "$FINDING" ]] && chmod 600 "$FINDING" 2>/dev/null || true
[[ -n "$LOG" && -f "$LOG" ]] && chmod 600 "$LOG" 2>/dev/null || true
echo "[triage] finding=$(safe "$FINDING")" >&2
