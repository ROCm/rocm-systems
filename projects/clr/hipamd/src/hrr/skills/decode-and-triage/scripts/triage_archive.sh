#!/usr/bin/env bash
# Decode + optional full replay + structured finding. Primary entry for agent triage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DECODE="$SCRIPT_DIR/decode_finding.sh"
REPLAY="$SCRIPT_DIR/run_hrr_replay.sh"

ARCHIVE=""
REPLAY_MODE="skip"   # skip | auto | native | docker
OUTPUT=""
FORMAT="markdown"

finding_ext() {
  if [[ "$FORMAT" == "json" ]]; then
    echo ".finding.json"
  else
    echo ".finding.md"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) ARCHIVE="$2"; shift 2 ;;
    --replay)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        REPLAY_MODE="native"
        shift
      else
        REPLAY_MODE="$2"
        shift 2
      fi
      ;;
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
FINDING=""

pick_replay_mode() {
  if [[ "$REPLAY_MODE" != "auto" ]]; then
    echo "$REPLAY_MODE"
    return
  fi
  if [[ -r /dev/kfd ]]; then
    # Prefer docker recipe when project script exists (MAF/vLLM captures).
    local docker_script="${HRR_ROOT:-$WORKDIR}/scripts/maf-hrr-docker-playback.sh"
    if [[ -f "$docker_script" ]] && command -v docker >/dev/null 2>&1; then
      echo "docker"
      return
    fi
    echo "native"
    return
  fi
  echo "skip"
}

mode="$(pick_replay_mode)"
echo "[triage_archive] archive=$ARCHIVE replay=$mode" >&2

if [[ "$mode" == "docker" ]]; then
  HRR_ROOT="${HRR_ROOT:-$WORKDIR}"
  export CLR_BUILD="${CLR_BUILD:-/var/lib/rancher/hrr-develop-wt/projects/clr/build-hrr}"
  LOG="$WORKDIR/hrr-replay-${name}-${ts}.log"
  echo "[triage_archive] docker replay -> $LOG" >&2
  set +e
  if sudo -n true 2>/dev/null; then
    sudo -n -E GPU="${GPU:-1}" HIP_HRR_REPLAY_PROGRESS_SECONDS="${HIP_HRR_REPLAY_PROGRESS_SECONDS:-30}" \
      bash "$HRR_ROOT/scripts/maf-hrr-docker-playback.sh" "$ARCHIVE" \
      2>&1 | tee "$LOG"
  else
    echo "error: docker replay requires passwordless sudo (sudo -n); use --replay native or --no-replay" >&2
    exit 1
  fi
  RC=${PIPESTATUS[0]}
  set -e
  echo "[triage_archive] replay exit=$RC" >&2
elif [[ "$mode" == "native" ]]; then
  LOG="$WORKDIR/hrr-replay-${name}-${ts}.log"
  set +e
  "$REPLAY" --archive "$ARCHIVE" --log "$LOG"
  RC=$?
  set -e
  echo "[triage_archive] replay exit=$RC" >&2
else
  echo "[triage_archive] skipping GPU replay (no /dev/kfd or --no-replay)" >&2
fi

FINDING="${OUTPUT:-$WORKDIR/${name}-${ts}$(finding_ext)}"
DECODE_ARGS=(--archive "$ARCHIVE" -o "$FINDING" --format "$FORMAT")
[[ -n "$LOG" && -f "$LOG" ]] && DECODE_ARGS+=(--log "$LOG")

"$DECODE" "${DECODE_ARGS[@]}"

echo "[triage_archive] finding=$FINDING" >&2
cat "$FINDING"
