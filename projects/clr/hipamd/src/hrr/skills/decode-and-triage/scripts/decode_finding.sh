#!/usr/bin/env bash
# Read-only HRR decode & triage: archive --info + optional replay log → structured finding.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ANALYZER="$SCRIPT_DIR/analyze_replay_finding.py"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

ARCHIVE=""
LOGS=()
OUTPUT=""
FORMAT="markdown"
EXTRA_ANALYZER=()

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
    --log) LOGS+=("$2"); shift 2 ;;
    -o|--output) OUTPUT="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    --) shift; EXTRA_ANALYZER+=("$@"); break ;;
    *) EXTRA_ANALYZER+=("$1"); shift ;;
  esac
done

[[ -n "$ARCHIVE" || ${#LOGS[@]} -gt 0 ]] || {
  echo "error: provide --archive and/or --log" >&2
  exit 1
}

resolve_playback() {
  local c candidates=()
  [[ -n "${HRR_PLAYBACK:-}" ]] && candidates+=("$HRR_PLAYBACK")
  if command -v hrr-playback >/dev/null 2>&1; then
    candidates+=("$(command -v hrr-playback)")
  fi
  if [[ -n "${CLR_BUILD:-}" ]]; then
    candidates+=("${CLR_BUILD}/hipamd/src/hrr/playback/hrr-playback")
  fi
  candidates+=(
    "/var/lib/rancher/hrr-develop-wt/projects/clr/build-hrr/hipamd/src/hrr/playback/hrr-playback"
    "$ROCM_PATH/bin/hrr-playback"
    "/opt/rocm/bin/hrr-playback"
  )
  local p
  for p in "${candidates[@]}"; do
    [[ -n "$p" && -x "$p" ]] || continue
    echo "$p"
    return
  done
  echo ""
}

if [[ -n "$ARCHIVE" ]]; then
  ARCHIVE="$(readlink -f "$ARCHIVE" 2>/dev/null || realpath "$ARCHIVE" 2>/dev/null || echo "$ARCHIVE")"
  [[ -d "$ARCHIVE" ]] || { echo "error: archive not found: $ARCHIVE" >&2; exit 1; }
fi

HRR_PLAY="$(resolve_playback)"
if [[ -n "$ARCHIVE" && -z "$HRR_PLAY" ]]; then
  echo "warning: hrr-playback not found; skipping --info (log-only triage still works)" >&2
  echo "warning: set HRR_PLAYBACK or install/build hrr-playback (see $SKILL_DIR/reference.md)" >&2
fi

CMD=(python3 "$ANALYZER" --format "$FORMAT")
[[ -n "$ARCHIVE" ]] && CMD+=(--archive "$ARCHIVE")
[[ -n "$HRR_PLAY" ]] && CMD+=(--hrr-playback "$HRR_PLAY")
for log in "${LOGS[@]}"; do
  CMD+=(--log "$log")
done
CMD+=("${EXTRA_ANALYZER[@]}")

if [[ -z "$OUTPUT" ]]; then
  ext="$(finding_ext)"
  if [[ ${#LOGS[@]} -gt 0 ]]; then
    base="${LOGS[0]%.log}"
    OUTPUT="${base}${ext}"
  elif [[ -n "$ARCHIVE" ]]; then
    name="$(basename "$ARCHIVE")"
    parent="$(dirname "$ARCHIVE")"
    candidate="${parent%/}/${name}${ext}"
    if [[ -w "$parent" ]]; then
      OUTPUT="$candidate"
    else
      OUTPUT="$(pwd)/${name}${ext}"
    fi
  fi
fi
[[ -n "$OUTPUT" ]] && CMD+=(-o "$OUTPUT")

echo "[decode_finding] skill=$SKILL_DIR playback=${HRR_PLAY:-none}" >&2
"${CMD[@]}"
[[ -n "$OUTPUT" ]] && echo "[decode_finding] finding=$OUTPUT" >&2
