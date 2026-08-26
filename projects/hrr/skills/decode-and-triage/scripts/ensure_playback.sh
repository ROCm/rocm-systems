#!/usr/bin/env bash
# Locate hrr-playback; optionally build the standalone projects/hrr playback tool (--build).
# Prints the absolute path to stdout on success.
#
# Discovery order (no build): explicit HRR_PLAYBACK, installed <ROCM_PATH>/bin, then PATH.
# --build: configure and build hrr-playback from projects/hrr into a dedicated HRR build
#          dir (never a CLR build dir), against a capture-enabled ROCm/HIP prefix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# scripts -> decode-and-triage -> skills -> hrr (the standalone HRR project root)
HRR_PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
DO_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    -h|--help)
      echo "usage: ensure_playback.sh [--build]" >&2
      echo "  default: find an existing hrr-playback (HRR_PLAYBACK, \$ROCM_PATH/bin, PATH)" >&2
      echo "  --build: configure and build hrr-playback from projects/hrr when missing" >&2
      exit 0
      ;;
    *) echo "error: unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ "${HRR_ENSURE_BUILD:-0}" == 1 ]] && DO_BUILD=1

# Discovery: explicit HRR_PLAYBACK, then installed <ROCM_PATH>/bin, then PATH.
find_existing_playback() {
  local p candidates=()
  [[ -n "${HRR_PLAYBACK:-}" && -x "${HRR_PLAYBACK}" ]] && candidates+=("$HRR_PLAYBACK")
  candidates+=("$ROCM_PATH/bin/hrr-playback")
  if command -v hrr-playback >/dev/null 2>&1; then
    candidates+=("$(command -v hrr-playback)")
  fi
  for p in "${candidates[@]}"; do
    [[ -n "$p" && -x "$p" ]] || continue
    echo "$p"
    return
  done
  echo ""
}

# A dedicated HRR build dir (never a CLR build dir).
resolve_hrr_build_dir() {
  [[ -n "${HRR_BUILD:-}" ]] && { echo "$HRR_BUILD"; return; }
  echo "$HRR_PROJECT_DIR/build-playback"
}

playback_from_build() {
  local build="$1"
  [[ -n "$build" && -x "$build/playback/hrr-playback" ]] || return 1
  echo "$build/playback/hrr-playback"
}

build_playback() {
  local build="$1"
  echo "[ensure_playback] configuring projects/hrr at $HRR_PROJECT_DIR (build=$build)" >&2
  cmake -S "$HRR_PROJECT_DIR" -B "$build" \
    -DROCM_PATH="$ROCM_PATH" \
    -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$ROCM_PATH}" \
    -DHRR_BUILD_PLAYBACK=ON -DHRR_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
  echo "[ensure_playback] building hrr-playback" >&2
  cmake --build "$build" --target hrr-playback -j"$(nproc)"
}

export_playback_env() {
  local play="$1" ld_parts=()
  export HRR_PLAYBACK="$play"
  [[ -d "$ROCM_PATH/lib" ]] && ld_parts+=("$ROCM_PATH/lib")
  if [[ ${#ld_parts[@]} -gt 0 ]]; then
    export LD_LIBRARY_PATH="$(IFS=:; echo "${ld_parts[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

fail_not_found() {
  echo "error: hrr-playback not found." >&2
  if [[ "$DO_BUILD" -eq 0 ]]; then
    echo "error: set HRR_PLAYBACK, install hrr-playback under $ROCM_PATH/bin, or re-run with --build." >&2
    echo "error: metadata-only: triage_archive.sh --archive <dir> --no-replay" >&2
  else
    echo "error: build projects/hrr manually or set HRR_PLAYBACK (see SKILL.md)." >&2
  fi
  return 1
}

main() {
  local existing play build

  existing="$(find_existing_playback)"
  if [[ -n "$existing" ]]; then
    export_playback_env "$existing"
    echo "$existing"
    return 0
  fi

  [[ "$DO_BUILD" -eq 1 ]] || fail_not_found

  [[ -f "$HRR_PROJECT_DIR/CMakeLists.txt" ]] || {
    echo "error: projects/hrr CMake project not found at $HRR_PROJECT_DIR" >&2
    fail_not_found
  }

  build="$(resolve_hrr_build_dir)"
  if play="$(playback_from_build "$build" 2>/dev/null)"; then
    export_playback_env "$play"
    echo "$play"
    return 0
  fi

  command -v cmake >/dev/null 2>&1 || {
    echo "error: cmake required to build hrr-playback" >&2
    fail_not_found
  }

  build_playback "$build"
  play="$(playback_from_build "$build")"
  [[ -n "$play" ]] || {
    echo "error: build finished but hrr-playback not found under $build" >&2
    fail_not_found
  }
  export_playback_env "$play"
  echo "$play"
}

main "$@"
