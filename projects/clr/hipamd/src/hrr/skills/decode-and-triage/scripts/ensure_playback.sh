#!/usr/bin/env bash
# Locate or build hrr-playback from the colocated ROCm CLR tree.
# Prints the absolute path to stdout; exports CLR_BUILD and HRR_PLAYBACK.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

playback_from_build() {
  local build="$1"
  [[ -n "$build" && -x "$build/hipamd/src/hrr/playback/hrr-playback" ]] || return 1
  echo "$build/hipamd/src/hrr/playback/hrr-playback"
}

resolve_repo_root() {
  local root=""
  [[ -n "${HRR_ROOT:-}" ]] && { echo "$HRR_ROOT"; return; }
  root="$(git -C "$SKILL_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
  echo "$root"
}

resolve_clr_root() {
  local repo candidates=()
  [[ -n "${CLR_ROOT:-}" ]] && candidates+=("$CLR_ROOT")
  # In-tree skill: projects/clr/hipamd/src/hrr/skills/decode-and-triage
  candidates+=("$(cd "$SKILL_DIR/../../../../../" 2>/dev/null && pwd || true)")
  repo="$(resolve_repo_root)"
  [[ -n "$repo" ]] && candidates+=("$repo/projects/clr")
  [[ -n "$repo" ]] && candidates+=("$repo/rocm-systems/projects/clr")
  local c
  for c in "${candidates[@]}"; do
    [[ -n "$c" && -f "$c/CMakeLists.txt" && -d "$c/hipamd/src/hrr" ]] || continue
    echo "$c"
    return
  done
  echo ""
}

resolve_build_dir() {
  local clr="$1"
  local candidates=()
  [[ -n "${CLR_BUILD:-}" ]] && candidates+=("$CLR_BUILD")
  candidates+=("$clr/build-hrr" "$clr/build")
  local b
  for b in "${candidates[@]}"; do
    [[ -n "$b" ]] || continue
    if playback_from_build "$b" >/dev/null 2>&1; then
      echo "$b"
      return
    fi
    [[ -d "$b" ]] && echo "$b" && return
  done
  echo "$clr/build-hrr"
}

resolve_hip_common_dir() {
  local clr="$1" repo candidates=()
  [[ -n "${HIP_COMMON_DIR:-}" ]] && candidates+=("$HIP_COMMON_DIR")
  candidates+=("$(cd "$clr/../hip" 2>/dev/null && pwd || true)")
  repo="$(resolve_repo_root)"
  [[ -n "$repo" ]] && candidates+=("$repo/projects/hip")
  [[ -n "$repo" ]] && candidates+=("$repo/rocm-systems/projects/hip")
  local h
  for h in "${candidates[@]}"; do
    [[ -n "$h" && -d "$h/include/hip" ]] || continue
    echo "$h"
    return
  done
  echo ""
}

find_existing_playback() {
  local p candidates=()
  [[ -n "${HRR_PLAYBACK:-}" && -x "${HRR_PLAYBACK}" ]] && candidates+=("$HRR_PLAYBACK")
  if command -v hrr-playback >/dev/null 2>&1; then
    candidates+=("$(command -v hrr-playback)")
  fi
  [[ -n "${CLR_BUILD:-}" ]] && candidates+=("${CLR_BUILD}/hipamd/src/hrr/playback/hrr-playback")
  candidates+=("$ROCM_PATH/bin/hrr-playback")
  for p in "${candidates[@]}"; do
    [[ -n "$p" && -x "$p" ]] || continue
    echo "$p"
    return
  done
  echo ""
}

build_playback() {
  local clr="$1" build="$2" hip="$3"
  echo "[ensure_playback] configuring CLR at $clr (build=$build)" >&2
  cmake -S "$clr" -B "$build" -GNinja \
    -DHIP_COMMON_DIR="$hip" \
    -DROCM_PATH="$ROCM_PATH" \
    -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
    -DCMAKE_BUILD_TYPE=Release
  echo "[ensure_playback] building amdhip64 hrr-playback" >&2
  ninja -C "$build" amdhip64 hrr-playback -j"$(nproc)"
}

export_playback_env() {
  local play="$1" build
  build="$(cd "$(dirname "$play")/../../../.." && pwd)"
  export CLR_BUILD="$build"
  export HRR_PLAYBACK="$play"
  export LD_LIBRARY_PATH="$build/hipamd/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
}

main() {
  local existing play clr build hip

  existing="$(find_existing_playback)"
  if [[ -n "$existing" ]]; then
    export_playback_env "$existing"
    echo "$existing"
    return 0
  fi

  clr="$(resolve_clr_root)"
  [[ -n "$clr" ]] || {
    echo "error: hrr-playback not found and CLR source tree not discoverable." >&2
    echo "error: set CLR_ROOT or HRR_ROOT to your rocm-systems checkout (see reference.md)." >&2
    return 1
  }

  build="$(resolve_build_dir "$clr")"
  if play="$(playback_from_build "$build" 2>/dev/null)"; then
    export_playback_env "$play"
    echo "$play"
    return 0
  fi

  hip="$(resolve_hip_common_dir "$clr")"
  [[ -n "$hip" ]] || {
    echo "error: HIP_COMMON_DIR not found (expected projects/hip next to projects/clr)." >&2
    return 1
  }

  for tool in cmake ninja; do
    command -v "$tool" >/dev/null 2>&1 || {
      echo "error: $tool required to build hrr-playback" >&2
      return 1
    }
  done

  build_playback "$clr" "$build" "$hip"
  play="$(playback_from_build "$build")"
  [[ -n "$play" ]] || {
    echo "error: build finished but hrr-playback not found under $build" >&2
    return 1
  }
  export_playback_env "$play"
  echo "$play"
}

main "$@"
