#!/usr/bin/env bash
# Locate hrr-playback; optionally build from the colocated CLR tree (--build).
# Prints the absolute path to stdout on success.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
DO_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    -h|--help)
      echo "usage: ensure_playback.sh [--build]" >&2
      echo "  default: find existing hrr-playback only (no cmake/ninja)" >&2
      echo "  --build: configure and build amdhip64 + hrr-playback when missing" >&2
      exit 0
      ;;
    *) echo "error: unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ "${HRR_ENSURE_BUILD:-0}" == 1 ]] && DO_BUILD=1

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
  candidates+=("$(cd "$SKILL_DIR/../../../" 2>/dev/null && pwd || true)")
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

resolve_rocr_lib_dir() {
  local clr="${1:-}" repo candidates=() d
  [[ -n "${ROCR_LIB:-}" ]] && { echo "$ROCR_LIB"; return; }
  repo="$(resolve_repo_root)"
  [[ -n "$repo" ]] && candidates+=("$repo/projects/rocr-runtime/build-local/rocr/lib")
  [[ -n "$clr" ]] && candidates+=("$clr/../rocr-runtime/build-local/rocr/lib")
  for d in "${candidates[@]}"; do
    [[ -n "$d" && -f "$d/libhsa-runtime64.so.1" ]] || continue
    echo "$d"
    return
  done
  echo ""
}

resolve_rocr_inc_dir() {
  local clr="${1:-}" repo rocr candidates=() d
  [[ -n "${ROCR_INC:-}" ]] && { echo "$ROCR_INC"; return; }
  repo="$(resolve_repo_root)"
  [[ -n "$repo" ]] && candidates+=("$repo/projects/rocr-runtime/runtime/hsa-runtime/inc")
  [[ -n "$clr" ]] && candidates+=("$clr/../rocr-runtime/runtime/hsa-runtime/inc")
  for d in "${candidates[@]}"; do
    [[ -n "$d" && -d "$d" ]] || continue
    echo "$d"
    return
  done
  echo ""
}

setup_rocr_header_wrap() {
  local rocr_inc="$1" wrap="$2" f
  mkdir -p "$wrap/hsa"
  for f in "$rocr_inc"/*.h; do
    [[ -f "$f" ]] || continue
    ln -sfn "$f" "$wrap/hsa/$(basename "$f")"
  done
}

find_playback_under_clr() {
  local clr="$1" b play
  [[ -n "$clr" ]] || return 1
  for b in "$clr"/build-hrr* "$clr/build"; do
    [[ -d "$b" ]] || continue
    if play="$(playback_from_build "$b" 2>/dev/null)"; then
      echo "$play"
      return 0
    fi
  done
  return 1
}

resolve_build_dir() {
  local clr="$1"
  local candidates=() b play
  [[ -n "${CLR_BUILD:-}" ]] && candidates+=("$CLR_BUILD")
  for b in "$clr"/build-hrr* "$clr/build"; do
    [[ -d "$b" ]] || continue
    candidates+=("$b")
  done
  candidates+=("$clr/build-hrr")
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
  local p candidates=() clr
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
  clr="$(resolve_clr_root)"
  play="$(find_playback_under_clr "$clr" 2>/dev/null || true)"
  if [[ -n "$play" ]]; then
    echo "$play"
    return
  fi
  echo ""
}

build_playback() {
  local clr="$1" build="$2" hip="$3"
  local rocr_inc rocr_lib wrap cxx_flags ld_flags cmake_args=()
  rocr_inc="$(resolve_rocr_inc_dir "$clr")"
  rocr_lib="$(resolve_rocr_lib_dir "$clr")"
  wrap="$build/.rocr-hsa-wrap"
  cxx_flags=""
  ld_flags=""
  if [[ -n "$rocr_inc" ]]; then
    setup_rocr_header_wrap "$rocr_inc" "$wrap"
    cxx_flags="-I$wrap"
    echo "[ensure_playback] using in-tree ROCR headers from $rocr_inc" >&2
  else
    echo "[ensure_playback] warning: in-tree ROCR headers not found; build may fail in rocdevice.cpp on older /opt/rocm" >&2
  fi
  if [[ -n "$rocr_lib" ]]; then
    ld_flags="-L$rocr_lib -Wl,-rpath,$rocr_lib"
    echo "[ensure_playback] linking in-tree ROCR from $rocr_lib" >&2
  fi
  cmake_args=(
    -S "$clr" -B "$build" -GNinja
    -DHIP_COMMON_DIR="$hip"
    -DROCM_PATH="$ROCM_PATH"
    -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd
    -DCMAKE_BUILD_TYPE=Release
  )
  [[ -n "$cxx_flags" ]] && cmake_args+=(-DCMAKE_CXX_FLAGS="$cxx_flags" -DCMAKE_C_FLAGS="$cxx_flags")
  [[ -n "$ld_flags" ]] && cmake_args+=(-DCMAKE_SHARED_LINKER_FLAGS="$ld_flags" -DCMAKE_EXE_LINKER_FLAGS="$ld_flags")
  echo "[ensure_playback] configuring CLR at $clr (build=$build)" >&2
  cmake "${cmake_args[@]}"
  echo "[ensure_playback] building amdhip64 hrr-playback" >&2
  ninja -C "$build" amdhip64 hrr-playback -j"$(nproc)"
}

export_playback_env() {
  local play="$1" build rocr_lib ld_parts=()
  build="$(cd "$(dirname "$play")/../../../.." && pwd)"
  rocr_lib="$(resolve_rocr_lib_dir "$(resolve_clr_root)" || true)"
  export CLR_BUILD="$build"
  export HRR_PLAYBACK="$play"
  [[ -d "$build/hipamd/lib" ]] && ld_parts+=("$build/hipamd/lib")
  [[ -n "$rocr_lib" ]] && ld_parts+=("$rocr_lib")
  ld_parts+=("$ROCM_PATH/lib")
  export LD_LIBRARY_PATH="$(IFS=:; echo "${ld_parts[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  # Not a trailing `&&`: under `set -e` this is the last statement of the
  # function, so an unresolved ROCR lib dir would make the function return 1
  # and abort the script before it prints the playback path.
  if [[ -n "$rocr_lib" ]]; then
    export ROCR_LIB="$rocr_lib"
  fi
}

fail_not_found() {
  echo "error: hrr-playback not found." >&2
  if [[ "$DO_BUILD" -eq 0 ]]; then
    echo "error: re-run with --build for full GPU replay, or set HRR_PLAYBACK." >&2
    echo "error: metadata-only: triage_archive.sh --archive <dir> --no-replay" >&2
  else
    echo "error: set CLR_ROOT/HRR_ROOT or fix the build environment (see SKILL.md)." >&2
  fi
  return 1
}

main() {
  local existing play clr build hip

  existing="$(find_existing_playback)"
  if [[ -n "$existing" ]]; then
    export_playback_env "$existing"
    echo "$existing"
    return 0
  fi

  [[ "$DO_BUILD" -eq 1 ]] || fail_not_found

  clr="$(resolve_clr_root)"
  [[ -n "$clr" ]] || {
    echo "error: CLR source tree not discoverable." >&2
    fail_not_found
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
    fail_not_found
  }

  for tool in cmake ninja; do
    command -v "$tool" >/dev/null 2>&1 || {
      echo "error: $tool required to build hrr-playback" >&2
      fail_not_found
    }
  done

  build_playback "$clr" "$build" "$hip"
  play="$(playback_from_build "$build")"
  [[ -n "$play" ]] || {
    echo "error: build finished but hrr-playback not found under $build" >&2
    fail_not_found
  }
  export_playback_env "$play"
  echo "$play"
}

main "$@"
