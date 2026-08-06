#!/usr/bin/env bash
#
# Build and run the RCCL CPU-only host unit tests (rccl-HostUnitTests).
#
# Single source of truth for every command the host-test pipeline needs, so the
# same steps run locally and in CI and nothing is scattered in the workflow YAML.
# CI invokes each phase as its own step for clear failure attribution; locally,
# `all` runs the whole pipeline end to end.
#
# Usage:
#   run_host_tests.sh [rccl-configure|hipify|configure|build|run|all] [extra gtest args]
#   (default phase: all)
#
# Phases:
#   rccl-configure  configure the RCCL tree (root) -- pins GPU_TARGETS so CMake
#                   never probes for a GPU; BUILD_TESTS=OFF (we only need hipify)
#   hipify          build the hipify_all target -> stages build/hipify/src, the
#                   prerequisite the host tests compile against
#   configure       configure test/host
#   build           build rccl-HostUnitTests
#   run             run the suite (timestamped log + JUnit XML)
#   all             rccl-configure -> hipify -> configure -> build -> run
#
# Knobs (environment variables, all optional):
#   ROCM_PATH     ROCm install prefix              (default: /opt/rocm)
#   GPU_TARGETS   arch for RCCL configure          (default: gfx942)
#   BUILD_TYPE    CMake build type                 (default: Debug)
#   BUILD_DIR     host-test build dir              (default: <script dir>/build)
#   GTEST_FILTER  gtest test filter (run phase)    (default: *  = all)
#   LOG_FILE      timestamped console log (run)    (default: <script dir>/host_tests.log)
#   XML_FILE      JUnit XML output (run)           (default: <script dir>/host_tests.xml)
# Any args after the phase are forwarded to the test binary, e.g.:
#   run_host_tests.sh run --gtest_filter='BitOps*' --gtest_repeat=5
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RCCL_BUILD_DIR="$RCCL_ROOT/build"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_TARGETS="${GPU_TARGETS:-gfx942}"
BUILD_TYPE="${BUILD_TYPE:-Debug}"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
GTEST_FILTER="${GTEST_FILTER:-*}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/host_tests.log}"
XML_FILE="${XML_FILE:-$SCRIPT_DIR/host_tests.xml}"
JOBS="$(nproc 2>/dev/null || echo 4)"

PHASE="${1:-all}"
[ $# -gt 0 ] && shift || true   # remaining args ($@) are forwarded to the binary

do_rccl_configure() {
  echo "==> RCCL configure  (GPU_TARGETS=$GPU_TARGETS)"
  cmake -S "$RCCL_ROOT" -B "$RCCL_BUILD_DIR" \
    -DGPU_TARGETS="$GPU_TARGETS" -DBUILD_TESTS=OFF
}

do_hipify() {
  echo "==> Stage hipified sources (hipify_all)  (-j$JOBS)"
  cmake --build "$RCCL_BUILD_DIR" --target hipify_all -j"$JOBS"
}

do_configure() {
  echo "==> Configure host tests  (BUILD_TYPE=$BUILD_TYPE  ROCM_PATH=$ROCM_PATH)"
  cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DROCM_PATH="$ROCM_PATH"
}

do_build() {
  echo "==> Build host tests  (-j$JOBS)"
  cmake --build "$BUILD_DIR" -j"$JOBS"
}

do_run() {
  echo "==> Run  (filter: $GTEST_FILTER)"
  # Prepend a real-UTC timestamp to each line via `ts` (moreutils) when available,
  # tee the full stdout+stderr to LOG_FILE, and preserve the test binary's exit
  # code (pipefail) so a failure still fails CI.
  local stamp
  if command -v ts >/dev/null 2>&1; then
    stamp=(env TZ=UTC ts '%Y-%m-%dT%H:%M:%.SZ')
  else
    stamp=(cat)
  fi
  "$BUILD_DIR/rccl-HostUnitTests" \
    --gtest_filter="$GTEST_FILTER" \
    --gtest_output="xml:$XML_FILE" \
    --gtest_color=no "$@" 2>&1 | "${stamp[@]}" | tee "$LOG_FILE"
}

case "$PHASE" in
  rccl-configure) do_rccl_configure ;;
  hipify)         do_hipify ;;
  configure)      do_configure ;;
  build)          do_build ;;
  run)            do_run "$@" ;;
  all)            do_rccl_configure; do_hipify; do_configure; do_build; do_run "$@" ;;
  *) echo "usage: $0 [rccl-configure|hipify|configure|build|run|all] [extra gtest args]" >&2; exit 2 ;;
esac
