#!/usr/bin/env bash
#
# Build and run the RCCL CPU-only host unit tests (rccl-HostUnitTests).
#
# Used by CI (.github/workflows/rccl-host-unit-tests.yml) and locally, so the
# same commands run in both places. Keeping the logic here (rather than inline in
# the workflow) lets us control what runs in the suite -- filters, extra gtest
# flags, build type -- in one place without editing CI.
#
# Usage:
#   run_host_tests.sh [configure|build|run|all] [extra gtest args]
#   (default phase: all)  CI invokes configure/build/run as separate steps so a
#   failure is clearly attributable; locally, `all` does the whole thing.
#
# Prerequisite: the RCCL hipify tree must exist (build/hipify/src). From the
# rccl root:  cmake -B build -DGPU_TARGETS=gfx942 && cmake --build build --target hipify_all
# (If it is missing, the configure phase fails fast with a clear CMake message.)
#
# Knobs (environment variables, all optional):
#   ROCM_PATH     ROCm install prefix              (default: /opt/rocm)
#   BUILD_TYPE    CMake build type                 (default: Debug)
#   BUILD_DIR     out-of-tree build dir            (default: <script dir>/build)
#   GTEST_FILTER  gtest test filter (run phase)    (default: *  = all)
#   LOG_FILE      timestamped console log (run)    (default: <script dir>/host_tests.log)
#   XML_FILE      JUnit XML output (run)           (default: <script dir>/host_tests.xml)
# Any args after the phase are forwarded to the test binary, e.g.:
#   run_host_tests.sh run --gtest_filter='BitOps*' --gtest_repeat=5
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
BUILD_TYPE="${BUILD_TYPE:-Debug}"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
GTEST_FILTER="${GTEST_FILTER:-*}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/host_tests.log}"
XML_FILE="${XML_FILE:-$SCRIPT_DIR/host_tests.xml}"
JOBS="$(nproc 2>/dev/null || echo 4)"

PHASE="${1:-all}"
[ $# -gt 0 ] && shift || true   # remaining args ($@) are forwarded to the binary

do_configure() {
  echo "==> Configure  (BUILD_TYPE=$BUILD_TYPE  ROCM_PATH=$ROCM_PATH)"
  cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DROCM_PATH="$ROCM_PATH"
}

do_build() {
  echo "==> Build  (-j$JOBS)"
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
  configure) do_configure ;;
  build)     do_build ;;
  run)       do_run "$@" ;;
  all)       do_configure; do_build; do_run "$@" ;;
  *) echo "usage: $0 [configure|build|run|all] [extra gtest args]" >&2; exit 2 ;;
esac
