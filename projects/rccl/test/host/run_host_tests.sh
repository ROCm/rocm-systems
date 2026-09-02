#!/usr/bin/env bash
#
# Build and run the RCCL CPU-only host unit tests: rccl-HostUnitTests plus the
# host-only microtests (rccl-UnitTestsMicro, rccl-UnitTestsMicroInit[-uncached]).
#
# Single source of truth for every command the host-test pipeline needs, so the
# same steps run locally and in CI and nothing is scattered in the workflow YAML.
# CI invokes each phase as its own step for clear failure attribution; locally,
# `all` runs the whole pipeline end to end.
#
# Usage:
#   run_host_tests.sh [deps|rccl-configure|hipify|configure|build|guards|run|all] [extra gtest args]
#   (default phase: all)
#
# Phases:
#   deps            install the host-test build/runtime dependencies via apt
#                   (cmake, toolchain, gtest, moreutils, python3-venv). CI
#                   runs this as its own step; not part of `all`.
#   rccl-configure  configure the RCCL tree (root) -- pins GPU_TARGETS so CMake
#                   never probes for a GPU; BUILD_TESTS=OFF (we only need hipify)
#   hipify          build the hipify_all target -> stages build/hipify/src, the
#                   prerequisite the host tests compile against
#   configure       configure test/host
#   build           build all host binaries (default target)
#   run             run every host binary (timestamped log + per-binary JUnit XML)
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
#   HOST_TEST_SHUFFLE  gtest ordering flag (run)   (default: --gtest_shuffle; set empty to disable
#                                                   when bisecting a failure. Deliberately NOT named
#                                                   GTEST_SHUFFLE: gtest owns that name and reads an
#                                                   empty value as TRUE, so clearing it would shuffle)
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
HOST_TEST_SHUFFLE="${HOST_TEST_SHUFFLE---gtest_shuffle}"  # `-` not `:-`: an explicit empty value disables it
# Split into argv elements. Passing the value as one quoted word makes gtest treat a multi-word
# setting as a single unknown flag: it prints its help, runs ZERO tests and exits 0 -- a silent green
# that `|| rc=1` cannot see. An empty value yields an empty array, which still disables shuffling.
read -ra HOST_TEST_SHUFFLE_ARGS <<< "$HOST_TEST_SHUFFLE"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/host_tests.log}"
XML_FILE="${XML_FILE:-$SCRIPT_DIR/host_tests.xml}"
JOBS="$(nproc 2>/dev/null || echo 4)"

PHASE="${1:-all}"
[ $# -gt 0 ] && shift || true   # remaining args ($@) are forwarded to the binary

# Install everything the host-test pipeline needs that the base ROCm dev image
# lacks: cmake + host toolchain, gtest, moreutils (ts), and python3-venv
# (the guards phase creates a venv + pip-installs pytest). Uses sudo when not
# already root so it works both in the root CI container and locally.
do_deps() {
  echo "==> Install host-test dependencies (apt)"
  local sudo=""
  [ "$(id -u)" -eq 0 ] || sudo="sudo"
  $sudo apt-get update
  $sudo apt-get install -y cmake git python3 python3-venv build-essential rocm-cmake \
    moreutils libgtest-dev libgmock-dev
}

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

do_host_tests() {
  echo "==> Run  (filter: $GTEST_FILTER)"
  # Prepend a real-UTC timestamp to each line via `ts` (moreutils) when available,
  # tee the full stdout+stderr to LOG_FILE, and preserve each binary's exit code
  # (pipefail) so a failure still fails CI.
  local stamp
  if command -v ts >/dev/null 2>&1; then
    stamp=(env TZ=UTC ts '%Y-%m-%dT%H:%M:%.SZ')
  else
    stamp=(cat)
  fi

  # Every host binary the build produces. Each writes its own JUnit XML
  # (host_tests*.xml, all uploaded) and appends to the single console LOG_FILE.
  # rccl-HostUnitTests keeps host_tests.xml for backward compatibility.
  local -a binaries=(
    "rccl-HostUnitTests:$XML_FILE"
    "rccl-UnitTestsMicro:$SCRIPT_DIR/host_tests_micro.xml"
    "rccl-UnitTestsMicroInit:$SCRIPT_DIR/host_tests_micro_init.xml"
    "rccl-UnitTestsMicroInit-uncached:$SCRIPT_DIR/host_tests_micro_init_uncached.xml"
  )

  : > "$LOG_FILE"   # truncate; each binary appends below
  local rc=0 entry name xml
  for entry in "${binaries[@]}"; do
    name="${entry%%:*}"
    xml="${entry#*:}"
    if [ ! -x "$BUILD_DIR/$name" ]; then
      echo "ERROR: expected binary not built: $BUILD_DIR/$name" | tee -a "$LOG_FILE"
      rc=1
      continue
    fi
    echo "----- $name -----" | tee -a "$LOG_FILE"
    # Shuffle every binary here, not just the micro ones: the init microtests share ~40 mutable
    # file-scope globals reset only in the fixture TearDown, and the older suites were audited to be
    # order-independent too. gtest prints the seed, so a failure stays reproducible; clear
    # HOST_TEST_SHUFFLE to run in declaration order while bisecting.
    "$BUILD_DIR/$name" \
      --gtest_filter="$GTEST_FILTER" \
      --gtest_output="xml:$xml" \
      ${HOST_TEST_SHUFFLE_ARGS[@]+"${HOST_TEST_SHUFFLE_ARGS[@]}"} \
      --gtest_color=no "$@" 2>&1 | "${stamp[@]}" | tee -a "$LOG_FILE" || rc=1
    # A binary that rejects its arguments prints usage, runs nothing and still exits 0, so the exit
    # status alone cannot tell "all green" from "never started". The report is the second signal:
    # gtest writes it before returning, and its absence means no suite ran at all.
    if [ ! -f "$xml" ] || ! grep -q "<testsuites" "$xml"; then
      echo "ERROR: $name wrote no test report ($xml) -- it likely rejected an argument" \
        | tee -a "$LOG_FILE"
      rc=1
    fi
  done
  return "$rc"
}

# Run the kernel-count guard pytest suite (test/kernel-count) in a local venv so
# the lean host-test image needs no system pytest. See that dir's README.
do_guards() {
  echo "==> Kernel-count guards (pytest: test/kernel-count)"
  local gd="$RCCL_ROOT/test/kernel-count"
  local venv="$gd/venv"
  if [ ! -x "$venv/bin/pytest" ]; then
    python3 -m venv "$venv"
    "$venv/bin/pip" install -q --disable-pip-version-check -r "$gd/requirements.txt"
  fi
  "$venv/bin/python" -m pytest "$gd/tests" -v
}

# The `run` phase aggregates every check the host-test pipeline executes: the
# gtest suite plus any CPU-only guards. The host-test workflow invokes `run`
# (and `all` ends with it), so adding a future check here makes both CI and
# local runs pick it up automatically -- no dispatch or workflow-YAML change.
# do_host_tests runs first so the JUnit XML artifact is always produced before a
# later guard can gate.
do_run() {
  do_host_tests "$@"
  do_guards
}

case "$PHASE" in
  deps)           do_deps ;;
  rccl-configure) do_rccl_configure ;;
  hipify)         do_hipify ;;
  configure)      do_configure ;;
  build)          do_build ;;
  guards)         do_guards ;;
  run)            do_run "$@" ;;
  all)            do_rccl_configure; do_hipify; do_configure; do_build; do_run "$@" ;;
  *) echo "usage: $0 [deps|rccl-configure|hipify|configure|build|run|guards|all] [extra gtest args]" >&2; exit 2 ;;
esac
