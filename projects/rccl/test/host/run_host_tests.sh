#!/usr/bin/env bash
#
# Build and run the RCCL CPU-only host unit tests: rccl-HostUnitTests plus the
# host-only microtests (rccl-UnitTestsMicro, rccl-UnitTestsMicroInit[-uncached],
# rccl-UnitTestsMicroEnqueue[-devlinker]).
#
# Single source of truth for every command the host-test pipeline needs, so the
# same steps run locally and in CI and nothing is scattered in the workflow YAML.
# CI invokes each phase as its own step for clear failure attribution; locally,
# `all` runs the whole pipeline end to end.
#
# Usage:
#   run_host_tests.sh [deps|rccl-configure|hipify|configure|build|guards|run|coverage|all] [extra gtest args]
#   (default phase: all)
#
# Phases:
#   deps            install the host-test build/runtime dependencies via apt
#                   (cmake, toolchain, gtest/fmt, moreutils, python3-venv). CI
#                   runs this as its own step; not part of `all`.
#   rccl-configure  configure the RCCL tree (root) -- pins GPU_TARGETS so CMake
#                   never probes for a GPU; BUILD_TESTS=OFF (we only need hipify)
#   hipify          build the hipify_all target -> stages build/hipify/src, the
#                   prerequisite the host tests compile against
#   configure       configure test/host
#   build           build all host binaries (default target)
#   run             run the suite (timestamped log + JUnit XML). Always emits
#                   llvm source-based coverage profiles (*.profraw) into
#                   <BUILD_DIR>/coverage (requires the host tests to be built
#                   with -DMICRO_COVERAGE=ON, the default)
#   coverage        turn the per-binary *.profraw profiles from `run` into
#                   reports: a per-binary text/HTML report + lcov tracefile
#                   (clean, no hash mismatch), plus an overall line/branch union
#                   across all binaries via lcov (the CI metric).
#   all             rccl-configure -> hipify -> configure -> build -> run -> coverage
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
# Coverage output always lives in a dedicated subdir we own under BUILD_DIR.
# Deliberately NOT a user knob: the `run` phase wipes this dir, and an
# arbitrary user-supplied path (e.g. '/', '.', or a source dir) would then
# recursively delete unrelated files.
COVERAGE_DIR="$BUILD_DIR/coverage"
JOBS="$(nproc 2>/dev/null || echo 4)"

PHASE="${1:-all}"
[ $# -gt 0 ] && shift || true   # remaining args ($@) are forwarded to the binary

# Install everything the host-test pipeline needs that the base ROCm dev image
# lacks: cmake + host toolchain, gtest/fmt, moreutils (ts), python3-venv (the
# guards phase creates a venv + pip-installs pytest), and lcov/genhtml (the
# coverage phase merges per-binary lcov tracefiles into one overall report).
# Uses sudo when not already root so it works both in the root CI container and
# locally.
do_deps() {
  echo "==> Install host-test dependencies (apt)"
  local sudo=""
  [ "$(id -u)" -eq 0 ] || sudo="sudo"
  $sudo apt-get update
  $sudo apt-get install -y cmake git python3 python3-venv build-essential rocm-cmake \
    moreutils libgtest-dev libgmock-dev libfmt-dev lcov llvm
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

  # Always emit source-based coverage profiles: instrumented binaries write
  # .profraw files that the `coverage` phase turns into reports without
  # re-running anything.
  #
  # Each binary writes into its OWN profraw subdir ($COVERAGE_DIR/<binary>/).
  # The host-test binaries compile some of the same source under different
  # #defines, so a shared function has a different coverage-mapping hash in each.
  # Keeping profiles isolated lets `coverage` emit a clean per-binary report and
  # a clean per-binary lcov tracefile; the tracefiles are then unioned at the
  # line/branch level into one overall report (hash-agnostic). Start clean.
  # COVERAGE_DIR is always <BUILD_DIR>/coverage, a dedicated subdir we own; the
  # ${VAR:?} guard is belt-and-suspenders against an empty BUILD_DIR. Start clean.
  mkdir -p "$COVERAGE_DIR"
  rm -rf "${COVERAGE_DIR:?}"/*

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
    "rccl-UnitTestsMicroEnqueue:$SCRIPT_DIR/host_tests_micro_enqueue.xml"
    # ENABLE_DEVICE_LINKER defaults ON, so this variant is the arm that ships;
    # enqueue.cc gates rcclShmemDynamicSize on RCCL_DEVICE_LINKER at the
    # preprocessor, so one compile cannot cover both. See test/host/CMakeLists.txt.
    "rccl-UnitTestsMicroEnqueue-devlinker:$SCRIPT_DIR/host_tests_micro_enqueue_devlinker.xml"
  )

  : > "$LOG_FILE"   # truncate; each binary appends below
  local rc=0 entry exe name xml profdir
  for entry in "${binaries[@]}"; do
    name="${entry%%:*}"
    xml="${entry#*:}"
    exe="$BUILD_DIR/$name"
    if [ ! -x "$exe" ]; then
      echo "ERROR: expected binary not built: $exe" | tee -a "$LOG_FILE"
      rc=1
      continue
    fi
    echo "----- $name -----" | tee -a "$LOG_FILE"
    # Direct this binary's llvm profiles into its OWN subdir so `coverage` finds
    # them at $COVERAGE_DIR/<binary>/. Without this, instrumented binaries write
    # default.profraw into the CWD and coverage sees nothing. %p (PID) + %m
    # (module hash) keep the process-isolated forks' files distinct; the runner
    # inherits this value (it sets LLVM_PROFILE_FILE with overwrite=0).
    profdir="$COVERAGE_DIR/$name"
    mkdir -p "$profdir"
    export LLVM_PROFILE_FILE="$profdir/$name-%p-%m.profraw"
    # Shuffle every binary here, not just the micro ones: the init microtests share ~40 mutable
    # file-scope globals reset only in the fixture TearDown, and the older suites were audited to be
    # order-independent too. gtest prints the seed, so a failure stays reproducible; clear
    # HOST_TEST_SHUFFLE to run in declaration order while bisecting.
    "$exe" \
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

# Turn the per-binary profraw sets produced by the `run` phase into coverage
# reports. Two levels of output:
#
#   1. Per binary ($COVERAGE_DIR/<binary>/): a text + HTML report and an lcov
#      tracefile, each built against ONLY that binary's own profile so llvm-cov
#      never warns about "mismatched data". This is what the inner dev loop wants
#      (focused on one binary / one file under test).
#
#   2. Overall ($COVERAGE_DIR/overall/): the per-binary lcov tracefiles unioned
#      at the line/branch level via `lcov`. A line/branch counts as covered if
#      ANY binary covered it, so source that is exercised in different binaries
#      under different #defines is credited correctly.
do_coverage() {
  echo "==> Coverage  (out: $COVERAGE_DIR)"

  # Prefer the toolchain's own llvm tools: versions match the build clang by construction.
  if [ -d "$ROCM_PATH/llvm/bin" ]; then
    PATH="$ROCM_PATH/llvm/bin:$PATH"
  fi
  local tool
  for tool in llvm-profdata llvm-cov; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: $tool not found -- coverage needs the LLVM that built the tests" >&2
      exit 1
    fi
  done

  local exe name dir rc=0
  local tracefiles=()
  # Each binary's profraw lives in its own $COVERAGE_DIR/<name>/ subdir, written
  # by the `run` phase. Iterate those so coverage tracks exactly what ran.
  for dir in "$COVERAGE_DIR"/*/; do
    name="$(basename "$dir")"
    [ "$name" = overall ] && continue
    exe="$BUILD_DIR/$name"
    if [ ! -x "$exe" ]; then
      echo "error: expected binary not built: $exe -- run the build phase first" >&2
      rc=1
      continue
    fi
    if ! compgen -G "$dir/*.profraw" >/dev/null; then
      echo "    skip $name -- no profraw (run the 'run' phase first)" >&2
      continue
    fi

    # Per-binary: merge its own profraw and report against its own object only.
    llvm-profdata merge -sparse "$dir"/*.profraw -o "$dir/merged.profdata"

    llvm-cov report "$exe" \
      -instr-profile="$dir/merged.profdata" \
      --show-branch-summary --show-region-summary \
      > "$dir/coverage.txt"

    llvm-cov show "$exe" \
      -instr-profile="$dir/merged.profdata" \
      -format=html -output-dir="$dir/html" \
      --show-branches=count

    # Per-binary lcov tracefile -- the hash-agnostic, line/branch-keyed form that
    # can be unioned across binaries. Tag it with the binary name so genhtml and
    # lcov diagnostics are legible.
    llvm-cov export "$exe" \
      -instr-profile="$dir/merged.profdata" \
      --ignore-filename-regex='(/test/|nvtx)' \
      -format=lcov > "$dir/coverage.lcov"

    echo "    $name text report: $dir/coverage.txt"
    echo "    $name html report: $dir/html/index.html"
    tracefiles+=("$dir/coverage.lcov")
  done

  if [ "${#tracefiles[@]}" -eq 0 ]; then
    echo "error: no .profraw files found under $COVERAGE_DIR -- run the 'run' phase first" >&2
    exit 1
  fi

  # --- Overall union across all binaries (the CI metric) -------------------
  if ! command -v lcov >/dev/null 2>&1; then
    echo "    note: lcov not installed -- skipping overall union report (run the 'deps' phase)" >&2
    return "$rc"
  fi

  local odir="$COVERAGE_DIR/overall"
  mkdir -p "$odir"

  # Common lcov flags. lcov 2.x enforces strict consistency checks that reject
  # llvm-cov's tracefiles (e.g. "line is hit but no branches evaluated"); disable
  # them and tolerate the format quirks. lcov 1.x has neither the checks nor the
  # error categories, so only pass these on >= 2.
  local lcov_args=()
  local lcov_major
  lcov_major="$(lcov --version 2>/dev/null | grep -oE '[0-9]+' | head -1)"
  if [ "${lcov_major:-0}" -ge 2 ]; then
    lcov_args+=(--rc branch_coverage=1
                --rc check_data_consistency=0
                --ignore-errors "inconsistent,unsupported,corrupt,format")
  else
    lcov_args+=(--rc lcov_branch_coverage=1)
  fi

  # lcov merges tracefiles by taking the union per line/branch: a line/branch is
  # covered if ANY input covered it.
  local add_args=()
  local tf
  for tf in "${tracefiles[@]}"; do
    add_args+=(--add-tracefile "$tf")
  done
  lcov "${lcov_args[@]}" "${add_args[@]}" -o "$odir/combined.lcov"

  # Machine-readable-ish overall summary (line + branch %) and a browsable HTML.
  lcov "${lcov_args[@]}" --summary "$odir/combined.lcov" \
    2>&1 | tee "$odir/summary.txt"
  if command -v genhtml >/dev/null 2>&1; then
    local genhtml_args=(--branch-coverage)
    [ "${lcov_major:-0}" -ge 2 ] && genhtml_args+=(--ignore-errors "inconsistent,unsupported,corrupt,format")
    genhtml "${genhtml_args[@]}" "$odir/combined.lcov" -o "$odir/html" \
      >/dev/null 2>&1 || true
    echo "    overall html report: $odir/html/index.html"
  fi
  echo "    overall tracefile:   $odir/combined.lcov"
  echo "    overall summary:     $odir/summary.txt"
  return "$rc"
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
  coverage)       do_coverage ;;
  all)            do_rccl_configure; do_hipify; do_configure; do_build; do_run "$@"; do_coverage ;;
  *) echo "usage: $0 [deps|rccl-configure|hipify|configure|build|run|guards|coverage|all] [extra gtest args]" >&2; exit 2 ;;
esac
