#!/usr/bin/env bash
# Run rccl-UnitTestsMicro under llvm source-based coverage and emit a
# report scoped to the file(s) compiled directly into the binary
# (currently just the unit-under-test hipify/src/transport/p2p_tmp.cc).
#
# Requirements:
#   - rccl-UnitTestsMicro built with llvm source-based coverage
#     (-fprofile-instr-generate -fcoverage-mapping). The in-RCCL-build target
#     is always instrumented; the standalone build is controlled by
#     -DMICRO_COVERAGE=ON (default ON).
#   - llvm-profdata and llvm-cov on PATH (or under /opt/rocm/llvm/bin).
#
# Usage:
#   test/host/coverage.sh                                # text summary to stdout
#   test/host/coverage.sh --html out/cov-html            # also emit HTML report
#   FUNC=ipcRegisterBuffer test/host/coverage.sh         # scope to one function
#   BUILD_DIR=build/debug test/host/coverage.sh          # non-default build tree
#   test/host/coverage.sh --html out -- --gtest_filter=X # forward args to the test
#
# Script options (--html DIR) are consumed here; everything after a literal
# `--` is forwarded verbatim to the test binary.
#
# HTML reports include inline branch counts (--show-branches=count) and a
# branch column in the per-file summary, so branch coverage is visible
# alongside line coverage.
#
# All paths are resolved relative to the rccl source root, regardless of
# where the script is invoked from.

set -euo pipefail

# --- Parse this script's own options up front. Everything after a literal
#     `--` (and any unrecognized leading args) is forwarded to the test binary,
#     so `--html DIR` is never handed to GoogleTest. ---
HTML_DIR=""
TEST_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --html)
      HTML_DIR="${2:?--html requires an output directory}"
      shift 2
      ;;
    --)
      shift
      TEST_ARGS+=("$@")
      break
      ;;
    *)
      TEST_ARGS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${RCCL_ROOT}/build/release}"
BIN="${BUILD_DIR}/test/host/rccl-UnitTestsMicro"
COV_DIR="${BUILD_DIR}/test/host/coverage"
PROFRAW="${COV_DIR}/micro.profraw"
PROFDATA="${COV_DIR}/micro.profdata"

# Prefer ROCm's bundled llvm tooling, fall back to system PATH.
LLVM_BIN="/opt/rocm/llvm/bin"
PROFDATA_TOOL="$(command -v "${LLVM_BIN}/llvm-profdata" || command -v llvm-profdata)"
COV_TOOL="$(command -v "${LLVM_BIN}/llvm-cov"      || command -v llvm-cov)"

if [[ ! -x "${BIN}" ]]; then
  echo "error: ${BIN} not found." >&2
  echo "       Build it first (standalone: cmake -B build -DRCCL_BUILD_DIR=... && cmake --build build;" >&2
  echo "       in-tree: ./install.sh -t). Coverage is on by default (standalone: -DMICRO_COVERAGE=ON)." >&2
  exit 1
fi

mkdir -p "${COV_DIR}"

# 1. Run the test binary with LLVM_PROFILE_FILE pointing at our .profraw.
#    Only forward test args (never this script's own --html) to the binary.
echo "==> Running ${BIN} ${TEST_ARGS[*]:-}"
LLVM_PROFILE_FILE="${PROFRAW}" "${BIN}" ${TEST_ARGS[@]+"${TEST_ARGS[@]}"}

# 2. Merge raw profile into indexed profdata.
echo "==> Merging profile -> ${PROFDATA}"
"${PROFDATA_TOOL}" merge -sparse "${PROFRAW}" -o "${PROFDATA}"

# 3. Scope the report to the source file(s) actually compiled into the binary.
#    The unit under test is included via P2P_CC_PATH, which points at the
#    unroll-transformed p2p_tmp.cc (there is no plain src/transport/p2p.cc in
#    the hipify tree). Scoping to a non-existent p2p.cc makes llvm-cov silently
#    fall back to whole-binary totals, so name p2p_tmp.cc explicitly. Add more
#    -sources here as the binary grows.
SOURCES=(
  "${BUILD_DIR}/hipify/src/transport/p2p_tmp.cc"
)

# Optional: scope the HTML / annotated-source output to a single function
# (exact match). Set via environment, e.g.
#   `FUNC=ipcRegisterBuffer ./coverage.sh --html out/`
HTML_NAME_FLAGS=()
if [[ -n "${FUNC:-}" ]]; then
  HTML_NAME_FLAGS+=(--name="${FUNC}")
  echo "==> Scoped to function: ${FUNC}"
fi

echo "==> Coverage summary (file totals)"
"${COV_TOOL}" report "${BIN}" \
  -instr-profile="${PROFDATA}" \
  --show-branch-summary --show-region-summary \
  "${SOURCES[@]}"

# When scoped to a specific function, also show the annotated source so
# branch hit/miss counts are visible in the terminal. (llvm-cov `report`
# aggregates per-file regardless of --name, so we use `show` for this.)
if [[ -n "${FUNC:-}" ]]; then
  echo ""
  echo "==> Annotated source for ${FUNC} (branch hit/miss counts inline)"
  "${COV_TOOL}" show "${BIN}" \
    -instr-profile="${PROFDATA}" \
    --name="${FUNC}" \
    --show-branches=count \
    "${SOURCES[@]}"
fi

# Optional HTML output: --html <dir> (parsed up top into HTML_DIR).
if [[ -n "${HTML_DIR}" ]]; then
  mkdir -p "${HTML_DIR}"
  echo "==> Writing HTML report to ${HTML_DIR}"
  # --show-branches=count puts hit/miss counts inline next to each branch;
  # --show-regions surfaces region boundaries (helpful when one source line
  # has multiple short-circuited conditions).
  "${COV_TOOL}" show "${BIN}" \
    -instr-profile="${PROFDATA}" \
    -format=html -output-dir="${HTML_DIR}" \
    -show-line-counts-or-regions \
    -show-regions \
    -show-branches=count \
    --show-region-summary --show-branch-summary \
    "${HTML_NAME_FLAGS[@]}" \
    "${SOURCES[@]}"
  echo "    Open: ${HTML_DIR}/index.html"
fi
