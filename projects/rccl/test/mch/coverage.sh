#!/usr/bin/env bash
# Run rccl-UnitTestsMch under llvm source-based coverage and emit a
# report scoped to the file(s) compiled directly into the binary
# (currently just hipify/src/transport/p2p.cc).
#
# Requirements:
#   - Configure with -DENABLE_MCH_COVERAGE=ON so the test binary is
#     built with -fprofile-instr-generate -fcoverage-mapping. (We use
#     our own knob rather than -DENABLE_CODE_COVERAGE because the
#     latter is Debug-only at the top level; the mch binary doesn't
#     need that constraint.)
#   - llvm-profdata and llvm-cov on PATH (or under /opt/rocm/llvm/bin).
#
# Usage:
#   test/mch/coverage.sh                       # text summary to stdout
#   test/mch/coverage.sh --html out/cov-html   # also emit HTML report
#   BUILD_DIR=build/debug test/mch/coverage.sh # non-default build tree
#
# All paths are resolved relative to the rccl source root, regardless of
# where the script is invoked from.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${RCCL_ROOT}/build/release}"
BIN="${BUILD_DIR}/test/mch/rccl-UnitTestsMch"
COV_DIR="${BUILD_DIR}/test/mch/coverage"
PROFRAW="${COV_DIR}/mch.profraw"
PROFDATA="${COV_DIR}/mch.profdata"

# Prefer ROCm's bundled llvm tooling, fall back to system PATH.
LLVM_BIN="/opt/rocm/llvm/bin"
PROFDATA_TOOL="$(command -v "${LLVM_BIN}/llvm-profdata" || command -v llvm-profdata)"
COV_TOOL="$(command -v "${LLVM_BIN}/llvm-cov"      || command -v llvm-cov)"

if [[ ! -x "${BIN}" ]]; then
  echo "error: ${BIN} not found." >&2
  echo "       Build with: cmake -DBUILD_TESTS=ON -DENABLE_CODE_COVERAGE=ON ..." >&2
  exit 1
fi

mkdir -p "${COV_DIR}"

# 1. Run the test binary with LLVM_PROFILE_FILE pointing at our .profraw.
echo "==> Running ${BIN}"
LLVM_PROFILE_FILE="${PROFRAW}" "${BIN}" "$@"

# 2. Merge raw profile into indexed profdata.
echo "==> Merging profile -> ${PROFDATA}"
"${PROFDATA_TOOL}" merge -sparse "${PROFRAW}" -o "${PROFDATA}"

# 3. Scope the report to the source file(s) actually under test. Add more
#    -sources as the binary grows.
SOURCES=(
  "${BUILD_DIR}/hipify/src/transport/p2p.cc"
)

echo "==> Coverage summary"
"${COV_TOOL}" report "${BIN}" \
  -instr-profile="${PROFDATA}" \
  "${SOURCES[@]}"

# Optional HTML output: --html <dir>
if [[ "${1:-}" == "--html" ]]; then
  HTML_DIR="${2:?--html requires an output directory}"
  mkdir -p "${HTML_DIR}"
  echo "==> Writing HTML report to ${HTML_DIR}"
  "${COV_TOOL}" show "${BIN}" \
    -instr-profile="${PROFDATA}" \
    -format=html -output-dir="${HTML_DIR}" \
    -show-line-counts-or-regions \
    "${SOURCES[@]}"
  echo "    Open: ${HTML_DIR}/index.html"
fi
