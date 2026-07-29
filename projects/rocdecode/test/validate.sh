#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# validate.sh — rocDecode code change validation script
# Runs CTest and conformance tests; prints combined summary and exits 0/1.
#
# Conformance streams are located via the ROCDECODE_CONFORMANCE_DIR environment
# variable (default: $HOME/rocDecodeConformance). That directory must contain the
# per-codec subdirectories AvcConformance, Av1Conformance, HevcConformance, and
# Vp9Conformance. Codecs whose directory is absent are reported as WARNING and
# skipped. Point the variable at your local stream collection, e.g.:
#   export ROCDECODE_CONFORMANCE_DIR="$HOME/Movies/rocDecodeConformance"
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
SKIP_CTEST=0
SKIP_CONFORMANCE=0

usage() {
  cat <<'EOF'
Usage: validate.sh [--build-dir DIR] [--skip-ctest] [--skip-conformance]

Runs the rocDecode CTest suite and per-codec conformance tests, then prints a
combined summary. Exits 0 if everything passed, 1 otherwise.

Options:
  --build-dir DIR      Path to the CMake build directory (default: ../build)
  --skip-ctest         Skip the CTest phase
  --skip-conformance   Skip all conformance phases
  -h, --help           Show this help and exit

Environment:
  ROCDECODE_CONFORMANCE_DIR   Parent directory holding the per-codec conformance
                              stream folders AvcConformance, Av1Conformance,
                              HevcConformance, Vp9Conformance.
                              Default: $HOME/rocDecodeConformance
EOF
}

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --build-dir)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --build-dir requires a path argument" >&2
        echo >&2
        usage >&2
        exit 1
      fi
      BUILD_DIR="$2"; shift 2
      ;;
    --skip-ctest) SKIP_CTEST=1; shift ;;
    --skip-conformance) SKIP_CONFORMANCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; echo; usage; exit 1 ;;
  esac
done

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="$SCRIPT_DIR/../validation_results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

# ANSI colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

declare -A RESULTS
declare -a PHASE_ORDER

# --- Phase 1: CTest ---
# Label carries the actual test count (varies with -DENABLE_EXTENDED_TESTS); query it up
# front so the summary is accurate whether or not the extended FFmpeg tests are enabled.
CTEST_PHASE="CTest"
if [[ -d "$BUILD_DIR" ]]; then
  TEST_COUNT=$(ctest --test-dir "$BUILD_DIR" -N 2>/dev/null | grep -oE "Total Tests: [0-9]+" | grep -oE "[0-9]+")
  [[ -n "$TEST_COUNT" ]] && CTEST_PHASE="CTest ($TEST_COUNT tests)"
fi
PHASE_ORDER+=("$CTEST_PHASE")
if [[ $SKIP_CTEST -eq 0 ]]; then
  echo "=== Running CTest ==="
  if [[ ! -d "$BUILD_DIR" ]]; then
    echo "ERROR: Build directory not found: $BUILD_DIR" >&2
    RESULTS["$CTEST_PHASE"]="FAILED"
  else
    make -C "$BUILD_DIR" test ARGS="-VV" 2>&1 | tee "$RESULTS_DIR/ctest_output.txt"
    if grep -qE "100% tests passed|0 tests failed" "$RESULTS_DIR/ctest_output.txt"; then
      RESULTS["$CTEST_PHASE"]="PASSED"
    else
      RESULTS["$CTEST_PHASE"]="FAILED"
    fi
  fi
else
  RESULTS["$CTEST_PHASE"]="SKIPPED"
fi

# --- Phase 2: Conformance ---
# Conformance streams live under a single parent dir (default: $HOME/rocDecodeConformance),
# overridable via ROCDECODE_CONFORMANCE_DIR, with one subdirectory per codec.
CONFORMANCE_DIR="${ROCDECODE_CONFORMANCE_DIR:-$HOME/rocDecodeConformance}"
declare -A CODEC_DIRS=(
  [AVC]="$CONFORMANCE_DIR/AvcConformance"
  [AV1]="$CONFORMANCE_DIR/Av1Conformance"
  [HEVC]="$CONFORMANCE_DIR/HevcConformance"
  [VP9]="$CONFORMANCE_DIR/Vp9Conformance"
)
CODEC_ORDER=(AVC AV1 HEVC VP9)

if [[ $SKIP_CONFORMANCE -eq 0 ]]; then
  pushd "$SCRIPT_DIR/testScripts" > /dev/null
  for CODEC in "${CODEC_ORDER[@]}"; do
    PHASE_ORDER+=("Conformance $CODEC")
    DIR="${CODEC_DIRS[$CODEC]}"
    if [[ ! -d "$DIR" ]]; then
      echo "WARNING: Conformance directory not found for $CODEC: $DIR (skipping)"
      echo "         Set ROCDECODE_CONFORMANCE_DIR to the parent of the *Conformance folders."
      echo "         (currently: ${ROCDECODE_CONFORMANCE_DIR:-<unset, using default \$HOME/rocDecodeConformance>})"
      RESULTS["Conformance $CODEC"]="WARNING"
      continue
    fi
    echo "=== Running Conformance: $CODEC ==="
    OUT="$RESULTS_DIR/conformance_${CODEC}_output.txt"
    python3 run_rocDecode_Conformance.py \
      --rocDecode_directory "$SCRIPT_DIR/.." \
      --files_directory "$DIR" \
      2>&1 | tee "$OUT"
    if grep -q "The number of failing streams is 0" "$OUT" && \
       grep -q "The number of streams that did not finish decoding is 0" "$OUT"; then
      RESULTS["Conformance $CODEC"]="PASSED"
    else
      RESULTS["Conformance $CODEC"]="FAILED"
    fi
  done
  popd > /dev/null
else
  for CODEC in "${CODEC_ORDER[@]}"; do
    PHASE_ORDER+=("Conformance $CODEC")
    RESULTS["Conformance $CODEC"]="SKIPPED"
  done
fi

# --- Summary ---
# Determine max phase name length for padding
MAX_LEN=0
for PHASE in "${PHASE_ORDER[@]}"; do
  if (( ${#PHASE} > MAX_LEN )); then
    MAX_LEN=${#PHASE}
  fi
done
# Column widths: left pad=2, phase name, gap=2, status label+2, right pad=2
STATUS_WIDTH=10  # " PASSED   " or " FAILED   " or " SKIPPED  " or " WARNING  "
BOX_WIDTH=$(( 2 + MAX_LEN + 2 + STATUS_WIDTH + 2 ))

pad_right() {
  local s="$1"
  local width="$2"
  printf "%-${width}s" "$s"
}

hr() {
  local left="$1" fill="$2" right="$3"
  printf "%s" "$left"
  printf '%0.s'"$fill" $(seq 1 $(( BOX_WIDTH - 2 )))
  printf "%s\n" "$right"
}

OVERALL_FAILED=0
echo ""
hr "╔" "═" "╗"
printf "║  %-$(( BOX_WIDTH - 4 ))s  ║\n" "rocDecode Validation Summary"
hr "╠" "═" "╣"

for PHASE in "${PHASE_ORDER[@]}"; do
  STATUS="${RESULTS[$PHASE]}"
  case "$STATUS" in
    PASSED)
      COLOR="$GREEN"
      LABEL="✓ PASSED"
      ;;
    FAILED)
      COLOR="$RED"
      LABEL="✗ FAILED"
      OVERALL_FAILED=1
      ;;
    SKIPPED)
      COLOR="$YELLOW"
      LABEL="- SKIPPED"
      ;;
    WARNING)
      COLOR="$YELLOW"
      LABEL="⚠ WARNING"
      ;;
    *)
      COLOR="$NC"
      LABEL="? UNKNOWN"
      OVERALL_FAILED=1
      ;;
  esac
  INNER_WIDTH=$(( BOX_WIDTH - 4 ))   # width between "║  " and "  ║"
  PHASE_FIELD_WIDTH=$(( INNER_WIDTH - ${#LABEL} - 1 ))
  printf "║  %-${PHASE_FIELD_WIDTH}s ${COLOR}%s${NC}  ║\n" "$PHASE" "$LABEL"
done

hr "╠" "═" "╣"

if [[ $OVERALL_FAILED -eq 0 ]]; then
  OVERALL_LABEL="✓ PASSED"
  OVERALL_COLOR="$GREEN"
else
  OVERALL_LABEL="✗ FAILED"
  OVERALL_COLOR="$RED"
fi
INNER_WIDTH=$(( BOX_WIDTH - 4 ))
PHASE_FIELD_WIDTH=$(( INNER_WIDTH - ${#OVERALL_LABEL} - 1 ))
printf "║  %-${PHASE_FIELD_WIDTH}s ${OVERALL_COLOR}%s${NC}  ║\n" "Overall" "$OVERALL_LABEL"

hr "╚" "═" "╝"
echo "Results saved to: $RESULTS_DIR/"
echo ""

exit $OVERALL_FAILED
