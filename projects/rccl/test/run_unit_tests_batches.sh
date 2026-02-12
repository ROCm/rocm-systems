#!/usr/bin/env bash
#
# Run rccl-UnitTests in batches (category.subcategory).
# Usage:
#   ./test/run_unit_tests_batches.sh list              # list all batches
#   ./test/run_unit_tests_batches.sh run BATCH [BATCH...]  # run one or more batches
#   ./test/run_unit_tests_batches.sh run-all            # run all batches one by one
#
# Run from repo root. Set RCCL_UNIT_TEST_BINARY to override the default binary.
# UT_* env vars are passed through (e.g. UT_MAX_GPUS=1).
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="${RCCL_UNIT_TEST_BINARY:-$RCCL_ROOT/build/release/test/rccl-UnitTests}"

if [[ ! -x "$BINARY" ]]; then
  echo "Error: rccl-UnitTests not found or not executable: $BINARY" >&2
  echo "Build with: ./install.sh -l --device-linker -t" >&2
  exit 1
fi

# Output batch names (Category.Subcategory) one per line, from --gtest_list_tests
get_batches() {
  "$BINARY" --gtest_list_tests 2>/dev/null | awk '
    /^[A-Za-z0-9_]+\.$/ { suite = substr($0, 1, length($0)-1); next }
    /^  [A-Za-z0-9_]+$/ { if (suite != "") print suite "." $1; next }
  '
}

cmd_list() {
  echo "Batches (use with: $0 run CATEGORY.SUBCATEGORY):"
  echo "---"
  get_batches | nl
}

cmd_run() {
  if [[ $# -eq 0 ]]; then
    echo "Usage: $0 run BATCH [BATCH ...]" >&2
    echo "Example: $0 run AllReduce.OutOfPlace AllGather.InPlace" >&2
    exit 1
  fi
  filter="$1"
  shift
  for b in "$@"; do
    filter="$filter:$b"
  done
  exec "$BINARY" --gtest_filter="$filter"
}

cmd_run_all() {
  total=0
  passed=0
  failed=0
  while IFS= read -r batch; do
    [[ -z "$batch" ]] && continue
    ((total++)) || true
    if "$BINARY" --gtest_filter="$batch" "$@" >/dev/null 2>&1; then
      ((passed++)) || true
      echo "  PASS $batch"
    else
      ((failed++)) || true
      echo "  FAIL $batch"
    fi
  done < <(get_batches)
  echo "---"
  echo "Total: $total  Passed: $passed  Failed: $failed"
  [[ $failed -eq 0 ]]
}

case "${1:-}" in
  list)
    cmd_list
    ;;
  run)
    shift
    cmd_run "$@"
    ;;
  run-all)
    shift
    cmd_run_all "$@"
    ;;
  *)
    echo "Usage: $0 list | run BATCH [BATCH ...] | run-all" >&2
    echo "" >&2
    echo "  list     - list all batches (Category.Subcategory)" >&2
    echo "  run      - run one or more batches (--gtest_filter=BATCH:BATCH:...)" >&2
    echo "  run-all  - run each batch separately and report pass/fail" >&2
    echo "" >&2
    echo "Override binary: RCCL_UNIT_TEST_BINARY=/path/to/rccl-UnitTests $0 ..." >&2
    echo "UT_* env vars (e.g. UT_MAX_GPUS=1) are passed through to the test binary." >&2
    exit 1
    ;;
esac
