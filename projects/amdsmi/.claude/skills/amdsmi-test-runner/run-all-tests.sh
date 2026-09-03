#!/usr/bin/env bash
# Run all amd-smi test suites (C++ GTest, Python unit/integration/CLI) and tee
# each to a timestamped log dir. Exits non-zero if any suite fails.
# Derives the project root from git; no hard-coded users or paths.
#
# Usage: run-all-tests.sh [-o outdir] [--no-cpp] [--no-python]
set -uo pipefail

OUTDIR=""
RUN_CPP=1
RUN_PY=1
need_val() { [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out)    need_val "$@"; OUTDIR="$2"; shift 2;;
    --no-cpp)    RUN_CPP=0; shift;;
    --no-python) RUN_PY=0; shift;;
    -h|--help)   tail -n +2 "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# Project root: the dir holding tests/ and build/ (handles the monorepo layout).
ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
[[ -d "$ROOT/projects/amdsmi" ]] && ROOT="$ROOT/projects/amdsmi"
[[ -d "$ROOT/tests" ]] || { echo "error: no tests/ under $ROOT" >&2; exit 2; }

STAMP=$(date +%Y%m%d-%H%M%S)
OUTDIR="${OUTDIR:-${TMPDIR:-/tmp}/amdsmi-tests-${STAMP}}"
mkdir -p "$OUTDIR"
echo "logs -> $OUTDIR" >&2

NAMES=()
RCS=()

# C++ GTest with the skip blacklist + ASIC filter.
if [[ "$RUN_CPP" == 1 ]]; then
  if [[ -x "$ROOT/build/tests/amdsmitst" ]]; then
    echo "=== C++ GTest ===" >&2
    ( cd "$ROOT/build/tests" \
      && source "$ROOT/tests/amd_smi_test/amdsmitst.exclude" \
      && source "$ROOT/tests/amd_smi_test/detect_asic_filter.sh" \
      && ./amdsmitst --gtest_filter="-${GTEST_EXCLUDE:-}" ) 2>&1 | tee "$OUTDIR/cpp-gtest.log"
    NAMES+=("C++ GTest"); RCS+=("${PIPESTATUS[0]}")
  else
    echo "skip C++: build/tests/amdsmitst not found (build with -DBUILD_TESTS=ON)" >&2
  fi
fi

# Python suites — run the file runners directly (each sys.exit(0/1)).
if [[ "$RUN_PY" == 1 ]]; then
  for pair in "Python unit:unit_tests.py" \
              "Python integration:integration_test.py" \
              "Python CLI:cli_unit_test.py"; do
    name="${pair%%:*}"; file="${pair##*:}"
    if [[ -f "$ROOT/tests/python/$file" ]]; then
      echo "=== $name ===" >&2
      log="$(echo "$name" | tr ' A-Z' '-a-z').log"
      ( cd "$ROOT/tests/python" && python3 "$file" -v ) 2>&1 | tee "$OUTDIR/$log"
      NAMES+=("$name"); RCS+=("${PIPESTATUS[0]}")
    else
      echo "skip $name: tests/python/$file not found" >&2
    fi
  done
fi

# Summary.
echo
echo "===== SUMMARY ($OUTDIR) ====="
[[ ${#NAMES[@]} -eq 0 ]] && { echo "  (no suites ran)"; exit 2; }
fail=0
for i in "${!NAMES[@]}"; do
  if [[ "${RCS[$i]}" == 0 ]]; then status=PASS; else status=FAIL; fail=1; fi
  printf '  %-20s %s (rc=%s)\n' "${NAMES[$i]}" "$status" "${RCS[$i]}"
done
exit $fail
