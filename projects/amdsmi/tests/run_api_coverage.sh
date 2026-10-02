#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Run the AMD SMI test suites with API-call tracing on, then turn the logs into
# the KPI coverage tables via api_summary.py + api_coverage_report.py.
#
# Tracing is what makes the coverage numbers work: `amdsmitst -v 1` and the
# Python runners' `-v` make DISPLAY_AMDSMI_API()/_build_call_msg() emit the
# "### amdsmi_<api>(" markers that api_summary.py counts.
#
# Must run as root -- the Python runners exit(1) when geteuid() != 0:
#
#     sudo projects/amdsmi/tests/run_api_coverage.sh
#     sudo projects/amdsmi/tests/run_api_coverage.sh -o ./kpi-logs
#     sudo projects/amdsmi/tests/run_api_coverage.sh -t ./build/tests   # build tree
#     sudo projects/amdsmi/tests/run_api_coverage.sh -n "post-XYZ fix"  # label the run
#
# Options:
#   -o, --out DIR        where logs and reports land
#                        (default ./api-coverage-results-<timestamp>)
#   -t, --tests-dir DIR  installed or build test dir holding amdsmitst
#   -H, --history FILE   accumulate the trend in one file instead of inheriting
#                        it from the previous run directory
#   -n, --notes TEXT     free-text note recorded with this run
#   -U, --unit-filter P  positive GTest pattern for the Unit column (default *Unit*)
#   -F, --func-filter P  positive GTest pattern for the Functional column
#                        (default *Functional*)
#   -X, --extra-exclude L  extra colon-separated GTest exclusions, appended to the
#                        per-ASIC list from amdsmitst.exclude
#   -A, --no-exclude     run everything: drop the per-ASIC blacklist entirely
#   -v, --verbose        mirror suite output to the terminal as well as the log
#       --no-cpp         skip the C++ suites
#       --no-python      skip the Python suites
#
# GTest selection is "<positive>-<negative>": the positive half comes from
# -U/-F, the negative half from amdsmitst.exclude via detect_asic_filter.sh
# (which picks the list matching the detected ASIC), plus anything in -X.
# The defaults *Unit* and *Functional* together match every registered suite,
# so the only thing dropping tests is the negative half -- use -A to disable it.
#
# Examples:
#     sudo ... run_api_coverage.sh -A                       # no filtering at all
#     sudo ... run_api_coverage.sh -U 'GpuUnit*' -F 'GpuFunctional*'
#     sudo ... run_api_coverage.sh -X 'GpuFunctionalReadWrite*' --no-python
#
# Produces a self-contained ./api-coverage-results-<timestamp>/ holding
# _api_coverage.md (report), _api_coverage.csv and _api_coverage.tsv (the
# report's tables, for opening and for pasting into a spreadsheet), the raw
# suite logs, and api_coverage_trend.csv carrying the trend forward from the
# newest earlier run directory. Nothing is written into the source tree.
#
# Every '_' file is disposable output. api_coverage_trend.csv has no underscore
# because it is the one generated file worth keeping: delete it and the trend
# is gone. It records hardware/driver configuration, never the hostname.
#
# Coverage is hardware- and driver-dependent: CPU/ESMI needs an EPYC host with
# HSMP, the NIC rows need a supported NIC, and IFoE needs ifoe.ko. On a consumer
# GPU those suites skip and their rows read 0%; the report's metadata block
# records the driver versions so a low number is explainable.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="${AMDSMI_PATH:-/opt/rocm/share/amd_smi}/tests"
OUTDIR=""
HISTORY=""
NOTES=""
# Positive GTest selection, split so the Unit and Functional columns stay
# separate. The per-ASIC exclusions from amdsmitst.exclude are appended as the
# negative half at run time.
UNIT_FILTER="*Unit*"
FUNC_FILTER="*Functional*"
EXTRA_EXCLUDE=""
NO_EXCLUDE=0
VERBOSE=0
SKIP_CPP=0
SKIP_PY=0

usage() { tail -n +5 "$0" | sed -n '/^#/!q; s/^# \{0,1\}//p'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o | --out)
      [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }
      OUTDIR="$2"; shift 2 ;;
    -t | --tests-dir)
      [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }
      TESTS_DIR="$2"; shift 2 ;;
    -H | --history)
      [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }
      HISTORY="$2"; shift 2 ;;
    -n | --notes)
      [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }
      NOTES="$2"; shift 2 ;;
    -U | --unit-filter)
      [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }
      UNIT_FILTER="$2"; shift 2 ;;
    -F | --func-filter)
      [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }
      FUNC_FILTER="$2"; shift 2 ;;
    -X | --extra-exclude)
      [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }
      EXTRA_EXCLUDE="$2"; shift 2 ;;
    -A | --no-exclude) NO_EXCLUDE=1; shift ;;
    -v | --verbose) VERBOSE=1; shift ;;
    --no-cpp) SKIP_CPP=1; shift ;;
    --no-python) SKIP_PY=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$(id -u)" -eq 0 ]] || echo "warning: not root -- the Python suites will exit(1)" >&2
[[ -x "$TESTS_DIR/amdsmitst" ]] || {
  echo "error: $TESTS_DIR/amdsmitst not found; pass --tests-dir or set AMDSMI_PATH" >&2
  exit 2
}

# Each run writes a self-contained, timestamped directory in the current
# directory. Nothing is written into the source tree, so generated data never
# lands in git; the trend carries forward from the newest earlier run dir.
OUTDIR="${OUTDIR:-$PWD/api-coverage-results-$(date +%Y-%m-%dT%H-%M-%S)}"
mkdir -p "$OUTDIR" || exit 2
OUTDIR="$(cd "$OUTDIR" && pwd)"
echo "logs -> $OUTDIR" >&2

# The installed tree names the Python suites python_unittest/; the source tree
# calls it python/. Accept either so this works before and after `make install`.
PYDIR="$TESTS_DIR/python_unittest"
[[ -d "$PYDIR" ]] || PYDIR="$TESTS_DIR/python"
[[ -d "$PYDIR" ]] || PYDIR="$SRC_ROOT/tests/python"

# The ASIC filter scripts ship next to the installed binary, but a build tree
# only holds amdsmitst -- fall back to the source copies so -t works there too.
# Not needed at all under -A, which never consults the blacklist.
FILTER_DIR="$TESTS_DIR"
[[ -f "$FILTER_DIR/amdsmitst.exclude" ]] || FILTER_DIR="$SRC_ROOT/tests/amd_smi_test"
if [[ "$NO_EXCLUDE" == 0 && ! -f "$FILTER_DIR/amdsmitst.exclude" ]]; then
  echo "error: amdsmitst.exclude not found in $TESTS_DIR or $SRC_ROOT/tests/amd_smi_test" >&2
  echo "       (pass -A to run without the per-ASIC blacklist)" >&2
  exit 2
fi

# ---- Suite status: recorded so a failed run cannot hide behind a full report ----
STATUS_CSV="$OUTDIR/_suite_status.csv"
echo "suite,exit_code,log" >"$STATUS_CSV"
record() { echo "$1,$2,$3" >>"$STATUS_CSV"; }

# Run a suite, capturing its output to a log. Under -v the output is mirrored to
# the terminal as well, so a long run can be watched live. The suite's own exit
# code must survive: in the tee pipeline $? is tee's, so take PIPESTATUS[0].
run_logged() {
  local log="$1"; shift
  if [[ "$VERBOSE" == 1 ]]; then
    "$@" 2>&1 | tee "$log"
    return "${PIPESTATUS[0]}"
  fi
  "$@" >"$log" 2>&1
}

# ---- C++ GTest: split unit vs functional so the columns stay separate ----
if [[ "$SKIP_CPP" == 0 ]]; then
  (
    cd "$TESTS_DIR" || exit 2
    exclude=""
    if [[ "$NO_EXCLUDE" == 1 ]]; then
      # Skip sourcing entirely rather than computing GTEST_EXCLUDE and throwing
      # it away: detect_asic_filter.sh prints its own "Detected ASIC" and "Final
      # gtest negative filter" lines, which would claim a blacklist is in force
      # when -A means none is.
      echo "-A: per-ASIC blacklist disabled, running every test" >&2
    else
      # shellcheck disable=SC1091
      source "$FILTER_DIR/amdsmitst.exclude"
      # shellcheck disable=SC1091
      source "$FILTER_DIR/detect_asic_filter.sh"
      exclude="${GTEST_EXCLUDE:-}"
    fi
    # -X appends, so a run can be narrowed without editing amdsmitst.exclude.
    [[ -n "$EXTRA_EXCLUDE" ]] && exclude="${exclude:+$exclude:}$EXTRA_EXCLUDE"
    # A trailing "-" with nothing after it is not a valid GTest filter, so only
    # add the negative half when there is something to exclude.
    unit_arg="$UNIT_FILTER${exclude:+-$exclude}"
    func_arg="$FUNC_FILTER${exclude:+-$exclude}"
    echo "=== C++ unit ===   filter=$unit_arg" >&2
    run_logged "$OUTDIR/_c_unit_test.log" ./amdsmitst -v 1 --gtest_filter="$unit_arg"
    echo "c_unit,$?,_c_unit_test.log" >>"$STATUS_CSV"
    echo "=== C++ functional ===   filter=$func_arg" >&2
    run_logged "$OUTDIR/_c_func_test.log" ./amdsmitst -v 1 --gtest_filter="$func_arg"
    echo "c_func,$?,_c_func_test.log" >>"$STATUS_CSV"
  )
fi

# ---- Python suites ----
run_py() {
  local script="$1" log="$2" label="$3" key="$4"
  if [[ -f "$PYDIR/$script" ]]; then
    echo "=== $label ===" >&2
    (cd "$PYDIR" && run_logged "$OUTDIR/$log" python3 "$script" -v)
    record "$key" "$?" "$log"
  else
    echo "skip $label: $PYDIR/$script not found" >&2
    record "$key" "skipped" "$log"
  fi
}
if [[ "$SKIP_PY" == 0 ]]; then
  run_py unit_tests.py _py_unit_test.log "Python unit" py_unit
  run_py integration_test.py _py_func_test.log "Python functional" py_func
  run_py cli_unit_test.py _py_cli_test.log "Python CLI" py_cli
fi

# ---- Reports ----
# api_summary.py always looks for six logs, but no runner drives tests/python/
# integration/, and there is no C++ integration suite -- so _c_intg_test.log and
# _py_intg_test.log can never exist. Its WARNING-level "Missing file" notes for
# those are structural noise, so raise the threshold and check the logs we do
# expect ourselves.
missing=0
for log in _c_unit_test.log _c_func_test.log _py_unit_test.log _py_func_test.log; do
  if [[ ! -s "$OUTDIR/$log" ]]; then
    echo "warning: expected log is missing or empty: $log" >&2
    missing=1
  fi
done
[[ "$missing" == 1 ]] && echo "warning: coverage numbers below are incomplete" >&2

python3 "$SCRIPT_DIR/api_summary.py" --verbose ERROR \
  --amdsmi "$SRC_ROOT/include/amd_smi/amdsmi.h" \
  --log_dir "$OUTDIR" --output_dir "$OUTDIR" || exit 1

# api_coverage_report.py writes _api_coverage.{md,csv,tsv} into $OUTDIR, plus
# api_coverage_trend.csv carrying forward the previous run dir's rows.
REPORT_ARGS=(--log_dir "$OUTDIR" --amdsmi "$SRC_ROOT/include/amd_smi/amdsmi.h")
[[ -n "$HISTORY" ]] && REPORT_ARGS+=(--history "$HISTORY")
[[ -n "$NOTES" ]] && REPORT_ARGS+=(--notes "$NOTES")
python3 "$SCRIPT_DIR/api_coverage_report.py" "${REPORT_ARGS[@]}" || exit 1

# Hand the logs back to the invoking user so they are readable after sudo.
if [[ -n "${SUDO_UID:-}" ]]; then
  chown -R "$SUDO_UID:${SUDO_GID:-$SUDO_UID}" "$OUTDIR"
  [[ -n "$HISTORY" && -f "$HISTORY" ]] && chown "$SUDO_UID:${SUDO_GID:-$SUDO_UID}" "$HISTORY"
fi

# ---- Suite summary: non-zero exit here if any suite failed ----
failed=$(awk -F, 'NR>1 && $2 != 0 && $2 != "skipped"' "$STATUS_CSV" | wc -l)
echo >&2
echo "===== SUITE STATUS =====" >&2
awk -F, 'NR>1 {printf "  %-10s %s\n", $1, ($2==0 ? "PASS" : ($2=="skipped" ? "SKIPPED" : "FAIL (rc="$2")"))}' \
  "$STATUS_CSV" >&2
echo "done -> $OUTDIR/_api_coverage.md" >&2
if [[ "$failed" -gt 0 ]]; then
  echo "WARNING: $failed suite(s) failed -- coverage numbers are incomplete" >&2
  exit 1
fi
