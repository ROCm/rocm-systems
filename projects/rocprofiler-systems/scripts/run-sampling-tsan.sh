#!/usr/bin/env bash
# run-sampling-tsan.sh — canonical TSAN invocation for sampling unit tests.
#
# Usage:
#   scripts/run-sampling-tsan.sh [binary] [gtest-args...]
#
# Examples:
#   scripts/run-sampling-tsan.sh
#   scripts/run-sampling-tsan.sh build/debug/bin/rocprof-sys-unit-tests
#   scripts/run-sampling-tsan.sh build/debug/bin/rocprof-sys-unit-tests \
#       --gtest_filter="steady_clock_smoke*"
#
# Exit codes:
#   0  — GTest passed AND no ThreadSanitizer warnings in output
#   1  — GTest failed  OR  ThreadSanitizer warning detected
#
# Environment:
#   TSAN_BINARY  — override default binary path
#   TSAN_LOG     — path to write combined TSAN output (default: /tmp/tsan-sampling.log)
#
# NFR-TS-1 / NFR-TS-3: halt_on_error=0 collects all races in one run;
# second_deadlock_stack=1 captures both acquisition stacks for lock-order bugs;
# exitcode=0 lets GTest's own exit code propagate so CI sees pass/fail.

set -euo pipefail

# ── Resolve binary ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_BINARY="${REPO_ROOT}/build/debug/bin/rocprof-sys-unit-tests"
BINARY="${TSAN_BINARY:-${1:-${DEFAULT_BINARY}}}"

if [[ $# -ge 1 && -f "$1" ]]; then
    shift  # consumed as BINARY; remaining args passed to binary
fi

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: binary not found or not executable: ${BINARY}" >&2
    echo "  Build first: cmake --build build/debug --target rocprof-sys-unit-tests" >&2
    exit 1
fi

LOG="${TSAN_LOG:-/tmp/tsan-sampling.log}"

echo "=== TSAN run ===" | tee "${LOG}"
echo "Binary : ${BINARY}" | tee -a "${LOG}"
echo "Args   : $*" | tee -a "${LOG}"
echo "Log    : ${LOG}" | tee -a "${LOG}"
echo "====================================" | tee -a "${LOG}"

# ── Canonical invocation ───────────────────────────────────────────────────────
# setarch -R: disable ASLR for deterministic addresses in TSAN stack traces.
# halt_on_error=0: collect ALL races/deadlocks (not just the first).
# second_deadlock_stack=1: capture both lock-acquisition stacks for lock-order bugs.
# exitcode=0: TSAN itself exits 0; GTest exit code propagates as the shell exit.
set +e
setarch "$(uname -m)" -R \
    env TSAN_OPTIONS="halt_on_error=0:second_deadlock_stack=1:exitcode=0" \
    "${BINARY}" "$@" 2>&1 | tee -a "${LOG}"
GTEST_EXIT=${PIPESTATUS[0]}
set -e

# ── NFR-TS-3 hard gate: any TSAN warning = failure ────────────────────────────
TSAN_WARNINGS=$(grep -c "WARNING: ThreadSanitizer:" "${LOG}" || true)

echo "" | tee -a "${LOG}"
echo "====================================" | tee -a "${LOG}"
echo "GTest exit code  : ${GTEST_EXIT}" | tee -a "${LOG}"
echo "TSAN warnings    : ${TSAN_WARNINGS}" | tee -a "${LOG}"

if [[ "${TSAN_WARNINGS}" -gt 0 ]]; then
    echo "RESULT: FAIL — ${TSAN_WARNINGS} ThreadSanitizer warning(s) detected" | tee -a "${LOG}"
    exit 1
fi

if [[ "${GTEST_EXIT}" -ne 0 ]]; then
    echo "RESULT: FAIL — GTest returned exit code ${GTEST_EXIT}" | tee -a "${LOG}"
    exit 1
fi

echo "RESULT: PASS — GTest passed, no TSAN warnings" | tee -a "${LOG}"
exit 0
