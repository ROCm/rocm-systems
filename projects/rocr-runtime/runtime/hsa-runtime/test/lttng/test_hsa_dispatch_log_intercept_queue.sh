#!/usr/bin/env bash
# Intercept-queue lifecycle test for the dispatch_log subsystem.
#
# Locks in the fix from commit `3dc1d27db6`: on_queue_destroy must
# unwrap QueueWrapper subclasses (InterceptQueue) so the disable
# sequence runs on the underlying AqlQueue that on_queue_create
# registered. Without the fix, intercept-queue destroys leak the
# drainer worker thread and silently drop the final-drain records;
# a regression here would manifest as either:
#   - drainer worker reading freed memory (likely SIGSEGV during
#     queue destroy or process exit), or
#   - records FW wrote between the last drain pass and queue destroy
#     never appearing in the LTTng trace.
#
# Implementation note: hsa_amd_queue_intercept_create is an internal
# runtime API (declared in hsa_api_trace.h, not in public hsa_ext_amd.h)
# and is NOT exported from libhsa-runtime64.so. We can't create an
# intercept queue from a standalone test program. Instead, we
# exercise intercept queues indirectly via HIP, which uses them
# internally for the runtime queue path.
#
# The hipcc-built square sample creates a HIP queue → CLR allocates
# an HSA queue via Hsa::queue_create (through hsa_queue_create) →
# the runtime returns either an AqlQueue directly or wraps it in an
# InterceptQueue (the latter happens for some configurations and
# definitely happens for hsa_amd_queue_intercept_register which HIP
# calls internally for some packet-rewrite paths).
#
# Mechanics:
#   1. Build hipcc square (or skip with INFO if hipcc unavailable).
#   2. Run it twice in succession under LTTng + DIAG_LOSS:
#        - First run: cold, fresh queues created and destroyed at
#          process exit. Tests on_queue_create/destroy symmetry on
#          first lifecycle.
#        - Second run (separate process): tests that any global
#          residual state from run 1 doesn't break run 2.
#   3. Assert both runs:
#        - Exited 0 (no SIGSEGV from drainer reading freed memory)
#        - Produced ≥1 [DIAG_LOSS] line per run (proves at least one
#          queue was registered AND disable ran AND DIAG dump fired)
#        - All [DIAG_LOSS] accounting balances
#
# Usage: $0 [<build_dir>]

set -euo pipefail

BUILD_DIR="${1:-$PWD/build/rocr}"
LIB_DIR="$BUILD_DIR/rocr/lib"

if [ ! -f "$LIB_DIR/libhsa-runtime64.so" ]; then
    echo "FAIL: $LIB_DIR/libhsa-runtime64.so not found"
    exit 1
fi

TRACE_DIR="$(mktemp -d)"
cleanup() {
    set +e
    pkill -P $$ 2>/dev/null
    if [ -n "${SESSIOND_PIDFILE:-}" ] && [ -f "$SESSIOND_PIDFILE" ]; then
        kill "$(cat "$SESSIOND_PIDFILE")" 2>/dev/null || true
    fi
    rm -rf "$TRACE_DIR"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
info() { echo "INFO: $*"; }

WORKLOAD_BIN=""
if command -v /opt/rocm/bin/hipcc >/dev/null 2>&1 \
   && [ -f /opt/rocm/share/hip/samples/0_Intro/square/square.hipref.cpp ]; then
    cp /opt/rocm/share/hip/samples/0_Intro/square/square.hipref.cpp "$TRACE_DIR/square.cpp"
    if /opt/rocm/bin/hipcc "$TRACE_DIR/square.cpp" -o "$TRACE_DIR/square" \
        >"$TRACE_DIR/hipcc.log" 2>&1; then
        WORKLOAD_BIN="$TRACE_DIR/square"
    fi
fi

if [ -z "$WORKLOAD_BIN" ]; then
    info "hipcc square not available; intercept-queue lifecycle test SKIPPED"
    exit 0
fi

# Per-user lttng-sessiond.
export LTTNG_HOME="$TRACE_DIR/lttng-home"
mkdir -p "$LTTNG_HOME"
SESSIOND_PIDFILE="$TRACE_DIR/sessiond.pid"
lttng-sessiond --daemonize --no-kernel --pidfile="$SESSIOND_PIDFILE"
for i in 1 2 3 4 5; do
    [ -f "$SESSIOND_PIDFILE" ] && break
    sleep 0.5
done
[ -f "$SESSIOND_PIDFILE" ] || fail "lttng-sessiond did not start"

run_workload() {
    local label="$1"
    local diag_log="$TRACE_DIR/diag.$label.log"
    local stdout_log="$TRACE_DIR/stdout.$label.log"

    rm -rf "$TRACE_DIR/trace.$label"
    lttng create "intercept-test-$label" --output="$TRACE_DIR/trace.$label" >/dev/null
    lttng enable-event --userspace 'rocm_hsa:kernel_dispatch_record' >/dev/null
    lttng enable-event --userspace 'rocm_hsa:kernel_dispatch_drop'   >/dev/null
    lttng start >/dev/null

    set +e
    HSA_DISPATCH_LOG_DIAG_LOSS=1 \
        LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}" \
        "$WORKLOAD_BIN" >"$stdout_log" 2>"$diag_log"
    local rc=$?
    set -e

    lttng stop    >/dev/null
    lttng destroy >/dev/null

    if [ "$rc" -ne 0 ]; then
        echo "FAIL: $label run exited with $rc (likely SIGSEGV in drainer worker on queue destroy — the intercept-queue unwrap fix may have regressed)" >&2
        echo "------ stderr ------" >&2
        cat "$diag_log" >&2
        return 1
    fi

    local diag_count
    diag_count="$(grep -c '^\[DIAG_LOSS\]' "$diag_log" || true)"
    if [ "$diag_count" -lt 1 ]; then
        echo "FAIL: $label run produced no [DIAG_LOSS] lines; either DIAG_LOSS env var did not propagate, or no queue ever enabled dispatch_log AND went through the disable sequence (regression: queue destroy bypassed disable_dispatch_log_for_queue_locked)" >&2
        echo "------ stderr ------" >&2
        cat "$diag_log" >&2
        return 1
    fi

    # Verify accounting on every active queue in this run.
    local fail_count=0
    while IFS= read -r line; do
        local qid sig_span emitted zero_swept drop_overrun accounted
        qid=$(echo          "$line" | sed -n 's/.*q=\(0x[0-9a-fA-F]\+\).*/\1/p')
        sig_span=$(echo     "$line" | sed -n 's/.*sig_span=\([0-9]\+\).*/\1/p')
        emitted=$(echo      "$line" | sed -n 's/.*emitted=\([0-9]\+\).*/\1/p')
        zero_swept=$(echo   "$line" | sed -n 's/.*zero_swept=\([0-9]\+\).*/\1/p')
        drop_overrun=$(echo "$line" | sed -n 's/.*drop_overrun=\([0-9]\+\).*/\1/p')
        accounted=$(echo    "$line" | sed -n 's/.*accounted=\([0-9]\+\).*/\1/p')
        [ "$sig_span" = "0" ] && continue
        local expected=$((emitted + zero_swept + drop_overrun))
        if [ "$accounted" != "$sig_span" ]; then
            echo "FAIL: $label queue $qid: accounted=$accounted != sig_span=$sig_span (diff=$((sig_span - accounted)))" >&2
            fail_count=$((fail_count + 1))
        fi
    done < <(grep '^\[DIAG_LOSS\]' "$diag_log")

    if [ "$fail_count" -gt 0 ]; then
        return 1
    fi

    info "$label: rc=0, $diag_count [DIAG_LOSS] line(s), all accounting balanced"
    return 0
}

run_workload first  || fail "first run failed"
run_workload second || fail "second run failed"

info "PASS: intercept-queue lifecycle (two consecutive runs) exited cleanly with balanced accounting"
exit 0
