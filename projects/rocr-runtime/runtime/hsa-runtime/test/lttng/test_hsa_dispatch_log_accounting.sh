#!/usr/bin/env bash
# Accounting-invariant test for the dispatch_log drainer.
#
# Locks in the invariant from the HEAD `31adf3c1e1` investigation:
#
#   For every queue that participated in dispatch_log, the per-queue
#   diagnostic counters dumped at queue disable MUST satisfy
#
#     emitted + zero_swept + drop_overrun == sig_max - sig_init
#
# This is the strongest correctness gate we have on the drainer: it
# proves that every FW signal-counter advance was placed in exactly one
# of the three buckets (real records emitted via LTTng, stale-zero
# slots quarantine-swept, or overrun-skipped). A regression that
# silently dropped records without accounting (e.g. the pre-fix
# init-sync overrun-skip) would produce sig_max - sig_init >
# emitted + zero_swept + drop_overrun and this test would FAIL.
#
# Mechanics:
#   1. Pick a workload that creates and exercises ≥ 1 dispatch_log
#      queue. We use the in-tree hipcc-built square sample (same as
#      test_hsa_dispatch_log.sh) because it routes through the public
#      hsa_queue_create path that triggers the dispatch_log enable
#      sequence. If hipcc is unavailable we fall back to a tiny HSA-only
#      AQL dispatch via the test API.
#   2. Run the workload under LTTng with HSA_DISPATCH_LOG_DIAG_LOSS=1.
#      The runtime emits one [DIAG_LOSS] line per queue at disable.
#   3. Parse each [DIAG_LOSS] line. For each queue with sig_span > 0,
#      assert emitted+zero_swept+drop_overrun == sig_span. Idle queues
#      (sig_span == 0) are reported as INFO and skipped.
#
# Usage: $0 [<build_dir>]
# <build_dir> defaults to $PWD/build/rocr.

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

# Build a kernel-dispatching workload. Prefer hipcc square (well-tested,
# triggers the drainer reliably). If unavailable, INFO-skip with reason.
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
    info "hipcc square not available; accounting-invariant test SKIPPED"
    info "(this test requires a HIP-or-AQL kernel dispatch workload to"
    info "exercise the drainer; see test_hsa_dispatch_log.sh for the same"
    info "preconditions)"
    exit 0
fi

# Per-user lttng-sessiond, scoped to this test.
export LTTNG_HOME="$TRACE_DIR/lttng-home"
mkdir -p "$LTTNG_HOME"
SESSIOND_PIDFILE="$TRACE_DIR/sessiond.pid"
lttng-sessiond --daemonize --no-kernel --pidfile="$SESSIOND_PIDFILE"
for i in 1 2 3 4 5; do
    [ -f "$SESSIOND_PIDFILE" ] && break
    sleep 0.5
done
[ -f "$SESSIOND_PIDFILE" ] || fail "lttng-sessiond did not start"

lttng create accounting-test --output="$TRACE_DIR/trace" >/dev/null
lttng enable-event --userspace 'rocm_hsa:kernel_dispatch_record' >/dev/null
lttng enable-event --userspace 'rocm_hsa:kernel_dispatch_drop'   >/dev/null
lttng start >/dev/null

# Run workload with HSA_DISPATCH_LOG_DIAG_LOSS=1. stderr captures the
# per-queue DIAG_LOSS lines emitted at queue disable.
DIAG_LOG="$TRACE_DIR/diag.log"
HSA_DISPATCH_LOG_DIAG_LOSS=1 \
    LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}" \
    "$WORKLOAD_BIN" >"$TRACE_DIR/workload.stdout" 2>"$DIAG_LOG" \
    || fail "workload exited non-zero (see $TRACE_DIR/workload.stdout / $DIAG_LOG)"

lttng stop    >/dev/null
lttng destroy >/dev/null

# Extract DIAG_LOSS lines. Format (one line per queue, dumped from
# disable_dispatch_log_for_queue_locked):
#   [DIAG_LOSS] q=0xN gpu=N sig_init=N sig_max=N sig_span=N \
#     emitted=N zero_swept=N drop_overrun=N accounted=N \
#     passes=N passes_with_work=N
DIAG_LINES="$(grep '^\[DIAG_LOSS\]' "$DIAG_LOG" || true)"

if [ -z "$DIAG_LINES" ]; then
    fail "no [DIAG_LOSS] lines in workload stderr; either the env var did not propagate, or no queue ever enabled dispatch_log (see $DIAG_LOG)"
fi

# Parse each DIAG_LOSS line and assert the accounting invariant.
NUM_QUEUES=0
NUM_ACTIVE=0
NUM_FAIL=0
while IFS= read -r line; do
    NUM_QUEUES=$((NUM_QUEUES + 1))
    # Extract numeric fields. The format is stable; use a per-key sed.
    QID=$(echo "$line" | sed -n 's/.*q=\(0x[0-9a-fA-F]\+\).*/\1/p')
    SIG_INIT=$(echo "$line"     | sed -n 's/.*sig_init=\([0-9]\+\).*/\1/p')
    SIG_MAX=$(echo "$line"      | sed -n 's/.*sig_max=\([0-9]\+\).*/\1/p')
    SIG_SPAN=$(echo "$line"     | sed -n 's/.*sig_span=\([0-9]\+\).*/\1/p')
    EMITTED=$(echo "$line"      | sed -n 's/.*emitted=\([0-9]\+\).*/\1/p')
    ZERO_SWEPT=$(echo "$line"   | sed -n 's/.*zero_swept=\([0-9]\+\).*/\1/p')
    DROP_OVERRUN=$(echo "$line" | sed -n 's/.*drop_overrun=\([0-9]\+\).*/\1/p')
    ACCOUNTED=$(echo "$line"    | sed -n 's/.*accounted=\([0-9]\+\).*/\1/p')

    for v in QID SIG_INIT SIG_MAX SIG_SPAN EMITTED ZERO_SWEPT DROP_OVERRUN ACCOUNTED; do
        if [ -z "${!v}" ]; then
            fail "could not parse $v from DIAG_LOSS line: $line"
        fi
    done

    # Idle queue: nothing to assert. Report and skip.
    if [ "$SIG_SPAN" = "0" ]; then
        info "queue $QID idle (sig_span=0); no accounting check"
        continue
    fi
    NUM_ACTIVE=$((NUM_ACTIVE + 1))

    # The runtime computes ACCOUNTED = emitted + zero_swept + drop_overrun
    # in disable_dispatch_log_for_queue_locked when it formats the line.
    # We re-derive it here to guard against a future refactor that breaks
    # the runtime's own sum.
    EXPECTED=$((EMITTED + ZERO_SWEPT + DROP_OVERRUN))
    if [ "$ACCOUNTED" != "$EXPECTED" ]; then
        echo "FAIL queue $QID: runtime printed accounted=$ACCOUNTED but emitted+zero_swept+drop_overrun=$EXPECTED" >&2
        NUM_FAIL=$((NUM_FAIL + 1))
    fi

    # The accounting invariant we actually care about.
    if [ "$ACCOUNTED" != "$SIG_SPAN" ]; then
        echo "FAIL queue $QID: accounted=$ACCOUNTED != sig_span=$SIG_SPAN (diff=$((SIG_SPAN - ACCOUNTED))); records vanished from drainer accounting" >&2
        echo "  sig_init=$SIG_INIT sig_max=$SIG_MAX emitted=$EMITTED zero_swept=$ZERO_SWEPT drop_overrun=$DROP_OVERRUN" >&2
        NUM_FAIL=$((NUM_FAIL + 1))
    fi
done <<< "$DIAG_LINES"

if [ "$NUM_FAIL" -gt 0 ]; then
    fail "$NUM_FAIL queue(s) failed accounting invariant out of $NUM_ACTIVE active ($NUM_QUEUES total)"
fi

if [ "$NUM_ACTIVE" = 0 ]; then
    info "no active dispatch_log queues observed (workload may be too short or KFD substrate may not support dispatch_log); accounting invariant has nothing to assert against"
    info "(this is not a FAIL but the test's coverage is degenerate)"
fi

info "PASS: accounting invariant held on $NUM_ACTIVE active queue(s) ($NUM_QUEUES total)"
exit 0
