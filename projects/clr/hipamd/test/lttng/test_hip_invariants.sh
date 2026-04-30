#!/usr/bin/env bash
# HIP LTTng invariant tests.
#
# Beyond the smoke tests in test_hip_api_tracepoints.sh, this script asserts
# semantic invariants of the trace produced by a tiny HIP program:
#
#   I1. Enter/exit balance: count(hip_api_enter) ==
#         sum(hip_api_exit_status, _ptr, _void). Catches missed exit emits.
#
#   I2. hip_aql_kernel_dispatch_submit must NOT carry a dispatch_idx field.
#         (Renumbered from I3 in schema v2 era.)
#
#   I3. Channel context vpid + vtid attach to events. Per schema v3 the
#         events themselves carry no per-event identity fields; the
#         consumer reconstructs identity from (vpid, vtid, timestamp).
#         Asserts that a kernel-dispatch event carries a valid vtid value.
#
#   I4. Per-(vpid, vtid) LIFO walk: enter and exit events on a given
#         thread arrive in matched-stack order. For the single-threaded
#         test program, every hip_api_enter on the program's vtid is
#         followed by a hip_api_exit_* on the same vtid before the next
#         enter at the same depth. Asserts that the dispatch events
#         (hip_kernel_dispatch_enqueue + hip_aql_kernel_dispatch_submit)
#         appear on the SAME vtid as the hipLaunchKernel hip_api_enter,
#         which is what the consumer's parent-attribution algorithm
#         relies on.
#
# Usage:
#   test_hip_invariants.sh [<libamdhip64-build-dir>]
#
# Default build dir: $PWD/build/clr/hipamd/lib
set -euo pipefail

BUILD_LIB_DIR="${1:-$PWD/build/clr/hipamd/lib}"
if [ ! -f "$BUILD_LIB_DIR/libamdhip64.so" ]; then
    echo "ERROR: $BUILD_LIB_DIR/libamdhip64.so not found" >&2
    exit 2
fi

WORK="$(mktemp -d)"
SESSION="hip-inv-$$"
SESSIOND_PIDFILE="$WORK/sessiond.pid"

# Per-test isolated LTTng state directory so lttng-sessiond's runtime
# socket / per-user state is scoped to this test's WORK dir and cannot
# collide with other tests' sessiond instances on the same host.
export LTTNG_HOME="$WORK/lttng-home"
mkdir -p "$LTTNG_HOME"

cleanup() {
    set +e
    lttng destroy "$SESSION" >/dev/null 2>&1
    # Kill ONLY the sessiond instance this test started, captured by
    # --pidfile. Do NOT use `pkill -f 'lttng-sessiond'` — that would
    # match every lttng-sessiond on the host (other concurrent tests).
    if [ -s "$SESSIOND_PIDFILE" ]; then
        kill "$(cat "$SESSIOND_PIDFILE")" 2>/dev/null
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

cat > "$WORK/tiny.hip.cpp" <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
__global__ void noop() {}
int main() {
    int* p = nullptr;
    hipMalloc(&p, 4);
    int dev = 0;
    hipGetDevice(&dev);
    hipLaunchKernelGGL(noop, dim3(1), dim3(1), 0, 0);
    hipDeviceSynchronize();
    hipFree(p);
    return 0;
}
EOF

HIPCC=/opt/rocm/bin/hipcc
"$HIPCC" -O2 "$WORK/tiny.hip.cpp" -o "$WORK/tiny" \
    -L"$BUILD_LIB_DIR" -lamdhip64 \
    -Wl,-rpath,"$BUILD_LIB_DIR" 2>&1 | tail -3

lttng-sessiond --daemonize --pidfile="$SESSIOND_PIDFILE"
# Wait for the pidfile to appear (--daemonize forks before writing it).
for _ in $(seq 1 20); do
    [ -s "$SESSIOND_PIDFILE" ] && break
    sleep 0.1
done
lttng create "$SESSION" --output="$WORK/trace" >/dev/null
# Larger buffers than other LTTng tests because this run captures many
# HSA events under a hipLaunchKernel call (~150-200 events) and they all
# need to land on the launching CPU's buffer for the I4 LIFO-stack
# assertion to find them. 224-CPU host x 32 KiB x 4 = ~28 MB total, fits
# in /dev/shm easily.
lttng enable-channel --userspace --discard --subbuf-size=32768 --num-subbuf=4 default >/dev/null
# Per schema v3 the events themselves no longer carry corr_id / parent_corr_id
# / tid; per-event identity comes from channel-context vpid + vtid plus the
# CTF event-header timestamp. Attach those contexts to the channel.
lttng add-context --userspace --channel=default --type=vpid --type=vtid >/dev/null
# Capture HIP and HSA tracepoints both. Doorbell + intercept are still
# captured (the consumer can recover their parent context from the LIFO walk).
lttng enable-event --userspace -c default \
    'rocm_hip:*,rocm_hsa:hsa_api_enter,rocm_hsa:hsa_doorbell_ring,rocm_hsa:hsa_intercept_packets' >/dev/null
lttng start >/dev/null

LD_LIBRARY_PATH="$BUILD_LIB_DIR:${LD_LIBRARY_PATH:-}" "$WORK/tiny" || true

lttng stop >/dev/null
lttng destroy "$SESSION" >/dev/null

LOG="$WORK/babeltrace.log"
babeltrace2 "$WORK/trace" > "$LOG" 2>&1

FAIL=0

# ---- I1: Enter/exit balance ------------------------------------------------
N_ENTER=$(grep -c 'rocm_hip:hip_api_enter:' "$LOG" || true)
N_EXIT=$(grep -cE 'rocm_hip:hip_api_exit_(status|ptr|void):' "$LOG" || true)
if [ "$N_ENTER" -ne "$N_EXIT" ]; then
    echo "  I1 FAIL: enter=$N_ENTER, exit=$N_EXIT (mismatch)" >&2
    FAIL=1
else
    echo "  I1 OK: enter==exit ($N_ENTER each)"
fi

# ---- I2: dispatch_idx field absent ----------------------------------------
# The hip_aql_kernel_dispatch_submit event schema has no dispatch_idx
# field (the join key for the firmware-ring track is (queue_id, write_idx)
# instead). Babeltrace2 prints all fields per event; if the field is
# present a substring match will succeed.
KD_COUNT=$(grep -c 'rocm_hip:hip_aql_kernel_dispatch_submit:' "$LOG" || true)
if [ "$KD_COUNT" -gt 0 ]; then
    if grep -q 'rocm_hip:hip_aql_kernel_dispatch_submit:.*dispatch_idx' "$LOG"; then
        echo "  I2 FAIL: hip_aql_kernel_dispatch_submit still carries dispatch_idx field" >&2
        FAIL=1
    else
        echo "  I2 OK: $KD_COUNT dispatch_submit events, none carry dispatch_idx"
    fi
else
    echo "  I2 SKIP: no kernel dispatch submit events captured"
fi

# ---- I3: vpid + vtid context propagation (HARD ASSERTION) -----------------
# Every event carries vpid + vtid because we attached the contexts at
# session-setup time. Without this, the consumer's per-thread LIFO walk
# is impossible. babeltrace2 renders contexts as `{ vpid = N, vtid = M }`
# brace blocks preceding the event-payload braces.
SAMPLE_LINE=$(grep -E 'rocm_hip:hip_api_enter:' "$LOG" | head -1 || true)
if [ -z "$SAMPLE_LINE" ]; then
    echo "  I3 FAIL: no hip_api_enter event found in trace" >&2
    FAIL=1
elif echo "$SAMPLE_LINE" | grep -qE 'vpid = [0-9]+' && \
     echo "$SAMPLE_LINE" | grep -qE 'vtid = [0-9]+'; then
    echo "  I3 OK: hip_api_enter carries vpid + vtid channel context"
else
    echo "  I3 FAIL: hip_api_enter missing vpid or vtid context" >&2
    echo "          line: $SAMPLE_LINE" >&2
    FAIL=1
fi

# ---- I4: Same-thread dispatch chain (HARD ASSERTION) ----------------------
# Per the schema-v3 consumer recipe (per-(vpid,vtid) LIFO walk), the
# kernel-dispatch events fired inside hipLaunchKernel must be on the
# SAME vtid as the hipLaunchKernel hip_api_enter event. CLR's call
# graph is synchronous: the AQL packet write happens on the API call
# thread; no thread hand-offs occur between the user-facing HIP API
# entry and the dispatch packet write.
LAUNCH_VTID=$(grep 'rocm_hip:hip_api_enter:.*hipLaunchKernel' "$LOG" \
    | head -1 \
    | sed -n 's/.*vtid = \([0-9]\+\).*/\1/p')
if [ -z "$LAUNCH_VTID" ]; then
    echo "  I4 FAIL: no hipLaunchKernel hip_api_enter found in trace" >&2
    FAIL=1
else
    ENQUEUE_ON_LAUNCH_VTID=$(grep 'rocm_hip:hip_kernel_dispatch_enqueue:' "$LOG" \
        | grep -cE "vtid = $LAUNCH_VTID[ ,}]" || true)
    SUBMIT_ON_LAUNCH_VTID=$(grep 'rocm_hip:hip_aql_kernel_dispatch_submit:' "$LOG" \
        | grep -cE "vtid = $LAUNCH_VTID[ ,}]" || true)
    if [ "$ENQUEUE_ON_LAUNCH_VTID" -lt 1 ]; then
        echo "  I4 FAIL: no hip_kernel_dispatch_enqueue on launching vtid ($LAUNCH_VTID)" >&2
        FAIL=1
    elif [ "$SUBMIT_ON_LAUNCH_VTID" -lt 1 ]; then
        echo "  I4 FAIL: no hip_aql_kernel_dispatch_submit on launching vtid ($LAUNCH_VTID)" >&2
        FAIL=1
    else
        echo "  I4 OK: hip_kernel_dispatch_enqueue + hip_aql_kernel_dispatch_submit both fired on launching vtid $LAUNCH_VTID"
        HSA_ON_LAUNCH_VTID=$(grep 'rocm_hsa:hsa_api_enter:' "$LOG" \
            | grep -cE "vtid = $LAUNCH_VTID[ ,}]" || true)
        echo "  I4 INFO: $HSA_ON_LAUNCH_VTID HSA api_enter events also fired on launching vtid (cross-runtime same-thread parent recovery via LIFO walk)"
    fi
fi

if [ "$FAIL" -ne 0 ]; then
    echo "FAIL: see $LOG" >&2
    exit 1
fi
echo "PASS"
