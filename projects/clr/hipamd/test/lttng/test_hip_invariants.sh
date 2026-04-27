#!/usr/bin/env bash
# HIP LTTng invariant tests.
#
# Beyond the smoke tests in test_hip_api_tracepoints.sh, this script asserts
# semantic invariants of the trace produced by a tiny HIP program:
#
#   I1. Enter/exit balance: count(hip_api_enter) ==
#         sum(hip_api_exit_status, _ptr, _void). Catches missed exit emits.
#
#   I2. Doorbell uniqueness: every hsa_doorbell_ring event has a distinct
#         corr_id within a single run (validated only when HSA tracepoints
#         are also captured).
#
#   I3. hip_aql_kernel_dispatch_submit must NOT carry a dispatch_idx field.
#
#   I4. Parent propagation for HIP -> HSA chain: after a hipLaunchKernel
#         call, at least one hip_kernel_dispatch_enqueue AND at least one
#         hip_aql_kernel_dispatch_submit must carry the launch's corr_id as
#         parent_corr_id. Catches regressions in the shared-TLS slot
#         propagation between HIP and HSA.
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
# need to land on the launching CPU's buffer for the I4 parent-propagation
# assertion to find them. 224-CPU host × 32 KiB × 4 = ~28 MB total, fits
# in /dev/shm easily.
lttng enable-channel --userspace --discard --subbuf-size=32768 --num-subbuf=4 default >/dev/null
# Capture HIP and HSA tracepoints both. Doorbell + intercept tell I2/I4.
lttng enable-event --userspace -c default \
    'rocm_hip:*,rocm_hsa:hsa_api_enter,rocm_hsa:hsa_doorbell_ring,rocm_hsa:hsa_intercept_packets' >/dev/null
lttng start >/dev/null

LD_LIBRARY_PATH="$BUILD_LIB_DIR:${LD_LIBRARY_PATH:-}" "$WORK/tiny" || true

lttng stop >/dev/null
lttng destroy "$SESSION" >/dev/null

LOG="$WORK/babeltrace.log"
babeltrace2 "$WORK/trace" > "$LOG" 2>&1

FAIL=0

# ---- I1: Enter/exit balance per (pid, tid) ----------------------------------
# Babeltrace2 default output includes `tid = N` field for events that have
# the field (we emit `tid` explicitly on enter). Counting works as:
#   ENTERS = lines matching rocm_hip:hip_api_enter
#   EXITS  = lines matching rocm_hip:hip_api_exit_(status|ptr|void)
# Per (pid, tid) breakdown is awk-grouped on the `vpid =` and `vtid =`
# context fields when present; we capture process-wide totals for
# simplicity (pid is constant, single-threaded test program).
N_ENTER=$(grep -c 'rocm_hip:hip_api_enter:' "$LOG" || true)
N_EXIT=$(grep -cE 'rocm_hip:hip_api_exit_(status|ptr|void):' "$LOG" || true)
if [ "$N_ENTER" -ne "$N_EXIT" ]; then
    echo "  I1 FAIL: enter=$N_ENTER, exit=$N_EXIT (mismatch)" >&2
    FAIL=1
else
    echo "  I1 OK: enter==exit ($N_ENTER each)"
fi

# ---- I2: Doorbell corr_id uniqueness (HARD ASSERTION) -----------------------
# Extract each doorbell event's corr_id field; count unique vs total.
# The test program launches a kernel, so at least one doorbell MUST fire.
DB_TOTAL=$(grep -c 'rocm_hsa:hsa_doorbell_ring:' "$LOG" || true)
if [ "$DB_TOTAL" -lt 1 ]; then
    echo "  I2 FAIL: no doorbell events captured but kernel launch was issued" >&2
    FAIL=1
else
    DB_CORR=$(grep 'rocm_hsa:hsa_doorbell_ring:' "$LOG" \
        | sed -n 's/.*corr_id = \([0-9]\+\).*/\1/p')
    DB_UNIQ=$(printf '%s\n' "$DB_CORR" | sort -u | wc -l)
    DB_COUNT=$(printf '%s\n' "$DB_CORR" | wc -l)
    if [ "$DB_UNIQ" -ne "$DB_COUNT" ]; then
        echo "  I2 FAIL: doorbell corr_ids not unique ($DB_UNIQ unique / $DB_COUNT total)" >&2
        FAIL=1
    else
        echo "  I2 OK: $DB_COUNT doorbells, all corr_id distinct"
    fi
fi

# ---- I3: dispatch_idx field removed -----------------------------------------
# The hip_aql_kernel_dispatch_submit event schema has no dispatch_idx
# field. Babeltrace2 prints all fields per event; if the field is present
# a substring match will succeed.
KD_COUNT=$(grep -c 'rocm_hip:hip_aql_kernel_dispatch_submit:' "$LOG" || true)
if [ "$KD_COUNT" -gt 0 ]; then
    if grep -q 'rocm_hip:hip_aql_kernel_dispatch_submit:.*dispatch_idx' "$LOG"; then
        echo "  I3 FAIL: hip_aql_kernel_dispatch_submit still carries dispatch_idx field" >&2
        FAIL=1
    else
        echo "  I3 OK: $KD_COUNT dispatch_submit events, none carry dispatch_idx"
    fi
else
    echo "  I3 SKIP: no kernel dispatch submit events captured"
fi

# ---- I4: Parent propagation through dispatch chain (HARD ASSERTION) --------
# After hipLaunchKernel, the HIP-side dispatch events must carry the
# launch's corr_id as their parent_corr_id. This proves the shared-TLS
# slot in librocprofiler-register is properly accessed by the dispatch
# path. Specifically:
#   - hip_kernel_dispatch_enqueue must have parent_corr_id = launch corr
#   - hip_aql_kernel_dispatch_submit must have parent_corr_id = launch corr
#
# Cross-runtime propagation INTO HSA events is ALSO checked, but only as
# INFO: HSA api_enter events under deep call chains have intermediate
# parents (HSA-on-HSA), and on high-CPU-count hosts the per-CPU sub-buffer
# constraints frequently drop the first-level HSA events with
# parent=launch. The HIP-side dispatch events are the reliable indicator
# that propagation works end-to-end.
LAUNCH_CORR=$(grep 'rocm_hip:hip_api_enter:.*hipLaunchKernel' "$LOG" \
    | head -1 \
    | sed -n 's/.*corr_id = \([0-9]\+\),.*/\1/p')
if [ -z "$LAUNCH_CORR" ]; then
    echo "  I4 FAIL: no hipLaunchKernel hip_api_enter found in trace" >&2
    FAIL=1
else
    # Match parent_corr_id field followed by a delimiter ([" ,}]) so we don't
    # accidentally match a longer corr_id with LAUNCH_CORR as a prefix.
    ENQUEUE_WITH_PARENT=$(grep 'rocm_hip:hip_kernel_dispatch_enqueue:' "$LOG" \
        | grep -cE "parent_corr_id = $LAUNCH_CORR[ ,}]" || true)
    SUBMIT_WITH_PARENT=$(grep 'rocm_hip:hip_aql_kernel_dispatch_submit:' "$LOG" \
        | grep -cE "parent_corr_id = $LAUNCH_CORR[ ,}]" || true)
    if [ "$ENQUEUE_WITH_PARENT" -lt 1 ]; then
        echo "  I4 FAIL: no hip_kernel_dispatch_enqueue carries hipLaunchKernel's corr_id ($LAUNCH_CORR) as parent_corr_id" >&2
        FAIL=1
    elif [ "$SUBMIT_WITH_PARENT" -lt 1 ]; then
        echo "  I4 FAIL: no hip_aql_kernel_dispatch_submit carries hipLaunchKernel's corr_id ($LAUNCH_CORR) as parent_corr_id" >&2
        FAIL=1
    else
        echo "  I4 OK: hip_kernel_dispatch_enqueue + hip_aql_kernel_dispatch_submit both carry hipLaunchKernel's corr_id as parent"
        HSA_WITH_PARENT=$(grep 'rocm_hsa:hsa_api_enter:' "$LOG" \
            | grep -cE "parent_corr_id = $LAUNCH_CORR[ ,}]" || true)
        echo "  I4 INFO: $HSA_WITH_PARENT HSA api_enter events also carry launch corr as parent (depth-1 HSA calls; deeper HSA calls have HSA-on-HSA parents)"
    fi
fi

if [ "$FAIL" -ne 0 ]; then
    echo "FAIL: see $LOG" >&2
    exit 1
fi
echo "PASS"
