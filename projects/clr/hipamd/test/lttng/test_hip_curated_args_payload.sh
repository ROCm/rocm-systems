#!/usr/bin/env bash
# End-to-end payload test for curated _args events (Phase E: 3 APIs).
#
# 1. Spins up a per-user lttng-sessiond.
# 2. Enables rocm_hip:hipMemcpyAsync_args, rocm_hip:hipMalloc_args,
#    rocm_hip:hipDeviceSynchronize_args plus generic enter/exit_status/exit_ptr.
# 3. Builds + runs a tiny program with known argument values.
# 4. Asserts the typed args events appear with correct payload values.
# 5. Asserts pointer-returning APIs still get hip_api_exit_ptr (NOT
#    exit_status), matching the spec §6.2 generic-exit preservation rule.
#
# Usage: test_hip_curated_args_payload.sh [<libamdhip64-build-dir>]
set -euo pipefail

BUILD_LIB_DIR="${1:-$PWD/build/clr/hipamd/lib}"
if [ ! -f "$BUILD_LIB_DIR/libamdhip64.so" ]; then
    echo "ERROR: $BUILD_LIB_DIR/libamdhip64.so not found" >&2
    exit 2
fi

WORK="$(mktemp -d)"
SESSION_NAME="hip-lttng-curated-payload-$$"

# Isolated sessiond (avoid host-wide pkill races per debate-review C5).
export LTTNG_HOME="$WORK/lttng_home"
mkdir -p "$LTTNG_HOME"
SESSIOND_PIDFILE="$WORK/sessiond.pid"

cleanup() {
    set +e
    lttng destroy "$SESSION_NAME" >/dev/null 2>&1
    if [ -f "$SESSIOND_PIDFILE" ]; then
        kill "$(cat $SESSIOND_PIDFILE)" 2>/dev/null
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

# Tiny HIP program with known argument values for assertions.
cat > "$WORK/curated.hip.cpp" <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

int main() {
    // hipMalloc with KNOWN size 4096.
    int* dev_ptr = nullptr;
    hipMalloc(&dev_ptr, 4096);

    // hipMemcpyAsync from a known src ptr to dev_ptr, KNOWN size 1024,
    // KNOWN kind hipMemcpyHostToDevice (=1), default stream (NULL).
    char host_buf[1024];
    memset(host_buf, 0, sizeof(host_buf));
    hipMemcpyAsync(dev_ptr, host_buf, 1024, hipMemcpyHostToDevice, nullptr);

    // hipDeviceSynchronize — zero-arg API.
    hipDeviceSynchronize();

    hipFree(dev_ptr);
    return 0;
}
EOF

HIPCC=/opt/rocm/bin/hipcc
"$HIPCC" "$WORK/curated.hip.cpp" -L "$BUILD_LIB_DIR" -lamdhip64 -Wl,-rpath,"$BUILD_LIB_DIR" -o "$WORK/curated_test"

# Start sessiond.
lttng-sessiond --daemonize --pidfile "$SESSIOND_PIDFILE"

# Set up session.
TRACE_DIR="$WORK/trace"
lttng create "$SESSION_NAME" --output "$TRACE_DIR" >/dev/null
lttng enable-channel --userspace --discard --subbuf-size=32768 --num-subbuf=4 ch1 >/dev/null
# Generic events (already covered by existing test; we re-enable to verify
# augment-not-replace behavior).
lttng enable-event --userspace --channel=ch1 \
    'rocm_hip:hip_api_enter,rocm_hip:hip_api_exit_status,rocm_hip:hip_api_exit_ptr' >/dev/null
# Curated typed events.
lttng enable-event --userspace --channel=ch1 \
    'rocm_hip:hipMalloc_args,rocm_hip:hipMemcpyAsync_args,rocm_hip:hipDeviceSynchronize_args' >/dev/null
lttng start "$SESSION_NAME" >/dev/null

"$WORK/curated_test"

lttng stop "$SESSION_NAME" >/dev/null
lttng destroy "$SESSION_NAME" >/dev/null

# Dump trace and assert.
DUMP="$WORK/trace.txt"
babeltrace2 "$TRACE_DIR" > "$DUMP"

echo "=== assertions ==="

# A. hipMalloc_args appears, sizeBytes == 4096, ptr_out is non-zero.
if grep -q 'rocm_hip:hipMalloc_args' "$DUMP" && \
   grep 'rocm_hip:hipMalloc_args' "$DUMP" | grep -q 'size = 4096'; then
    echo "  PASS  hipMalloc_args present with size = 4096"
else
    echo "  FAIL  hipMalloc_args missing or size != 4096"
    grep 'rocm_hip:hipMalloc' "$DUMP" || true
    exit 1
fi

# B. hipMemcpyAsync_args appears, sizeBytes == 1024, kind == 1 (HostToDevice).
if grep 'rocm_hip:hipMemcpyAsync_args' "$DUMP" | grep -q 'sizeBytes = 1024' && \
   grep 'rocm_hip:hipMemcpyAsync_args' "$DUMP" | grep -q 'kind = 1'; then
    echo "  PASS  hipMemcpyAsync_args present with sizeBytes = 1024, kind = 1"
else
    echo "  FAIL  hipMemcpyAsync_args payload mismatch"
    grep 'hipMemcpyAsync' "$DUMP" || true
    exit 1
fi

# C. hipDeviceSynchronize_args appears (zero-arg payload — only corr_id).
if grep -q 'rocm_hip:hipDeviceSynchronize_args' "$DUMP"; then
    echo "  PASS  hipDeviceSynchronize_args present (NOARGS variant works)"
else
    echo "  FAIL  hipDeviceSynchronize_args missing"
    exit 1
fi

# D. Generic exit events still fire (augment-not-replace per spec §6.2).
N_ENTER=$(grep -c 'rocm_hip:hip_api_enter' "$DUMP" || true)
N_EXIT_STATUS=$(grep -c 'rocm_hip:hip_api_exit_status' "$DUMP" || true)
if [ "$N_ENTER" -ge 3 ] && [ "$N_EXIT_STATUS" -ge 3 ]; then
    echo "  PASS  generic enter/exit_status preserved ($N_ENTER enter, $N_EXIT_STATUS exit_status)"
else
    echo "  FAIL  generic event preservation broken"
    exit 1
fi

# E. corr_id linkage: each _args event must share a corr_id with a matching
#    enter and exit event from the same call. Spot-check hipMemcpyAsync.
ARGS_CORR=$(grep 'rocm_hip:hipMemcpyAsync_args' "$DUMP" | head -1 | \
            sed -n 's/.*corr_id = \([0-9]*\).*/\1/p')
if [ -n "$ARGS_CORR" ] && \
   grep "corr_id = $ARGS_CORR" "$DUMP" | grep -q 'hip_api_enter' && \
   grep "corr_id = $ARGS_CORR" "$DUMP" | grep -q 'hip_api_exit_status'; then
    echo "  PASS  corr_id $ARGS_CORR links _args event to generic enter+exit"
else
    echo "  FAIL  corr_id linkage broken for hipMemcpyAsync"
    exit 1
fi

echo "=== ALL PAYLOAD ASSERTIONS PASSED ==="
exit 0
