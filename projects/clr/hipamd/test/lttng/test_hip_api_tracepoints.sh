#!/usr/bin/env bash
# End-to-end validation of the rocm_hip LTTng tracepoints.
#
# 1. Spins up a per-user lttng-sessiond.
# 2. Enables rocm_hip:* tracepoints with --discard and conservative
#    sub-buffer sizing.
# 3. Builds and runs a tiny HIP program that calls 3 well-known APIs
#    (hipMalloc, hipFree, hipGetDevice) and launches a no-op kernel.
# 4. Parses the resulting CTF trace with babeltrace2 and asserts each
#    tracepoint we expected appears.
#
# Usage:
#   test_hip_api_tracepoints.sh [<libamdhip64-build-dir>]
#
# Default build dir: $PWD/build/clr/hipamd/lib
#
# Requires: lttng-tools, babeltrace2, hipcc (in /opt/rocm/bin).
set -euo pipefail

BUILD_LIB_DIR="${1:-$PWD/build/clr/hipamd/lib}"
if [ ! -f "$BUILD_LIB_DIR/libamdhip64.so" ]; then
    echo "ERROR: $BUILD_LIB_DIR/libamdhip64.so not found" >&2
    exit 2
fi

WORK="$(mktemp -d)"
SESSION_NAME="hip-lttng-test-$$"

cleanup() {
    set +e
    lttng destroy "$SESSION_NAME" >/dev/null 2>&1
    # Scoped sessiond kill: only this test's daemon (started below with
    # a pidfile and isolated LTTNG_HOME). Avoid host-wide pkill -f which
    # would kill any concurrent test's lttng-sessiond.
    if [ -n "${SESSIOND_PIDFILE:-}" ] && [ -f "$SESSIOND_PIDFILE" ]; then
        kill "$(cat "$SESSIOND_PIDFILE")" 2>/dev/null
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

# Tiny HIP program: 3 status-returning APIs + a no-op kernel launch.
# When EXPECT_DISPATCH=1 the launch is required to emit
# rocm_hip:hip_kernel_dispatch_enqueue (otherwise the API-only assertions
# stand on their own).
cat > "$WORK/tiny.hip.cpp" <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void noop() {}

int main() {
    // API-only path: hipMalloc / hipGetDevice / hipFree exercise STATUS
    // wrappers without going through HSA dispatch. Always run.
    int* p = nullptr;
    hipMalloc(&p, 4);
    int dev = 0;
    hipGetDevice(&dev);
    hipFree(p);

#ifdef EXPECT_DISPATCH
    // Kernel-dispatch path: requires a working HSA runtime ABI. Skipped
    // unless EXPECT_DISPATCH is defined at compile time.
    int* q = nullptr;
    hipMalloc(&q, 4);
    hipLaunchKernelGGL(noop, dim3(1), dim3(1), 0, 0);
    hipDeviceSynchronize();
    hipFree(q);
#endif
    return 0;
}
EOF

# Build with hipcc, linking the just-built libamdhip64.so. Define
# EXPECT_DISPATCH at compile time when the env var is set so the test
# program includes the kernel launch.
HIPCC=/opt/rocm/bin/hipcc
if [ ! -x "$HIPCC" ]; then
    echo "ERROR: hipcc not at $HIPCC" >&2
    exit 2
fi
EXTRA_DEFS=""
if [ "${EXPECT_DISPATCH:-0}" = "1" ]; then
    EXTRA_DEFS="-DEXPECT_DISPATCH=1"
fi
"$HIPCC" -O2 $EXTRA_DEFS "$WORK/tiny.hip.cpp" -o "$WORK/tiny" \
    -L"$BUILD_LIB_DIR" -lamdhip64 \
    -Wl,-rpath,"$BUILD_LIB_DIR" 2>&1

# Per-user sessiond + session.
# Use isolated LTTNG_HOME and pidfile so cleanup can target THIS test's
# daemon only (host-wide pkill would interfere with concurrent tests).
export LTTNG_HOME="$WORK/lttng-home"
mkdir -p "$LTTNG_HOME"
SESSIOND_PIDFILE="$WORK/sessiond.pid"
lttng-sessiond --daemonize --pidfile="$SESSIOND_PIDFILE"
# Wait briefly for pidfile to appear.
for i in 1 2 3 4 5; do
    [ -f "$SESSIOND_PIDFILE" ] && break
    sleep 0.5
done
lttng create "$SESSION_NAME" --output="$WORK/trace"

# Enable a per-CPU userspace channel with conservative sizing (stock
# channel sizing overflows /dev/shm on hosts with high CPU count).
lttng enable-channel --userspace --discard \
    --subbuf-size=4096 --num-subbuf=2 default

lttng enable-event --userspace --channel=default \
    'rocm_hip:hip_api_enter,rocm_hip:hip_api_exit_status,rocm_hip:hip_api_exit_ptr,rocm_hip:hip_api_exit_void,rocm_hip:hip_kernel_dispatch_enqueue'

lttng start

# Run with the just-built libamdhip64.so.
LD_LIBRARY_PATH="$BUILD_LIB_DIR:${LD_LIBRARY_PATH:-}" "$WORK/tiny"

lttng stop
lttng destroy "$SESSION_NAME"

# Parse and assert.
LOG="$WORK/babeltrace.log"
babeltrace2 "$WORK/trace" > "$LOG" 2>&1

FAIL=0
for api in hipMalloc hipFree hipGetDevice; do
    if grep -q "rocm_hip:hip_api_enter.*api_name = \"$api\"" "$LOG"; then
        echo "  OK: hip_api_enter for $api"
    else
        echo "  FAIL: missing hip_api_enter for $api" >&2
        FAIL=1
    fi
    if grep -q "rocm_hip:hip_api_exit_status.*api_name = \"$api\"" "$LOG"; then
        echo "  OK: hip_api_exit_status for $api"
    else
        echo "  FAIL: missing hip_api_exit_status for $api" >&2
        FAIL=1
    fi
done

# Kernel dispatch enqueue tracepoint. Soft assertion by default; set
# EXPECT_DISPATCH=1 to make it fatal.
if grep -q 'rocm_hip:hip_kernel_dispatch_enqueue' "$LOG"; then
    echo "  OK: hip_kernel_dispatch_enqueue observed"
else
    echo "  WARN: hip_kernel_dispatch_enqueue not observed" >&2
    if [ "${EXPECT_DISPATCH:-0}" = "1" ]; then
        FAIL=1
    fi
fi

if [ "$FAIL" -ne 0 ]; then
    echo "FAIL: see $LOG" >&2
    exit 1
fi

echo "PASS"
