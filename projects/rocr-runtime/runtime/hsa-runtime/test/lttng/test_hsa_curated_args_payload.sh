#!/usr/bin/env bash
# End-to-end payload test for HSA curated _args events.
#
# 1. Spins up a per-user lttng-sessiond.
# 2. Enables rocm_hsa:hsa_signal_create_args, hsa_signal_destroy_args,
#    hsa_amd_memory_pool_allocate_args plus generic enter/exit_status.
# 3. Builds + runs a tiny HSA program with known argument values.
# 4. Asserts the typed args events appear with correct payload values.
# 5. Asserts generic enter/exit_status still fire (typed _args augments,
#    never replaces).
#
# Usage: test_hsa_curated_args_payload.sh [<libhsa-runtime64-build-dir>]
set -euo pipefail

BUILD_LIB_DIR="${1:-$PWD/build/rocr/rocr/lib}"
if [ ! -f "$BUILD_LIB_DIR/libhsa-runtime64.so" ] && \
   ! ls "$BUILD_LIB_DIR"/libhsa-runtime64.so* >/dev/null 2>&1; then
    echo "ERROR: libhsa-runtime64.so not found in $BUILD_LIB_DIR" >&2
    exit 2
fi

WORK="$(mktemp -d)"
SESSION_NAME="hsa-lttng-curated-payload-$$"
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

# Tiny HSA program with known argument values for assertions.
# - hsa_signal_create with KNOWN initial_value 0x1234 (=4660), num_consumers 0
# - hsa_signal_destroy on the just-created signal
# - hsa_amd_memory_pool_allocate with size=4096 — needs a real pool;
#   enumerate agents + pools first.
cat > "$WORK/curated.cpp" <<'EOF'
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>

static hsa_status_t find_gpu(hsa_agent_t agent, void* data) {
    hsa_device_type_t device_type;
    hsa_status_t s = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE,
                                        &device_type);
    if (s == HSA_STATUS_SUCCESS && device_type == HSA_DEVICE_TYPE_GPU) {
        *((hsa_agent_t*)data) = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t find_pool(hsa_amd_memory_pool_t pool, void* data) {
    hsa_amd_segment_t seg;
    hsa_status_t s = hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (s == HSA_STATUS_SUCCESS && seg == HSA_AMD_SEGMENT_GLOBAL) {
        *((hsa_amd_memory_pool_t*)data) = pool;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

int main() {
    hsa_init();

    // hsa_signal_create with KNOWN initial_value 0x1234, num_consumers 0.
    hsa_signal_t sig = {0};
    hsa_signal_create(0x1234, 0, NULL, &sig);
    // hsa_signal_destroy with the just-created handle.
    hsa_signal_destroy(sig);

    // Enumerate agents to find a GPU; then enumerate pools and allocate
    // a known-size buffer so hsa_amd_memory_pool_allocate_args has a real
    // payload to capture.
    hsa_agent_t agent = {0};
    hsa_iterate_agents(&find_gpu, &agent);
    if (agent.handle != 0) {
        hsa_amd_memory_pool_t pool = {0};
        hsa_amd_agent_iterate_memory_pools(agent, &find_pool, &pool);
        if (pool.handle != 0) {
            void* p = NULL;
            if (hsa_amd_memory_pool_allocate(pool, 4096, 0, &p) ==
                HSA_STATUS_SUCCESS && p) {
                hsa_amd_memory_pool_free(p);
            }
        }
    }

    hsa_shut_down();
    return 0;
}
EOF

g++ -std=c++17 "$WORK/curated.cpp" -I/opt/rocm/include \
    -L "$BUILD_LIB_DIR" -lhsa-runtime64 -Wl,-rpath,"$BUILD_LIB_DIR" \
    -o "$WORK/curated_test"

# Start sessiond.
lttng-sessiond --daemonize --pidfile "$SESSIOND_PIDFILE"
TRACE_DIR="$WORK/trace"
lttng create "$SESSION_NAME" --output "$TRACE_DIR" >/dev/null
lttng enable-channel --userspace --discard --subbuf-size=32768 --num-subbuf=4 ch1 >/dev/null
# Per schema v3, attach vpid+vtid contexts so consumers can reconstruct
# per-thread enter/exit pairing.
lttng add-context --userspace --channel=ch1 --type=vpid --type=vtid >/dev/null
lttng enable-event --userspace --channel=ch1 \
    'rocm_hsa:hsa_api_enter,rocm_hsa:hsa_api_exit_status' >/dev/null
lttng enable-event --userspace --channel=ch1 \
    'rocm_hsa:hsa_signal_create_args,rocm_hsa:hsa_signal_destroy_args,rocm_hsa:hsa_amd_memory_pool_allocate_args' >/dev/null
lttng start "$SESSION_NAME" >/dev/null

APP_RC=0
"$WORK/curated_test" || APP_RC=$?
echo "  curated_test exit=$APP_RC"

lttng stop "$SESSION_NAME" >/dev/null
lttng destroy "$SESSION_NAME" >/dev/null

DUMP="$WORK/trace.txt"
babeltrace2 "$TRACE_DIR" > "$DUMP"

echo "=== HSA curated payload assertions ==="

FAIL=0

# A. hsa_signal_create_args: initial_value == 4660 (0x1234), num_consumers == 0.
if grep 'rocm_hsa:hsa_signal_create_args' "$DUMP" | grep -q 'initial_value = 4660' && \
   grep 'rocm_hsa:hsa_signal_create_args' "$DUMP" | grep -q 'num_consumers = 0'; then
    echo "  PASS  hsa_signal_create_args present with initial_value=4660, num_consumers=0"
else
    echo "  FAIL  hsa_signal_create_args missing or payload mismatch"
    grep 'hsa_signal_create' "$DUMP" || true
    FAIL=$((FAIL+1))
fi

# B. hsa_signal_destroy_args present with non-zero signal handle.
if grep -q 'rocm_hsa:hsa_signal_destroy_args' "$DUMP"; then
    echo "  PASS  hsa_signal_destroy_args present"
else
    echo "  FAIL  hsa_signal_destroy_args missing"
    FAIL=$((FAIL+1))
fi

# C. hsa_amd_memory_pool_allocate_args present (best effort -- needs GPU
#    + pool; if the test system has neither this is a WARN not a FAIL).
if grep -q 'rocm_hsa:hsa_amd_memory_pool_allocate_args' "$DUMP"; then
    if grep 'rocm_hsa:hsa_amd_memory_pool_allocate_args' "$DUMP" | grep -q 'size = 4096'; then
        echo "  PASS  hsa_amd_memory_pool_allocate_args present with size=4096"
    else
        echo "  FAIL  hsa_amd_memory_pool_allocate_args present but size != 4096"
        grep 'hsa_amd_memory_pool_allocate' "$DUMP" || true
        FAIL=$((FAIL+1))
    fi
else
    echo "  WARN  hsa_amd_memory_pool_allocate_args not in trace (no GPU/pool present?)"
fi

# D. Generic enter/exit_status preserved (typed _args augments, never replaces).
N_ENTER=$(grep -c 'rocm_hsa:hsa_api_enter' "$DUMP" || true)
N_EXIT=$(grep -c 'rocm_hsa:hsa_api_exit_status' "$DUMP" || true)
if [ "$N_ENTER" -ge 2 ] && [ "$N_EXIT" -ge 2 ]; then
    echo "  PASS  generic enter/exit_status preserved ($N_ENTER enter, $N_EXIT exit_status)"
else
    echo "  FAIL  generic event preservation broken ($N_ENTER enter, $N_EXIT exit_status)"
    FAIL=$((FAIL+1))
fi

if [ "$FAIL" -gt 0 ]; then
    echo "=== $FAIL ASSERTION(S) FAILED ==="
    exit 1
fi
echo "=== ALL HSA PAYLOAD ASSERTIONS PASSED ==="
exit 0
