#!/usr/bin/env bash
# Validation of the test-only dispatch_log introspection API
# (hsa_amd_dispatch_log_test_enable / hsa_amd_dispatch_log_test_get_state).
#
# These symbols are intentionally exported from libhsa-runtime64.so for
# standalone test programs that need to inspect the per-queue FW dispatch
# record buffer directly (i.e. without going through the LTTng UST emit
# path). This test exercises them and verifies the documented contract:
#
#   1. _enable() returns SUCCESS on a freshly-created HSA queue (any
#      AqlQueue regardless of GPU agent type).
#   2. _get_state() after _enable() returns:
#        - non-null buffer_base
#        - num_records == DISPATCH_LOG_RECORD_COUNT (65536) — matches
#          AqlQueue::SetProfiling's hard-coded ring count
#        - non-null wptr_ptr and signal_ptr
#   3. _enable() / _get_state() reject nullptr / non-AQL-queue arguments
#      with HSA_STATUS_ERROR_INVALID_*.
#
# The test does NOT submit kernels or assert on FW writes — that path is
# covered by test_hsa_dispatch_log.sh which runs end-to-end through the
# LTTng pipeline. This is a focused unit-style test of the API surface
# the dlog_test_* programs use.
#
# Usage:
#   test_hsa_dispatch_log_test_api.sh [<build_dir>]
#
# <build_dir> defaults to $PWD/build/rocr.
#
# Skips with INFO if no GPU agent is present (the runtime can be
# initialized but no AqlQueue can be created).

set -euo pipefail

BUILD_DIR="${1:-$PWD/build/rocr}"
LIB_DIR="$BUILD_DIR/rocr/lib"

if [ ! -f "$LIB_DIR/libhsa-runtime64.so" ]; then
    echo "FAIL: $LIB_DIR/libhsa-runtime64.so not found"
    exit 1
fi

TRACE_DIR="$(mktemp -d)"
trap 'rm -rf "$TRACE_DIR"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
info() { echo "INFO: $*"; }

cat > "$TRACE_DIR/test_api.cpp" <<'EOF'
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" {
hsa_status_t hsa_amd_dispatch_log_test_enable(hsa_queue_t* queue);
hsa_status_t hsa_amd_dispatch_log_test_get_state(
    hsa_queue_t* queue,
    void** buffer_base,
    uint32_t* num_records,
    const volatile uint64_t** wptr_ptr,
    const volatile uint64_t** signal_ptr);
}

static hsa_status_t find_gpu(hsa_agent_t agent, void* data) {
    hsa_device_type_t dt;
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dt) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    if (dt == HSA_DEVICE_TYPE_GPU) {
        *static_cast<hsa_agent_t*>(data) = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

#define EXPECT_OK(call)                                                  \
    do {                                                                 \
        hsa_status_t __s = (call);                                       \
        if (__s != HSA_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "FAIL: " #call " -> %d\n", __s);        \
            return 2;                                                    \
        }                                                                \
    } while (0)

#define EXPECT_EQ(a, b, msg)                                             \
    do {                                                                 \
        if ((a) != (b)) {                                                \
            std::fprintf(stderr, "FAIL: %s: %lld != %lld\n",             \
                         msg, (long long)(a), (long long)(b));           \
            return 2;                                                    \
        }                                                                \
    } while (0)

int main() {
    EXPECT_OK(hsa_init());

    hsa_agent_t gpu = {0};
    hsa_status_t s = hsa_iterate_agents(find_gpu, &gpu);
    if (gpu.handle == 0) {
        std::fprintf(stdout, "INFO: no GPU agent; skipping\n");
        hsa_shut_down();
        return 0;
    }
    (void)s;

    uint32_t qsize = 0;
    EXPECT_OK(hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &qsize));
    if (qsize == 0) qsize = 1024;
    if (qsize > 4096) qsize = 4096;

    hsa_queue_t* queue = nullptr;
    EXPECT_OK(hsa_queue_create(gpu, qsize, HSA_QUEUE_TYPE_SINGLE,
                               nullptr, nullptr, 0, 0, &queue));

    /* --- Negative tests (must come BEFORE enable while args are valid). */
    if (hsa_amd_dispatch_log_test_enable(nullptr)
            != HSA_STATUS_ERROR_INVALID_ARGUMENT) {
        std::fprintf(stderr, "FAIL: _enable(nullptr) didn't reject\n");
        return 2;
    }

    void* buf = (void*)0xdeadbeef;
    uint32_t nrec = 0xdeadbeef;
    const volatile uint64_t* w = nullptr;
    const volatile uint64_t* sig = nullptr;
    if (hsa_amd_dispatch_log_test_get_state(nullptr, &buf, &nrec, &w, &sig)
            != HSA_STATUS_ERROR_INVALID_ARGUMENT) {
        std::fprintf(stderr, "FAIL: _get_state(nullptr queue) didn't reject\n");
        return 2;
    }
    if (hsa_amd_dispatch_log_test_get_state(queue, nullptr, &nrec, &w, &sig)
            != HSA_STATUS_ERROR_INVALID_ARGUMENT) {
        std::fprintf(stderr, "FAIL: _get_state(nullptr buffer_base) didn't reject\n");
        return 2;
    }

    /* --- Positive tests. */
    EXPECT_OK(hsa_amd_dispatch_log_test_enable(queue));
    buf = nullptr; nrec = 0; w = nullptr; sig = nullptr;
    EXPECT_OK(hsa_amd_dispatch_log_test_get_state(queue, &buf, &nrec, &w, &sig));

    if (buf == nullptr) { std::fprintf(stderr, "FAIL: buffer_base null\n"); return 2; }
    /* DISPATCH_LOG_RECORD_COUNT in dispatch_log.h. */
    EXPECT_EQ(nrec, 65536u, "num_records mismatch");
    if (w == nullptr) { std::fprintf(stderr, "FAIL: wptr_ptr null\n"); return 2; }
    if (sig == nullptr) { std::fprintf(stderr, "FAIL: signal_ptr null\n"); return 2; }

    EXPECT_OK(hsa_queue_destroy(queue));
    EXPECT_OK(hsa_shut_down());
    std::fprintf(stdout, "OK\n");
    return 0;
}
EOF

# Build against the freshly-built libhsa-runtime64.so so dispatch_log
# code under review is what we exercise.
c++ "$TRACE_DIR/test_api.cpp" -o "$TRACE_DIR/test_api" \
    -std=c++17 \
    -I/opt/rocm/include -L"$LIB_DIR" -lhsa-runtime64 \
    -Wl,-rpath,"$LIB_DIR" \
    || fail "compile failed"

LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}" \
    "$TRACE_DIR/test_api" > "$TRACE_DIR/out.log" 2>&1
RC=$?
cat "$TRACE_DIR/out.log"

if [ $RC -ne 0 ]; then
    fail "test_api binary returned $RC"
fi

if grep -q '^INFO: no GPU agent' "$TRACE_DIR/out.log"; then
    info "test skipped (no GPU agent present)"
elif ! grep -q '^OK' "$TRACE_DIR/out.log"; then
    fail "test_api binary did not print OK"
fi

echo "PASS hsa_dispatch_log_test_api"
