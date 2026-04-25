/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * HIP-internal LTTng emit abstraction. Each tracepoint is wrapped behind a
 * `static inline` function that checks `tracepoint_enabled()` first so the
 * steady-state cost when no LTTng session is active is one atomic load and
 * one unlikely-branch.
 *
 * The header has TWO modes:
 *   HIP_ENABLE_LTTNG_UST=1 -> real tracepoint emission
 *   otherwise              -> all functions become empty no-ops
 *
 * The two backends (LTTng-UST today, user_events future) are encapsulated
 * by the contents of these inline functions. To swap a future backend in,
 * the only change is the function bodies - the call sites stay identical.
 */
#ifndef ROCM_HIP_TRACE_EMIT_H_
#define ROCM_HIP_TRACE_EMIT_H_

#include <stdint.h>
#include "rocm_trace_tid.h"

#if defined(HIP_ENABLE_LTTNG_UST) && HIP_ENABLE_LTTNG_UST

#include "rocm_hip_tp.h"

static inline void rocm_trace_emit_hip_api_enter(const char* api_name,
                                                 uint64_t    corr_id) {
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_api_enter)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_api_enter,
                                api_name, corr_id, rocm_trace_current_tid());
    }
}

/* Status-returning APIs (hipError_t, int, etc.) */
static inline void rocm_trace_emit_hip_api_exit_status(const char* api_name,
                                                       uint64_t    corr_id,
                                                       int32_t     status) {
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_api_exit_status)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_api_exit_status,
                                api_name, corr_id, status);
    }
}

/* Pointer-returning APIs (hipApiName, __hipRegisterFatBinary, etc.) */
static inline void rocm_trace_emit_hip_api_exit_ptr(const char* api_name,
                                                    uint64_t    corr_id,
                                                    const void* retval) {
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_api_exit_ptr)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_api_exit_ptr,
                                api_name, corr_id, (uint64_t)(uintptr_t)retval);
    }
}

/* Void-returning APIs */
static inline void rocm_trace_emit_hip_api_exit_void(const char* api_name,
                                                     uint64_t    corr_id) {
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_api_exit_void)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_api_exit_void, api_name, corr_id);
    }
}

/* Pack three 16-bit dims into a 64-bit field. Bits [15:0]=x, [31:16]=y,
 * [47:32]=z, [63:48]=0. Real HIP launches never exceed 2^16-1 per dim;
 * if they did the high bits would saturate (lossy but not corruption). */
static inline uint64_t rocm_trace_pack_dims3(uint32_t x, uint32_t y, uint32_t z) {
    return ((uint64_t)(x & 0xffffu))
         | ((uint64_t)(y & 0xffffu) << 16)
         | ((uint64_t)(z & 0xffffu) << 32);
}

static inline void rocm_trace_emit_hip_kernel_dispatch_enqueue(
    const char* kernel_name, uint64_t corr_id, void* stream,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    uint32_t shared_mem_bytes) {
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_kernel_dispatch_enqueue)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_kernel_dispatch_enqueue,
                                kernel_name, corr_id, rocm_trace_current_tid(),
                                stream,
                                rocm_trace_pack_dims3(grid_x, grid_y, grid_z),
                                rocm_trace_pack_dims3(block_x, block_y, block_z),
                                shared_mem_bytes);
    }
}

#else /* !HIP_ENABLE_LTTNG_UST */

static inline void rocm_trace_emit_hip_api_enter(const char* a, uint64_t c) { (void)a; (void)c; }
static inline void rocm_trace_emit_hip_api_exit_status(const char* a, uint64_t c, int32_t s) {
    (void)a; (void)c; (void)s;
}
static inline void rocm_trace_emit_hip_api_exit_ptr(const char* a, uint64_t c, const void* p) {
    (void)a; (void)c; (void)p;
}
static inline void rocm_trace_emit_hip_api_exit_void(const char* a, uint64_t c) {
    (void)a; (void)c;
}
static inline void rocm_trace_emit_hip_kernel_dispatch_enqueue(
    const char* a, uint64_t b, void* c, uint32_t d, uint32_t e, uint32_t f,
    uint32_t g, uint32_t h, uint32_t i, uint32_t j) {
    (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; (void)i; (void)j;
}

#endif

#endif /* ROCM_HIP_TRACE_EMIT_H_ */
