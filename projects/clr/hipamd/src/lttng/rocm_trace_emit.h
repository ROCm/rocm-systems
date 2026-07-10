/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * HIP-internal LTTng emit abstraction. Each tracepoint is wrapped behind a
 * `static inline` function that checks `tracepoint_enabled()` first so the
 * steady-state cost when no LTTng session is active is one atomic load and
 * an unlikely branch. When HIP_ENABLE_LTTNG_UST is unset every function
 * collapses to an empty no-op.
 *
 * Per-event identity (vpid, vtid) and timestamp are provided by LTTng-UST's
 * native channel contexts, configured session-side via
 *   lttng add-context --userspace --type vpid --type vtid
 * The CTF event-header timestamp is always present. Consumers reconstruct
 * enter/exit pairing by walking the per-(vpid, vtid) event stream sorted by
 * timestamp using a LIFO stack; the enclosing parent of any nested call is
 * the topmost unmatched enter on that stack at the time of the inner event.
 */
#ifndef ROCM_HIP_TRACE_EMIT_H_
#define ROCM_HIP_TRACE_EMIT_H_

#include <stdint.h>

#if defined(HIP_ENABLE_LTTNG_UST) && HIP_ENABLE_LTTNG_UST

#include <atomic>
#include "rocm_hip_tp.h"

/* Runtime-wide kill switch defined in rocm_trace_init.cpp. When true,
 * every emit helper short-circuits before touching LTTng state. */
extern std::atomic<bool> rocm_hip_trace_g_disabled;

#ifndef ROCM_TRACE_DISABLED_DEFINED
#define ROCM_TRACE_DISABLED_DEFINED
static inline bool rocm_trace_disabled(void) {
    return rocm_hip_trace_g_disabled.load(std::memory_order_relaxed);
}
#endif

static inline void rocm_trace_emit_hip_api_enter(const char* api_name) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_api_enter)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_api_enter, api_name);
    }
}

/* Status-returning APIs (hipError_t, int, etc.) */
static inline void rocm_trace_emit_hip_api_exit_status(const char* api_name,
                                                       int32_t     status) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_api_exit_status)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_api_exit_status,
                                api_name, status);
    }
}

/* Pointer-returning APIs (hipApiName, __hipRegisterFatBinary, etc.) */
static inline void rocm_trace_emit_hip_api_exit_ptr(const char* api_name,
                                                    const void* retval) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_api_exit_ptr)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_api_exit_ptr,
                                api_name, (uint64_t)(uintptr_t)retval);
    }
}

/* Void-returning APIs */
static inline void rocm_trace_emit_hip_api_exit_void(const char* api_name) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_api_exit_void)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_api_exit_void, api_name);
    }
}

/* Pack a (x, y, z) triplet into a 64-bit field using the same lane
 * layout as ROCM_DIM3_PACK (rocm_dim3_pack.h):
 *   bits  0..31  : x  (full 32 bits)
 *   bits 32..47  : y  (16-bit lane, saturates to 0xFFFF on overflow)
 *   bits 48..62  : z  (15-bit lane, saturates to 0x7FFF on overflow)
 *   bit  63      : overflow flag (set iff y or z exceeded its lane;
 *                  z >= 0x8000 is treated as overflow because the lane
 *                  is intentionally 15 bits to keep bit 63 unambiguous)
 *
 * Same uint32_t triplet signature is kept (vs. ROCM_DIM3_PACK's dim3
 * argument) so this header stays self-contained — no <hip/hip_runtime.h>
 * dependency. Lane semantics MUST stay byte-compatible with
 * ROCM_DIM3_PACK so consumers decode both with one routine. */
static inline uint64_t rocm_trace_pack_dims3(uint32_t x, uint32_t y, uint32_t z) {
    const uint64_t x64 = (uint64_t)x;
    const uint64_t y64 = (y > 0xFFFFu) ? 0xFFFFu : (uint64_t)y;
    const uint64_t z64 = (z > 0x7FFFu) ? 0x7FFFu : (uint64_t)z;
    const uint64_t overflow = ((y > 0xFFFFu) || (z > 0x7FFFu))
                                  ? (1ULL << 63) : 0ULL;
    return x64 | (y64 << 32) | (z64 << 48) | overflow;
}

/* hip_kernel_dispatch_enqueue: emitted just before command->enqueue() in
 * ihipModuleLaunchKernel. The enclosing HIP API (if any) is the topmost
 * unmatched hip_api_enter on the same (vpid, vtid). */
static inline void rocm_trace_emit_hip_kernel_dispatch_enqueue(
    const char* kernel_name, void* stream,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    uint32_t shared_mem_bytes) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_kernel_dispatch_enqueue)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_kernel_dispatch_enqueue,
                                kernel_name, stream,
                                rocm_trace_pack_dims3(grid_x, grid_y, grid_z),
                                rocm_trace_pack_dims3(block_x, block_y, block_z),
                                shared_mem_bytes);
    }
}

/* hip_aql_kernel_dispatch_submit: one emit per AQL kernel-dispatch packet
 * written into a queue ring. Join key for the firmware-ring track is
 * (queue_id, write_idx); see rocm_hip_tp.h. */
static inline void rocm_trace_emit_hip_aql_kernel_dispatch_submit(
    uint32_t queue_id, uint64_t write_idx,
    uint64_t kernel_object, uint64_t completion_signal) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_aql_kernel_dispatch_submit)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_aql_kernel_dispatch_submit,
                                queue_id, write_idx,
                                kernel_object, completion_signal);
    }
}

#else /* !HIP_ENABLE_LTTNG_UST */

static inline void rocm_trace_emit_hip_api_enter(const char* a) { (void)a; }
static inline void rocm_trace_emit_hip_api_exit_status(const char* a, int32_t s) {
    (void)a; (void)s;
}
static inline void rocm_trace_emit_hip_api_exit_ptr(const char* a, const void* p) {
    (void)a; (void)p;
}
static inline void rocm_trace_emit_hip_api_exit_void(const char* a) {
    (void)a;
}
static inline void rocm_trace_emit_hip_kernel_dispatch_enqueue(
    const char* a, void* b, uint32_t c, uint32_t d, uint32_t e,
    uint32_t f, uint32_t g, uint32_t h, uint32_t i) {
    (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; (void)i;
}
static inline void rocm_trace_emit_hip_aql_kernel_dispatch_submit(
    uint32_t a, uint64_t b, uint64_t c, uint64_t d) {
    (void)a; (void)b; (void)c; (void)d;
}

#endif

#endif /* ROCM_HIP_TRACE_EMIT_H_ */
