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
 *
 * Cross-runtime same-thread correlation:
 *   emit_hip_api_enter pushes the entering API's corr_id onto the shared
 *   librocprofiler-register TLS slot and emit_hip_api_exit_* pops it. This
 *   is the slot HSA's emit helpers read at emit time to populate
 *   `parent_corr_id` on HSA tracepoints fired during a HIP API call on the
 *   same thread.
 */
#ifndef ROCM_HIP_TRACE_EMIT_H_
#define ROCM_HIP_TRACE_EMIT_H_

#include <stdint.h>
#include "rocm_trace_tid.h"

#if defined(HIP_ENABLE_LTTNG_UST) && HIP_ENABLE_LTTNG_UST

#include <atomic>
#include "rocm_hip_tp.h"

/* Runtime-wide kill switch defined in rocm_trace_init.cpp. Set true when
 * ROCM_LTTNG_UST_DISABLE=1 or when running in a non-default link namespace
 * (dlmopen mitigation). When true every emit helper short-circuits before
 * touching any LTTng state OR the auto-stack. We deliberately skip the
 * push/pop too: when emission is disabled the parent_corr_id chain is
 * unobservable, so leaving the stack untouched is correct AND saves a
 * pair of TLS writes per API call. The enter and exit helpers both check
 * the same flag so the depth stays balanced.
 *
 * The symbol is named per-DSO (`rocm_hip_...` in libamdhip64,
 * `rocm_hsa_...` in libhsa-runtime64) so ELF symbol interposition cannot
 * cross-bind the two runtimes' flags. */
extern std::atomic<bool> rocm_hip_trace_g_disabled;

static inline bool rocm_trace_disabled(void) {
    return rocm_hip_trace_g_disabled.load(std::memory_order_relaxed);
}

/* HIP emit_enter: capture parent_corr_id (= the active slot value BEFORE
 * the push) and emit it as the new parent_corr_id field. The push then
 * makes this HIP call's own corr_id the active slot value, so any nested
 * HSA / HIP tracepoints fired from within this HIP API's body see this
 * call as their parent.
 *
 * The push always runs regardless of tracepoint enable state so that
 * nested tracepoints (which may be enabled even when this one is not) see
 * correct parent values. The matching pop happens in the exit emit helpers
 * (and in the CATCH macro for the exception path). Pattern matches HSA's
 * emit_hsa_api_enter so HIP <-> HSA chains have consistent semantics. */
static inline void rocm_trace_emit_hip_api_enter(const char* api_name,
                                                 uint64_t    corr_id) {
    if (rocm_trace_disabled()) return;
    const uint64_t parent = rocp_reg_auto_push(corr_id);
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_api_enter)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_api_enter,
                                api_name, corr_id, rocm_trace_current_tid(),
                                parent);
    }
}

/* Status-returning APIs (hipError_t, int, etc.).
 * Pop first, then read active. Post-pop active equals pre-push parent, so
 * the exit event's parent_corr_id matches the matching enter's
 * parent_corr_id. Same pattern as HSA's emit_hsa_api_exit_status. */
static inline void rocm_trace_emit_hip_api_exit_status(const char* api_name,
                                                       uint64_t    corr_id,
                                                       int32_t     status) {
    if (rocm_trace_disabled()) return;
    rocp_reg_auto_pop();
    const uint64_t parent = rocp_reg_active_corr_id_get();
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_api_exit_status)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_api_exit_status,
                                api_name, corr_id, status, parent);
    }
}

/* Pointer-returning APIs (hipApiName, __hipRegisterFatBinary, etc.) */
static inline void rocm_trace_emit_hip_api_exit_ptr(const char* api_name,
                                                    uint64_t    corr_id,
                                                    const void* retval) {
    if (rocm_trace_disabled()) return;
    rocp_reg_auto_pop();
    const uint64_t parent = rocp_reg_active_corr_id_get();
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_api_exit_ptr)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_api_exit_ptr,
                                api_name, corr_id, (uint64_t)(uintptr_t)retval,
                                parent);
    }
}

/* Void-returning APIs */
static inline void rocm_trace_emit_hip_api_exit_void(const char* api_name,
                                                     uint64_t    corr_id) {
    if (rocm_trace_disabled()) return;
    rocp_reg_auto_pop();
    const uint64_t parent = rocp_reg_active_corr_id_get();
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_api_exit_void)) {
        lttng_ust_do_tracepoint(rocm_hip, hip_api_exit_void,
                                api_name, corr_id, parent);
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

/* hip_kernel_dispatch_enqueue: emitted just before command->enqueue() in
 * ihipModuleLaunchKernel. corr_id is freshly minted for THIS dispatch
 * enqueue event so the dispatch step has its own identity, distinct from
 * the launching HIP API. parent_corr_id is the active TLS slot at emit
 * time -- the surrounding HIP API launch's corr_id when called from
 * inside an HIP API body, or 0 if launched outside any HIP API context. */
static inline void rocm_trace_emit_hip_kernel_dispatch_enqueue(
    const char* kernel_name, void* stream,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    uint32_t shared_mem_bytes) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_kernel_dispatch_enqueue)) {
        const uint64_t self_corr   = rocp_reg_next_corr_id();
        const uint64_t parent_corr = rocp_reg_active_corr_id_get();
        lttng_ust_do_tracepoint(rocm_hip, hip_kernel_dispatch_enqueue,
                                kernel_name, self_corr, rocm_trace_current_tid(),
                                stream,
                                rocm_trace_pack_dims3(grid_x, grid_y, grid_z),
                                rocm_trace_pack_dims3(block_x, block_y, block_z),
                                shared_mem_bytes,
                                parent_corr);
    }
}

/* hip_aql_kernel_dispatch_submit: emit at the HIP CLR per-packet write site,
 * once per AQL kernel-dispatch packet written to a queue ring. Mints its
 * own corr_id (distinct identity for this submit step). parent_corr_id is
 * the active TLS slot at emit time -- the surrounding HIP API's corr_id
 * (which the wrapper pushed on entry) when called from inside a HIP API
 * call body; otherwise 0.
 *
 * Join key for the firmware-ring track is (queue_id, write_idx); see
 * rocm_hip_tp.h for the rationale. There is no separate software
 * dispatch counter. */
static inline void rocm_trace_emit_hip_aql_kernel_dispatch_submit(
    uint32_t queue_id, uint64_t write_idx,
    uint64_t kernel_object, uint64_t completion_signal) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hip, hip_aql_kernel_dispatch_submit)) {
        const uint64_t self_corr   = rocp_reg_next_corr_id();
        const uint64_t parent_corr = rocp_reg_active_corr_id_get();
        lttng_ust_do_tracepoint(rocm_hip, hip_aql_kernel_dispatch_submit,
                                queue_id, write_idx,
                                self_corr, parent_corr, rocm_trace_current_tid(),
                                kernel_object, completion_signal);
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
