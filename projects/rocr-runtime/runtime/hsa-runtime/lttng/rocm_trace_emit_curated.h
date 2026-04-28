/* AUTO-GENERATED from curated_apis.yaml. Do not edit by hand.
 * SHA256(curated_apis.yaml) at generation: 048f5ab0f9e96dbcd93746951b5a77bd03a38a8a7db021f676a607645ceaed1c
 *
 * Per-API typed emit helpers for curated parameter capture. Every helper
 * takes (uint64_t corr_id, <captured-args...>, <status_type> status);
 * status is the call's success result, used to gate OUT-param deref.
 * All-IN APIs accept it but mark it unused.
 */
#ifndef ROCM_HSA_TRACE_EMIT_CURATED_H_
#define ROCM_HSA_TRACE_EMIT_CURATED_H_

// NOTE: When the codegen is regenerated (via PR2's lttng_curated_codegen.py),
// ensure "rocm_trace_tid.h" is emitted INSIDE the *_ENABLE_LTTNG_UST guard.
// The unconditional include breaks Windows builds where rocprofiler-register
// is not on the include path. See PR #5475 CI fix history.

#include <stdint.h>
#include <stddef.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST

#include "rocm_trace_tid.h"
#include <atomic>
#include "rocm_hsa_curated_tp.h"

extern std::atomic<bool> rocm_hsa_trace_g_disabled;
#ifndef ROCM_TRACE_DISABLED_DEFINED
#define ROCM_TRACE_DISABLED_DEFINED
static inline bool rocm_trace_disabled(void) {
    return rocm_hsa_trace_g_disabled.load(std::memory_order_relaxed);
}
#endif

        static inline void rocm_trace_emit_hsa_queue_create_args(
            uint64_t corr_id,
    uint64_t agent,
    uint32_t size,
    uint32_t type,
    const void* callback,
    const void* data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t ** queue_out_ptr,
    hsa_status_t status) {
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_queue_create_args)) {
        const uint64_t queue_val =
    (status == HSA_STATUS_SUCCESS && queue_out_ptr != NULL)
        ? (uint64_t)((uint64_t)(uintptr_t)(*queue_out_ptr)) : 0ULL;
    lttng_ust_do_tracepoint(rocm_hsa, hsa_queue_create_args, corr_id,
(uint64_t)(uintptr_t)(agent),
(uint32_t)(size),
(uint32_t)(type),
(uint64_t)(uintptr_t)(callback),
(uint64_t)(uintptr_t)(data),
(uint32_t)(private_segment_size),
(uint32_t)(group_segment_size),
queue_val);
            }
        }

        static inline void rocm_trace_emit_hsa_queue_destroy_args(
            uint64_t corr_id,
    uint64_t queue,
    hsa_status_t /*status*/ /* unused: all-IN API */) {
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_queue_destroy_args)) {
            lttng_ust_do_tracepoint(rocm_hsa, hsa_queue_destroy_args, corr_id,
(uint64_t)(uintptr_t)(queue));
            }
        }

        static inline void rocm_trace_emit_hsa_amd_queue_intercept_create_args(
            uint64_t corr_id,
    uint64_t agent_handle,
    uint32_t size,
    uint32_t type,
    const void* callback,
    const void* data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t ** queue_out_ptr,
    hsa_status_t status) {
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_amd_queue_intercept_create_args)) {
        const uint64_t queue_val =
    (status == HSA_STATUS_SUCCESS && queue_out_ptr != NULL)
        ? (uint64_t)((uint64_t)(uintptr_t)(*queue_out_ptr)) : 0ULL;
    lttng_ust_do_tracepoint(rocm_hsa, hsa_amd_queue_intercept_create_args, corr_id,
(uint64_t)(uintptr_t)(agent_handle),
(uint32_t)(size),
(uint32_t)(type),
(uint64_t)(uintptr_t)(callback),
(uint64_t)(uintptr_t)(data),
(uint32_t)(private_segment_size),
(uint32_t)(group_segment_size),
queue_val);
            }
        }

        static inline void rocm_trace_emit_hsa_signal_create_args(
            uint64_t corr_id,
    int64_t initial_value,
    uint32_t num_consumers,
    const void* consumers,
    hsa_signal_t * signal_out_ptr,
    hsa_status_t status) {
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_signal_create_args)) {
        const uint64_t signal_val =
    (status == HSA_STATUS_SUCCESS && signal_out_ptr != NULL)
        ? (uint64_t)((signal_out_ptr->handle)) : 0ULL;
    lttng_ust_do_tracepoint(rocm_hsa, hsa_signal_create_args, corr_id,
(int64_t)(initial_value),
(uint32_t)(num_consumers),
(uint64_t)(uintptr_t)(consumers),
signal_val);
            }
        }

        static inline void rocm_trace_emit_hsa_signal_destroy_args(
            uint64_t corr_id,
    uint64_t signal,
    hsa_status_t /*status*/ /* unused: all-IN API */) {
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_signal_destroy_args)) {
            lttng_ust_do_tracepoint(rocm_hsa, hsa_signal_destroy_args, corr_id,
(uint64_t)(uintptr_t)(signal));
            }
        }

        static inline void rocm_trace_emit_hsa_amd_signal_create_args(
            uint64_t corr_id,
    int64_t initial_value,
    uint32_t num_consumers,
    const void* consumers,
    uint64_t attributes,
    hsa_signal_t * signal_out_ptr,
    hsa_status_t status) {
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_amd_signal_create_args)) {
        const uint64_t signal_val =
    (status == HSA_STATUS_SUCCESS && signal_out_ptr != NULL)
        ? (uint64_t)((signal_out_ptr->handle)) : 0ULL;
    lttng_ust_do_tracepoint(rocm_hsa, hsa_amd_signal_create_args, corr_id,
(int64_t)(initial_value),
(uint32_t)(num_consumers),
(uint64_t)(uintptr_t)(consumers),
(uint64_t)(attributes),
signal_val);
            }
        }

        static inline void rocm_trace_emit_hsa_amd_memory_pool_allocate_args(
            uint64_t corr_id,
    uint64_t memory_pool,
    size_t size,
    uint32_t flags,
    void ** ptr_out_ptr,
    hsa_status_t status) {
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_amd_memory_pool_allocate_args)) {
        const uint64_t ptr_val =
    (status == HSA_STATUS_SUCCESS && ptr_out_ptr != NULL)
        ? (uint64_t)((uint64_t)(uintptr_t)(*ptr_out_ptr)) : 0ULL;
    lttng_ust_do_tracepoint(rocm_hsa, hsa_amd_memory_pool_allocate_args, corr_id,
(uint64_t)(uintptr_t)(memory_pool),
(uint64_t)(size),
(uint32_t)(flags),
ptr_val);
            }
        }

        static inline void rocm_trace_emit_hsa_amd_memory_pool_free_args(
            uint64_t corr_id,
    const void* ptr,
    hsa_status_t /*status*/ /* unused: all-IN API */) {
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_amd_memory_pool_free_args)) {
            lttng_ust_do_tracepoint(rocm_hsa, hsa_amd_memory_pool_free_args, corr_id,
(uint64_t)(uintptr_t)(ptr));
            }
        }

        static inline void rocm_trace_emit_hsa_amd_memory_async_copy_args(
            uint64_t corr_id,
    const void* dst,
    uint64_t dst_agent,
    const void* src,
    uint64_t src_agent,
    size_t size,
    uint32_t num_dep_signals,
    const void* dep_signals,
    uint64_t completion_signal,
    hsa_status_t /*status*/ /* unused: all-IN API */) {
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_amd_memory_async_copy_args)) {
            lttng_ust_do_tracepoint(rocm_hsa, hsa_amd_memory_async_copy_args, corr_id,
(uint64_t)(uintptr_t)(dst),
(uint64_t)(uintptr_t)(dst_agent),
(uint64_t)(uintptr_t)(src),
(uint64_t)(uintptr_t)(src_agent),
(uint64_t)(size),
(uint32_t)(num_dep_signals),
(uint64_t)(uintptr_t)(dep_signals),
(uint64_t)(uintptr_t)(completion_signal));
            }
        }

        static inline void rocm_trace_emit_hsa_amd_memory_async_copy_on_engine_args(
            uint64_t corr_id,
    const void* dst,
    uint64_t dst_agent,
    const void* src,
    uint64_t src_agent,
    size_t size,
    uint32_t num_dep_signals,
    uint64_t completion_signal,
    int32_t engine_id,
    int force_copy_on_sdma,
    hsa_status_t /*status*/ /* unused: all-IN API */) {
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_amd_memory_async_copy_on_engine_args)) {
            lttng_ust_do_tracepoint(rocm_hsa, hsa_amd_memory_async_copy_on_engine_args, corr_id,
(uint64_t)(uintptr_t)(dst),
(uint64_t)(uintptr_t)(dst_agent),
(uint64_t)(uintptr_t)(src),
(uint64_t)(uintptr_t)(src_agent),
(uint64_t)(size),
(uint32_t)(num_dep_signals),
(uint64_t)(uintptr_t)(completion_signal),
(int32_t)(engine_id),
(uint32_t)(!!(force_copy_on_sdma)));
            }
        }


#else  /* HSA_ENABLE_LTTNG_UST not defined — all helpers are no-ops */

static inline void rocm_trace_emit_hsa_queue_create_args(uint64_t, uint64_t, uint32_t, uint32_t, const void*, const void*, uint32_t, uint32_t, hsa_queue_t **, hsa_status_t) {}
static inline void rocm_trace_emit_hsa_queue_destroy_args(uint64_t, uint64_t, hsa_status_t) {}
static inline void rocm_trace_emit_hsa_amd_queue_intercept_create_args(uint64_t, uint64_t, uint32_t, uint32_t, const void*, const void*, uint32_t, uint32_t, hsa_queue_t **, hsa_status_t) {}
static inline void rocm_trace_emit_hsa_signal_create_args(uint64_t, int64_t, uint32_t, const void*, hsa_signal_t *, hsa_status_t) {}
static inline void rocm_trace_emit_hsa_signal_destroy_args(uint64_t, uint64_t, hsa_status_t) {}
static inline void rocm_trace_emit_hsa_amd_signal_create_args(uint64_t, int64_t, uint32_t, const void*, uint64_t, hsa_signal_t *, hsa_status_t) {}
static inline void rocm_trace_emit_hsa_amd_memory_pool_allocate_args(uint64_t, uint64_t, size_t, uint32_t, void **, hsa_status_t) {}
static inline void rocm_trace_emit_hsa_amd_memory_pool_free_args(uint64_t, const void*, hsa_status_t) {}
static inline void rocm_trace_emit_hsa_amd_memory_async_copy_args(uint64_t, const void*, uint64_t, const void*, uint64_t, size_t, uint32_t, const void*, uint64_t, hsa_status_t) {}
static inline void rocm_trace_emit_hsa_amd_memory_async_copy_on_engine_args(uint64_t, const void*, uint64_t, const void*, uint64_t, size_t, uint32_t, uint64_t, int32_t, int, hsa_status_t) {}

#endif  /* HSA_ENABLE_LTTNG_UST */

#endif  /* ROCM_HSA_TRACE_EMIT_CURATED_H_ */
