#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER rocm_hsa

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "rocm_hsa_tp.h"

#if !defined(_ROCM_HSA_TP_H) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define _ROCM_HSA_TP_H

#include <lttng/tracepoint.h>
#include <stdint.h>

/* parent_corr_id (uint64) is added to every HSA event. It carries the
 * value of the shared librocprofiler-register active-corr-id TLS slot at
 * emit time. For events fired inside an HSA wrapper that was entered from
 * a HIP API call on the same thread, this field carries the HIP API's
 * corr_id, providing an explicit join key for HIP -> HSA same-thread
 * correlation chains without timestamp heuristics. A value of 0 means
 * no enclosing correlation context. CTF schema additions are
 * backward-compatible: older consumers ignore the new field. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_api_enter,
    LTTNG_UST_TP_ARGS(const char*, api_name, uint64_t, corr_id, uint32_t, tid,
                      uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer(uint32_t, tid, tid)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

/* hsa_api_exit_status: for hsa_status_t / int returns. Use the typed
 * hsa_api_exit_u64 / _i64 events below for 64-bit integer returns from
 * hsa_signal_* / hsa_queue_* index ops -- those preserve full width. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_api_exit_status,
    LTTNG_UST_TP_ARGS(const char*, api_name, uint64_t, corr_id, int32_t, status,
                      uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer(int32_t, status, status)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

/* hsa_api_exit_u64: for unsigned 64-bit integer returns. Used by queue
 * index ops (hsa_queue_load_*_index_*, hsa_queue_cas_write_index_*,
 * hsa_queue_add_write_index_*) -- their return type is uint64_t. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_api_exit_u64,
    LTTNG_UST_TP_ARGS(const char*, api_name, uint64_t, corr_id,
                      uint64_t, retval, uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer(uint64_t, retval, retval)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

/* hsa_api_exit_i64: for signed 64-bit integer returns. Used by signal value
 * ops (hsa_signal_load_*, hsa_signal_cas_*, hsa_signal_exchange_*,
 * hsa_signal_wait_*) -- their return type is hsa_signal_value_t (int64_t). */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_api_exit_i64,
    LTTNG_UST_TP_ARGS(const char*, api_name, uint64_t, corr_id,
                      int64_t, retval, uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer(int64_t, retval, retval)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

/* hsa_api_exit_ptr: for pointer-returning HSA APIs. Captured as uint64_t hex
 * (0 if NULL). */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_api_exit_ptr,
    LTTNG_UST_TP_ARGS(const char*, api_name, uint64_t, corr_id, uint64_t, retval_ptr,
                      uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer_hex(uint64_t, retval_ptr, retval_ptr)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

/* hsa_api_exit_void: for void-returning HSA APIs (e.g., signal store ops). */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_api_exit_void,
    LTTNG_UST_TP_ARGS(const char*, api_name, uint64_t, corr_id,
                      uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

/* hsa_doorbell_ring carries an explicit packet_type discriminator so consumers
 * can filter at the LTTng-event level via `--filter 'packet_type == 0'`
 * rather than after CTF parse. Values:
 *   0 = KERNEL_DISPATCH
 *   1 = AGENT_DISPATCH
 *   2 = BARRIER_AND
 *   3 = BARRIER_OR
 *   4 = PM4 (set by the PM4 IB submit path; not an AQL packet type)
 *   255 = UNKNOWN
 *
 * The signed write_idx (hsa_signal_value_t is int64_t) is preserved as
 * int64_t in the schema to maintain sign semantics.
 */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_doorbell_ring,
    LTTNG_UST_TP_ARGS(uint32_t, queue_id, int64_t, write_idx,
                      uint8_t, packet_type, uint64_t, corr_id, uint32_t, tid,
                      uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(uint32_t, queue_id, queue_id)
        lttng_ust_field_integer(int64_t, write_idx, write_idx)
        lttng_ust_field_integer(uint8_t, packet_type, packet_type)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer(uint32_t, tid, tid)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_intercept_packets,
    LTTNG_UST_TP_ARGS(uint32_t, queue_id, uint64_t, pkt_index,
                      uint32_t, pkt_count, uint8_t, packet_type,
                      uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(uint32_t, queue_id, queue_id)
        lttng_ust_field_integer(uint64_t, pkt_index, pkt_index)
        lttng_ust_field_integer(uint32_t, pkt_count, pkt_count)
        lttng_ust_field_integer(uint8_t, packet_type, packet_type)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

/* Curated per-API typed tracepoint events. Generated by
 * lttng_curated_codegen.py from curated_apis.yaml. See spec §5.1. */
#include "rocm_hsa_curated_tp.h"

#endif /* _ROCM_HSA_TP_H */

#include <lttng/tracepoint-event.h>
