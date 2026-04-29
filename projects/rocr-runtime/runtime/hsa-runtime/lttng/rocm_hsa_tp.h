#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER rocm_hsa

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "rocm_hsa_tp.h"

#if !defined(_ROCM_HSA_TP_H) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define _ROCM_HSA_TP_H

#include <lttng/tracepoint.h>
#include <stdint.h>

/* Schema version. Bump on any breaking change to event field layout
 * (field add/remove/rename/type-change) so consumers can detect they
 * are reading a stream produced by a different schema generation.
 *
 * Version history:
 *   2 - corr_id / parent_corr_id / tid carried as explicit event fields.
 *   3 - corr_id / parent_corr_id / tid removed from every event; identity
 *       and parent attribution derive from the (vpid, vtid, timestamp)
 *       channel contexts. See rocm_trace_emit.h for the consumer recipe.
 */
#define ROCM_HSA_TP_SCHEMA_VERSION 3

/* Per-event identity (process id, thread id, timestamp) is provided by
 * LTTng-UST's native channel contexts -- consumers configure their channel
 * with `lttng add-context --userspace --type vpid --type vtid` (and the CTF
 * event-header timestamp is always present). HIP -> HSA same-thread
 * correlation is reconstructed by walking the per-(vpid, vtid) merged
 * event stream sorted by timestamp; the enclosing HIP API is the topmost
 * unmatched hip_api_enter on the same thread at the time the HSA event
 * fires. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_api_enter,
    LTTNG_UST_TP_ARGS(const char*, api_name),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
    )
)

/* hsa_api_exit_status: for hsa_status_t / int returns. Use the typed
 * hsa_api_exit_u64 / _i64 events below for 64-bit integer returns from
 * hsa_signal_* / hsa_queue_* index ops -- those preserve full width. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_api_exit_status,
    LTTNG_UST_TP_ARGS(const char*, api_name, int32_t, status),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(int32_t, status, status)
    )
)

/* hsa_api_exit_u64: for unsigned 64-bit integer returns. Used by queue
 * index ops (hsa_queue_load_*_index_*, hsa_queue_cas_write_index_*,
 * hsa_queue_add_write_index_*) -- their return type is uint64_t. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_api_exit_u64,
    LTTNG_UST_TP_ARGS(const char*, api_name, uint64_t, retval),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(uint64_t, retval, retval)
    )
)

/* hsa_api_exit_i64: for signed 64-bit integer returns. Used by signal value
 * ops (hsa_signal_load_*, hsa_signal_cas_*, hsa_signal_exchange_*,
 * hsa_signal_wait_*) -- their return type is hsa_signal_value_t (int64_t). */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_api_exit_i64,
    LTTNG_UST_TP_ARGS(const char*, api_name, int64_t, retval),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(int64_t, retval, retval)
    )
)

/* hsa_api_exit_ptr: for pointer-returning HSA APIs. Captured as uint64_t hex
 * (0 if NULL). */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_api_exit_ptr,
    LTTNG_UST_TP_ARGS(const char*, api_name, uint64_t, retval_ptr),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer_hex(uint64_t, retval_ptr, retval_ptr)
    )
)

/* hsa_api_exit_void: for void-returning HSA APIs (e.g., signal store ops). */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_api_exit_void,
    LTTNG_UST_TP_ARGS(const char*, api_name),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
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
                      uint8_t, packet_type),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(uint32_t, queue_id, queue_id)
        lttng_ust_field_integer(int64_t, write_idx, write_idx)
        lttng_ust_field_integer(uint8_t, packet_type, packet_type)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, hsa_intercept_packets,
    LTTNG_UST_TP_ARGS(uint32_t, queue_id, uint64_t, pkt_index,
                      uint32_t, pkt_count, uint8_t, packet_type),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(uint32_t, queue_id, queue_id)
        lttng_ust_field_integer(uint64_t, pkt_index, pkt_index)
        lttng_ust_field_integer(uint32_t, pkt_count, pkt_count)
        lttng_ust_field_integer(uint8_t, packet_type, packet_type)
    )
)

/* Curated per-API typed tracepoint events (generated header). */
#include "rocm_hsa_curated_tp.h"

#endif /* _ROCM_HSA_TP_H */

#include <lttng/tracepoint-event.h>
