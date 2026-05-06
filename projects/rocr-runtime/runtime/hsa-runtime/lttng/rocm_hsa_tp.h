#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER rocm_hsa

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "rocm_hsa_tp.h"

#if !defined(_ROCM_HSA_TP_H) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define _ROCM_HSA_TP_H

#include <lttng/tracepoint.h>
#include <stdint.h>

/* parent_corr_id (uint64) on every HSA event carries the value of the
 * shared librocprofiler-register active-corr-id TLS slot at emit time.
 * For events fired inside an HSA wrapper that was entered from a HIP API
 * call on the same thread, this is the HIP API's corr_id -- an explicit
 * join key for HIP -> HSA same-thread correlation chains without
 * timestamp heuristics. A value of 0 means no enclosing context. */
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

/* Curated per-API typed tracepoint events (generated header). */
#include "rocm_hsa_curated_tp.h"

/* kernel_dispatch_record: BATCHED FORM — emitted by the firmware-dispatch-log
 * drainer thread once per drain pass with a packed array of all records
 * read in that pass.
 *
 * Each event carries:
 *   - queue_id (one per batch — all records in a batch belong to one queue)
 *   - count: number of records in `records[]`
 *   - records: packed array of `count` × 16-byte FW records, exactly the
 *     mec_dispatch_record_16 struct layout (ts_lo, ts_hi, record_type,
 *     dispatch_idx). Consumer un-packs this to get per-dispatch
 *     {gpu_ts, record_type, dispatch_idx} tuples.
 *
 * Batching motivation: per-record tracepoint calls were the host-side
 * bottleneck under sustained FW write rates (~1M records/sec on busy
 * queues during graph workloads). Each LTTng tracepoint call costs ~1-2
 * us; at 1M/sec that fills a CPU core entirely. Batching N records per
 * call reduces the per-record cost to ~1/N microseconds plus the constant
 * memcpy of N*16 bytes into the LTTng UST buffer. The drainer's typical
 * batch size is bounded by either the per-pass record count or the
 * configured kBatchMax (see drain_one_queue).
 *
 * Per-record `corr_id` field is removed in the batched form: the drainer
 * has no API context to attach (it runs on its own thread), and per-record
 * unique IDs aren't needed for the (queue_id, dispatch_idx) join most
 * stream consumers do. Consumers that need a per-record monotonic ID can
 * derive one from (queue_id, batch_seq, record_index_within_batch).
 *
 * record_type interpretation: the MEC firmware writes one record at
 * dispatch start (rt=2 on gfx950) and one at dispatch end (rt=1 on
 * gfx950). HSA does NOT interpret record_type — the stream consumer
 * joins records on (queue_id, dispatch_idx) and either orders by gpu_ts
 * or applies its own FW-version-specific record_type interpretation.
 *
 * gpu_ts is the raw hardware clock counter written by FW. It is NOT
 * translated to the host system clock domain. The consumer must use the
 * rocm_hsa:clock_sync tracepoint to correlate raw GPU timestamps with
 * the host system clock. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, kernel_dispatch_record,
    LTTNG_UST_TP_ARGS(uint32_t, queue_id, uint32_t, count,
                      const uint8_t*, records, size_t, records_len),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(uint32_t, queue_id, queue_id)
        lttng_ust_field_integer(uint32_t, count, count)
        lttng_ust_field_sequence(uint8_t, records, records,
                                 size_t, records_len)
    )
)

/* kernel_dispatch_drop: tracepoint defined but currently unemittable. The
 * sentinel-scan drainer (core/runtime/dispatch_log.cpp::drain_one_queue)
 * cannot detect ring overruns because the substrate publishes no
 * host-visible FW write pointer to compare against the host read cursor.
 * Reserved for a future overrun-detection mechanism (e.g. a per-slot
 * sequence-number gap check, or a substrate extension that publishes
 * wptr); see rocm_trace_emit.h::rocm_trace_emit_hsa_kernel_dispatch_drop
 * and design spec §10. Per-queue enable failures are reported via stderr
 * WARNING (see Phase A spec §5), not via this tracepoint. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, kernel_dispatch_drop,
    LTTNG_UST_TP_ARGS(uint32_t, queue_id, uint64_t, bytes_lost),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(uint32_t, queue_id, queue_id)
        lttng_ust_field_integer(uint64_t, bytes_lost, bytes_lost)
    )
)

/* clock_sync: emitted periodically to correlate raw GPU timestamps with the
 * host system clock.
 *
 * gpu_id is the node_id of the GPU agent.
 * gpu_ts is the raw hardware clock counter.
 * system_ts is the corresponding host system clock counter.
 */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, clock_sync,
    LTTNG_UST_TP_ARGS(uint64_t, gpu_id, uint64_t, gpu_ts, uint64_t, system_ts),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(uint64_t, gpu_id, gpu_id)
        lttng_ust_field_integer(uint64_t, gpu_ts, gpu_ts)
        lttng_ust_field_integer(uint64_t, system_ts, system_ts)
    )
)

#endif /* _ROCM_HSA_TP_H */

#include <lttng/tracepoint-event.h>
