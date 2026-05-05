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

/* kernel_dispatch_record: emitted by the firmware-dispatch-log drainer
 * thread (NOT the application thread) once per FW-written 16-byte record.
 *
 * The MEC firmware writes two records per kernel dispatch: the two
 * records share the same dispatch_idx and have different record_type
 * values (currently 1 and 2). HSA does NOT interpret which record_type
 * means start vs end; that interpretation is left to the stream
 * consumer, which can:
 *   (a) join records on (queue_id, dispatch_idx) to find pairs, and
 *   (b) infer start vs end from gpu_ts ordering (smaller ts = start)
 *       OR from record_type if the consumer has FW-version-specific
 *       knowledge.
 *
 * Empirical observation on the gfx950 MEC firmware (gc_9_5_0_mec.bin)
 * shows record_type==2 carries the EARLIER GPU clock (dispatch start)
 * and record_type==1 carries the LATER GPU clock (EOP / dispatch end).
 * This is OPPOSITE of what cpc_tracing's mec_dispatch_record.h comment
 * claims. Rather than baking a host-side polarity decision against a
 * non-versioned FW contract, HSA emits both records and lets the
 * consumer decide. Future FW revisions that change the polarity (or add
 * a third record_type) cost only a consumer change.
 *
 * gpu_ts is the raw hardware clock counter written by the FW. It is NOT
 * translated to the host system clock domain. The consumer must use the
 * rocm_hsa:clock_sync tracepoint to correlate this raw GPU timestamp with
 * the system clock.
 *
 * record_type is the raw value FW wrote (uint8_t — the FW field is
 * 32 bits but the values seen so far fit in uint8_t and the on-the-wire
 * payload size matters; see drain_one_queue for the narrowing).
 *
 * dispatch_idx is the low 32 bits of the AQL read_dispatch_id at the
 * FW's moment of dispatch processing; stream-side joiners against
 * rocm_hsa:hsa_doorbell_ring MUST mask the doorbell write_idx to 32
 * bits before comparing.
 *
 * self_corr is freshly minted per emission so this event has its own
 * identity; parent_corr_id is always 0 (the drainer thread has no API
 * context — real parent recovery is consumer-side via the doorbell
 * join). See Phase A spec §6 for full rationale. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hsa, kernel_dispatch_record,
    LTTNG_UST_TP_ARGS(uint32_t, queue_id, uint32_t, dispatch_idx,
                      uint64_t, gpu_ts, uint8_t, record_type,
                      uint64_t, self_corr, uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(uint32_t, queue_id, queue_id)
        lttng_ust_field_integer(uint32_t, dispatch_idx, dispatch_idx)
        lttng_ust_field_integer(uint64_t, gpu_ts, gpu_ts)
        lttng_ust_field_integer(uint8_t, record_type, record_type)
        lttng_ust_field_integer(uint64_t, corr_id, self_corr)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
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
