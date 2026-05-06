#ifndef ROCM_HSA_TRACE_EMIT_H_
#define ROCM_HSA_TRACE_EMIT_H_

#include <stdint.h>
#include <stddef.h>

/* Packet-type values matching the schema in rocm_hsa_tp.h */
enum {
    ROCM_PKT_KERNEL_DISPATCH = 0,
    ROCM_PKT_AGENT_DISPATCH  = 1,
    ROCM_PKT_BARRIER_AND     = 2,
    ROCM_PKT_BARRIER_OR      = 3,
    ROCM_PKT_PM4             = 4,
    ROCM_PKT_UNKNOWN         = 255,
};

/* TLS hint used by the PM4 doorbell-ring path to tell StoreRelaxed that
 * the ringing packet is PM4 rather than AQL. We use a separate "valid"
 * boolean alongside the value because 0 (ROCM_PKT_KERNEL_DISPATCH) is a
 * valid hint value, so it cannot also serve as the "unset" sentinel.
 * Consumed-on-read (valid is cleared back to 0). */
static __thread uint8_t g_rocm_packet_type_hint;
static __thread uint8_t g_rocm_packet_type_hint_valid;
static inline void rocm_trace_set_packet_type_hint(uint8_t t) {
    g_rocm_packet_type_hint = t;
    g_rocm_packet_type_hint_valid = 1;
}
/* Returns the hint if set, ROCM_PKT_UNKNOWN otherwise. Always clears
 * the valid flag. */
static inline uint8_t rocm_trace_consume_packet_type_hint(void) {
    if (!g_rocm_packet_type_hint_valid) return ROCM_PKT_UNKNOWN;
    uint8_t t = g_rocm_packet_type_hint;
    g_rocm_packet_type_hint_valid = 0;
    return t;
}

/* Sniff the AQL packet at queue_base[(write_idx - 1) & queue_size_mask].
 * The first 16 bits are the AQL packet header; bits 0..7 are the packet type
 * (HSA_PACKET_TYPE_*). Returns one of the ROCM_PKT_* values above. */
static inline uint8_t rocm_trace_sniff_packet_type(
    const void* queue_base, uint64_t queue_size_mask, int64_t write_idx) {
    if (write_idx <= 0 || queue_base == NULL) return ROCM_PKT_UNKNOWN;
    /* Each AQL packet is 64 bytes; the just-written packet is at write_idx-1. */
    const uint16_t* hdr = (const uint16_t*)((const char*)queue_base
        + (((uint64_t)(write_idx - 1)) & queue_size_mask) * 64);
    uint8_t aql_type = (uint8_t)(*hdr & 0xFF);
    /* HSA AQL packet type values per HSA spec (hsa_packet_type_t):
     *   HSA_PACKET_TYPE_VENDOR_SPECIFIC = 0
     *   HSA_PACKET_TYPE_INVALID         = 1
     *   HSA_PACKET_TYPE_KERNEL_DISPATCH = 2
     *   HSA_PACKET_TYPE_BARRIER_AND     = 3
     *   HSA_PACKET_TYPE_AGENT_DISPATCH  = 4
     *   HSA_PACKET_TYPE_BARRIER_OR      = 5
     */
    switch (aql_type) {
        case 2:  return ROCM_PKT_KERNEL_DISPATCH;
        case 4:  return ROCM_PKT_AGENT_DISPATCH;
        case 3:  return ROCM_PKT_BARRIER_AND;
        case 5:  return ROCM_PKT_BARRIER_OR;
        default: return ROCM_PKT_UNKNOWN;
    }
}

#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST

/* rocm_trace_tid.h transitively includes <rocprofiler-register/correlation.h>
 * which is only on the include path when LTTng is enabled. The no-op stubs
 * in the #else branch don't use anything from rocm_trace_tid.h, so this
 * include lives inside the LTTng guard. */
#include "rocm_trace_tid.h"
#include <atomic>
#include "rocm_hsa_tp.h"

/* Runtime-wide kill switch defined in rocm_trace_init.cpp. When true,
 * every emit helper short-circuits before touching LTTng state or the
 * auto-stack -- enter and exit both check the same flag so the auto-stack
 * depth stays balanced. */
extern std::atomic<bool> rocm_hsa_trace_g_disabled;

#ifndef ROCM_TRACE_DISABLED_DEFINED
#define ROCM_TRACE_DISABLED_DEFINED
static inline bool rocm_trace_disabled(void) {
    return rocm_hsa_trace_g_disabled.load(std::memory_order_relaxed);
}
#endif

/* HSA emit_enter: capture parent_corr_id (= the active slot value BEFORE
 * the push), then push this call's corr_id so any nested HSA / non-HSA
 * tracepoints see this call as their parent. The push runs regardless of
 * this tracepoint's enable state so nested (possibly enabled) tracepoints
 * still get correct parent values; the matching pop happens in the exit
 * emit helpers. */
static inline void rocm_trace_emit_hsa_api_enter(const char* api_name,
                                                 uint64_t    corr_id) {
    if (rocm_trace_disabled()) return;
    const uint64_t parent = rocp_reg_auto_push(corr_id);
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_api_enter)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_api_enter,
                                api_name, corr_id, rocm_trace_current_tid(),
                                parent);
    }
}

/* Status-returning APIs (hsa_status_t, int, etc.) */
static inline void rocm_trace_emit_hsa_api_exit_status(const char* api_name,
                                                       uint64_t    corr_id,
                                                       int32_t     status) {
    if (rocm_trace_disabled()) return;
    /* Pop first, then read active: post-pop active == pre-push parent, so
     * the exit event's parent_corr_id matches the matching enter's. */
    rocp_reg_auto_pop();
    const uint64_t parent = rocp_reg_active_corr_id_get();
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_api_exit_status)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_api_exit_status,
                                api_name, corr_id, status, parent);
    }
}

/* Unsigned 64-bit integer-returning APIs (queue index ops). Preserves
 * full width of the return value. */
static inline void rocm_trace_emit_hsa_api_exit_u64(const char* api_name,
                                                    uint64_t    corr_id,
                                                    uint64_t    retval) {
    if (rocm_trace_disabled()) return;
    rocp_reg_auto_pop();
    const uint64_t parent = rocp_reg_active_corr_id_get();
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_api_exit_u64)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_api_exit_u64,
                                api_name, corr_id, retval, parent);
    }
}

/* Signed 64-bit integer-returning APIs (signal value ops). Preserves
 * full width and sign of the return value. */
static inline void rocm_trace_emit_hsa_api_exit_i64(const char* api_name,
                                                    uint64_t    corr_id,
                                                    int64_t     retval) {
    if (rocm_trace_disabled()) return;
    rocp_reg_auto_pop();
    const uint64_t parent = rocp_reg_active_corr_id_get();
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_api_exit_i64)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_api_exit_i64,
                                api_name, corr_id, retval, parent);
    }
}

/* Pointer-returning APIs */
static inline void rocm_trace_emit_hsa_api_exit_ptr(const char* api_name,
                                                    uint64_t    corr_id,
                                                    const void* retval) {
    if (rocm_trace_disabled()) return;
    rocp_reg_auto_pop();
    const uint64_t parent = rocp_reg_active_corr_id_get();
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_api_exit_ptr)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_api_exit_ptr,
                                api_name, corr_id, (uint64_t)retval, parent);
    }
}

/* Void-returning APIs */
static inline void rocm_trace_emit_hsa_api_exit_void(const char* api_name,
                                                     uint64_t    corr_id) {
    if (rocm_trace_disabled()) return;
    rocp_reg_auto_pop();
    const uint64_t parent = rocp_reg_active_corr_id_get();
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_api_exit_void)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_api_exit_void,
                                api_name, corr_id, parent);
    }
}

/* Doorbell ring: mints a fresh self_corr so the schema's `corr_id`
 * uniquely identifies this doorbell occurrence (would otherwise alias the
 * surrounding HSA wrapper's corr_id and make corr_id == parent_corr_id).
 * parent_corr_id is the surrounding HSA wrapper's corr_id when called
 * from within one, or 0 otherwise. */
static inline void rocm_trace_emit_hsa_doorbell_ring(uint32_t queue_id,
                                                     int64_t  write_idx,
                                                     uint8_t  packet_type) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_doorbell_ring)) {
        const uint64_t self_corr = rocp_reg_next_corr_id();
        const uint64_t parent    = rocp_reg_active_corr_id_get();
        lttng_ust_do_tracepoint(rocm_hsa, hsa_doorbell_ring,
                                queue_id, write_idx, packet_type,
                                self_corr, rocm_trace_current_tid(), parent);
    }
}

static inline void rocm_trace_emit_hsa_intercept_packets(uint32_t queue_id,
                                                         uint64_t pkt_index,
                                                         uint32_t pkt_count,
                                                         uint8_t  packet_type) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_intercept_packets)) {
        const uint64_t parent = rocp_reg_active_corr_id_get();
        lttng_ust_do_tracepoint(rocm_hsa, hsa_intercept_packets,
                                queue_id, pkt_index, pkt_count, packet_type,
                                parent);
    }
}

/* Curated per-API typed emit helpers (generated header). */
#include "rocm_trace_emit_curated.h"

/* kernel_dispatch_record emit helper. Called ONLY by the firmware-dispatch-
 * log drainer thread (NOT the application thread) from
 * dispatch_log::drain_one_queue. Emits one event per FW-written 16-byte
 * record (no host-side START/END pairing — see rocm_hsa_tp.h
 * kernel_dispatch_record doc comment for the consumer-side join contract).
 *
 * force_emit=true bypasses the per-tracepoint lttng_ust_tracepoint_enabled()
 * guard but NOT the rocm_trace_disabled() kill switch. The bypass is
 * REQUIRED for the disable-edge final drain (Phase A spec §4): by the time
 * final drain runs, the user has already disabled the tracepoint (that
 * disablement is what triggered the disable edge), so the steady-state
 * guard would otherwise return false and the "records the firmware has
 * already written before wait_for_idle returns are emitted" loss guarantee
 * would be unachievable. Steady-state drainer passes always pass
 * force_emit=false.
 *
 * gpu_ts is the raw FW-written GPU clock. It is NOT translated to the host
 * system clock domain by the drainer. The consumer must use the rocm_hsa:clock_sync
 * tracepoint to correlate this raw GPU timestamp with the system clock.
 * record_type is the raw FW-written tag, narrowed to uint8_t at
 * the call site (current FW values are 1 and 2; see drain_one_queue).
 *
 * parent_corr_id is always 0 because the drainer thread has no API context
 * (real parent is recovered consumer-side via the
 * rocm_hsa:hsa_doorbell_ring join on (queue_id, dispatch_idx)).
 */
/* Batched form: emit `count` records (each a 16-byte mec_dispatch_record_16
 * laid out [ts_lo, ts_hi, record_type, dispatch_idx]) under one tracepoint
 * call. The drainer accumulates records during a drain pass and calls this
 * once per pass (or once per kBatchMax records, whichever is smaller).
 *
 * `records` is a pointer to a contiguous packed array of `count` × 16
 * bytes; `records_len` is `count * 16` (the byte length the LTTng
 * sequence field needs).
 *
 * Per-record corr_id is no longer emitted: per-batch unique IDs aren't
 * useful for the (queue_id, dispatch_idx) join most consumers do, and
 * the per-record corr_id allocation was a meaningful overhead at sustained
 * rates.
 */
static inline void rocm_trace_emit_hsa_kernel_dispatch_record(
    uint32_t queue_id, uint32_t count,
    const uint8_t* records, size_t records_len,
    bool     force_emit /* bypasses the tracepoint-enabled check */) {
    if (rocm_trace_disabled()) return;
    if (force_emit ||
        lttng_ust_tracepoint_enabled(rocm_hsa, kernel_dispatch_record)) {
        lttng_ust_do_tracepoint(rocm_hsa, kernel_dispatch_record,
                                queue_id, count, records, records_len);
    }
}

/* kernel_dispatch_drop emit helper. The current sentinel-scan drainer
 * (core/runtime/dispatch_log.cpp::drain_one_queue) cannot detect ring
 * overruns because the substrate publishes no host-visible FW write
 * pointer to compare against the host read cursor. As a result no code
 * path emits this tracepoint today. The definition is retained so a
 * future overrun-detection mechanism (e.g. a sequence-number gap check
 * across slots, or a substrate extension that publishes wptr) can emit
 * it without re-introducing the tracepoint definition. Per-queue enable
 * failures use stderr WARNING per spec §5. force_emit not supported here:
 * drop events would always be observed via the tracepoint-enabled gate,
 * since they would only fire while the drainer is actively consuming a
 * ring (i.e. while the tracepoint is enabled). */
static inline void rocm_trace_emit_hsa_kernel_dispatch_drop(uint32_t queue_id,
                                                            uint64_t bytes_lost) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hsa, kernel_dispatch_drop)) {
        lttng_ust_do_tracepoint(rocm_hsa, kernel_dispatch_drop,
                                queue_id, bytes_lost);
    }
}

/* clock_sync emit helper. Emits a CPU/GPU clock pair for offline translation. */
static inline void rocm_trace_emit_hsa_clock_sync(uint64_t gpu_id,
                                                  uint64_t gpu_ts,
                                                  uint64_t system_ts) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hsa, clock_sync)) {
        lttng_ust_do_tracepoint(rocm_hsa, clock_sync,
                                gpu_id, gpu_ts, system_ts);
    }
}

#else /* !HSA_ENABLE_LTTNG_UST */

static inline void rocm_trace_emit_hsa_api_enter(const char* a, uint64_t c) {
    (void)a; (void)c;
}
static inline void rocm_trace_emit_hsa_api_exit_status(const char* a, uint64_t c, int32_t s) {
    (void)a; (void)c; (void)s;
}
static inline void rocm_trace_emit_hsa_api_exit_u64(const char* a, uint64_t c, uint64_t v) {
    (void)a; (void)c; (void)v;
}
static inline void rocm_trace_emit_hsa_api_exit_i64(const char* a, uint64_t c, int64_t v) {
    (void)a; (void)c; (void)v;
}
static inline void rocm_trace_emit_hsa_api_exit_ptr(const char* a, uint64_t c, const void* p) {
    (void)a; (void)c; (void)p;
}
static inline void rocm_trace_emit_hsa_api_exit_void(const char* a, uint64_t c) {
    (void)a; (void)c;
}
static inline void rocm_trace_emit_hsa_doorbell_ring(uint32_t a, int64_t b, uint8_t c) {
    (void)a; (void)b; (void)c;
}
static inline void rocm_trace_emit_hsa_intercept_packets(uint32_t a, uint64_t b, uint32_t c, uint8_t d) {
    (void)a; (void)b; (void)c; (void)d;
}
static inline void rocm_trace_emit_hsa_kernel_dispatch_record(
    uint32_t a, uint32_t b, const uint8_t* c, size_t d, bool e) {
    (void)a; (void)b; (void)c; (void)d; (void)e;
}
static inline void rocm_trace_emit_hsa_kernel_dispatch_drop(uint32_t a, uint64_t b) {
    (void)a; (void)b;
}
static inline void rocm_trace_emit_hsa_clock_sync(uint64_t a, uint64_t b, uint64_t c) {
    (void)a; (void)b; (void)c;
}

#endif /* HSA_ENABLE_LTTNG_UST */

#endif /* ROCM_HSA_TRACE_EMIT_H_ */
