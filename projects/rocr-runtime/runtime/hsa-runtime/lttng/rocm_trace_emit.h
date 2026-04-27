#ifndef ROCM_HSA_TRACE_EMIT_H_
#define ROCM_HSA_TRACE_EMIT_H_

#include <stdint.h>
#include <stddef.h>
#include "rocm_trace_tid.h"

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

#endif

#endif /* ROCM_HSA_TRACE_EMIT_H_ */
