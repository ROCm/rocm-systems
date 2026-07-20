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
static thread_local uint8_t g_rocm_packet_type_hint;
static thread_local uint8_t g_rocm_packet_type_hint_valid;
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
 * every emit helper short-circuits before touching LTTng state. */
extern std::atomic<bool> rocm_hsa_trace_g_disabled;

#ifndef ROCM_TRACE_DISABLED_DEFINED
#define ROCM_TRACE_DISABLED_DEFINED
static inline bool rocm_trace_disabled(void) {
    return rocm_hsa_trace_g_disabled.load(std::memory_order_relaxed);
}
#endif

/* Doorbell ring: identity (vpid, vtid, timestamp) comes from the channel
 * context; a surrounding HSA wrapper is identified by its shared
 * hsa_api_enter event on the same thread. */
static inline void rocm_trace_emit_hsa_doorbell_ring(uint32_t queue_id,
                                                     int64_t  write_idx,
                                                     uint8_t  packet_type) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_doorbell_ring)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_doorbell_ring,
                                queue_id, write_idx, packet_type);
    }
}

static inline void rocm_trace_emit_hsa_intercept_packets(uint32_t queue_id,
                                                         uint64_t pkt_index,
                                                         uint32_t pkt_count,
                                                         uint8_t  packet_type) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_intercept_packets)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_intercept_packets,
                                queue_id, pkt_index, pkt_count, packet_type);
    }
}

#else /* !HSA_ENABLE_LTTNG_UST */

static inline void rocm_trace_emit_hsa_doorbell_ring(uint32_t a, int64_t b, uint8_t c) {
    (void)a; (void)b; (void)c;
}
static inline void rocm_trace_emit_hsa_intercept_packets(uint32_t a, uint64_t b, uint32_t c, uint8_t d) {
    (void)a; (void)b; (void)c; (void)d;
}

#endif

/* Curated per-API typed emit helpers (generated header). Self-contained:
 * makes its own HSA_ENABLE_LTTNG_UST decision and provides no-op stubs
 * when LTTng is disabled (e.g. on Windows), so this include lives
 * OUTSIDE the LTTng guard. */
#include "rocm_trace_emit_curated.h"

#endif /* ROCM_HSA_TRACE_EMIT_H_ */
