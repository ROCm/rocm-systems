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

/* TLS hint used by the PM4 doorbell-ring path to tell StoreRelaxed that the
 * ringing packet is PM4 rather than AQL. Consumed-on-read (reset to 0). */
static __thread uint8_t g_rocm_packet_type_hint;
static inline void rocm_trace_set_packet_type_hint(uint8_t t) { g_rocm_packet_type_hint = t; }
static inline uint8_t rocm_trace_consume_packet_type_hint(void) {
    uint8_t t = g_rocm_packet_type_hint;
    g_rocm_packet_type_hint = 0;
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

#include "rocm_hsa_tp.h"

static inline void rocm_trace_emit_hsa_api_enter(const char* api_name,
                                                 uint64_t    corr_id) {
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_api_enter)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_api_enter,
                                api_name, corr_id, rocm_trace_current_tid());
    }
}

/* Status-returning APIs (hsa_status_t, int, etc.) */
static inline void rocm_trace_emit_hsa_api_exit_status(const char* api_name,
                                                       uint64_t    corr_id,
                                                       int32_t     status) {
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_api_exit_status)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_api_exit_status,
                                api_name, corr_id, status);
    }
}

/* Pointer-returning APIs */
static inline void rocm_trace_emit_hsa_api_exit_ptr(const char* api_name,
                                                    uint64_t    corr_id,
                                                    const void* retval) {
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_api_exit_ptr)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_api_exit_ptr,
                                api_name, corr_id, (uint64_t)retval);
    }
}

/* Void-returning APIs */
static inline void rocm_trace_emit_hsa_api_exit_void(const char* api_name,
                                                     uint64_t    corr_id) {
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_api_exit_void)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_api_exit_void, api_name, corr_id);
    }
}

static inline void rocm_trace_emit_hsa_doorbell_ring(uint32_t queue_id,
                                                     int64_t  write_idx,
                                                     uint8_t  packet_type,
                                                     uint64_t corr_id) {
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_doorbell_ring)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_doorbell_ring,
                                queue_id, write_idx, packet_type,
                                corr_id, rocm_trace_current_tid());
    }
}

static inline void rocm_trace_emit_hsa_intercept_packets(uint32_t queue_id,
                                                         uint64_t pkt_index,
                                                         uint32_t pkt_count,
                                                         uint8_t  packet_type) {
    if (lttng_ust_tracepoint_enabled(rocm_hsa, hsa_intercept_packets)) {
        lttng_ust_do_tracepoint(rocm_hsa, hsa_intercept_packets,
                                queue_id, pkt_index, pkt_count, packet_type);
    }
}

#else /* !HSA_ENABLE_LTTNG_UST */

static inline void rocm_trace_emit_hsa_api_enter(const char* a, uint64_t c) {
    (void)a; (void)c;
}
static inline void rocm_trace_emit_hsa_api_exit_status(const char* a, uint64_t c, int32_t s) {
    (void)a; (void)c; (void)s;
}
static inline void rocm_trace_emit_hsa_api_exit_ptr(const char* a, uint64_t c, const void* p) {
    (void)a; (void)c; (void)p;
}
static inline void rocm_trace_emit_hsa_api_exit_void(const char* a, uint64_t c) {
    (void)a; (void)c;
}
static inline void rocm_trace_emit_hsa_doorbell_ring(uint32_t a, int64_t b, uint8_t c, uint64_t d) {
    (void)a; (void)b; (void)c; (void)d;
}
static inline void rocm_trace_emit_hsa_intercept_packets(uint32_t a, uint64_t b, uint32_t c, uint8_t d) {
    (void)a; (void)b; (void)c; (void)d;
}

#endif

#endif /* ROCM_HSA_TRACE_EMIT_H_ */
