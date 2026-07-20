#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER rocm_hsa

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "rocm_hsa_tp.h"

#if !defined(_ROCM_HSA_TP_H) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define _ROCM_HSA_TP_H

#include <lttng/tracepoint.h>
#include <stdint.h>

/* Tracepoint schema version. Bump on any breaking change to event field
 * layout (field add/remove/rename/type-change) so consumers can detect a
 * different schema generation.
 *
 *   1 - initial release: one combined event per curated API
 *       (rocm_hsa:<api>), emitted at entry and exit with a `phase` field
 *       (0=enter, 1=exit); enabling one event name captures both. Typed
 *       per-API arguments + return value; the event name is the API
 *       identity (no api_name string field). Call identity / parent
 *       attribution derive from the (vpid, vtid, timestamp) channel
 *       contexts (add-context --type vpid --type vtid).
 */
#define ROCM_HSA_TP_SCHEMA_VERSION 1

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
