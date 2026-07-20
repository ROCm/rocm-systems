/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * LTTng-UST tracepoint provider for the HIP CLR runtime.
 *
 * This header is included two ways:
 *   1. From rocm_hip_tp.cpp (the tracepoint-provider package), with
 *      LTTNG_UST_TRACEPOINT_CREATE_PROBES + LTTNG_UST_TRACEPOINT_DEFINE
 *      defined; this generates the per-event registration symbols.
 *   2. From rocm_trace_emit.h (every callsite), without the CREATE_PROBES /
 *      DEFINE macros; this expands the tracepoint() macros at the call sites.
 *
 * Per-event identity (process id, thread id, timestamp) is provided by
 * LTTng-UST's native channel contexts -- consumers configure their channel
 * with `lttng add-context --userspace --type vpid --type vtid` (and the CTF
 * event-header timestamp is always present). The schema therefore does NOT
 * carry an explicit corr_id / parent_corr_id / tid field per event. Each
 * curated API emits a single combined event (rocm_hip:<api>) at entry and
 * exit, discriminated by a `phase` field, carrying its typed argument and
 * return values directly.
 */
#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER rocm_hip

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "rocm_hip_tp.h"

#if !defined(_ROCM_HIP_TP_H) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define _ROCM_HIP_TP_H

#include <lttng/tracepoint.h>
#include <stdint.h>

/* Tracepoint schema version. Bump on any breaking change to event field
 * layout (field add/remove/rename/type-change) so consumers can detect a
 * different schema generation.
 *
 *   1 - initial release: one combined event per curated API
 *       (rocm_hip:<api>), emitted at entry and exit with a `phase` field
 *       (0=enter, 1=exit); enabling one event name captures both. Typed
 *       per-API arguments + return value; the event name is the API
 *       identity (no api_name string field). Call identity / parent
 *       attribution derive from the (vpid, vtid, timestamp) channel
 *       contexts (add-context --type vpid --type vtid).
 */
#define ROCM_HIP_TP_SCHEMA_VERSION 1

/* hip_kernel_dispatch_enqueue: emitted just before command->enqueue() in
 * ihipModuleLaunchKernel. The 6 launch dims (grid_xyz, block_xyz) are packed
 * into two uint64_t (each 16 bits per dim, hipDim3 max is 2^31 but real
 * launches never exceed 2^16-1 per dim) to stay under LTTng-UST's 10-field
 * (20-arg) tracepoint limit. The thread that enqueued the launch is
 * identified by the channel-context vtid. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip,
    hip_kernel_dispatch_enqueue,
    LTTNG_UST_TP_ARGS(
        const char*, kernel_name,
        void*,       stream,
        uint64_t,    grid_xyz_packed,
        uint64_t,    block_xyz_packed,
        uint32_t,    shared_mem_bytes
    ),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(kernel_name, kernel_name)
        lttng_ust_field_integer_hex(uint64_t, stream, (uint64_t)stream)
        lttng_ust_field_integer_hex(uint64_t, grid_xyz_packed, grid_xyz_packed)
        lttng_ust_field_integer_hex(uint64_t, block_xyz_packed, block_xyz_packed)
        lttng_ust_field_integer(uint32_t, shared_mem_bytes, shared_mem_bytes)
    )
)

/* hip_aql_kernel_dispatch_submit: emitted at the HIP CLR packet-write site
 * (rocvirtual.cpp:dispatchGenericAqlPacket and dispatchAqlPacketBatchFlat),
 * once per AQL kernel-dispatch packet written into a queue ring. Captures
 * HIP's intent BEFORE any HSA intercept-queue rewrite.
 *
 * Join key for the firmware-ring track is (queue_id, write_idx); the HSA
 * AQL queue's hardware ring slot index is unique per-queue and matches
 * what firmware records as its dispatch index. A per-VirtualGPU software
 * counter would be wrong here because VirtualGPUs frequently share a
 * single hsa_queue_t via HIP's queue-pool reuse path.
 */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip,
    hip_aql_kernel_dispatch_submit,
    LTTNG_UST_TP_ARGS(
        uint32_t,    queue_id,
        uint64_t,    write_idx,
        uint64_t,    kernel_object,
        uint64_t,    completion_signal
    ),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(uint32_t, queue_id, queue_id)
        lttng_ust_field_integer(uint64_t, write_idx, write_idx)
        lttng_ust_field_integer_hex(uint64_t, kernel_object, kernel_object)
        lttng_ust_field_integer_hex(uint64_t, completion_signal, completion_signal)
    )
)

/* Curated per-API typed tracepoint events (generated header). */
#include "rocm_hip_curated_tp.h"

#endif /* _ROCM_HIP_TP_H */

#include <lttng/tracepoint-event.h>
