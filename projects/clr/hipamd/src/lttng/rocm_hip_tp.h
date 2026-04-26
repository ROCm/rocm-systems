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
 */
#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER rocm_hip

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "rocm_hip_tp.h"

#if !defined(_ROCM_HIP_TP_H) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define _ROCM_HIP_TP_H

#include <lttng/tracepoint.h>
#include <stdint.h>

/* hip_api_enter: corr_id is freshly minted for THIS HIP call.
 * parent_corr_id is the active corr_id slot value BEFORE this call's push --
 * i.e., the calling context's corr_id (an outer HIP API for nested calls,
 * or 0 for top-level user calls). */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip,
    hip_api_enter,
    LTTNG_UST_TP_ARGS(const char*, api_name, uint64_t, corr_id, uint32_t, tid,
                      uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer(uint32_t, tid, tid)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

/* hip_api_exit_status: for hipError_t / int / other 32-bit-status returns.
 * parent_corr_id mirrors the matching enter's parent_corr_id (i.e., the
 * pre-push slot value, recovered by popping before the read). */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip,
    hip_api_exit_status,
    LTTNG_UST_TP_ARGS(const char*, api_name, uint64_t, corr_id, int32_t, status,
                      uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer(int32_t, status, status)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

/* hip_api_exit_ptr: for pointer-returning APIs (hipApiName,
 * __hipRegisterFatBinary, etc.). The pointer is captured as a uint64_t
 * hex field (0 if NULL). parent_corr_id semantics same as exit_status. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip,
    hip_api_exit_ptr,
    LTTNG_UST_TP_ARGS(const char*, api_name, uint64_t, corr_id, uint64_t, retval_ptr,
                      uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer_hex(uint64_t, retval_ptr, retval_ptr)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

/* hip_api_exit_void: for void-returning APIs. parent_corr_id semantics
 * same as exit_status. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip,
    hip_api_exit_void,
    LTTNG_UST_TP_ARGS(const char*, api_name, uint64_t, corr_id,
                      uint64_t, parent_corr_id),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(api_name, api_name)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

/* hip_kernel_dispatch_enqueue: emitted just before command->enqueue() in
 * ihipModuleLaunchKernel. The 6 launch dims (grid_xyz, block_xyz) are packed
 * into two uint64_t (each 16 bits per dim, hipDim3 max is 2^31 but real
 * launches never exceed 2^16-1 per dim) to stay under LTTng-UST's 10-field
 * (20-arg) tracepoint limit. tid is available via LTTng's vtid context.
 *
 * corr_id is freshly minted for THIS dispatch enqueue event (so the
 * dispatch step has its own identity, distinct from the launching HIP API).
 * parent_corr_id is the active TLS slot at emit time -- typically the
 * surrounding HIP API launch's corr_id when called from inside an HIP API
 * body; 0 if launched outside any HIP API context. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip,
    hip_kernel_dispatch_enqueue,
    LTTNG_UST_TP_ARGS(
        const char*, kernel_name,
        uint64_t,    corr_id,
        uint32_t,    tid,
        void*,       stream,
        uint64_t,    grid_xyz_packed,
        uint64_t,    block_xyz_packed,
        uint32_t,    shared_mem_bytes,
        uint64_t,    parent_corr_id
    ),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(kernel_name, kernel_name)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer(uint32_t, tid, tid)
        lttng_ust_field_integer_hex(uint64_t, stream, (uint64_t)stream)
        lttng_ust_field_integer_hex(uint64_t, grid_xyz_packed, grid_xyz_packed)
        lttng_ust_field_integer_hex(uint64_t, block_xyz_packed, block_xyz_packed)
        lttng_ust_field_integer(uint32_t, shared_mem_bytes, shared_mem_bytes)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
    )
)

/* hip_aql_kernel_dispatch_submit: emitted at the HIP CLR packet-write site
 * (rocvirtual.cpp:dispatchGenericAqlPacket and dispatchAqlPacketBatchFlat),
 * once per AQL kernel-dispatch packet written into a queue ring. Captures
 * HIP's intent BEFORE any HSA intercept-queue rewrite, with corr_id from
 * TLS naturally available. This is the join key for the firmware-ring track.
 *
 * parent_corr_id (uint64) carries the active TLS slot value at emit time;
 * for dispatches submitted from inside a HIP API call body this is the
 * surrounding HIP API's corr_id. */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip,
    hip_aql_kernel_dispatch_submit,
    LTTNG_UST_TP_ARGS(
        uint32_t,    queue_id,
        uint64_t,    write_idx,
        uint64_t,    dispatch_idx,
        uint64_t,    corr_id,
        uint64_t,    parent_corr_id,
        uint32_t,    tid,
        uint64_t,    kernel_object,
        uint64_t,    completion_signal
    ),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(uint32_t, queue_id, queue_id)
        lttng_ust_field_integer(uint64_t, write_idx, write_idx)
        lttng_ust_field_integer(uint64_t, dispatch_idx, dispatch_idx)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
        lttng_ust_field_integer(uint32_t, tid, tid)
        lttng_ust_field_integer_hex(uint64_t, kernel_object, kernel_object)
        lttng_ust_field_integer_hex(uint64_t, completion_signal, completion_signal)
    )
)

#endif /* _ROCM_HIP_TP_H */

#include <lttng/tracepoint-event.h>
