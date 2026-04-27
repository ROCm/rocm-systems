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

/* Schema version. Bump on any breaking change to event field layout
 * (field add/remove/rename/type-change) so consumers can detect they
 * are reading a stream produced by a different schema generation.
 *
 * Version history:
 *   1 - initial schema (development branch only; never released)
 *   2 - hip_aql_kernel_dispatch_submit dropped `dispatch_idx` field
 *       (was a per-VirtualGPU counter; non-unique across VirtualGPUs
 *       sharing a queue via queue pooling). Join key for the
 *       firmware-ring track is now (queue_id, write_idx).
 *
 * NOTE: This LTTng instrumentation has NEVER shipped in any released
 * ROCm version; it lives entirely on development branch
 * `users/bewelton/lttng`. Schema-version bumps here are advisory only —
 * there are zero downstream CTF consumers in the wild that need to
 * tolerate the change. Once this work merges to mainline ROCm, this
 * version becomes the contract baseline and future bumps require
 * proper deprecation handling. */
#define ROCM_HIP_TP_SCHEMA_VERSION 2

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
 * Join key for the firmware-ring track is (queue_id, write_idx). The HSA
 * AQL queue's hardware ring slot index is unique per-queue and matches
 * what firmware records as its dispatch index. No separate software
 * dispatch counter is needed (and a per-VirtualGPU counter would be wrong
 * across VirtualGPUs that share a single hsa_queue_t via queue pooling).
 *
 * parent_corr_id (uint64) carries the active TLS slot value at emit time;
 * for dispatches submitted from inside a HIP API call body this is the
 * surrounding HIP API's corr_id.
 *
 * ============================================================================
 * SCHEMA BREAKING CHANGE (development-branch only) — bumped to v2:
 *
 *   v1 of this event included a `dispatch_idx` field (uint64) — a software
 *   counter incremented per-VirtualGPU at packet write time. That field
 *   was REMOVED in v2 because:
 *
 *     - VirtualGPU instances frequently share a single hsa_queue_t via
 *       HIP's queue-pool reuse path, so per-VirtualGPU counters DUPLICATE
 *       across VirtualGPUs that submit to the same hardware queue. The
 *       counter therefore could not serve as a unique key on the firmware
 *       ring (multiple events with the same dispatch_idx but on the same
 *       (queue_id, write_idx) slot).
 *
 *     - (queue_id, write_idx) IS a unique key per ring submission, and
 *       firmware records the same write_idx as its own dispatch index, so
 *       the join key on the firmware-ring track is naturally available
 *       without the separate software counter.
 *
 *   This is a backward-incompatible CTF schema change. The change is
 *   acceptable here because this LTTng instrumentation has NEVER shipped
 *   in any released ROCm version — it lives entirely on development
 *   branch `users/bewelton/lttng`. There are zero downstream CTF consumers
 *   in the wild. Consumers building against this branch must use the
 *   current (v2) schema; see ROCM_HIP_TP_SCHEMA_VERSION above. Once this
 *   work merges to mainline, future schema changes will require proper
 *   deprecation handling.
 * ============================================================================
 */
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip,
    hip_aql_kernel_dispatch_submit,
    LTTNG_UST_TP_ARGS(
        uint32_t,    queue_id,
        uint64_t,    write_idx,
        uint64_t,    corr_id,
        uint64_t,    parent_corr_id,
        uint32_t,    tid,
        uint64_t,    kernel_object,
        uint64_t,    completion_signal
    ),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(uint32_t, queue_id, queue_id)
        lttng_ust_field_integer(uint64_t, write_idx, write_idx)
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer(uint64_t, parent_corr_id, parent_corr_id)
        lttng_ust_field_integer(uint32_t, tid, tid)
        lttng_ust_field_integer_hex(uint64_t, kernel_object, kernel_object)
        lttng_ust_field_integer_hex(uint64_t, completion_signal, completion_signal)
    )
)

/* Curated per-API typed tracepoint events. Generated by
 * lttng_curated_codegen.py from curated_apis.yaml. See spec §5.1. */
#include "rocm_hip_curated_tp.h"

#endif /* _ROCM_HIP_TP_H */

#include <lttng/tracepoint-event.h>
