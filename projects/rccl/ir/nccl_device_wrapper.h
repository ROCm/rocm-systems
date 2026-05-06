/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Adapted from NVIDIA NCCL ir/nccl_device_wrapper.h (v2.29.2-1).
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_WRAPPER_H_
#define _NCCL_DEVICE_WRAPPER_H_

/*
 * RCCL Device API C-style wrapper functions.
 *
 * Public surface for the LLVM IR / bitcode artifact (librccl_device.bc).
 * Each entry point is an extern "C" __device__ thunk that downstream code
 * generators (Triton, MLIR, custom JITs, etc.) can call into without
 * recompiling RCCL from source. The thunks themselves are defined in the
 * companion translation unit nccl_device_wrapper__impl.h, which clang
 * compiles to bitcode under -D__clang_llvm_bitcode_lib__.
 *
 * Sectioning of this header (mirrored in __impl.h):
 *   [A] Always-on    — APIs whose RCCL prerequisites already exist.
 *   [B] Coop-gated   — APIs that take ncclCoopAny by value; require the
 *                      ncclCoopAny type to be enabled in coop.h, controlled
 *                      by -DRCCL_ENABLE_NCCL_COOP_ANY=1.
 *   [C] Stubbed-out  — APIs whose underlying RCCL primitives (ncclGin*,
 *                      composite ncclBarrierSession, ncclGinFenceLevel)
 *                      do not yet exist in RCCL. Wrapped in `#if 0` and
 *                      ready to enable once those land via a future sync
 *                      with NCCL.
 */

#include "nccl_device.h"

/* ------------------------------------------------------------------------
 * NCCL_IR_EXTERN_C: extern "C" only when this header is included while
 * building the bitcode artifact (clang -D__clang_llvm_bitcode_lib__);
 * otherwise empty. Provided defensively here in case nccl_device/utility.h
 * has not yet been updated with the same definition (that update is part
 * of the broader IR enablement work; see the porting plan).
 * ----------------------------------------------------------------------*/
#ifndef NCCL_IR_EXTERN_C
  #ifdef __clang_llvm_bitcode_lib__
    #define NCCL_IR_EXTERN_C extern "C"
  #else
    #define NCCL_IR_EXTERN_C
  #endif
#endif

/* ========================================================================
 * [A] Always-on APIs
 *
 * Prerequisites already in RCCL today:
 *   - ncclGetPeerPointer(ncclWindow_t, size_t, ncclTeam, int)
 *     (src/include/nccl_device/impl/core__funcs.h)
 * ======================================================================*/

/* Peer pointer API */
NCCL_IR_EXTERN_C __device__ void* ncclGetPeerPointerTeam(
    ncclWindow_t w, size_t offset, ncclTeam tm, int peer);


/* ========================================================================
 * [B] APIs that depend on ncclCoopAny
 *
 * Underlying templated RCCL types are present:
 *   - ncclLsaBarrierSession<Coop>  (src/include/nccl_device/mem_barrier.h)
 *   - ncclLsaBarrierHandle, ncclMultimemHandle, ncclDevComm, ncclTeam
 *
 * The only missing piece is ncclCoopAny itself, which lives in
 * src/include/nccl_device/coop.h gated on RCCL_ENABLE_NCCL_COOP_ANY.
 * Enable both with: -DRCCL_ENABLE_NCCL_COOP_ANY=1
 *
 * Memory-order parameter: NCCL upstream uses cuda::memory_order, but RCCL's
 * mem_barrier__funcs.h uses std::memory_order on the HIP/AMD path (see the
 * __HIP_PLATFORM_AMD__ branch in nccl_device/utility.h). The wrapper
 * matches RCCL's HIP convention.
 * ======================================================================*/
#if RCCL_ENABLE_NCCL_COOP_ANY

/* Struct definitions */
struct ncclLsaBarrierSession_C {
  ncclLsaBarrierSession<ncclCoopAny> bar;
};

/* Coop initialization and utility */
NCCL_IR_EXTERN_C __device__ void ncclCoopAnyInitThread(ncclCoopAny* coop);
NCCL_IR_EXTERN_C __device__ void ncclCoopAnyInitWarp(ncclCoopAny* coop);
NCCL_IR_EXTERN_C __device__ void ncclCoopAnyInitLanes(ncclCoopAny* coop, uint32_t lane_mask);
NCCL_IR_EXTERN_C __device__ void ncclCoopAnyInitWarpSpan(ncclCoopAny* coop, int warp0, int nWarps, int id);
NCCL_IR_EXTERN_C __device__ void ncclCoopAnyInitCta(ncclCoopAny* coop);

NCCL_IR_EXTERN_C __device__ int  ncclCoopThreadRank(const ncclCoopAny* coop);
NCCL_IR_EXTERN_C __device__ int  ncclCoopSize(const ncclCoopAny* coop);
NCCL_IR_EXTERN_C __device__ int  ncclCoopNumThreads(const ncclCoopAny* coop);
NCCL_IR_EXTERN_C __device__ void ncclCoopSync(const ncclCoopAny* coop);

/* LSA Barrier Session APIs */
NCCL_IR_EXTERN_C __device__ void ncclLsaBarrierSessionInit(
    ncclLsaBarrierSession_C* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclLsaBarrierHandle handle,
    uint32_t index,
    bool multimem = false,
    ncclMultimemHandle mmHandle = {});

NCCL_IR_EXTERN_C __device__
void ncclLsaBarrierSessionArrive(ncclLsaBarrierSession_C* session,
                                 ncclCoopAny coop,
                                 std::memory_order order);
NCCL_IR_EXTERN_C __device__
void ncclLsaBarrierSessionWait(ncclLsaBarrierSession_C* session,
                               ncclCoopAny coop,
                               std::memory_order order);
NCCL_IR_EXTERN_C __device__
void ncclLsaBarrierSessionSync(ncclLsaBarrierSession_C* session,
                               ncclCoopAny coop,
                               std::memory_order order);

#endif  /* RCCL_ENABLE_NCCL_COOP_ANY */


/* ========================================================================
 * [C] APIs whose RCCL prerequisites do not exist yet
 *
 * Required but missing in RCCL:
 *   - ncclGin_C, ncclGinBarrierSession<Coop>, ncclGinBarrierHandle,
 *     ncclGinFenceLevel  (no GPU-Initiated Networking in RCCL today)
 *   - ncclBarrierSession<Coop>  (composite inner-LSA + outer-GIN barrier)
 *
 * Kept here verbatim from NCCL v2.29.2-1 so that a future sync that
 * imports the GIN / composite-barrier infrastructure into RCCL only has
 * to flip the `#if 0` to `#if RCCL_ENABLE_NCCL_COOP_ANY` (these all need
 * ncclCoopAny too).
 * ======================================================================*/
#if 0  /* TODO(rccl-ir): enable once RCCL grows ncclGin* / composite Barrier APIs */

/* Struct definitions */
struct ncclGinBarrierSession_C {
  ncclGinBarrierSession<ncclCoopAny> bar;
};

struct ncclBarrierSession_C {
  ncclBarrierSession<ncclCoopAny> bar;
};

/* GIN Barrier Session APIs */
NCCL_IR_EXTERN_C __device__ void ncclGinBarrierSessionInit(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    ncclGin_C net,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index);

NCCL_IR_EXTERN_C __device__ void ncclGinBarrierSessionSync(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    std::memory_order order,
    ncclGinFenceLevel fence);

/* Composite (LSA + GIN) Barrier Session APIs */
NCCL_IR_EXTERN_C __device__ void ncclBarrierSessionInit(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    ncclTeam innerTeam,
    ncclTeam outerTeam,
    ncclGin_C net,
    ncclLsaBarrierHandle const innerBarHandle,
    ncclGinBarrierHandle const outerBarHandle,
    uint32_t index,
    bool multimem = false,
    ncclMultimemHandle const innerMmHandle = {});

NCCL_IR_EXTERN_C __device__ void ncclBarrierSessionSync(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    std::memory_order order,
    ncclGinFenceLevel fence);

#endif  /* 0 — GIN / composite Barrier wrappers */

#endif  /* _NCCL_DEVICE_WRAPPER_H_ */
