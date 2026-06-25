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
 *   [B] Always-on    — APIs that take ncclCoopAny by value.
 *   [C] Stubbed-out  — APIs whose underlying RCCL primitives (ncclGin*,
 *                      composite ncclBarrierSession, ncclGinFenceLevel)
 *                      do not yet exist in RCCL. Wrapped in `#if 0` and
 *                      ready to enable once those land via a future sync
 *                      with NCCL.
 */

#include "nccl_device.h"

/* ------------------------------------------------------------------------
 * NCCL_IR_EXPORT: full attribute set for an exported C-ABI thunk in the
 * bitcode artifact.
 *
 *   - `extern "C"`              : unmangled symbol name in the .bc.
 *   - `__device__`              : HIP device function.
 *   - `__attribute__((used))`   : prevent clang -O1 from DCE'ing the body
 *                                 when there is no caller in this TU.
 *   - NO `always_inline`/`inline` : the thunks are the public ABI; they
 *                                 must retain external linkage so opt's
 *                                 `internalize` pass keeps them on the
 *                                 public-api list. With `always_inline`
 *                                 (as `NCCL_DEVICE_INLINE` expands to in
 *                                 bitcode mode), clang's GlobalOpt
 *                                 downgrades the thunk to `internal`
 *                                 linkage before opt even runs.
 *
 * Callees of the thunk (e.g. ncclGetPeerPointer from core__funcs.h) still
 * carry `NCCL_DEVICE_INLINE` and therefore still inline into the thunk
 * body; this macro only controls the linkage of the thunk itself.
 * ----------------------------------------------------------------------*/
#ifndef NCCL_IR_EXPORT
  #ifdef __clang_llvm_bitcode_lib__
    /* Building the bitcode artifact: applied to the definitions in
     * nccl_device_wrapper__impl.h. Three attributes are needed:
     *   `used`                -> @llvm.compiler.used : block compiler-driven DCE.
     *   `retain`              -> @llvm.used          : block clang -O1 GlobalOpt
     *                                                  from downgrading linkage
     *                                                  to `internal`. Without it
     *                                                  the thunk arrives at opt
     *                                                  already internal and the
     *                                                  public-api-list cannot
     *                                                  promote it back.
     *   `visibility("default")` -> visible to AMDGPU lld at object-level link.
     *                              HIP-Clang defaults `__device__` symbols to
     *                              `hidden` visibility, which is fine for
     *                              `llvm-link`-style consumers but causes
     *                              `undefined hidden symbol` errors when
     *                              downstream apps let the driver handle
     *                              the bitcode link (the more common case).
     */
    #define NCCL_IR_EXPORT extern "C" __device__ \
        __attribute__((used, retain, visibility("default")))
  #else
    /* Downstream consumer use of this header: applied to declarations
     * only. Must be `extern "C" __device__` so a __global__ kernel can
     * call the thunk directly; the body is supplied at link time by
     * linking librccl_device.bc into the device-side image. */
    #define NCCL_IR_EXPORT extern "C" __device__
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
NCCL_IR_EXPORT void* ncclGetPeerPointerTeam(
    ncclWindow_t w, size_t offset, ncclTeam tm, int peer);


/* ========================================================================
 * [B] APIs that depend on ncclCoopAny
 *
 * Underlying templated RCCL types:
 *   - ncclLsaBarrierSession<Coop>  (src/include/nccl_device/mem_barrier.h)
 *   - ncclLsaBarrierHandle, ncclMultimemHandle, ncclDevComm, ncclTeam
 *
 * Memory-order parameter: cuda::memory_order (HIP aliases provided by
 * hip_compat.h in nccl_device/utility.h).
 * ======================================================================*/

/* Struct definitions */
struct ncclLsaBarrierSession_C {
  ncclLsaBarrierSession<ncclCoopAny> bar;
};

/* Coop initialization and utility */
NCCL_IR_EXPORT void ncclCoopAnyInitThread(ncclCoopAny* coop);
NCCL_IR_EXPORT void ncclCoopAnyInitWarp(ncclCoopAny* coop);
NCCL_IR_EXPORT void ncclCoopAnyInitLanes(ncclCoopAny* coop, ncclCoopMask_t lane_mask);
NCCL_IR_EXPORT void ncclCoopAnyInitWarpSpan(ncclCoopAny* coop, int warp0, int nWarps, int id);
NCCL_IR_EXPORT void ncclCoopAnyInitCta(ncclCoopAny* coop);

NCCL_IR_EXPORT int  ncclCoopThreadRank(const ncclCoopAny* coop);
NCCL_IR_EXPORT int  ncclCoopSize(const ncclCoopAny* coop);
NCCL_IR_EXPORT int  ncclCoopNumThreads(const ncclCoopAny* coop);
NCCL_IR_EXPORT void ncclCoopSync(const ncclCoopAny* coop);

/* LSA Barrier Session APIs */
NCCL_IR_EXPORT void ncclLsaBarrierSessionInit(
    ncclLsaBarrierSession_C* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclLsaBarrierHandle handle,
    uint32_t index,
    bool multimem = false,
    ncclMultimemHandle mmHandle = {});

NCCL_IR_EXPORT
void ncclLsaBarrierSessionArrive(ncclLsaBarrierSession_C* session,
                                 ncclCoopAny coop,
                                 cuda::memory_order order);
NCCL_IR_EXPORT
void ncclLsaBarrierSessionWait(ncclLsaBarrierSession_C* session,
                               ncclCoopAny coop,
                               cuda::memory_order order);
NCCL_IR_EXPORT
void ncclLsaBarrierSessionSync(ncclLsaBarrierSession_C* session,
                               ncclCoopAny coop,
                               cuda::memory_order order);


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
 * to remove the `#if 0` guard (these all need ncclCoopAny, which is
 * unconditionally available).
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
NCCL_IR_EXPORT void ncclGinBarrierSessionInit(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    ncclGin_C net,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index);

NCCL_IR_EXPORT void ncclGinBarrierSessionSync(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence);

/* Composite (LSA + GIN) Barrier Session APIs */
NCCL_IR_EXPORT void ncclBarrierSessionInit(
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

NCCL_IR_EXPORT void ncclBarrierSessionSync(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence);

#endif  /* 0 — GIN / composite Barrier wrappers */

#endif  /* _NCCL_DEVICE_WRAPPER_H_ */
