/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Adapted from NVIDIA NCCL ir/nccl_device_wrapper__impl.h (v2.29.2-1).
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_WRAPPER__IMPL_H_
#define _NCCL_DEVICE_WRAPPER__IMPL_H_

/*
 * RCCL Device API force-instantiation translation unit for the LLVM IR /
 * bitcode build (librccl_device.bc).
 *
 * This file is the single source clang feeds to -emit-llvm under the
 * driver:
 *
 *     clang++ -x hip --offload-device-only --offload-arch=<gfx*>          \
 *             -D__clang_llvm_bitcode_lib__ -D__HIP_PLATFORM_AMD__=1       \
 *             -emit-llvm -O1 -c nccl_device_wrapper__impl.h               \
 *             -o librccl_device.bc.unoptimized
 *
 * The body of every extern "C" thunk declared in nccl_device_wrapper.h is
 * defined here. By referring to ncclLsaBarrierSession<ncclCoopAny> (and
 * its GIN/Barrier siblings, in the disabled bucket) the compiler is forced
 * to instantiate those templates so the bitcode actually contains the
 * device-side code.
 *
 * Bucket layout mirrors nccl_device_wrapper.h exactly:
 *   [A] Always-on    — no preprocessor gate.
 *   [B] Always-on    — ncclCoopAny + LSA barrier session thunks.
 *   [C] Always-on    — GIN + composite ncclBarrierSession thunks, enabled
 *                      now that the NCCL v2.29.x GIN sync landed them.
 */

#include "nccl_device_wrapper.h"
#include <new>          /* placement new */

/* The thunks below use NCCL_IR_EXPORT (extern "C" __device__ __attribute__((used)))
 * rather than NCCL_DEVICE_INLINE. NCCL_DEVICE_INLINE expands to
 * `__device__ __forceinline__` (or `__device__ __attribute__((always_inline))`
 * if we override it in bitcode mode), and clang's GlobalOpt at -O1 downgrades
 * any always_inline __device__ function to `internal` linkage -- which would
 * defeat the public ABI of the bitcode artifact. The callees of each thunk
 * (e.g. ncclGetPeerPointer) still carry NCCL_DEVICE_INLINE inside the
 * nccl_device headers and therefore still inline into the thunk body.
 */

#if NCCL_CHECK_CUDACC   /* Defined in hip_compat.h (via nccl_device/utility.h); true under
                           HIP-Clang -x hip (__HIPCC__) and NVCC (__CUDACC__). HIP-Clang does
                           NOT define __CUDACC__, so a bare `#if __CUDACC__` here would
                           silently drop the entire device body from the bitcode. */

/* ========================================================================
 * [A] Always-on bodies
 * ======================================================================*/

NCCL_IR_EXPORT
void* ncclGetPeerPointerTeam(ncclWindow_t w, size_t offset, ncclTeam tm, int peer) {
  return ncclGetPeerPointer(w, offset, tm, peer);
}


/* ========================================================================
 * [B] ncclCoopAny + LSA Barrier Session bodies
 * ======================================================================*/

/* ---- coop init thunks (placement-new the static coop into the storage) ---- */
NCCL_IR_EXPORT void ncclCoopAnyInitThread(ncclCoopAny* coop) {
  ::new (coop) ncclCoopAny(ncclCoopThread{});
}
NCCL_IR_EXPORT void ncclCoopAnyInitWarp(ncclCoopAny* coop) {
  ::new (coop) ncclCoopAny(ncclCoopWarp{});
}
NCCL_IR_EXPORT void ncclCoopAnyInitLanes(ncclCoopAny* coop, ncclCoopMask_t lane_mask) {
  ::new (coop) ncclCoopAny(ncclCoopLanes(lane_mask));
}
NCCL_IR_EXPORT void ncclCoopAnyInitWarpSpan(ncclCoopAny* coop, int warp0, int nWarps, int id) {
  ::new (coop) ncclCoopAny(ncclCoopWarpSpan(warp0, nWarps, id));
}
NCCL_IR_EXPORT void ncclCoopAnyInitCta(ncclCoopAny* coop) {
  ::new (coop) ncclCoopAny(ncclCoopCta{});
}

/* ---- coop accessors ---- */
NCCL_IR_EXPORT int ncclCoopThreadRank(const ncclCoopAny* coop) {
  return coop->thread_rank();
}
NCCL_IR_EXPORT int ncclCoopSize(const ncclCoopAny* coop) {
  return coop->size();
}
NCCL_IR_EXPORT int ncclCoopNumThreads(const ncclCoopAny* coop) {
  return coop->num_threads();
}
NCCL_IR_EXPORT void ncclCoopSync(const ncclCoopAny* coop) {
  /* sync() is non-const on ncclCoopAny; const_cast matches NCCL upstream. */
  const_cast<ncclCoopAny*>(coop)->sync();
}

/* ---- LSA barrier session thunks ---- */
/* The placement-new of ncclLsaBarrierSession<ncclCoopAny> below is what
 * forces template instantiation of every method on the session class --
 * this is the whole point of the bitcode build.
 *
 * Memory-order parameter type matches lsa_barrier.h: cuda::memory_order.
 * On HIP, cuda::memory_order is provided by hip_compat.h. */

NCCL_IR_EXPORT void ncclLsaBarrierSessionInit(
    ncclLsaBarrierSession_C* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclLsaBarrierHandle handle,
    uint32_t index,
    bool multimem,
    ncclMultimemHandle mmHandle) {
  ::new (&(session->bar)) ncclLsaBarrierSession<ncclCoopAny>(
      coop, comm, team, handle, index, multimem, mmHandle);
}

NCCL_IR_EXPORT
void ncclLsaBarrierSessionArrive(ncclLsaBarrierSession_C* session,
                                 ncclCoopAny coop,
                                 cuda::memory_order order) {
  session->bar.arrive(coop, order);
}

NCCL_IR_EXPORT
void ncclLsaBarrierSessionWait(ncclLsaBarrierSession_C* session,
                               ncclCoopAny coop,
                               cuda::memory_order order) {
  session->bar.wait(coop, order);
}

NCCL_IR_EXPORT
void ncclLsaBarrierSessionSync(ncclLsaBarrierSession_C* session,
                               ncclCoopAny coop,
                               cuda::memory_order order) {
  session->bar.sync(coop, order);
}


/* ========================================================================
 * [C] GIN + composite Barrier Session bodies
 *
 * Enabled now that the NCCL v2.29.x GIN sync landed the prerequisites in
 * RCCL:
 *   - ncclGin / ncclGin_C                         (nccl_device/gin.h)
 *   - ncclGinBarrierSession<Coop>, ncclGinBarrierHandle, ncclGinFenceLevel
 *                                                 (nccl_device/gin_barrier.h)
 *   - composite ncclBarrierSession<Coop>          (nccl_device/barrier.h)
 *
 * These templates are gated on the upstream __CUDACC__ in the shared
 * headers; hipify rewrites it to __HIPCC__ in the staged copy fed to this
 * -x hip bitcode build, so they instantiate without any shared-header
 * change. ncclCoopAny is unconditionally available.
 * ======================================================================*/

/* GIN barrier session */
NCCL_IR_EXPORT void ncclGinBarrierSessionInit(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    ncclGin_C net,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index) {
  ::new (&(session->bar)) ncclGinBarrierSession<ncclCoopAny>(
      coop, reinterpret_cast<ncclGin const&>(net), team, handle, index);
}

NCCL_IR_EXPORT void ncclGinBarrierSessionSync(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence) {
  session->bar.sync(coop, order, fence);
}

/* Composite (LSA + GIN) barrier session */
NCCL_IR_EXPORT void ncclBarrierSessionInit(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    ncclTeam innerTeam,
    ncclTeam outerTeam,
    ncclGin_C net,
    ncclLsaBarrierHandle const innerBarHandle,
    ncclGinBarrierHandle const outerBarHandle,
    uint32_t index,
    bool multimem,
    ncclMultimemHandle const innerMmHandle) {
  ::new (&(session->bar)) ncclBarrierSession<ncclCoopAny>(
      coop, innerTeam, outerTeam, reinterpret_cast<ncclGin const&>(net),
      innerBarHandle, outerBarHandle, index, multimem, innerMmHandle);
}

NCCL_IR_EXPORT void ncclBarrierSessionSync(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence) {
  session->bar.sync(coop, order, fence);
}

#endif  /* NCCL_CHECK_CUDACC */

#endif  /* _NCCL_DEVICE_WRAPPER__IMPL_H_ */
