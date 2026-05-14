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
 *   [B] Coop-gated   — #if RCCL_ENABLE_NCCL_COOP_ANY (same switch as
 *                      ncclCoopAny in src/include/nccl_device/coop.h).
 *   [C] Stubbed-out  — #if 0; ready to enable once RCCL syncs in
 *                      ncclGin* / composite ncclBarrierSession.
 */

#include "nccl_device_wrapper.h"
#include <new>          /* placement new */

/* ------------------------------------------------------------------------
 * NCCL_DEVICE_INLINE: bitcode-mode override.
 *
 * RCCL's nccl_device/utility.h currently defines NCCL_DEVICE_INLINE as
 * `__device__ __forceinline__`. clang's HIP frontend has known issues
 * honoring __forceinline__ in --offload-device-only mode, which produces
 * unresolved symbols inside the generated bitcode. NCCL solved this by
 * adding a NCCL_CHECK_CUDACC / __clang_llvm_bitcode_lib__ branch in
 * utility.h that maps the macro to __attribute__((always_inline)).
 *
 * We have not (yet) made that change to RCCL's utility.h, so this TU
 * applies the override locally and only when actually building the
 * bitcode artifact. No effect on any other RCCL build.
 * ----------------------------------------------------------------------*/
#if defined(__clang_llvm_bitcode_lib__)
  #undef  NCCL_DEVICE_INLINE
  #define NCCL_DEVICE_INLINE __device__ __attribute__((always_inline))
#endif

#if NCCL_CHECK_CUDACC   /* Defined in nccl_device/utility.h; true under HIP-Clang -x hip
                           (__HIPCC__) and NVCC (__CUDACC__). HIP-Clang does NOT
                           define __CUDACC__, so a bare `#if __CUDACC__` here would
                           silently drop the entire device body from the bitcode. */

/* ========================================================================
 * [A] Always-on bodies
 * ======================================================================*/

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE
void* ncclGetPeerPointerTeam(ncclWindow_t w, size_t offset, ncclTeam tm, int peer) {
  return ncclGetPeerPointer(w, offset, tm, peer);
}


/* ========================================================================
 * [B] ncclCoopAny + LSA Barrier Session bodies
 *
 * Active when both:
 *   - ncclCoopAny is enabled in coop.h (RCCL_ENABLE_NCCL_COOP_ANY=1)
 *   - This wrapper is being instantiated (true for the bitcode build)
 * ======================================================================*/
#if RCCL_ENABLE_NCCL_COOP_ANY

/* ---- coop init thunks (placement-new the static coop into the storage) ---- */
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclCoopAnyInitThread(ncclCoopAny* coop) {
  ::new (coop) ncclCoopAny(ncclCoopThread{});
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclCoopAnyInitWarp(ncclCoopAny* coop) {
  ::new (coop) ncclCoopAny(ncclCoopWarp{});
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclCoopAnyInitLanes(ncclCoopAny* coop, uint32_t lane_mask) {
  ::new (coop) ncclCoopAny(ncclCoopLanes(static_cast<ncclCoopMask_t>(lane_mask)));
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclCoopAnyInitWarpSpan(ncclCoopAny* coop, int warp0, int nWarps, int id) {
  ::new (coop) ncclCoopAny(ncclCoopWarpSpan(warp0, nWarps, id));
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclCoopAnyInitCta(ncclCoopAny* coop) {
  ::new (coop) ncclCoopAny(ncclCoopCta{});
}

/* ---- coop accessors ---- */
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE int ncclCoopThreadRank(const ncclCoopAny* coop) {
  return coop->thread_rank();
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE int ncclCoopSize(const ncclCoopAny* coop) {
  return coop->size();
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE int ncclCoopNumThreads(const ncclCoopAny* coop) {
  return coop->num_threads();
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclCoopSync(const ncclCoopAny* coop) {
  /* sync() is non-const on ncclCoopAny; const_cast matches NCCL upstream. */
  const_cast<ncclCoopAny*>(coop)->sync();
}

/* ---- LSA barrier session thunks ---- */
/* The placement-new of ncclLsaBarrierSession<ncclCoopAny> below is what
 * forces template instantiation of every method on the session class --
 * this is the whole point of the bitcode build.
 *
 * Memory-order parameter type matches wrapper.h: std::memory_order on
 * the HIP path (RCCL has no <cuda/atomic>; mem_barrier__funcs.h provides
 * std::memory_order overloads of arrive/wait/sync). */

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclLsaBarrierSessionInit(
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

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE
void ncclLsaBarrierSessionArrive(ncclLsaBarrierSession_C* session,
                                 ncclCoopAny coop,
                                 std::memory_order order) {
  session->bar.arrive(coop, order);
}

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE
void ncclLsaBarrierSessionWait(ncclLsaBarrierSession_C* session,
                               ncclCoopAny coop,
                               std::memory_order order) {
  session->bar.wait(coop, order);
}

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE
void ncclLsaBarrierSessionSync(ncclLsaBarrierSession_C* session,
                               ncclCoopAny coop,
                               std::memory_order order) {
  session->bar.sync(coop, order);
}

#endif  /* RCCL_ENABLE_NCCL_COOP_ANY */


/* ========================================================================
 * [C] GIN + composite Barrier Session bodies (disabled)
 *
 * Carried over from NCCL v2.29.2-1 with `cuda::memory_order` rewritten
 * to `std::memory_order` to match RCCL's HIP convention. Will become
 * usable once RCCL imports:
 *   - ncclGin / ncclGin_C
 *   - ncclGinBarrierSession<Coop>, ncclGinBarrierHandle, ncclGinFenceLevel
 *   - composite ncclBarrierSession<Coop>
 *
 * Flip the #if 0 to #if RCCL_ENABLE_NCCL_COOP_ANY when those land
 * (these all need ncclCoopAny too).
 * ======================================================================*/
#if 0  /* TODO(rccl-ir): enable once RCCL grows ncclGin* / composite Barrier */

/* GIN barrier session */
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinBarrierSessionInit(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    ncclGin_C net,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index) {
  ::new (&(session->bar)) ncclGinBarrierSession<ncclCoopAny>(
      coop, reinterpret_cast<ncclGin const&>(net), team, handle, index);
}

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinBarrierSessionSync(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    std::memory_order order,
    ncclGinFenceLevel fence) {
  session->bar.sync(coop, order, fence);
}

/* Composite (LSA + GIN) barrier session */
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclBarrierSessionInit(
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

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclBarrierSessionSync(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    std::memory_order order,
    ncclGinFenceLevel fence) {
  session->bar.sync(coop, order, fence);
}

#endif  /* 0 -- GIN / composite Barrier bodies */

#endif  /* NCCL_CHECK_CUDACC */

#endif  /* _NCCL_DEVICE_WRAPPER__IMPL_H_ */
