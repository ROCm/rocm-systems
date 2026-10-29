/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

/*
 * NCCL Device API force instantiation and C style APIs for LLVM IR generation
 */

// nccl_device_ir.h must come first: it installs the bitcode-build linkage
// override (always_inline / attribute-free, external) that the device-API
// definitions in nccl_device.h below are then compiled with, so the library
// emits real symbols. It also supplies NCCL_IR_EXPORT and -- crucially -- the
// consumer-visible `extern "C"` declarations of every thunk defined here, so
// each definition below is checked against the exact ABI a consumer will see.
#include "nccl_device_ir.h"
// The bitcode library is device-only: suppress "nccl_device/host.h" by
// pre-defining its include guard, so the host entrypoints never enter this TU.
#define _NCCL_DEVICE_HOST_H_
#include "nccl_device.h"
#ifdef NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER
#error "nccl_device/host.h leaked in: its include guard was renamed, update the #define above"
#endif
#include "util.h"
#include <new>

#if defined(__CUDACC__) || defined(__HIPCC__)

/* Opaque session storage <-> real C++ session.
 *
 * The ABI hands out ncclIr*BarrierSession, a fixed-size byte blob whose extent
 * nccl_device_ir.h static_asserts against the template it stands in for. These
 * helpers are the single place that launders the blob back into the live
 * object, so the reinterpret_cast appears once per session kind rather than
 * once per entry point. Deliberately `static`: they are an implementation
 * detail of this TU and must not reach the artifact's symbol table. */
template<typename Session, typename Storage>
__device__ __attribute__((always_inline))
static Session& ncclIrSession(Storage* s) {
  return *reinterpret_cast<Session*>(s);
}
#define NCCL_IR_LSA_SESSION(s)  ncclIrSession<ncclLsaBarrierSession<ncclCoopAny>>(s)
#define NCCL_IR_GIN_SESSION(s)  ncclIrSession<ncclGinBarrierSession<ncclCoopAny>>(s)
#define NCCL_IR_BAR_SESSION(s)  ncclIrSession<ncclBarrierSession<ncclCoopAny>>(s)

/* ncclDevComm field accessors */
NCCL_IR_EXPORT int                  ncclIrDevComm_Rank(ncclDevComm const* comm)                 { return comm->rank; }
NCCL_IR_EXPORT int                  ncclIrDevComm_NRanks(ncclDevComm const* comm)               { return comm->nRanks; }
NCCL_IR_EXPORT int                  ncclIrDevComm_LsaRank(ncclDevComm const* comm)              { return comm->lsaRank; }
NCCL_IR_EXPORT int                  ncclIrDevComm_LsaSize(ncclDevComm const* comm)              { return comm->lsaSize; }
NCCL_IR_EXPORT ncclLsaBarrierHandle ncclIrDevComm_LsaBarrier(ncclDevComm const* comm)            { return comm->lsaBarrier; }
NCCL_IR_EXPORT ncclGinBarrierHandle ncclIrDevComm_RailGinBarrier(ncclDevComm const* comm)        { return comm->railGinBarrier; }
NCCL_IR_EXPORT ncclLsaBarrierHandle ncclIrDevComm_HybridLsaBarrier(ncclDevComm const* comm)      { return comm->hybridLsaBarrier; }
NCCL_IR_EXPORT ncclGinBarrierHandle ncclIrDevComm_HybridRailGinBarrier(ncclDevComm const* comm)  { return comm->hybridRailGinBarrier; }
NCCL_IR_EXPORT ncclGinBarrierHandle ncclIrDevComm_WorldGinBarrier(ncclDevComm const* comm)       { return comm->worldGinBarrier; }
NCCL_IR_EXPORT ncclMultimemHandle   ncclIrDevComm_LsaMultimem(ncclDevComm const* comm)           { return comm->lsaMultimem; }

NCCL_IR_EXPORT void* ncclIrGetPeerPointerTeam(ncclWindow_t w, size_t offset, ncclTeam tm, int peer) {
    return ncclGetPeerPointer(w, offset, tm, peer);
}

/* coop */
NCCL_IR_EXPORT void ncclIrCoopAnyInitThread(ncclCoopAny* coop) {
    ::new (coop) ncclCoopAny(ncclCoopThread());
}
NCCL_IR_EXPORT void ncclIrCoopAnyInitWarp(ncclCoopAny* coop) {
    ::new (coop) ncclCoopAny(ncclCoopWarp());
}
NCCL_IR_EXPORT void ncclIrCoopAnyInitLanes(ncclCoopAny* coop, ncclCoopMask_t lane_mask) {
    ::new (coop) ncclCoopAny(ncclCoopLanes(lane_mask));
}
NCCL_IR_EXPORT void ncclIrCoopAnyInitWarpSpan(
    ncclCoopAny* coop, int warp0, int nWarps, int id, void* barrierLds) {
    ::new (coop) ncclCoopAny(
        ncclCoopWarpSpan(warp0, nWarps, id, static_cast<ncclCoopNamedBarrierSlot*>(barrierLds)));
}
NCCL_IR_EXPORT void ncclIrCoopAnyInitCta(ncclCoopAny* coop) {
    ::new (coop) ncclCoopAny(ncclCoopCta());
}

NCCL_IR_EXPORT int ncclIrCoopThreadRank(const ncclCoopAny* coop) {
    return coop->thread_rank();
}
NCCL_IR_EXPORT int ncclIrCoopSize(const ncclCoopAny* coop) {
    return coop->size();
}
NCCL_IR_EXPORT int ncclIrCoopNumThreads(const ncclCoopAny* coop) {
    return coop->num_threads();
}
NCCL_IR_EXPORT void ncclIrCoopSync(const ncclCoopAny* coop) {
    const_cast<ncclCoopAny*>(coop)->sync();
}

/* Session storage sizes.
 *
 * A compile-time consumer uses sizeof(ncclIrXxxBarrierSession) directly; these
 * exist for the consumer that cannot, because it never sees this header -- a
 * JIT or dynamic binding that links the artifact and must learn at runtime how
 * much storage to alloca. NCCL exports the same three as
 * ncclXxxBarrierSession_C_size(); the name differs only because the `_C_`
 * infix refers to wrapper structs RCCL does not have. The static_asserts in
 * nccl_device_ir.h are what keep these equal to the real C++ session sizes. */
NCCL_IR_EXPORT size_t ncclIrLsaBarrierSessionSize() { return sizeof(ncclIrLsaBarrierSession); }
NCCL_IR_EXPORT size_t ncclIrGinBarrierSessionSize() { return sizeof(ncclIrGinBarrierSession); }
NCCL_IR_EXPORT size_t ncclIrBarrierSessionSize()    { return sizeof(ncclIrBarrierSession); }

/* lsa barrier session */
NCCL_IR_EXPORT void ncclIrLsaBarrierSessionInit(
    ncclIrLsaBarrierSession* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclLsaBarrierHandle handle,
    uint32_t index,
    bool multimem,
    ncclMultimemHandle mmHandle) {
    ::new (session->storage) ncclLsaBarrierSession<ncclCoopAny>(coop, comm, team, handle, index, multimem, mmHandle);
}
NCCL_IR_EXPORT
void ncclIrLsaBarrierSessionArrive(ncclIrLsaBarrierSession* session, ncclCoopAny coop, cuda::memory_order order) {
    NCCL_IR_LSA_SESSION(session).arrive(coop, order);
}
NCCL_IR_EXPORT
void ncclIrLsaBarrierSessionWait(ncclIrLsaBarrierSession* session, ncclCoopAny coop, cuda::memory_order order) {
    NCCL_IR_LSA_SESSION(session).wait(coop, order);
}
NCCL_IR_EXPORT
void ncclIrLsaBarrierSessionSync(ncclIrLsaBarrierSession* session, ncclCoopAny coop, cuda::memory_order order) {
    NCCL_IR_LSA_SESSION(session).sync(coop, order);
}
NCCL_IR_EXPORT
void ncclIrLsaBarrierSessionDestroy(ncclIrLsaBarrierSession* session) {
    using Session = ncclLsaBarrierSession<ncclCoopAny>;
    NCCL_IR_LSA_SESSION(session).~Session();
}

/* GIN barrier session */
NCCL_IR_EXPORT void ncclIrGinBarrierSessionInit(
    ncclIrGinBarrierSession* session,
    ncclCoopAny coop,
    ncclGin_C const* net,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index) {
    ::new (session->storage) ncclGinBarrierSession<ncclCoopAny>(coop, reinterpret_cast<ncclGin const&>(*net), team, handle, index);
}

NCCL_IR_EXPORT void ncclIrGinBarrierSessionInitAllContexts(
    ncclIrGinBarrierSession* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index) {
    ::new (session->storage) ncclGinBarrierSession<ncclCoopAny>(coop, ncclGinAllContexts(comm), team, handle, index);
}

/* GIN barrier session */
NCCL_IR_EXPORT void ncclIrGinBarrierSessionSync(
    ncclIrGinBarrierSession* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence) {
    NCCL_IR_GIN_SESSION(session).sync(coop, order, fence);
}
NCCL_IR_EXPORT
void ncclIrGinBarrierSessionDestroy(ncclIrGinBarrierSession* session) {
    using Session = ncclGinBarrierSession<ncclCoopAny>;
    NCCL_IR_GIN_SESSION(session).~Session();
}

/* Barrier Session */
NCCL_IR_EXPORT void ncclIrBarrierSessionInit(
    ncclIrBarrierSession* session,
    ncclCoopAny coop,
    ncclTeam innerTeam,
    ncclTeam outerTeam,
    ncclGin_C const* net,
    ncclLsaBarrierHandle const innerBarHandle,
    ncclGinBarrierHandle const outerBarHandle,
    uint32_t index,
    bool multimem, ncclMultimemHandle const innerMmHandle) {
    ::new (session->storage) ncclBarrierSession<ncclCoopAny>(coop, innerTeam, outerTeam, reinterpret_cast<ncclGin const&>(*net),
           innerBarHandle, outerBarHandle, index, multimem, innerMmHandle);
}

NCCL_IR_EXPORT void ncclIrBarrierSessionSync(
    ncclIrBarrierSession* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence) {
    NCCL_IR_BAR_SESSION(session).sync(coop, order, fence);
}
NCCL_IR_EXPORT
void ncclIrBarrierSessionDestroy(ncclIrBarrierSession* session) {
    using Session = ncclBarrierSession<ncclCoopAny>;
    NCCL_IR_BAR_SESSION(session).~Session();
}

// ReduceCopy APIs
#define NCCL_IR_DEFINE_ncclIrLsaReduceSum(suffix, type) \
  NCCL_IR_EXPORT void ncclIrLsaReduceSum_##suffix( \
      ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset, \
      type* dst, size_t count, ncclTeam team) { \
    ncclLsaReduceSum<type, ncclCoopAny, size_t>( \
        coop, srcWindow, srcOffset, dst, count, team); \
  }
#define NCCL_IR_DEFINE_ncclIrMultimemReduceSum(suffix, type) \
  NCCL_IR_EXPORT void ncclIrMultimemReduceSum_##suffix( \
      ncclCoopAny coop, type* mcSrc, type* dst, size_t count) { \
    ncclMultimemReduceSum<type, ncclCoopAny, size_t>( \
        coop, mcSrc, dst, count); \
  }
#define NCCL_IR_DEFINE_ncclIrLsaCopy(suffix, type) \
  NCCL_IR_EXPORT void ncclIrLsaCopy_##suffix( \
      ncclCoopAny coop, type* src, ncclWindow_t dstWindow, \
      size_t dstOffset, size_t count, ncclTeam team) { \
    ncclLsaCopy<type, ncclCoopAny, size_t>( \
        coop, src, dstWindow, dstOffset, count, team); \
  }
#define NCCL_IR_DEFINE_ncclIrMultimemCopy(suffix, type) \
  NCCL_IR_EXPORT void ncclIrMultimemCopy_##suffix( \
      ncclCoopAny coop, type* src, type* mcDst, size_t count) { \
    ncclMultimemCopy<type, ncclCoopAny, size_t>( \
        coop, src, mcDst, count); \
  }
#define NCCL_IR_DEFINE_ncclIrLsaReduceSumCopy(suffix, type) \
  NCCL_IR_EXPORT void ncclIrLsaReduceSumCopy_##suffix( \
      ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset, \
      ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team) { \
    ncclLsaReduceSumCopy<type, ncclCoopAny, size_t>( \
        coop, srcWindow, srcOffset, dstWindow, dstOffset, count, team); \
  }
#define NCCL_IR_DEFINE_ncclIrMultimemReduceSumCopy(suffix, type) \
  NCCL_IR_EXPORT void ncclIrMultimemReduceSumCopy_##suffix( \
      ncclCoopAny coop, type* mcSrc, type* mcDst, size_t count) { \
    ncclMultimemReduceSumCopy<type, ncclCoopAny, size_t>( \
        coop, mcSrc, mcDst, count); \
  }
#define NCCL_IR_DEFINE_ncclIrLocalReduceSumCopy(suffix, type) \
  NCCL_IR_EXPORT void ncclIrLocalReduceSumCopy_##suffix( \
      ncclCoopAny coop, int nSrc, type* srcBase, size_t srcDispl, \
      int nDst, type* dstBase, size_t dstDispl, size_t count) { \
    ncclLocalReduceSumCopy<type, ncclCoopAny, size_t>( \
        coop, nSrc, srcBase, srcDispl, nDst, dstBase, dstDispl, count); \
  }

NCCL_IR_DEFINE_API_ALL_TYPES(ncclIrLsaReduceSum)
NCCL_IR_DEFINE_API_MULTIMEM_TYPES(ncclIrMultimemReduceSum)
NCCL_IR_DEFINE_API_ALL_TYPES(ncclIrLsaCopy)
NCCL_IR_DEFINE_API_MULTIMEM_TYPES(ncclIrMultimemCopy)
NCCL_IR_DEFINE_API_ALL_TYPES(ncclIrLsaReduceSumCopy)
NCCL_IR_DEFINE_API_MULTIMEM_TYPES(ncclIrMultimemReduceSumCopy)
NCCL_IR_DEFINE_API_ALL_TYPES(ncclIrLocalReduceSumCopy)
#endif //  __CUDACC__
