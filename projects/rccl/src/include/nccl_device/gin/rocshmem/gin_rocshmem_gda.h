/*************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ROCSHMEM_GDA_H_
#define _NCCL_DEVICE_GIN_ROCSHMEM_GDA_H_

#include "../gin_device_common.h"
#include "gin_rocshmem_device_host_common_gda.h"
#include "gda/queue_pair.hpp"

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes, bool hasSignal,
                                      ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given) {
    using nccl::utility::loadConst;
    ncclGinRocshmemGdaGPUContext* rsCtx = (ncclGinRocshmemGdaGPUContext*)ctx.handle;
    rocshmem::QueuePair* qp = loadConst(loadConst(&rsCtx->qps) + peer);
    rocshmem::ActiveWFInfo wf_info(peer, rocshmem::ThreadScope::thread);

    if (hasWins) {
      ncclGinRocshmemGdaMemHandle* dstMh = (ncclGinRocshmemGdaMemHandle*)dstWin;
      ncclGinRocshmemGdaMemHandle* srcMh = (ncclGinRocshmemGdaMemHandle*)srcWin;

      uint32_t dstRkey = loadConst(loadConst(&dstMh->rkeys) + peer);
      uint32_t srcLkey = loadConst(&srcMh->lkey);

      qp->put_nbi_with_keys((void*)dstOff, dstRkey, (void*)srcOff, srcLkey, bytes, wf_info, !hasSignal);
    }

    if (hasSignal) {
      if (signalOp == ncclGinSignalInc) signalOpArg = 1;
      uintptr_t sigAddr = loadConst(loadConst(&rsCtx->signal_raddrs) + peer) + sizeof(uint64_t) * signalId;
      uint32_t sigRkey = loadConst(loadConst(&rsCtx->signal_rkeys) + peer);
      qp->atomic_add_with_keys((void*)sigAddr, sigRkey, (int64_t)signalOpArg, wf_info, /*fence=*/true);
    } else if (hasCounter) {
      qp->quiet(wf_info);
    }

    if (hasCounter) {
      atomicAdd((unsigned long long*)&loadConst(&rsCtx->counters)[counterId], 1ULL);
    }
  }
};

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal, bool hasSignal,
                                      ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given) {
    using nccl::utility::loadConst;
    ncclGinRocshmemGdaGPUContext* rsCtx = (ncclGinRocshmemGdaGPUContext*)ctx.handle;
    rocshmem::QueuePair* qp = loadConst(loadConst(&rsCtx->qps) + peer);
    rocshmem::ActiveWFInfo wf_info(peer, rocshmem::ThreadScope::thread);

    ncclGinRocshmemGdaMemHandle* dstMh = (ncclGinRocshmemGdaMemHandle*)dstWin;
    uint32_t dstRkey = loadConst(loadConst(&dstMh->rkeys) + peer);

    // lkey=0: put_nbi_with_keys copies srcVal inline into the WQE
    // (inline_threshold >= sizeof(T)), so no registered MR is needed.
    qp->put_nbi_with_keys((void*)dstOff, dstRkey, &srcVal, 0, sizeof(T), wf_info, !hasSignal);

    if (hasSignal) {
      if (signalOp == ncclGinSignalInc) signalOpArg = 1;
      uintptr_t sigAddr = loadConst(loadConst(&rsCtx->signal_raddrs) + peer) + sizeof(uint64_t) * signalId;
      uint32_t sigRkey = loadConst(loadConst(&rsCtx->signal_rkeys) + peer);
      qp->atomic_add_with_keys((void*)sigAddr, sigRkey, (int64_t)signalOpArg, wf_info, /*fence=*/true);
    }
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinRocshmemGdaGPUContext* rsCtx = (ncclGinRocshmemGdaGPUContext*)ctx.handle;
    return nccl::utility::loadConst(&rsCtx->counters) + counterId;
  }
};

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinRocshmemGdaGPUContext* rsCtx = (ncclGinRocshmemGdaGPUContext*)ctx.handle;
    nccl::utility::loadConst(&rsCtx->counters)[counterId] = 0;
  }
};

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    ncclGinRocshmemGdaGPUContext* rsCtx = (ncclGinRocshmemGdaGPUContext*)ctx.handle;
    return nccl::utility::loadConst(&rsCtx->signals) + signalId;
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    ncclGinRocshmemGdaGPUContext* rsCtx = (ncclGinRocshmemGdaGPUContext*)ctx.handle;
    nccl::utility::loadConst(&rsCtx->signals)[signalId] = 0;
  }
};

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, cuda::memory_order ord) {
    using nccl::utility::loadConst;
    ncclGinRocshmemGdaGPUContext* rsCtx = (ncclGinRocshmemGdaGPUContext*)ctx.handle;
    rocshmem::QueuePair** qps = loadConst(&rsCtx->qps);
#pragma unroll 1
    for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
      rocshmem::ActiveWFInfo wf_info(peer, rocshmem::ThreadScope::thread);
      loadConst(qps + peer)->quiet(wf_info);
    }
  }
};

#endif /* _NCCL_DEVICE_GIN_ROCSHMEM_GDA_H_ */
