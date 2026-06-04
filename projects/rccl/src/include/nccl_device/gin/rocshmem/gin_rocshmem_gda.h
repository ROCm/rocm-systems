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

    printf("GDA Enter: rank=%d peer=%d hasWins=%d hasSignal=%d hasCounter=%d bytes=%zu dstOff=%zu srcOff=%zu tid=%d bid=%d\n",
           ctx.rank, peer, (int)hasWins, (int)hasSignal, (int)hasCounter, bytes, dstOff, srcOff,
           threadIdx.x, blockIdx.x);

    if (hasWins) {
      ncclGinRocshmemGdaMemHandle* dstMh = (ncclGinRocshmemGdaMemHandle*)dstWin;
      ncclGinRocshmemGdaMemHandle* srcMh = (ncclGinRocshmemGdaMemHandle*)srcWin;

      uintptr_t dstAddr = loadConst(loadConst(&dstMh->remote_vas) + peer) + dstOff;
      uintptr_t srcAddr = loadConst(&srcMh->local_va) + srcOff;
      uint32_t dstRkey = loadConst(loadConst(&dstMh->rkeys) + peer);
      uint32_t srcLkey = loadConst(&srcMh->lkey);

      printf("GDA Put: rank=%d peer=%d dst=%p src=%p bytes=%zu dstRkey=0x%x srcLkey=0x%x ring_db=%d\n",
             ctx.rank, peer, (void*)dstAddr, (void*)srcAddr, bytes, dstRkey, srcLkey, (int)!hasSignal);
      qp->put_nbi_with_keys((void*)dstAddr, dstRkey, (void*)srcAddr, srcLkey, bytes, wf_info, !hasSignal);
      printf("GDA Put done: rank=%d peer=%d\n", ctx.rank, peer);
    }

    if (hasSignal) {
      if (signalOp == ncclGinSignalInc) signalOpArg = 1;
      uintptr_t sigAddr = loadConst(loadConst(&rsCtx->signal_raddrs) + peer) + sizeof(uint64_t) * signalId;
      uint32_t sigRkey = loadConst(loadConst(&rsCtx->signal_rkeys) + peer);
      printf("GDA Signal: rank=%d peer=%d sigAddr=%p sigRkey=0x%x val=%lld fence=1\n",
             ctx.rank, peer, (void*)sigAddr, sigRkey, (long long)signalOpArg);
      qp->atomic_add_with_keys((void*)sigAddr, sigRkey, (int64_t)signalOpArg, wf_info, /*fence=*/true);
      printf("GDA Signal posted: rank=%d peer=%d, calling quiet...\n", ctx.rank, peer);
      qp->quiet(wf_info);
      printf("GDA Signal+quiet done: rank=%d peer=%d\n", ctx.rank, peer);
    } else if (hasCounter) {
      printf("GDA Counter quiet: rank=%d peer=%d\n", ctx.rank, peer);
      qp->quiet(wf_info);
      printf("GDA Counter quiet done: rank=%d peer=%d\n", ctx.rank, peer);
    }

    if (hasCounter) {
      atomicAdd((unsigned long long*)&loadConst(&rsCtx->counters)[counterId], 1ULL);
    }

    printf("GDA Exit: rank=%d peer=%d\n", ctx.rank, peer);
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
    uintptr_t dstAddr = loadConst(loadConst(&dstMh->remote_vas) + peer) + dstOff;
    uint32_t dstRkey = loadConst(loadConst(&dstMh->rkeys) + peer);

    // lkey=0: put_nbi_with_keys copies srcVal inline into the WQE
    // (inline_threshold >= sizeof(T)), so no registered MR is needed.
    qp->put_nbi_with_keys((void*)dstAddr, dstRkey, &srcVal, 0, sizeof(T), wf_info, !hasSignal);

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
    uint64_t* ptr = nccl::utility::loadConst(&rsCtx->signals) + signalId;
    static __device__ uint64_t last_val = ~0ULL;
    uint64_t v = *ptr;
    if (v != last_val) {
      printf("GDA GetSignalPtr: rank=%d signalId=%u ptr=%p val=%llu\n",
             ctx.rank, signalId, (void*)ptr, (unsigned long long)v);
      last_val = v;
    }
    return ptr;
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
    printf("GDA Flush enter: rank=%d tid=%d bid=%d nRanks=%d\n",
           ctx.rank, threadIdx.x, blockIdx.x, ctx.nRanks);
#pragma unroll 1
    for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
      printf("GDA Flush quiet: rank=%d peer=%d\n", ctx.rank, peer);
      rocshmem::ActiveWFInfo wf_info(peer, rocshmem::ThreadScope::thread);
      loadConst(qps + peer)->quiet(wf_info);
      printf("GDA Flush quiet done: rank=%d peer=%d\n", ctx.rank, peer);
    }
    printf("GDA Flush exit: rank=%d\n", ctx.rank);
  }
};

#endif /* _NCCL_DEVICE_GIN_ROCSHMEM_GDA_H_ */
