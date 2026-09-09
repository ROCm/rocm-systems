/*************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ROCSHMEM_GDA_H_
#define _NCCL_DEVICE_GIN_ROCSHMEM_GDA_H_

#include "../gin_device_common.h"
#include "gin_rocshmem_device_host_common_gda.h"
#include "queue_pair_device.h"

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                      size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasCounter, ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                      cuda::thread_scope given, uint32_t optFlags = ncclGinOptFlagsDefault) {
    using nccl::utility::loadConst;
    bool hasSignal = signal.type != NCCL_GIN_SIGNAL_TYPE_NONE;

    coop.sync();
    if (coop.thread_rank() == 0) {
      ncclGinRocshmemGdaGPUContext* rsCtx = (ncclGinRocshmemGdaGPUContext*)ctx.handle;
      rocshmem::QueuePair* qp = loadConst(loadConst(&rsCtx->qps) + peer);
      rocshmem::ActiveWFInfo wf_info(peer, rocshmem::ThreadScope::thread);

      // HIP thread_scope (hip_compat.h): system is the MAX value, so a caller that
      // only guaranteed a weaker scope has given < required -> add a system fence.
      if ((required == cuda::thread_scope_system) && (given < required)) {
        NCCL_GIN_THREADFENCE_SYSTEM();
      }

      // Skip zero-length RDMA writes (0-byte put_nbi can stall quiet()/flush() -> deadlock); signal still delivered, matching native rocSHMEM/PROXY.
      if (hasWins && bytes != 0) {
        ncclGinRocshmemGdaMemHandle* dstMh = (ncclGinRocshmemGdaMemHandle*)dstWin;
        ncclGinRocshmemGdaMemHandle* srcMh = (ncclGinRocshmemGdaMemHandle*)srcWin;

        uintptr_t dstAddr = loadConst(loadConst(&dstMh->remote_vas) + peer) + dstOff;
        uintptr_t srcAddr = loadConst(&srcMh->local_va) + srcOff;
        uint32_t dstRkey = loadConst(loadConst(&dstMh->rkeys) + peer);
        uint32_t srcLkey = loadConst(&srcMh->lkey);

        qp->put_nbi((void*)dstAddr, dstRkey, (void*)srcAddr, srcLkey, bytes, wf_info, !hasSignal);
      }

      if (hasSignal) {
        if (signalOp == ncclGinSignalInc) signalOpArg = 1;
        uintptr_t sigAddr =
          loadConst(loadConst(&rsCtx->signal_raddrs) + peer) + sizeof(uint64_t) * signal.indexedSignal.signalId;
        uint32_t sigRkey = loadConst(loadConst(&rsCtx->signal_rkeys) + peer);
        qp->atomic_add((void*)sigAddr, sigRkey, (int64_t)signalOpArg, wf_info);
      } else if (hasCounter) {
        qp->quiet(wf_info);
      }

      if (hasCounter) {
        atomicAdd((unsigned long long*)&loadConst(&rsCtx->counters)[counterId], 1ULL);
      }
    }
    coop.sync();
  }
};

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin, size_t dstOff,
                                      T srcVal, ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    using nccl::utility::loadConst;
    bool hasSignal = signal.type != NCCL_GIN_SIGNAL_TYPE_NONE;

    coop.sync();
    if (coop.thread_rank() == 0) {
      ncclGinRocshmemGdaGPUContext* rsCtx = (ncclGinRocshmemGdaGPUContext*)ctx.handle;
      rocshmem::QueuePair* qp = loadConst(loadConst(&rsCtx->qps) + peer);
      rocshmem::ActiveWFInfo wf_info(peer, rocshmem::ThreadScope::thread);

      ncclGinRocshmemGdaMemHandle* dstMh = (ncclGinRocshmemGdaMemHandle*)dstWin;
      uintptr_t dstAddr = loadConst(loadConst(&dstMh->remote_vas) + peer) + dstOff;
      uint32_t dstRkey = loadConst(loadConst(&dstMh->rkeys) + peer);

      // HIP thread_scope (hip_compat.h): system is the MAX value, so a caller that
      // only guaranteed a weaker scope has given < required -> add a system fence.
      if ((required == cuda::thread_scope_system) && (given < required)) {
        NCCL_GIN_THREADFENCE_SYSTEM();
      }

      // lkey=0: put_nbi copies srcVal inline into the WQE
      // (inline_threshold >= sizeof(T)), so no registered MR is needed.
      qp->put_nbi((void*)dstAddr, dstRkey, &srcVal, 0, sizeof(T), wf_info, !hasSignal);

      if (hasSignal) {
        if (signalOp == ncclGinSignalInc) signalOpArg = 1;
        uintptr_t sigAddr =
          loadConst(loadConst(&rsCtx->signal_raddrs) + peer) + sizeof(uint64_t) * signal.indexedSignal.signalId;
        uint32_t sigRkey = loadConst(loadConst(&rsCtx->signal_rkeys) + peer);
        qp->atomic_add((void*)sigAddr, sigRkey, (int64_t)signalOpArg, wf_info);
      }
    }
    coop.sync();
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinRocshmemGdaGPUContext* rsCtx = (ncclGinRocshmemGdaGPUContext*)ctx.handle;
    return {nccl::utility::loadConst(&rsCtx->counters) + counterId, 0};
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
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    ncclGinRocshmemGdaGPUContext* rsCtx = (ncclGinRocshmemGdaGPUContext*)ctx.handle;
    return {nccl::utility::loadConst(&rsCtx->signals) + signalId, 0};
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    ncclGinRocshmemGdaGPUContext* rsCtx = (ncclGinRocshmemGdaGPUContext*)ctx.handle;
    if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED)
      nccl::utility::loadConst(&rsCtx->signals)[signal.indexedSignal.signalId] = 0;
  }
};

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::memory_order ord, uint32_t* abortFlag) {
    (void)hasDescriptor;
    (void)descriptor;
    (void)ord;
    (void)abortFlag;
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

template <>
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, Coop, int, ncclGinWindow_t, size_t, ncclGinWindow_t, size_t, size_t,
                                      bool, ncclGinDescriptorSmem*, uint32_t = ncclGinOptFlagsDefault) {
    __builtin_trap();
  }
};

template <>
struct ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, int, ncclGinRequest_t*, bool, ncclGinDescriptorSmem*, uint32_t) {
    __builtin_trap();
  }
};

template <>
struct ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, ncclGinRequest_t&, bool, ncclGinDescriptorSmem*, cuda::memory_order,
                                      uint32_t*) {
    __builtin_trap();
  }
};

#endif /* _NCCL_DEVICE_GIN_ROCSHMEM_GDA_H_ */
