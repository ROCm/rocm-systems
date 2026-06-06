/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_H_
#define _NCCL_DEVICE_GIN_ANVIL_H_

#include "../gin_device_common.h"
#include "gin_anvil_device_host_common.h"

#include "sdma/anvil_device.hpp"

NCCL_DEVICE_INLINE static uintptr_t ncclGinAnvilRankPtr(uintptr_t rank0Base, uint64_t strideBytes, int rank) {
  return rank0Base + (uintptr_t)rank * (uintptr_t)strideBytes;
}

NCCL_DEVICE_INLINE static ncclGinAnvilGPUContext* ncclGinAnvilGetCtx(ncclGinCtx ctx) {
  void* handle = nccl::utility::loadConst(&ctx.handle);
  if (handle == nullptr) return nullptr;
  return &((ncclGinAnvilGPUContext*)handle)[ctx.contextId];
}

// Indexed signal cell on peer P. Uses imported local VA + GPU atomics (not SDMA atomics).
NCCL_DEVICE_INLINE static uint64_t* ncclGinAnvilPeerSignalPtr(ncclGinAnvilGPUContext* aCtx, int peer,
                                                               ncclGinSignal_t signalId) {
  using nccl::utility::loadConst;
  uint64_t** signalsBaseArr = (uint64_t**)loadConst(&aCtx->signalsBase);
  if (signalsBaseArr == nullptr) return nullptr;
  uint64_t* peerSignals = (uint64_t*)loadConst(&signalsBaseArr[peer]);
  if (peerSignals == nullptr) return nullptr;
  uint32_t ctxOff = loadConst(&aCtx->signalsContextOffset);
  return peerSignals + (size_t)ctxOff + (size_t)signalId;
}

NCCL_DEVICE_INLINE static uint64_t* ncclGinAnvilLocalSignalPtr(ncclGinAnvilGPUContext* aCtx,
                                                               ncclGinSignal_t signalId) {
  uint64_t* signals = nccl::utility::loadConst(&aCtx->signals);
  if (signals == nullptr) return nullptr;
  return signals + signalId;
}

NCCL_DEVICE_INLINE static void ncclGinAnvilLocalSignalOp(uint64_t* sigPtr, ncclGinSignalOp_t signalOp,
                                                         uint64_t signalOpArg) {
  if (sigPtr == nullptr) return;
  if (signalOp == ncclGinSignalInc) signalOpArg = 1;
  if (signalOp == ncclGinSignalInc || signalOp == ncclGinSignalAdd)
    atomicAdd((unsigned long long*)sigPtr, (unsigned long long)signalOpArg);
}

NCCL_DEVICE_INLINE static void ncclGinAnvilRemoteSignalOp(ncclGinAnvilGPUContext* aCtx, int peer,
                                                            ncclGinSignal_t signalId,
                                                            ncclGinSignalOp_t signalOp,
                                                            uint64_t signalOpArg) {
  uint64_t* sigPtr = ncclGinAnvilPeerSignalPtr(aCtx, peer, signalId);
  ncclGinAnvilLocalSignalOp(sigPtr, signalOp, signalOpArg);
}

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool,
                                      ncclGinDescriptorSmem*,
                                      cuda::thread_scope, cuda::thread_scope,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    using nccl::utility::loadConst;
    ncclGinAnvilGPUContext* aCtx = ncclGinAnvilGetCtx(ctx);
    if (aCtx == nullptr) return;

    bool hasSignal = signal.type != NCCL_GIN_SIGNAL_TYPE_NONE;
    ncclGinSignal_t signalId = 0;
    if (hasSignal && signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED)
      signalId = signal.indexedSignal.signalId;

    if (peer == ctx.rank) {
      if (!hasWins) {
        if (hasSignal)
          ncclGinAnvilLocalSignalOp(ncclGinAnvilLocalSignalPtr(aCtx, signalId), signalOp, signalOpArg);
        return;
      }
      if (dstWin == nullptr || srcWin == nullptr) return;

      auto* dstMh = (ncclGinAnvilMemHandle*)dstWin;
      auto* srcMh = (ncclGinAnvilMemHandle*)srcWin;
      uintptr_t dstRank0Base = loadConst(&dstMh->lsaRank0Base);
      uintptr_t srcRank0Base = loadConst(&srcMh->lsaRank0Base);
      uint64_t stride = loadConst(&dstMh->lsaStrideBytes);
      if (dstRank0Base == 0 || srcRank0Base == 0 || stride == 0) return;

      char* dst = (char*)(ncclGinAnvilRankPtr(dstRank0Base, stride, ctx.rank) + dstOff);
      char* src = (char*)(ncclGinAnvilRankPtr(srcRank0Base, stride, ctx.rank) + srcOff);
      for (size_t i = 0; i < bytes; ++i) dst[i] = src[i];

      if (hasSignal)
        ncclGinAnvilLocalSignalOp(ncclGinAnvilLocalSignalPtr(aCtx, signalId), signalOp, signalOpArg);
      if (hasCounter)
        atomicAdd((unsigned long long*)(loadConst(&aCtx->counters) + counterId), 1ULL);
      return;
    }

    void** queues = (void**)loadConst(&aCtx->queues);
    if (queues == nullptr) return;
    auto* q = (rocshmem::anvil::SdmaQueueDeviceHandle*)loadConst(&queues[peer]);
    if (q == nullptr) return;

    uint64_t* sigPtr = nullptr;
    if (hasSignal) {
      sigPtr = ncclGinAnvilPeerSignalPtr(aCtx, peer, signalId);
      if (sigPtr == nullptr) return;
      if (signalOp == ncclGinSignalInc) signalOpArg = 1;
    }

    if (!hasWins) {
      if (hasSignal) ncclGinAnvilRemoteSignalOp(aCtx, peer, signalId, signalOp, signalOpArg);
      return;
    }

    if (dstWin == nullptr || srcWin == nullptr) return;

    auto* dstMh = (ncclGinAnvilMemHandle*)dstWin;
    auto* srcMh = (ncclGinAnvilMemHandle*)srcWin;
    uintptr_t dstRank0Base = loadConst(&dstMh->lsaRank0Base);
    uintptr_t srcRank0Base = loadConst(&srcMh->lsaRank0Base);
    uint64_t stride = loadConst(&dstMh->lsaStrideBytes);
    if (dstRank0Base == 0 || srcRank0Base == 0 || stride == 0) return;

    void* dst = (void*)(ncclGinAnvilRankPtr(dstRank0Base, stride, peer) + dstOff);
    void* src = (void*)(ncclGinAnvilRankPtr(srcRank0Base, stride, ctx.rank) + srcOff);

    uint64_t* counterPtr = nullptr;
    if (hasCounter) counterPtr = (uint64_t*)(loadConst(&aCtx->counters) + counterId);

    if (hasSignal && hasCounter) {
      rocshmem::anvil::put(*q, dst, src, bytes);
      rocshmem::anvil::quiet(*q);
      ncclGinAnvilLocalSignalOp(sigPtr, signalOp, signalOpArg);
      atomicAdd((unsigned long long*)counterPtr, 1ULL);
    } else if (hasSignal) {
      rocshmem::anvil::put(*q, dst, src, bytes);
      rocshmem::anvil::quiet(*q);
      ncclGinAnvilLocalSignalOp(sigPtr, signalOp, signalOpArg);
    } else if (hasCounter) {
      rocshmem::anvil::putCounter(*q, dst, src, bytes, counterPtr);
    } else {
      rocshmem::anvil::put(*q, dst, src, bytes);
    }
  }
};

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_ANVIL> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, Coop, int, ncclGinWindow_t,
                                      size_t, T,
                                      ncclGinSignalDescriptor, ncclGinSignalOp_t,
                                      uint64_t, bool,
                                      ncclGinDescriptorSmem*,
                                      cuda::thread_scope, cuda::thread_scope,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    __builtin_unreachable();
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_ANVIL> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinAnvilGPUContext* aCtx = ncclGinAnvilGetCtx(ctx);
    if (aCtx == nullptr) return nullptr;
    uint64_t* counters = nccl::utility::loadConst(&aCtx->counters);
    if (counters == nullptr) return nullptr;
    return counters + counterId;
  }
};

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_ANVIL> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinAnvilGPUContext* aCtx = ncclGinAnvilGetCtx(ctx);
    if (aCtx == nullptr) return;
    nccl::utility::loadConst(&aCtx->counters)[counterId] = 0;
  }
};

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_ANVIL> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    ncclGinAnvilGPUContext* aCtx = ncclGinAnvilGetCtx(ctx);
    if (aCtx == nullptr) return nullptr;
    return ncclGinAnvilLocalSignalPtr(aCtx, signalId);
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_ANVIL> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, ncclGinSignalDescriptor signal) {
    (void)signal;
  }
};

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ANVIL> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop, cuda::memory_order,
                                      uint32_t* abortFlag) {
    using nccl::utility::loadConst;
    ncclGinAnvilGPUContext* aCtx = ncclGinAnvilGetCtx(ctx);
    if (aCtx == nullptr) return;
    void** queues = (void**)loadConst(&aCtx->queues);
    if (queues == nullptr) return;
    for (int p = 0; p < ctx.nRanks; p++) {
      if (p == ctx.rank) continue;
      auto* q = (rocshmem::anvil::SdmaQueueDeviceHandle*)loadConst(&queues[p]);
      if (q == nullptr) continue;
      rocshmem::anvil::quiet(*q);
    }
  }
};

#endif /* _NCCL_DEVICE_GIN_ANVIL_H_ */
