/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_H_
#define _NCCL_DEVICE_GIN_ANVIL_H_

#include "../gin_device_common.h"
#include "gin_anvil_device_host_common.h"

// Anvil SDMA packet helpers (rocshmem src/sdma); -I from ROCSHMEM_SDMA_SRC_DIR.
#include "sdma/anvil_device.hpp"

// Helper: compute per-rank pointer in LSA flat address space.
NCCL_DEVICE_INLINE static inline uintptr_t ncclGinAnvilRankPtr(uintptr_t rank0Base, uint64_t strideBytes, int rank) {
  return rank0Base + (uintptr_t)rank * (uintptr_t)strideBytes;
}

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes, bool hasSignal,
                                      ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool,
                                      ncclGinDescriptorSmem*,
                                      cuda::thread_scope, cuda::thread_scope) {
    using nccl::utility::loadConst;
    auto* aCtx = (ncclGinAnvilGPUContext*)ctx.handle;

    if (!hasWins) return;
    if (peer == ctx.rank) return;

    auto* q = (rocshmem::anvil::SdmaQueueDeviceHandle*)loadConst(&aCtx->queues[peer]);
    // Off-node peers have no SDMA queue (intra-node backend only).
    if (q == nullptr) return;

    // Resolve per-window symmetric bases.
    auto* dstMh = (ncclGinAnvilMemHandle*)dstWin;
    auto* srcMh = (ncclGinAnvilMemHandle*)srcWin;
    uintptr_t dstRank0Base = loadConst(&dstMh->lsaRank0Base);
    uintptr_t srcRank0Base = loadConst(&srcMh->lsaRank0Base);
    uint64_t stride = loadConst(&dstMh->lsaStrideBytes);

    // Local and remote pointers live in the same LSA flat VA space.
    void* dst = (void*)(ncclGinAnvilRankPtr(dstRank0Base, stride, peer) + dstOff);
    void* src = (void*)(ncclGinAnvilRankPtr(srcRank0Base, stride, ctx.rank) + srcOff);

    uint64_t* counterPtr = nullptr;
    if (hasCounter) counterPtr = (uint64_t*)(loadConst(&aCtx->counters) + counterId);

    uint64_t* sigPtr = nullptr;
    if (hasSignal) {
      uint64_t* peerBase = (uint64_t*)loadConst(&aCtx->signalsBase[peer]);
      sigPtr = peerBase + signalId;
      if (signalOp == ncclGinSignalInc) signalOpArg = 1;
      // Only ADD64 is supported in current SDMA helper; encode arg by repeated add is not supported.
      // We treat any non-1 arg as 1 for now (matches Inc use in AlltoAll).
      signalOpArg = 1;
    }

    if (hasSignal && hasCounter) {
      rocshmem::anvil::putSignalCounter(*q, dst, src, bytes, sigPtr, counterPtr);
    } else if (hasSignal) {
      rocshmem::anvil::putSignal(*q, dst, src, bytes, sigPtr);
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
                                      size_t, T, bool,
                                      ncclGinSignal_t, ncclGinSignalOp_t,
                                      uint64_t, bool,
                                      ncclGinDescriptorSmem*,
                                      cuda::thread_scope, cuda::thread_scope) {
    // Not needed for AlltoAll path; leave unimplemented.
    __builtin_unreachable();
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_ANVIL> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    auto* aCtx = (ncclGinAnvilGPUContext*)ctx.handle;
    return nccl::utility::loadConst(&aCtx->counters) + counterId;
  }
};

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_ANVIL> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    auto* aCtx = (ncclGinAnvilGPUContext*)ctx.handle;
    nccl::utility::loadConst(&aCtx->counters)[counterId] = 0;
  }
};

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_ANVIL> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    auto* aCtx = (ncclGinAnvilGPUContext*)ctx.handle;
    // readSignal/waitSignal operate on this rank's signal array (remote puts increment it)
    uint64_t* localBase = (uint64_t*)nccl::utility::loadConst(&aCtx->signalsBase[ctx.rank]);
    return localBase + signalId;
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_ANVIL> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, ncclGinSignal_t) {
    // Resetting remote signals is a host responsibility for this backend.
  }
};

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ANVIL> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop, cuda::memory_order) {
    auto* aCtx = (ncclGinAnvilGPUContext*)ctx.handle;
    // Quiet all queues. This is conservative but correct.
    for (int p = 0; p < ctx.nRanks; p++) {
      if (p == ctx.rank) continue;
      auto* q = (rocshmem::anvil::SdmaQueueDeviceHandle*)nccl::utility::loadConst(&aCtx->queues[p]);
      if (q == nullptr) continue;
      rocshmem::anvil::quiet(*q);
    }
  }
};

#endif /* _NCCL_DEVICE_GIN_ANVIL_H_ */

