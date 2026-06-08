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

// Match ROCSHMEM IPC SDMA policy (see ROCSHMEM_SDMA_THRESHOLD, default 256): below this
// size, GPU load/store to peer-mapped LSA memory beats SDMA queue submit latency.
constexpr size_t ncclGinAnvilSdmaThresholdBytes = 256;
// Split large SDMA puts into chunks to pipeline ring-buffer reuse and spread load across channels.
constexpr size_t ncclGinAnvilSdmaChunkBytes = 4 * 1024 * 1024;

NCCL_DEVICE_INLINE static int ncclGinAnvilSelectSdmaChannel(ncclGinCtx ctx, ncclGinAnvilGPUContext* aCtx) {
  using nccl::utility::loadConst;
  uint32_t numCh = loadConst(&aCtx->numSdmaChannels);
  if (numCh <= 1) return 0;
  // Spread CTAs across channels; contextId is typically blockIdx.x for multi-CTA kernels.
  return (int)(ctx.contextId % numCh);
}

NCCL_DEVICE_INLINE static rocshmem::anvil::SdmaQueueDeviceHandle* ncclGinAnvilPeerQueue(
    ncclGinAnvilGPUContext* aCtx, int peer, int channel) {
  using nccl::utility::loadConst;
  void** queues = (void**)loadConst(&aCtx->queues);
  if (queues == nullptr) return nullptr;
  uint32_t numCh = loadConst(&aCtx->numSdmaChannels);
  if (numCh < 1) numCh = 1;
  return (rocshmem::anvil::SdmaQueueDeviceHandle*)loadConst(
      &queues[(size_t)peer * numCh + (size_t)channel]);
}

NCCL_DEVICE_INLINE static void ncclGinAnvilMarkSdmaDirty(ncclGinAnvilGPUContext* aCtx, int peer) {
  if (peer >= 0 && peer < 32)
    atomicOr((unsigned int*)&aCtx->sdmaDirtyMask, 1u << peer);
}

// Enqueue one or more SDMA COPY_LINEAR packets; signal is issued separately by the caller.
NCCL_DEVICE_INLINE static void ncclGinAnvilSdmaPutChunks(rocshmem::anvil::SdmaQueueDeviceHandle* q,
                                                         void* dst, void* src, size_t bytes) {
  if (bytes == 0) return;
  char* d = (char*)dst;
  char* s = (char*)src;
  while (bytes > ncclGinAnvilSdmaChunkBytes) {
    rocshmem::anvil::put(*q, d, s, ncclGinAnvilSdmaChunkBytes);
    d += ncclGinAnvilSdmaChunkBytes;
    s += ncclGinAnvilSdmaChunkBytes;
    bytes -= ncclGinAnvilSdmaChunkBytes;
  }
  rocshmem::anvil::put(*q, d, s, bytes);
}

// Intra-rank / local-LSA copy (Put self path and sub-threshold remote puts).
NCCL_DEVICE_INLINE static void ncclGinAnvilMemcpy(void* dst, void const* src, size_t bytes) {
  if (bytes == 0) return;
  char* d = (char*)dst;
  char const* s = (char const*)src;
  if (bytes >= sizeof(uint64_t) &&
      ((uintptr_t)d | (uintptr_t)s) % alignof(uint64_t) == 0) {
    uint64_t* d64 = (uint64_t*)d;
    uint64_t const* s64 = (uint64_t const*)s;
    size_t n64 = bytes / sizeof(uint64_t);
#if defined(__HIPCC__) || defined(__clang__)
    for (size_t i = 0; i < n64; ++i) d64[i] = s64[i];
#else
    for (size_t i = 0; i < n64; ++i) d64[i] = s64[i];
#endif
    size_t tail = n64 * sizeof(uint64_t);
    for (size_t i = tail; i < bytes; ++i) d[i] = s[i];
    return;
  }
#if defined(__HIPCC__) || defined(__clang__)
  __builtin_memcpy(dst, src, bytes);
#else
  for (size_t i = 0; i < bytes; ++i) d[i] = s[i];
#endif
}

NCCL_DEVICE_INLINE static ncclGinAnvilGPUContext* ncclGinAnvilGetCtx(ncclGinCtx ctx) {
  void* handle = nccl::utility::loadConst(&ctx.handle);
  if (handle == nullptr) return nullptr;
  return &((ncclGinAnvilGPUContext*)handle)[ctx.contextId];
}

// Indexed signal cell on peer P via imported cuMem (local RW VA for GPU atomics).
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

// Remote signal fallback: GPU atomics on the imported cuMem view when SDMA putSignal is unavailable.
NCCL_DEVICE_INLINE static void ncclGinAnvilRemoteGpuSignalOp(uint64_t* sigPtr, ncclGinSignalOp_t signalOp,
                                                               uint64_t signalOpArg) {
  if (sigPtr == nullptr) return;
  if (signalOp == ncclGinSignalInc) signalOpArg = 1;
  if (signalOp == ncclGinSignalInc || signalOp == ncclGinSignalAdd) {
#if defined(__HIP_PLATFORM_AMD__)
    __hip_atomic_fetch_add((unsigned long long*)sigPtr, (unsigned long long)signalOpArg,
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    atomicAdd((unsigned long long*)sigPtr, (unsigned long long)signalOpArg);
#endif
  }
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
      ncclGinAnvilMemcpy(dst, src, bytes);

      if (hasSignal)
        ncclGinAnvilLocalSignalOp(ncclGinAnvilLocalSignalPtr(aCtx, signalId), signalOp, signalOpArg);

      if (hasCounter)
        atomicAdd((unsigned long long*)(loadConst(&aCtx->counters) + counterId), 1ULL);
      return;
    }

    int sdmaChannel = ncclGinAnvilSelectSdmaChannel(ctx, aCtx);
    auto* q = ncclGinAnvilPeerQueue(aCtx, peer, sdmaChannel);
    if (q == nullptr) return;

    uint64_t* sigPtr = nullptr;
    if (hasSignal) {
      sigPtr = ncclGinAnvilPeerSignalPtr(aCtx, peer, signalId);
      if (sigPtr == nullptr) return;
      if (signalOp == ncclGinSignalInc) signalOpArg = 1;
    }

    if (!hasWins) {
      if (hasSignal) ncclGinAnvilRemoteGpuSignalOp(sigPtr, signalOp, signalOpArg);
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

    if (bytes < ncclGinAnvilSdmaThresholdBytes) {
      ncclGinAnvilMemcpy(dst, src, bytes);
      if (hasSignal) ncclGinAnvilRemoteGpuSignalOp(sigPtr, signalOp, signalOpArg);
      if (hasCounter) atomicAdd((unsigned long long*)counterPtr, 1ULL);
      return;
    }

    if (hasSignal) {
      ncclGinAnvilSdmaPutChunks(q, dst, src, bytes);
      ncclGinAnvilRemoteGpuSignalOp(sigPtr, signalOp, signalOpArg);
      if (hasCounter) atomicAdd((unsigned long long*)counterPtr, 1ULL);
    } else if (hasCounter) {
      // putCounter does not support chunking; use single packet when counter is required.
      rocshmem::anvil::putCounter(*q, dst, src, bytes, counterPtr);
    } else {
      ncclGinAnvilSdmaPutChunks(q, dst, src, bytes);
    }
    ncclGinAnvilMarkSdmaDirty(aCtx, peer);
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
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, cuda::memory_order,
                                      uint32_t* abortFlag) {
    using nccl::utility::loadConst;
    ncclGinAnvilGPUContext* aCtx = ncclGinAnvilGetCtx(ctx);
    if (aCtx == nullptr) return;
    uint32_t numCh = loadConst(&aCtx->numSdmaChannels);
    if (numCh < 1) numCh = 1;
    uint32_t dirty = atomicExch((unsigned int*)&aCtx->sdmaDirtyMask, 0u);
    for (int p = coop.thread_rank(); p < ctx.nRanks; p += coop.size()) {
      if (p == ctx.rank || (dirty & (1u << p)) == 0) continue;
      for (int ch = 0; ch < (int)numCh; ch++) {
        auto* q = ncclGinAnvilPeerQueue(aCtx, p, ch);
        if (q == nullptr) continue;
        rocshmem::anvil::quiet(*q);
      }
    }
    coop.sync();
    (void)abortFlag;
  }
};

#endif /* _NCCL_DEVICE_GIN_ANVIL_H_ */
