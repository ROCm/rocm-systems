/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Runtime (nRanks-flexible) GPU barrier for the DDA fabric/VMM path.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cuda.h>
#include <cstdint>
#include <memory>

#include "algorithms/dda/device/dda_kernel_params.h"
#include "algorithms/dda/device/device_buffer.h"
#include "algorithms/dda/fabric/fabric_mem_handler.h"
#include "algorithms/dda/ipc/ipc_gpu_barrier.h"

struct ncclMemManager;

namespace dda::common {

// Upper bound on the number of ranks the fabric DDA path supports (DPX).
// Note: LL scratch scales as 2 * nRanks * 16 MiB; at 144 ranks this is ~4.5 GiB per GPU.
constexpr int kDdaMaxNranks = 144;

class FabricGpuBarrier;

struct FabricGpuBarrierResources {
  std::unique_ptr<ncclFabricMemHandler> fabricMemHandler;
  // This rank's own flag buffer (VMM-backed when available).
  std::unique_ptr<DeviceBuffer> selfFlagBuf;
  // Device-resident array of nRanks flag-buffer pointers (FlagType*[nRanks]).
  std::unique_ptr<DeviceBuffer> peerFlagsDev;
};

class FabricGpuBarrier {
public:
  using FlagType = uint32_t;

  __host__ FabricGpuBarrier() = default;

  // Allocates this rank's flag buffer, exchanges peer flag-buffer pointers via
  // fabric shareable handles and builds a device-resident pointer table.
  static __host__ std::pair<std::unique_ptr<FabricGpuBarrierResources>, FabricGpuBarrier> mallocAndInit(
    int nRanks, int nBlocks, int selfRank, void* bootstrap, struct ncclMemManager* manager, DdaKernelParams params);

  template <bool hasPreviousMemAccess, bool hasSubsequentMemAccess>
  __device__ __forceinline__ void syncOnSameBlockIdx() {
    enum class MemFenceType {
      RELEASE_ACQUIRE,
      RELEASE_ONLY,
      ACQUIRE_ONLY,
    };

    static_assert(hasPreviousMemAccess || hasSubsequentMemAccess);

    constexpr MemFenceType fenceType =
      hasPreviousMemAccess && hasSubsequentMemAccess ?
        MemFenceType::RELEASE_ACQUIRE :
        (!hasPreviousMemAccess ? MemFenceType::ACQUIRE_ONLY : MemFenceType::RELEASE_ONLY);

    if constexpr (hasPreviousMemAccess) {
      __syncthreads();
    }
    // Each thread handles one or more peers in a strided loop so the barrier
    // stays correct when blockDim.x < nRanks_. A small count can launch as few
    // as 64 threads while a clique may have up to kDdaMaxNranks ranks; with a
    // single thread per peer, peers with rank >= blockDim.x would never be
    // signaled or waited on, hanging the barrier. Every rank walks peers in the
    // same increasing order, so the interleaved signal/wait cannot deadlock.
    FlagType* selfBuf = peerFlags_[selfRank_];
    bool gaveUp = false;
    for (int peerRank = threadIdx.x; !aborted_ && peerRank < nRanks_; peerRank += blockDim.x) {
      FlagType* peerBuf = peerFlags_[peerRank];
      bool signaled = false;

      // Signal the peer that this rank reached the barrier for this block.
      if constexpr (fenceType == MemFenceType::ACQUIRE_ONLY) {
        signaled = putFlag<std::memory_order_relaxed>(peerBuf + getFlagIdx(selfRank_, blockIdx.x), params_);
      } else {
        signaled = putFlag<std::memory_order_release>(peerBuf + getFlagIdx(selfRank_, blockIdx.x), params_);
      }
      if (!signaled) {
        gaveUp = true;
        break;
      }

      // Wait for the peer's signal in this rank's own buffer.
      bool arrived = false;
      if constexpr (fenceType == MemFenceType::RELEASE_ONLY) {
        arrived = waitFlag<std::memory_order_relaxed>(selfBuf + getFlagIdx(peerRank, blockIdx.x), params_);
      } else {
        arrived = waitFlag<std::memory_order_acquire>(selfBuf + getFlagIdx(peerRank, blockIdx.x), params_);
      }
      if (!arrived) {
        gaveUp = true;
        break;
      }
    }
    // With nRanks_ <= warpSize every signaling lane is in wave 0, so a wave vote
    // suffices; larger cliques span waves and need the block-wide vote.
    const bool singleWave = nRanks_ <= warpSize;
    if (singleWave) {
      aborted_ = __any(aborted_ || gaveUp);
    }
    if constexpr (hasSubsequentMemAccess) {
      if (singleWave) {
        __syncthreads();
      } else {
        aborted_ = __syncthreads_or(aborted_ || gaveUp);
      }
    }
  }

  // Waits until every store this wave has issued has completed, so a system-scope store has reached its
  // destination.
  __device__ __forceinline__ void orderDestWrites() {
#if defined(__gfx1250__)
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    asm volatile("s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0" ::: "memory");
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
#else
    __threadfence_system();
#endif
  }

  // Two-rank form of syncOnSameBlockIdx<true, true>: posts this rank's flag into dstRank's slot for block dstBlock
  // and waits for srcRank's flag in this rank's slot for this block. A flag slot holds one token, consumed by its
  // wait, so every post must be matched by exactly one wait within the same kernel.
  __device__ __forceinline__ void signalWaitPeer(int dstRank, int srcRank, int dstBlock) {
    // Every wave drains its own stores, then the block syncs so thread 0 only posts once all of them are done.
    orderDestWrites();
    __syncthreads();
    if (threadIdx.x == 0) {
      putFlag<std::memory_order_release>(peerFlags_[dstRank] + getFlagIdx(selfRank_, dstBlock));
      waitFlag<std::memory_order_acquire>(peerFlags_[selfRank_] + getFlagIdx(srcRank, blockIdx.x));
    }
    __syncthreads();
  }

  // The two halves of signalWaitPeer, for pipelined callers that post several pieces. A flag slot holds one token and
  // a post waits until the previous token has been consumed, so every post must be matched by exactly one wait on
  // the receiving rank within the same kernel.
  __device__ __forceinline__ void postPeer(int dstRank, int dstBlock) {
    orderDestWrites();
    __syncthreads();
    if (threadIdx.x == 0) {
      putFlag<std::memory_order_release>(peerFlags_[dstRank] + getFlagIdx(selfRank_, dstBlock));
    }
  }
  __device__ __forceinline__ void waitPeer(int srcRank) {
    if (threadIdx.x == 0) {
      waitFlag<std::memory_order_acquire>(peerFlags_[selfRank_] + getFlagIdx(srcRank, blockIdx.x));
    }
    __syncthreads();
  }

  // Epoch flags of the peer-major push alltoall (own region; the toggle flags are untouched): a post stores a growing
  // value and never waits, a wait polls for at least a value, so every wave can poll one slot. A block's launch uses
  // epoch .. epoch + pieces + 1 (entry posts epoch, piece p posts epoch + 2 + p); epoch is the block's cell, advanced by
  // epochEnd, so it survives CUDA graph replays. Scratch accesses are system-scope: no fence besides orderDestWrites().
  __device__ __forceinline__ FlagType epochBegin() const { return epochCell()[blockIdx.x] + 1; }
  // Must be called by every thread of the block after its last use of the epoch flags.
  __device__ __forceinline__ void epochEnd(FlagType epoch, unsigned pieces) {
    __syncthreads();
    if (threadIdx.x == 0) epochCell()[blockIdx.x] = epoch + pieces + 1;
  }
  // Entry handshake: posts epoch to rank `to` and waits for rank `from` to post at least epoch.
  __device__ __forceinline__ void epochSignalWait(int to, int from, FlagType epoch) {
    orderDestWrites();
    __syncthreads();
    if (threadIdx.x == 0) {
      __scoped_atomic_store_n(peerFlags_[to] + getEpochFlagIdx(selfRank_, blockIdx.x), epoch, __ATOMIC_RELEASE,
                              __MEMORY_SCOPE_SYSTEM);
      const FlagType* flag = epochFlagFrom(from);
      while (static_cast<int32_t>(__scoped_atomic_load_n(flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) - epoch) < 0) {
      }
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
    }
    __syncthreads();
  }
  // Posts value to rank `to` once every store of the block has completed.
  __device__ __forceinline__ void epochPost(int to, FlagType value) {
    orderDestWrites();
    __syncthreads();
    if (threadIdx.x == 0) {
      __scoped_atomic_store_n(peerFlags_[to] + getEpochFlagIdx(selfRank_, blockIdx.x), value, __ATOMIC_RELAXED,
                              __MEMORY_SCOPE_SYSTEM);
    }
  }
  // Piece `piece` of a launch is posted as epoch + 2 + piece. Every thread calls epochWaitPiece and polls the slot
  // itself (`ready` is epochFlagFrom(source)), so a wave continues as soon as the piece is there, with no block-wide sync.
  __device__ __forceinline__ void epochPostPiece(int to, FlagType epoch, unsigned piece) {
    epochPost(to, epoch + 2 + piece);
  }
  __device__ __forceinline__ void epochWaitPiece(const FlagType* ready, FlagType epoch, unsigned piece) {
    const FlagType want = epoch + 2 + piece;
    while (static_cast<int32_t>(__scoped_atomic_load_n(ready, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) - want) < 0) {
    }
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
  }
  // The slot rank `from` posts to from this block index, in this rank's buffer. Every wave polls it.
  __device__ __forceinline__ const FlagType* epochFlagFrom(int from) const {
    return peerFlags_[selfRank_] + getEpochFlagIdx(from, blockIdx.x);
  }

private:
  int nBlocks_{-1};
  int selfRank_{-1};
  int nRanks_{-1};
  FlagType** peerFlags_{nullptr};
  DdaKernelParams params_{};
  // Sticky for the rest of the kernel and uniform across the signaling lanes; see IpcGpuBarrier::aborted_.
  bool aborted_{false};

  __host__ FabricGpuBarrier(int nBlocks, int selfRank, int nRanks, FlagType** peerFlags, DdaKernelParams params)
    : nBlocks_(nBlocks), selfRank_(selfRank), nRanks_(nRanks), peerFlags_(peerFlags), params_(params) {}

  __device__ inline int getFlagIdx(int rank, int block) {
    return block * nRanks_ + rank;
  }

  // Buffer layout: [0, n * b) toggle flags, [n * b, 2 * n * b) epoch flags, then b epoch cells (n ranks, b blocks).
  __device__ inline int getEpochFlagIdx(int rank, int block) const {
    return nBlocks_ * nRanks_ + block * nRanks_ + rank;
  }
  __device__ inline FlagType* epochCell() const {
    return peerFlags_[selfRank_] + 2 * nBlocks_ * nRanks_;
  }
};

} // namespace dda::common
