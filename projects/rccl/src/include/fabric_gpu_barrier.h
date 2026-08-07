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

#include "device_buffer.h"
#include "fabric_mem_handler.h"
#include "ipc_gpu_barrier.h"

struct ncclMemManager;

namespace meta::comms {

// Upper bound on the number of ranks the fabric DDA path supports.
constexpr int kDdaMaxNranks = 72;

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
    int nRanks, int nBlocks, int selfRank, void* bootstrap, struct ncclMemManager* manager);

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
    for (int peerRank = threadIdx.x; peerRank < nRanks_; peerRank += blockDim.x) {
      FlagType* peerBuf = peerFlags_[peerRank];

      // Signal the peer that this rank reached the barrier for this block.
      if constexpr (fenceType == MemFenceType::ACQUIRE_ONLY) {
        putFlag<std::memory_order_relaxed>(peerBuf + getFlagIdx(selfRank_, blockIdx.x));
      } else {
        putFlag<std::memory_order_release>(peerBuf + getFlagIdx(selfRank_, blockIdx.x));
      }

      // Wait for the peer's signal in this rank's own buffer.
      if constexpr (fenceType == MemFenceType::RELEASE_ONLY) {
        waitFlag<std::memory_order_relaxed>(selfBuf + getFlagIdx(peerRank, blockIdx.x));
      } else {
        waitFlag<std::memory_order_acquire>(selfBuf + getFlagIdx(peerRank, blockIdx.x));
      }
    }
    if constexpr (hasSubsequentMemAccess) {
      __syncthreads();
    }
  }

private:
  int nBlocks_{-1};
  int selfRank_{-1};
  int nRanks_{-1};
  FlagType** peerFlags_{nullptr};

  __host__ FabricGpuBarrier(int nBlocks, int selfRank, int nRanks, FlagType** peerFlags)
    : nBlocks_(nBlocks), selfRank_(selfRank), nRanks_(nRanks), peerFlags_(peerFlags) {}

  __device__ inline int getFlagIdx(int rank, int block) {
    return block * nRanks_ + rank;
  }
};

} // namespace meta::comms
