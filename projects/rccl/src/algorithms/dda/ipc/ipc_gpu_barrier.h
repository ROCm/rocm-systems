/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/IpcGpuBarrier.cuh.
 * Bootstrap/IPC adapted to ncclIpcMemHandler (void* bootstrap).
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cuda.h>
#include <cstdint>
#include <memory>

#include "algorithms/dda/device/device_buffer.h"
#include "algorithms/dda/ipc/ipc_mem_handler.h"

namespace dda::common {

namespace {

template <std::memory_order Sem>
__device__ __forceinline__ uint32_t cas(uint32_t* addr, uint32_t compare, uint32_t val) {
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> ref(*addr);
  ref.compare_exchange_strong(compare, val, ::cuda::std::memory_order(Sem));
  return compare;
#elif defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
  __atomic_compare_exchange_n(addr, &compare, val, false, static_cast<int>(Sem), __ATOMIC_RELAXED);
  return compare;
#endif
}

template <std::memory_order Sem>
__device__ __forceinline__ void putFlag(uint32_t* addr) {
  while (cas<Sem>(addr, 0, 1) != 0);
}

template <std::memory_order Sem>
__device__ __forceinline__ void waitFlag(uint32_t* addr) {
  while (cas<Sem>(addr, 1, 0) != 1);
}

} // namespace

// Default/baseline clique size for the DDA IPC path: the 8-GPU SPX case, and the
// size the DDA unit tests build their fake comms around. Neither the barrier nor
// the kernels are limited to it -- the dispatch gates accept 2..kDdaMaxNranks and
// specialize the kernels per clique size, falling back to a runtime rank count.
constexpr int NRANKS = 8;

// Upper bound on the number of ranks the DDA IPC and fabric paths support.
// Lives here rather than in fabric_gpu_barrier.h because that header includes
// this one (for putFlag/waitFlag), so both paths can share a single definition.
constexpr int kDdaMaxNranks = 72;

class IpcGpuBarrier;

struct IpcGpuBarrierResources {
  std::unique_ptr<ncclIpcMemHandler> ipcMemHandler;
  // This rank's own flag buffer, exported to peers over IPC.
  std::unique_ptr<DeviceBuffer> selfFlagBuf;
  // Device-resident array of nRanks flag-buffer pointers (FlagType*[nRanks]).
  std::unique_ptr<DeviceBuffer> peerFlagsDev;
};

class IpcGpuBarrier {
public:
  using FlagType = uint32_t;
  __host__ IpcGpuBarrier() = default;

  static __host__ std::pair<std::unique_ptr<IpcGpuBarrierResources>, IpcGpuBarrier> mallocAndInit(
    int nRanks, int nBlocks, int selfRank, void* bootstrap);

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

  __host__ IpcGpuBarrier(int nBlocks, int selfRank, int nRanks, FlagType** peerFlags)
    : nBlocks_(nBlocks), selfRank_(selfRank), nRanks_(nRanks), peerFlags_(peerFlags) {}

  __device__ inline int getFlagIdx(int rank, int block) {
    return block * nRanks_ + rank;
  }
};

} // namespace dda::common
