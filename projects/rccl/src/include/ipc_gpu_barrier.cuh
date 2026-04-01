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
#include <array>
#include <memory>

#include "device_buffer.h"
#include "ipc_mem_handler.h"

namespace meta::comms {

namespace {

template <std::memory_order Sem>
__device__ __forceinline__ uint32_t
cas(uint32_t* addr, uint32_t compare, uint32_t val) {
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> ref(*addr);
  ref.compare_exchange_strong(compare, val, ::cuda::std::memory_order(Sem));
  return compare;
#elif defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
  __atomic_compare_exchange_n(
      addr, &compare, val, false, static_cast<int>(Sem), __ATOMIC_RELAXED);
  return compare;
#endif
}

template <std::memory_order Sem>
__device__ __forceinline__ void putFlag(uint32_t* addr) {
  while (cas<Sem>(addr, 0, 1) != 0)
    ;
}

template <std::memory_order Sem>
__device__ __forceinline__ void waitFlag(uint32_t* addr) {
  while (cas<Sem>(addr, 1, 0) != 1)
    ;
}

constexpr int NRANKS = 8;

} // namespace

class DeviceMailbox {
 public:
  using FlagType = uint32_t;
  static __host__ std::pair<std::unique_ptr<DeviceBuffer>, DeviceMailbox>
  mallocAndInit(int nRanks, int nBlocks);

  DeviceMailbox() = default;

  __host__ DeviceMailbox(int nRanks, int nBlocks, void* flagsBuf);

  __device__ inline void setFlagNoMemFence(int senderRank, int senderBlock) {
    putFlag<std::memory_order_relaxed>(
        flags_ + getFlagIdx(senderRank, senderBlock));
  }

  __device__ inline void waitFlagNoMemFence(int senderRank, int senderBlock) {
    waitFlag<std::memory_order_relaxed>(
        flags_ + getFlagIdx(senderRank, senderBlock));
  }

  __device__ inline void setFlagWithMemFence(int senderRank, int senderBlock) {
    putFlag<std::memory_order_release>(
        flags_ + getFlagIdx(senderRank, senderBlock));
  }

  __device__ inline void waitFlagWithMemFence(int senderRank, int senderBlock) {
    waitFlag<std::memory_order_acquire>(
        flags_ + getFlagIdx(senderRank, senderBlock));
  }

 private:
  int nBlocks_;
  FlagType* flags_;

  __device__ inline int getFlagIdx(int rank, int block) {
    return block * NRANKS + rank;
  }
};

class IpcGpuBarrier;

struct IpcGpuBarrierResources {
  std::unique_ptr<ncclIpcMemHandler> ipcMemHandler;
  std::unique_ptr<DeviceBuffer> selfMailboxBuf;
};

class IpcGpuBarrier {
 public:
  using FlagType = DeviceMailbox::FlagType;
  __host__ IpcGpuBarrier() = default;

  static __host__
      std::pair<std::unique_ptr<IpcGpuBarrierResources>, IpcGpuBarrier>
      mallocAndInit(
          int nRanks,
          int nBlocks,
          int selfRank,
          void* bootstrap);

  template <bool hasPreviousMemAccess, bool hasSubsequentMemAccess>
  __device__ __forceinline__ void syncOnSameBlockIdx() {
    enum class MemFenceType {
      RELEASE_ACQUIRE,
      RELEASE_ONLY,
      ACQUIRE_ONLY,
    };

    static_assert(hasPreviousMemAccess || hasSubsequentMemAccess);

    constexpr MemFenceType fenceType =
        hasPreviousMemAccess && hasSubsequentMemAccess
        ? MemFenceType::RELEASE_ACQUIRE
        : (!hasPreviousMemAccess ? MemFenceType::ACQUIRE_ONLY
                                 : MemFenceType::RELEASE_ONLY);

    if constexpr (hasPreviousMemAccess) {
      __syncthreads();
    }
    if (threadIdx.x < NRANKS) {
      auto peerRank = threadIdx.x;
      if constexpr (fenceType == MemFenceType::ACQUIRE_ONLY) {
        allMailboxes_[peerRank].setFlagNoMemFence(selfRank_, blockIdx.x);
      } else {
        allMailboxes_[peerRank].setFlagWithMemFence(selfRank_, blockIdx.x);
      }

      if constexpr (fenceType == MemFenceType::RELEASE_ONLY) {
        allMailboxes_[selfRank_].waitFlagNoMemFence(peerRank, blockIdx.x);
      } else {
        allMailboxes_[selfRank_].waitFlagWithMemFence(peerRank, blockIdx.x);
      }
    }
    if constexpr (hasSubsequentMemAccess) {
      __syncthreads();
    }
  }

 private:
  int nBlocks_{-1};
  int selfRank_{-1};
  std::array<DeviceMailbox, NRANKS> allMailboxes_;

  __host__ IpcGpuBarrier(
      int nRanks,
      int nBlocks,
      int selfRank,
      const std::array<DeviceMailbox, NRANKS>& allMailboxes);
};

} // namespace meta::comms
