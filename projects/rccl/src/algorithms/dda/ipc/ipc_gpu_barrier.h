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

#include "algorithms/dda/device/dda_kernel_params.h"
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
__device__ __forceinline__ bool putFlag(uint32_t* addr, DdaKernelParams params) {
  int spins = 0;
  while (cas<Sem>(addr, 0, 1) != 0) {
    if (params.abortSpinTick(spins)) return false;
  }
  return true;
}

template <std::memory_order Sem>
__device__ __forceinline__ bool waitFlag(uint32_t* addr, DdaKernelParams params) {
  int spins = 0;
  while (cas<Sem>(addr, 1, 0) != 1) {
    if (params.abortSpinTick(spins)) return false;
  }
  return true;
}

constexpr int NRANKS = 8;

} // namespace

class DeviceMailbox {
public:
  using FlagType = uint32_t;
  static __host__ std::pair<std::unique_ptr<DeviceBuffer>, DeviceMailbox> mallocAndInit(int nRanks, int nBlocks);

  DeviceMailbox() = default;

  __host__ DeviceMailbox(int nRanks, int nBlocks, void* flagsBuf);

  __device__ inline bool setFlagNoMemFence(int senderRank, int senderBlock, DdaKernelParams params) {
    return putFlag<std::memory_order_relaxed>(flags_ + getFlagIdx(senderRank, senderBlock), params);
  }

  __device__ inline bool waitFlagNoMemFence(int senderRank, int senderBlock, DdaKernelParams params) {
    return waitFlag<std::memory_order_relaxed>(flags_ + getFlagIdx(senderRank, senderBlock), params);
  }

  __device__ inline bool setFlagWithMemFence(int senderRank, int senderBlock, DdaKernelParams params) {
    return putFlag<std::memory_order_release>(flags_ + getFlagIdx(senderRank, senderBlock), params);
  }

  __device__ inline bool waitFlagWithMemFence(int senderRank, int senderBlock, DdaKernelParams params) {
    return waitFlag<std::memory_order_acquire>(flags_ + getFlagIdx(senderRank, senderBlock), params);
  }

private:
  // Initialized in-class so a default-constructed mailbox is deterministic.
  // IpcGpuBarrier holds a std::array<DeviceMailbox, NRANKS> but fills only the
  // first nRanks entries when RCCL_DDA_NRANKS_RELAX admits a smaller clique;
  // without these the unused tail would carry indeterminate values into the
  // kernel argument.
  int nBlocks_{0};
  FlagType* flags_{nullptr};

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

  static __host__ std::pair<std::unique_ptr<IpcGpuBarrierResources>, IpcGpuBarrier> mallocAndInit(
    int nRanks, int nBlocks, int selfRank, void* bootstrap, DdaKernelParams params);

  // Active-peer bound: kNRanks when > 0 (the calling kernel's compile-time clique
  // size; must equal the nRanks this barrier was built with), else nRanksRt (the
  // NRANKS == 0 runtime kernel under RCCL_DDA_NRANKS_RELAX). Deliberately not a
  // member: the kernel keeps its by-value barrier copy in scratch, so a member
  // read here is a scratch load + vmcnt(0) wait on every barrier, and growing
  // the struct also slows the full-clique kernels (measured ~2-3% busbw at
  // 64K-1M, 8 ranks, MI355X).
  template <bool hasPreviousMemAccess, bool hasSubsequentMemAccess, int kNRanks = 0>
  __device__ __forceinline__ void syncOnSameBlockIdx(int nRanksRt = NRANKS) {
    static_assert(kNRanks >= 0 && kNRanks <= NRANKS, "kNRanks out of range");
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
    const int activeRanks = (kNRanks > 0) ? kNRanks : nRanksRt;
    if (threadIdx.x < activeRanks && !aborted_) {  // active peers only; NRANKS would deadlock for <8
      auto peerRank = threadIdx.x;
      bool signaled = false;
      if constexpr (fenceType == MemFenceType::ACQUIRE_ONLY) {
        signaled = allMailboxes_[peerRank].setFlagNoMemFence(selfRank_, blockIdx.x, params_);
      } else {
        signaled = allMailboxes_[peerRank].setFlagWithMemFence(selfRank_, blockIdx.x, params_);
      }

      bool arrived = false;
      if (signaled) {
        if constexpr (fenceType == MemFenceType::RELEASE_ONLY) {
          arrived = allMailboxes_[selfRank_].waitFlagNoMemFence(peerRank, blockIdx.x, params_);
        } else {
          arrived = allMailboxes_[selfRank_].waitFlagWithMemFence(peerRank, blockIdx.x, params_);
        }
      }
      aborted_ = __any(!arrived);
    }
    if constexpr (hasSubsequentMemAccess) {
      __syncthreads();
    }
  }

private:
  int nBlocks_{-1};
  int selfRank_{-1};
  std::array<DeviceMailbox, NRANKS> allMailboxes_;
  DdaKernelParams params_{};
  // Sticky for the rest of the kernel and uniform across the signaling lanes,
  // which all sit in wave 0: once any of them gives up a barrier on abort, the
  // block signals no later barrier, so a late peer cannot pair its earlier flag
  // with this block's next one and read data this rank never finished producing.
  bool aborted_{false};

  __host__ IpcGpuBarrier(int nRanks, int nBlocks, int selfRank, const std::array<DeviceMailbox, NRANKS>& allMailboxes,
                         DdaKernelParams params);
};

} // namespace dda::common
