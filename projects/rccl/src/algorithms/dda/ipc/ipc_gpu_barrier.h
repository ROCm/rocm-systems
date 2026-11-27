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

// Staging bank control words in a small zeroed buffer of the owning rank, never IPC-exported: call count, arrivals.
constexpr int kIpcStagingBankCallWord = 0;
constexpr int kIpcStagingBankArrivedWord = 1;
constexpr size_t kIpcStagingBankCtlBytes = 64;
static_assert(kIpcStagingBankCtlBytes >= (kIpcStagingBankArrivedWord + 1) * sizeof(uint32_t));

__host__ __device__ constexpr int ipcStagingBankOfCall(uint32_t call) {
  return static_cast<int>(call & 1u);
}

// The last block to reach a banked call's first barrier bumps the call counter; every block has read it by then.
__host__ __device__ constexpr bool ipcStagingBankLastArrival(uint32_t arrivedBefore, uint32_t nBlocks) {
  return arrivedBefore + 1 == nBlocks;
}

class DeviceMailbox {
public:
  using FlagType = uint32_t;
  static __host__ std::pair<std::unique_ptr<DeviceBuffer>, DeviceMailbox> mallocAndInit(int nRanks, int nBlocks);

  DeviceMailbox() = default;

  __host__ DeviceMailbox(int nRanks, int nBlocks, void* flagsBuf);

  __device__ inline bool setFlagNoMemFence(int senderRank, int senderBlock, DdaKernelParams params) const {
    return putFlag<std::memory_order_relaxed>(flags_ + getFlagIdx(senderRank, senderBlock), params);
  }

  __device__ inline bool waitFlagNoMemFence(int senderRank, int senderBlock, DdaKernelParams params) const {
    return waitFlag<std::memory_order_relaxed>(flags_ + getFlagIdx(senderRank, senderBlock), params);
  }

  __device__ inline bool setFlagWithMemFence(int senderRank, int senderBlock, DdaKernelParams params) const {
    return putFlag<std::memory_order_release>(flags_ + getFlagIdx(senderRank, senderBlock), params);
  }

  __device__ inline bool waitFlagWithMemFence(int senderRank, int senderBlock, DdaKernelParams params) const {
    return waitFlag<std::memory_order_acquire>(flags_ + getFlagIdx(senderRank, senderBlock), params);
  }

private:
  int nBlocks_;
  FlagType* flags_;

  __device__ inline int getFlagIdx(int rank, int block) const {
    return block * NRANKS + rank;
  }
};

class IpcGpuBarrier;

struct IpcGpuBarrierResources {
  std::unique_ptr<ncclIpcMemHandler> ipcMemHandler;
  std::unique_ptr<DeviceBuffer> selfMailboxBuf;
  std::unique_ptr<DeviceBuffer> stagingBankCtlBuf;
};

class IpcGpuBarrier {
public:
  using FlagType = DeviceMailbox::FlagType;
  __host__ IpcGpuBarrier() = default;

  static __host__ std::pair<std::unique_ptr<IpcGpuBarrierResources>, IpcGpuBarrier> mallocAndInit(
    int nRanks, int nBlocks, int selfRank, void* bootstrap, DdaKernelParams params);

  // `aborted` is the caller-held sticky abort state of this block: false at kernel
  // start and passed to every barrier of the kernel. Once any signaling lane gives
  // up a barrier on abort, the block signals no later barrier, so a late peer cannot
  // pair its earlier flag with this block's next one and read data this rank never
  // finished producing. The signaling lanes all sit in wave 0, so it is uniform
  // across them.
  //
  // It is held by the caller, not by this object, on purpose. Kernels take the
  // barrier by value; writing a member of a by-value kernel argument makes the
  // compiler copy the whole object (mailbox table included) to per-lane scratch at
  // every launch, because allMailboxes_ is indexed per lane. Keeping the barrier
  // const lets it be read straight from the kernarg segment.
  //
  // kStagingBankArrive (first barrier of a banked call): lane 0 counts the block in after every wave read the call
  // counter; the last block bumps it.
  template <bool hasPreviousMemAccess, bool hasSubsequentMemAccess, bool kStagingBankArrive = false>
  __device__ __forceinline__ void syncOnSameBlockIdx(bool& aborted) const {
    enum class MemFenceType {
      RELEASE_ACQUIRE,
      RELEASE_ONLY,
      ACQUIRE_ONLY,
    };

    static_assert(hasPreviousMemAccess || hasSubsequentMemAccess);
    static_assert(!kStagingBankArrive || hasPreviousMemAccess, "the arrival must follow every wave's counter read");

    constexpr MemFenceType fenceType =
      hasPreviousMemAccess && hasSubsequentMemAccess ?
        MemFenceType::RELEASE_ACQUIRE :
        (!hasPreviousMemAccess ? MemFenceType::ACQUIRE_ONLY : MemFenceType::RELEASE_ONLY);

    if constexpr (hasPreviousMemAccess) {
      __syncthreads();
    }
    uint32_t arrivedBefore = 0;
    if (threadIdx.x < NRANKS && !aborted) {
      auto peerRank = threadIdx.x;
      bool signaled = false;
      if constexpr (fenceType == MemFenceType::ACQUIRE_ONLY) {
        signaled = allMailboxes_[peerRank].setFlagNoMemFence(selfRank_, blockIdx.x, params_);
      } else {
        signaled = allMailboxes_[peerRank].setFlagWithMemFence(selfRank_, blockIdx.x, params_);
      }
      if constexpr (kStagingBankArrive) {
        if (peerRank == 0) {
          // Relaxed: each used counter read fed a staging store before the barrier, so it returned before this arrival.
          arrivedBefore = __scoped_atomic_fetch_add(stagingBankCtl_ + kIpcStagingBankArrivedWord, 1u, __ATOMIC_RELAXED,
                                                    __MEMORY_SCOPE_DEVICE);
        }
      }

      bool arrived = false;
      if (signaled) {
        if constexpr (fenceType == MemFenceType::RELEASE_ONLY) {
          arrived = allMailboxes_[selfRank_].waitFlagNoMemFence(peerRank, blockIdx.x, params_);
        } else {
          arrived = allMailboxes_[selfRank_].waitFlagWithMemFence(peerRank, blockIdx.x, params_);
        }
      }
      aborted = __any(!arrived);
    }
    if constexpr (kStagingBankArrive) {
      if (threadIdx.x == 0 && ipcStagingBankLastArrival(arrivedBefore, gridDim.x)) {
        // RMWs keep an overlapping call's arrivals. Two banked kernels in flight on a comm break its banks for good.
        FlagType* const ctl = stagingBankCtl_;
        __scoped_atomic_fetch_sub(ctl + kIpcStagingBankArrivedWord, gridDim.x, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
        __scoped_atomic_fetch_add(ctl + kIpcStagingBankCallWord, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      }
    }
    if constexpr (hasSubsequentMemAccess) {
      __syncthreads();
    }
  }

  // Banked: bufs holds staging bank 0's peer pointers, then bank 1's; the call counter picks this call's bank.
  template <bool kBanked, typename T>
  __device__ __forceinline__ T* const* stagingBuffers(T* const* bufs) const {
    if constexpr (kBanked) {
      const uint32_t call = __scoped_atomic_load_n(stagingBankCtl_ + kIpcStagingBankCallWord, __ATOMIC_RELAXED,
                                                   __MEMORY_SCOPE_DEVICE);
      return bufs + NRANKS * ipcStagingBankOfCall(call);
    } else {
      return bufs;
    }
  }

  // Ends an in-kernel staged call. A banked call skips it: its bank is next written two banked calls later, past
  // barrier 1 of the one between.
  template <bool kBanked>
  __device__ __forceinline__ void syncAfterPeerReads(bool& aborted) const {
    if constexpr (!kBanked) {
      syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>(aborted);
    }
  }

private:
  int nBlocks_{-1};
  int selfRank_{-1};
  std::array<DeviceMailbox, NRANKS> allMailboxes_;
  DdaKernelParams params_{};
  FlagType* stagingBankCtl_{nullptr};

  __host__ IpcGpuBarrier(int nRanks, int nBlocks, int selfRank, const std::array<DeviceMailbox, NRANKS>& allMailboxes,
                         DdaKernelParams params, FlagType* stagingBankCtl);
};

} // namespace dda::common
