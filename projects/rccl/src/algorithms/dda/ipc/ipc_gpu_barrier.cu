/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/IpcGpuBarrier.cu.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "algorithms/dda/ipc/ipc_gpu_barrier.h"

#include "bootstrap.h"
#include "checks.h"
#include "debug.h"

#include <cuda_runtime.h>

namespace dda::common {

__host__ DeviceMailbox::DeviceMailbox(int nRanks, int nBlocks, void* flagsBuf)
  : nBlocks_(nBlocks), flags_(static_cast<FlagType*>(flagsBuf)) {
  // nRanks may be < NRANKS (any 2..NRANKS) when RCCL_DDA_NRANKS_RELAX is set; the flag
  // buffer is always sized/strided for NRANKS so the layout is count-independent.
  assert(nRanks <= NRANKS);
  (void)nRanks;
}

/* static */ __host__ std::pair<std::unique_ptr<DeviceBuffer>, DeviceMailbox> DeviceMailbox::mallocAndInit(
  int nRanks, int nBlocks) {
  assert(nRanks <= NRANKS);
  // Always allocate/stride for NRANKS so getFlagIdx(rank, block) stays valid
  // regardless of the active participant count.
  auto flagBuf = std::make_unique<DeviceBuffer>(NRANKS * nBlocks * sizeof(FlagType));
  if (flagBuf == nullptr) {
    ERROR("DeviceMailbox::mallocAndInit: allocation failed");
    return {nullptr, DeviceMailbox{}};
  }
  cudaError_t err = cudaMemset(flagBuf->get(), 0, NRANKS * nBlocks * sizeof(FlagType));
  if (err != cudaSuccess) {
    WARN("DeviceMailbox::mallocAndInit: cudaMemset failed (%s)", cudaGetErrorString(err));
    return {nullptr, DeviceMailbox{}};
  }
  DeviceMailbox mailbox{nRanks, nBlocks, static_cast<FlagType*>(flagBuf->get())};
  return {std::move(flagBuf), mailbox};
}

__host__ IpcGpuBarrier::IpcGpuBarrier(int nRanks, int nBlocks, int selfRank,
                                      const std::array<DeviceMailbox, NRANKS>& allMailboxes, DdaKernelParams params)
  : nBlocks_(nBlocks), selfRank_(selfRank), nRanks_(nRanks), allMailboxes_(allMailboxes), params_(params) {
  assert(nRanks <= NRANKS);
}

// True only if every rank reports localReady. Each rank contributes one byte to
// a bootstrap all-gather, so every rank reaches the same answer.
static bool barrierAllRanksReady(void* bootstrap, int nRanks, int selfRank, bool localReady) {
  std::vector<uint8_t> ready(static_cast<size_t>(nRanks), 0);
  ready[static_cast<size_t>(selfRank)] = localReady ? 1 : 0;
  if (bootstrapAllGather(bootstrap, ready.data(), sizeof(uint8_t)) != ncclSuccess) {
    WARN("IpcGpuBarrier::mallocAndInit: readiness all-gather failed");
    return false;
  }
  for (int i = 0; i < nRanks; ++i) {
    if (ready[static_cast<size_t>(i)] == 0) {
      return false;
    }
  }
  return true;
}

/* static */ __host__ std::pair<std::unique_ptr<IpcGpuBarrierResources>, IpcGpuBarrier> IpcGpuBarrier::mallocAndInit(
  int nRanks, int nBlocks, int selfRank, void* bootstrap, DdaKernelParams params) {
  assert(nRanks <= NRANKS);
  // exchangeMemPtrs() below is a bootstrap all-gather, so a rank that bailed out
  // on its own before it would leave every peer blocked there. Do all the
  // rank-local work first -- including exporting the mailbox's IPC handle,
  // which exchangeMemPtrs() would otherwise fail on just before its all-gather
  // -- and agree on the outcome before anyone enters the exchange.
  auto selfAlloc = DeviceMailbox::mallocAndInit(nRanks, nBlocks);
  auto& selfMboxBuf = selfAlloc.first;
  auto& selfMbox = selfAlloc.second;
  bool localReady = selfMboxBuf != nullptr && selfMboxBuf->get() != nullptr;
  if (localReady) {
    cudaIpcMemHandle_t probeHandle;
    cudaError_t err = cudaIpcGetMemHandle(&probeHandle, selfMboxBuf->get());
    if (err != cudaSuccess) {
      WARN("IpcGpuBarrier::mallocAndInit: cudaIpcGetMemHandle failed (%s)", cudaGetErrorString(err));
      localReady = false;
    }
  }
  if (!barrierAllRanksReady(bootstrap, nRanks, selfRank, localReady)) {
    return {nullptr, IpcGpuBarrier{}};
  }

  auto memHandler = std::make_unique<ncclIpcMemHandler>(bootstrap, selfRank, nRanks);

  ncclResult_t result = memHandler->addSelfDeviceMemPtr(selfMboxBuf->get());
  if (result != ncclSuccess && result != ncclInProgress) {
    if (ncclDebugNoWarn == 0) {
      INFO(NCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, result);
    }
    return {nullptr, IpcGpuBarrier{}};
  }
  result = memHandler->exchangeMemPtrs();
  if (result != ncclSuccess && result != ncclInProgress) {
    if (ncclDebugNoWarn == 0) {
      INFO(NCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, result);
    }
    return {nullptr, IpcGpuBarrier{}};
  }

  std::array<DeviceMailbox, NRANKS> allMailboxes;
  for (int i = 0; i < nRanks; i++) {
    if (i == selfRank) {
      allMailboxes[i] = selfMbox;
    } else {
      void* peerPtr = nullptr;
      result = memHandler->getPeerDeviceMemPtr(i, &peerPtr);

      if (result != ncclSuccess && result != ncclInProgress) {
        if (ncclDebugNoWarn == 0) {
          INFO(NCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, result);
        }
        return {nullptr, IpcGpuBarrier{}};
      }
      allMailboxes[i] = DeviceMailbox(nRanks, nBlocks, peerPtr);
    }
  }

  IpcGpuBarrier barrier(nRanks, nBlocks, selfRank, allMailboxes, params);

  auto resources = std::make_unique<IpcGpuBarrierResources>();
  resources->ipcMemHandler = std::move(memHandler);
  resources->selfMailboxBuf = std::move(selfMboxBuf);
  return {std::move(resources), barrier};
}

} // namespace dda::common
