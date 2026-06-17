/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/IpcGpuBarrier.cu.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include <cassert>
#include <memory>

#include "ipc_gpu_barrier.h"

#include "checks.h"
#include "debug.h"

#include <cuda_runtime.h>

namespace meta::comms {

__host__ DeviceMailbox::DeviceMailbox(int nRanks, int nBlocks, void* flagsBuf)
    : nBlocks_(nBlocks), flags_(static_cast<FlagType*>(flagsBuf)) {
  assert(nRanks == NRANKS);
}

/* static */ __host__ std::pair<std::unique_ptr<DeviceBuffer>, DeviceMailbox>
DeviceMailbox::mallocAndInit(int nRanks, int nBlocks) {
  assert(nRanks == NRANKS);
  auto flagBuf =
      std::make_unique<DeviceBuffer>(nRanks * nBlocks * sizeof(FlagType));
  if (flagBuf == nullptr) {
    ERROR("DeviceMailbox::mallocAndInit: allocation failed");
    return {nullptr, DeviceMailbox{}};
  }
  cudaError_t err = cudaMemset(
      flagBuf->get(), 0, nRanks * nBlocks * sizeof(FlagType));
  if (err != cudaSuccess) {
    WARN("DeviceMailbox::mallocAndInit: cudaMemset failed (%s)",
         cudaGetErrorString(err));
    return {nullptr, DeviceMailbox{}};
  }
  DeviceMailbox mailbox{
      nRanks, nBlocks, static_cast<FlagType*>(flagBuf->get())};
  return {std::move(flagBuf), mailbox};
}

__host__ IpcGpuBarrier::IpcGpuBarrier(
    int nRanks,
    int nBlocks,
    int selfRank,
    const std::array<DeviceMailbox, NRANKS>& allMailboxes)
    : nBlocks_(nBlocks), selfRank_(selfRank), allMailboxes_(allMailboxes) {
  assert(nRanks == NRANKS);
}

/* static */ __host__
    std::pair<std::unique_ptr<IpcGpuBarrierResources>, IpcGpuBarrier>
    IpcGpuBarrier::mallocAndInit(
        int nRanks,
        int nBlocks,
        int selfRank,
        void* bootstrap) {
  assert(nRanks == NRANKS);
  auto selfAlloc = DeviceMailbox::mallocAndInit(nRanks, nBlocks);
  auto& selfMboxBuf = selfAlloc.first;
  auto& selfMbox = selfAlloc.second;
  if (!selfMboxBuf) {
    return {nullptr, IpcGpuBarrier{}};
  }

  auto memHandler =
      std::make_unique<ncclIpcMemHandler>(bootstrap, selfRank, nRanks);
  
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

  IpcGpuBarrier barrier(nRanks, nBlocks, selfRank, allMailboxes);

  auto resources = std::make_unique<IpcGpuBarrierResources>();
  resources->ipcMemHandler = std::move(memHandler);
  resources->selfMailboxBuf = std::move(selfMboxBuf);
  return {std::move(resources), barrier};

}

} // namespace meta::comms
