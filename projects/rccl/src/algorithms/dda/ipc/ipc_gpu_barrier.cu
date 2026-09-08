/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/IpcGpuBarrier.cu.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include <memory>
#include <vector>

#include "algorithms/dda/ipc/ipc_gpu_barrier.h"

#include "checks.h"
#include "debug.h"

#include <cuda_runtime.h>

namespace dda::common {

/* static */ __host__ std::pair<std::unique_ptr<IpcGpuBarrierResources>, IpcGpuBarrier> IpcGpuBarrier::mallocAndInit(
  int nRanks, int nBlocks, int selfRank, void* bootstrap) {
  if (nRanks <= 0 || nRanks > kDdaMaxNranks) {
    WARN("IpcGpuBarrier::mallocAndInit: nRanks %d out of range (1..%d)", nRanks, kDdaMaxNranks);
    return {nullptr, IpcGpuBarrier{}};
  }

  const size_t flagBytes = static_cast<size_t>(nRanks) * nBlocks * sizeof(FlagType);

  // This rank's flag buffer, exported to peers over IPC. Must stay a legacy
  // fine-grained allocation (DeviceBuffer's default) so cudaIpcGetMemHandle can
  // export it; the fabric path is the one that needs VMM here.
  auto selfFlagBuf = std::make_unique<DeviceBuffer>(flagBytes);
  if (selfFlagBuf == nullptr || selfFlagBuf->get() == nullptr) {
    ERROR("IpcGpuBarrier::mallocAndInit: flag buffer allocation failed");
    return {nullptr, IpcGpuBarrier{}};
  }
  cudaError_t err = cudaMemset(selfFlagBuf->get(), 0, flagBytes);
  if (err != cudaSuccess) {
    WARN("IpcGpuBarrier::mallocAndInit: cudaMemset failed (%s)", cudaGetErrorString(err));
    return {nullptr, IpcGpuBarrier{}};
  }

  auto memHandler = std::make_unique<ncclIpcMemHandler>(bootstrap, selfRank, nRanks);

  ncclResult_t result = memHandler->addSelfDeviceMemPtr(selfFlagBuf->get());
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

  // Gather every rank's flag-buffer device pointer.
  std::vector<FlagType*> hostPeerFlags(nRanks, nullptr);
  for (int i = 0; i < nRanks; i++) {
    if (i == selfRank) {
      hostPeerFlags[i] = static_cast<FlagType*>(selfFlagBuf->get());
      continue;
    }
    void* peerPtr = nullptr;
    result = memHandler->getPeerDeviceMemPtr(i, &peerPtr);
    if (result != ncclSuccess && result != ncclInProgress) {
      if (ncclDebugNoWarn == 0) {
        INFO(NCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, result);
      }
      return {nullptr, IpcGpuBarrier{}};
    }
    hostPeerFlags[i] = static_cast<FlagType*>(peerPtr);
  }

  // Stage the pointer table into device memory so the barrier can index it.
  auto peerFlagsDev = std::make_unique<DeviceBuffer>(static_cast<size_t>(nRanks) * sizeof(FlagType*));
  if (peerFlagsDev == nullptr || peerFlagsDev->get() == nullptr) {
    ERROR("IpcGpuBarrier::mallocAndInit: peer pointer table allocation failed");
    return {nullptr, IpcGpuBarrier{}};
  }
  err = cudaMemcpy(peerFlagsDev->get(), hostPeerFlags.data(), static_cast<size_t>(nRanks) * sizeof(FlagType*),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    WARN("IpcGpuBarrier::mallocAndInit: cudaMemcpy(table) failed (%s)", cudaGetErrorString(err));
    return {nullptr, IpcGpuBarrier{}};
  }

  IpcGpuBarrier barrier(nBlocks, selfRank, nRanks, static_cast<FlagType**>(peerFlagsDev->get()));

  auto resources = std::make_unique<IpcGpuBarrierResources>();
  resources->ipcMemHandler = std::move(memHandler);
  resources->selfFlagBuf = std::move(selfFlagBuf);
  resources->peerFlagsDev = std::move(peerFlagsDev);
  return {std::move(resources), barrier};
}

} // namespace dda::common
