/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include <cassert>
#include <memory>
#include <vector>

#include "fabric_gpu_barrier.h"

#include "checks.h"
#include "debug.h"

#include <cuda_runtime.h>

namespace meta::comms {

/* static */ __host__
    std::pair<std::unique_ptr<FabricGpuBarrierResources>, FabricGpuBarrier>
    FabricGpuBarrier::mallocAndInit(
        int nRanks,
        int nBlocks,
        int selfRank,
        void* bootstrap,
        struct ncclMemManager* manager) {
  if (nRanks <= 0 || nRanks > kDdaMaxNranks) {
    WARN("FabricGpuBarrier::mallocAndInit: nRanks %d out of range (1..%d)",
         nRanks, kDdaMaxNranks);
    return {nullptr, FabricGpuBarrier{}};
  }

  const size_t flagBytes =
      static_cast<size_t>(nRanks) * nBlocks * sizeof(FlagType);

  // This rank's flag buffer, shared with peers (VMM-backed when available).
  auto selfFlagBuf =
      std::make_unique<DeviceBuffer>(flagBytes, /*useVmm=*/true, manager);
  if (selfFlagBuf == nullptr || selfFlagBuf->get() == nullptr) {
    ERROR("FabricGpuBarrier::mallocAndInit: flag buffer allocation failed");
    return {nullptr, FabricGpuBarrier{}};
  }
  cudaError_t err = cudaMemset(selfFlagBuf->get(), 0, flagBytes);
  if (err != cudaSuccess) {
    WARN("FabricGpuBarrier::mallocAndInit: cudaMemset failed (%s)",
         cudaGetErrorString(err));
    return {nullptr, FabricGpuBarrier{}};
  }

  // Fabric handle export requires a VMM (cuMem) allocation. If DeviceBuffer fell
  // back to a legacy allocation, the fabric path cannot proceed.
  if (!selfFlagBuf->isVmm()) {
    WARN("FabricGpuBarrier::mallocAndInit: flag buffer is not VMM-backed; "
         "fabric handle exchange unavailable");
    return {nullptr, FabricGpuBarrier{}};
  }

  auto memHandler = std::make_unique<ncclFabricMemHandler>(
      bootstrap, selfRank, nRanks, manager);

  ncclResult_t result = memHandler->addSelfDeviceMem(
      selfFlagBuf->get(), selfFlagBuf->vmmHandle(), flagBytes);
  if (result != ncclSuccess && result != ncclInProgress) {
    if (ncclDebugNoWarn == 0) {
      INFO(NCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, result);
    }
    return {nullptr, FabricGpuBarrier{}};
  }
  result = memHandler->exchangeMemPtrs();
  if (result != ncclSuccess && result != ncclInProgress) {
    if (ncclDebugNoWarn == 0) {
      INFO(NCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, result);
    }
    return {nullptr, FabricGpuBarrier{}};
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
      return {nullptr, FabricGpuBarrier{}};
    }
    hostPeerFlags[i] = static_cast<FlagType*>(peerPtr);
  }

  // Stage the pointer table into device memory so the barrier can index it.
  auto peerFlagsDev = std::make_unique<DeviceBuffer>(
      static_cast<size_t>(nRanks) * sizeof(FlagType*));
  if (peerFlagsDev == nullptr || peerFlagsDev->get() == nullptr) {
    ERROR("FabricGpuBarrier::mallocAndInit: peer pointer table allocation failed");
    return {nullptr, FabricGpuBarrier{}};
  }
  err = cudaMemcpy(
      peerFlagsDev->get(),
      hostPeerFlags.data(),
      static_cast<size_t>(nRanks) * sizeof(FlagType*),
      cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    WARN("FabricGpuBarrier::mallocAndInit: cudaMemcpy(table) failed (%s)",
         cudaGetErrorString(err));
    return {nullptr, FabricGpuBarrier{}};
  }

  FabricGpuBarrier barrier(
      nBlocks,
      selfRank,
      nRanks,
      static_cast<FlagType**>(peerFlagsDev->get()));

  auto resources = std::make_unique<FabricGpuBarrierResources>();
  resources->fabricMemHandler = std::move(memHandler);
  resources->selfFlagBuf = std::move(selfFlagBuf);
  resources->peerFlagsDev = std::move(peerFlagsDev);
  return {std::move(resources), barrier};
}

} // namespace meta::comms
