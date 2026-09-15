/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gin/gin_fabric_a2a_host.h"
#include "alloc.h"
#include "bootstrap.h"
#include "gin/gin_bootstrap_agree.h"
#include "checks.h"
#include "comm.h"
#include "nccl_device/gin/anvil_sdma/gin_fabric_ll_policy.h"
#include "nccl_device/impl/comm__types.h"

#include <hip/hip_runtime.h>

#include <cstring>
#include <map>
#include <mutex>

static std::mutex ginFabricA2ALaneMutex;
static std::map<void*, ncclGinFabricA2ALane> ginFabricA2ALanes;

void ncclGinFabricA2ALanePublish(void* ginHandle, ncclGinFabricA2ALane const& lane) {
  if (ginHandle == nullptr) return;
  std::lock_guard<std::mutex> lock(ginFabricA2ALaneMutex);
  ginFabricA2ALanes[ginHandle] = lane;
}

void ncclGinFabricA2ALaneErase(void* ginHandle) {
  if (ginHandle == nullptr) return;
  std::lock_guard<std::mutex> lock(ginFabricA2ALaneMutex);
  ginFabricA2ALanes.erase(ginHandle);
}

void ncclGinFabricA2ALaneClearAll() {
  std::lock_guard<std::mutex> lock(ginFabricA2ALaneMutex);
  ginFabricA2ALanes.clear();
}

ncclResult_t ncclGinFabricA2ALaneAgreeEnabled(struct ncclComm* comm, int localEnabled, int* allEnabled) {
  if (allEnabled == nullptr) return ncclInvalidArgument;
  // Cleared up front so a failed allgather never leaves a stale 1 behind; the
  // shared helper writes the reduction only when the allgather succeeded.
  *allEnabled = 0;
  return ncclGinBootstrapAgreeAll(comm, localEnabled, allEnabled);
}

ncclResult_t ginFabricA2ALaneBuildPeerScratchDev(struct ncclComm* comm, void*** outPeerDev, size_t* outRegionBytes) {
  if (outPeerDev == nullptr || outRegionBytes == nullptr) return ncclInvalidArgument;
  *outPeerDev = nullptr;
  *outRegionBytes = 0;
  if (comm == nullptr || comm->nRanks < 2 || comm->ddaPeerPtrsHost == nullptr) return ncclInvalidArgument;

  const size_t ginRegion = gin::fabric::ginFabricLlA2AGinRegionBytes(
      comm->nRanks, gin::fabric::resolveGinFabricLLThresholdAlltoAll());
  const size_t allocBytes = comm->ddaScratchAllocBytes ? comm->ddaScratchAllocBytes : comm->ddaScratchBytes;
  const size_t carveOff = gin::fabric::ginFabricLlA2ACarveOffset(allocBytes, ginRegion);
  if (ginRegion == 0 || carveOff == 0 || allocBytes < carveOff + ginRegion) return ncclInvalidArgument;

  void** hostPtrs = nullptr;
  NCCLCHECK(ncclCalloc(&hostPtrs, comm->nRanks));
  for (int i = 0; i < comm->nRanks; ++i) {
    hostPtrs[i] = reinterpret_cast<char*>(comm->ddaPeerPtrsHost[i]) + carveOff;
  }

  void** devPtrs = nullptr;
  ncclResult_t ret = ncclSuccess;
  CUDACHECKGOTO(hipMalloc(&devPtrs, comm->nRanks * sizeof(void*)), ret, fail);
  CUDACHECKGOTO(hipMemcpy(devPtrs, hostPtrs, comm->nRanks * sizeof(void*), hipMemcpyHostToDevice), ret, fail);
  free(hostPtrs);
  *outPeerDev = devPtrs;
  *outRegionBytes = ginRegion;
  return ncclSuccess;

fail:
  if (devPtrs) CUDACHECKIGNORE(hipFree(devPtrs));
  free(hostPtrs);
  return ret;
}

extern "C" __attribute__((visibility("default")))
ncclResult_t ncclGinQueryFabricA2ALane(struct ncclDevComm const* devComm,
                                       struct ncclGinFabricA2ALane* out) {
  if (out == nullptr) return ncclInvalidArgument;
  memset(out, 0, sizeof(*out));
  if (devComm == nullptr || devComm->ginHandles[0] == nullptr) return ncclSuccess;
  std::lock_guard<std::mutex> lock(ginFabricA2ALaneMutex);
  auto it = ginFabricA2ALanes.find(devComm->ginHandles[0]);
  if (it != ginFabricA2ALanes.end()) *out = it->second;
  return ncclSuccess;
}
