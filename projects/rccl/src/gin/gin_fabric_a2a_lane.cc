/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gin/gin_fabric_a2a_host.h"
#include "alloc.h"
#include "bootstrap.h"
#include "comm.h"
#include "nccl_device/impl/comm__types.h"

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
  *allEnabled = 0;
  if (comm == nullptr || comm->nRanks < 1 || comm->rank < 0 || comm->rank >= comm->nRanks) {
    return ncclInvalidArgument;
  }
  int* oks = nullptr;
  NCCLCHECK(ncclCalloc(&oks, comm->nRanks));
  oks[comm->rank] = localEnabled ? 1 : 0;
  ncclResult_t ret = bootstrapAllGather(comm->bootstrap, oks, sizeof(int));
  int all = 1;
  if (ret == ncclSuccess) {
    for (int i = 0; i < comm->nRanks; i++) {
      if (!oks[i]) all = 0;
    }
  }
  free(oks);
  if (ret != ncclSuccess) return ret;
  *allEnabled = all;
  return ncclSuccess;
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
