/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gin/gin_fabric_a2a_host.h"
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
