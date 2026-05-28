/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#include "cpu_mem.h"

#include <hip/hip_runtime.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

struct StagingBuf {
  void* devicePtr;
  size_t bytes;
  std::vector<char> host;
};

static std::mutex g_stagingMutex;
static thread_local std::unordered_map<void*, StagingBuf> g_staging;

}  // namespace

ncclResult_t rcclCpuMapDevicePtr(void* devicePtr, size_t bytes, void** hostPtr, bool* needsUnmap) {
  if (hostPtr == nullptr || needsUnmap == nullptr) return ncclInvalidArgument;
  *needsUnmap = false;
  if (devicePtr == nullptr || bytes == 0) {
    *hostPtr = devicePtr;
    return ncclSuccess;
  }

  hipPointerAttribute_t attr{};
  hipError_t err = hipPointerGetAttributes(&attr, devicePtr);
  if (err == hipSuccess && (attr.type == hipMemoryTypeHost || attr.isManaged)) {
    *hostPtr = devicePtr;
    return ncclSuccess;
  }

  StagingBuf& s = g_staging[devicePtr];
  if (s.host.size() < bytes) s.host.resize(bytes);
  s.devicePtr = devicePtr;
  s.bytes = bytes;
  CUDACHECK(hipMemcpy(s.host.data(), devicePtr, bytes, hipMemcpyDeviceToHost));
  *hostPtr = s.host.data();
  *needsUnmap = true;
  return ncclSuccess;
}

void rcclCpuUnmapDevicePtr(void* devicePtr, void* hostPtr, bool needsUnmap) {
  if (!needsUnmap || devicePtr == nullptr) return;
  auto it = g_staging.find(devicePtr);
  if (it == g_staging.end()) return;
  (void)hipMemcpy(devicePtr, it->second.host.data(), it->second.bytes, hipMemcpyHostToDevice);
}
