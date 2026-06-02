/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#include "cpu_mem.h"
#include "cpu_device_guard.h"

#include <hip/hip_runtime.h>

ncclResult_t rcclCpuCopyBytes(int cudaDev, void* dst, void const* src, size_t bytes) {
  if (bytes == 0 || dst == src) return ncclSuccess;
  if (dst == nullptr || src == nullptr) return ncclInvalidArgument;
  rcclCpuDeviceGuard guard(cudaDev);
  CUDACHECK(hipMemcpy(dst, src, bytes, hipMemcpyDefault));
  return ncclSuccess;
}

ncclResult_t rcclCpuLoadDevU64(int cudaDev, uint64_t const* devPtr, uint64_t* out) {
  if (devPtr == nullptr || out == nullptr) return ncclInvalidArgument;
  rcclCpuDeviceGuard guard(cudaDev);
  CUDACHECK(hipMemcpy(out, devPtr, sizeof(uint64_t), hipMemcpyDeviceToHost));
  return ncclSuccess;
}

ncclResult_t rcclCpuStoreDevU64(int cudaDev, uint64_t* devPtr, uint64_t val) {
  if (devPtr == nullptr) return ncclInvalidArgument;
  rcclCpuDeviceGuard guard(cudaDev);
  CUDACHECK(hipMemcpy(devPtr, &val, sizeof(uint64_t), hipMemcpyHostToDevice));
  return ncclSuccess;
}

ncclResult_t rcclCpuStoreDevU32(int cudaDev, uint32_t* devPtr, uint32_t val) {
  if (devPtr == nullptr) return ncclInvalidArgument;
  rcclCpuDeviceGuard guard(cudaDev);
  CUDACHECK(hipMemcpy(devPtr, &val, sizeof(uint32_t), hipMemcpyHostToDevice));
  return ncclSuccess;
}
