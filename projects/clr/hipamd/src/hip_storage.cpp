/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/amd_detail/hip_storage.h>

#include <hip/hip_runtime.h>
#include "hip_internal.hpp"

hipError_t hipAmdFileRead(hipAmdFileHandle_t handle, void* devicePtr, uint64_t size, int64_t file_offset,
                       uint64_t* size_copied, int32_t* status) {
  if (size == 0) {
    // Skip if nothing needs reading.
    return hipSuccess;
  }
  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  if (device == nullptr) {
    LogError("Failed to get current device");
    return hipErrorInvalidDevice;
  }
#if defined(_WIN32)
  amd::Os::FileDesc opaque = handle.handle;
#else
  amd::Os::FileDesc opaque = handle.fd;
#endif
  if (!device->amdFileRead(opaque, devicePtr, size, file_offset, size_copied, status)) {
    LogError("Failed to perform file read operation");
    return hipErrorUnknown;
  }
  return hipSuccess;
}

hipError_t hipAmdFileWrite(hipAmdFileHandle_t handle, void* devicePtr, uint64_t size, int64_t file_offset,
                       uint64_t* size_copied, int32_t* status) {
  if (size == 0) {
    // Skip if nothing needs writing.
    return hipSuccess;
  }
  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  if (device == nullptr) {
    LogError("Failed to get current device");
    return hipErrorInvalidDevice;
  }
#if defined(_WIN32)
  amd::Os::FileDesc opaque = handle.handle;
#else
  amd::Os::FileDesc opaque = handle.fd;
#endif
  if (!device->amdFileWrite(opaque, devicePtr, size, file_offset, size_copied, status)) {
    LogError("Failed to perform file write operation");
    return hipErrorUnknown;
  }
  return hipSuccess;
}
