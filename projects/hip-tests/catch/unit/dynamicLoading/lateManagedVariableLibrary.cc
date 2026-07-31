/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

namespace {

constexpr int kExpectedValue = 42;
__managed__ int managedValue;

__global__ void ReadManagedValue(int* result) { *result = managedValue; }

}  // namespace

extern "C" hipError_t launchLateManagedVariable(int* deviceResult, hipStream_t stream, int value) {
  managedValue = value;
  ReadManagedValue<<<1, 1, 0, stream>>>(deviceResult);
  return hipGetLastError();
}

extern "C" int verifyLateManagedVariable() {
  int* deviceResult = nullptr;
  if (hipMalloc(&deviceResult, sizeof(*deviceResult)) != hipSuccess) {
    return 0;
  }

  int result = 0;
  hipError_t status = launchLateManagedVariable(deviceResult, nullptr, kExpectedValue);
  if (status == hipSuccess) {
    status = hipMemcpy(&result, deviceResult, sizeof(result), hipMemcpyDeviceToHost);
  }

  const hipError_t freeStatus = hipFree(deviceResult);
  return status == hipSuccess && freeStatus == hipSuccess && result == kExpectedValue;
}
