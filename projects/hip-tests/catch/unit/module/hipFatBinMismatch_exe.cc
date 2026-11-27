/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Built only for gfx900 and gfx906 (see CMakeLists.txt), so on other GPUs its fat binary has no
// code object for the device.

#include <hip/hip_runtime.h>
#include <iostream>

// Exit codes, also used by Unit_FatBin_NoCompatibleCodeObject_StaticKernel
constexpr int kPassed = 0;
constexpr int kFailed = 1;

__device__ int device_var = 1;
__global__ void Kernel(int* out) { *out = device_var; }

int main() {
  // The first query loads the fat binary: its error must be the load error, not a later
  // lookup error
  hipFuncAttributes attr{};
  const hipError_t attr_status = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(Kernel));
  std::cout << "hipFuncGetAttributes: " << hipGetErrorName(attr_status) << std::endl;
  if (attr_status != hipErrorInvalidImage && attr_status != hipErrorNoBinaryForGpu) {
    return kFailed;
  }

  void* ptr = nullptr;
  const hipError_t symbol_status = hipGetSymbolAddress(&ptr, HIP_SYMBOL(device_var));
  std::cout << "hipGetSymbolAddress: " << hipGetErrorName(symbol_status) << std::endl;
  if (symbol_status == hipSuccess) {
    return kFailed;
  }

  Kernel<<<1, 1>>>(nullptr);
  const hipError_t launch_status = hipGetLastError();
  std::cout << "launch: " << hipGetErrorName(launch_status) << std::endl;
  if (launch_status == hipSuccess) {
    return kFailed;
  }
  return kPassed;
}
