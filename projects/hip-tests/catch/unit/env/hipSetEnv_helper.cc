/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

#include <iostream>

// Child for the GPU_ENABLE_PAL tests. The parent sets the environment
// before spawn. This process only initializes HIP so the parent can read
// the backend log.
#ifndef __HIP_PLATFORM_AMD__
#error "This test is AMD-only (GPU_ENABLE_PAL controls ROCr vs PAL backend)"
#endif

int main() {
  int deviceCount = 0;
  hipError_t err = hipGetDeviceCount(&deviceCount);
  if (err != hipSuccess) {
    std::cerr << "ERROR: hipGetDeviceCount failed: " << hipGetErrorString(err) << std::endl;
    return 255;
  }
  return 0;
}
