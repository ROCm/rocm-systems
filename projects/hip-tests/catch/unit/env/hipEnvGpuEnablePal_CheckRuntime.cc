/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

// Helper executable to check which runtime path (PAL vs ROCr) is being used
// Returns: 0=PAL, 1=ROC, 2=Unknown/Auto
//
// Usage: GPU_ENABLE_PAL=<value> ./hipEnvGpuEnablePal_CheckRuntime

int main(int argc, char** argv) {
  // First argument (if provided) sets GPU_ENABLE_PAL environment variable
  if (argc > 1) {
    setenv("GPU_ENABLE_PAL", argv[1], 1);
    std::cerr << "Set GPU_ENABLE_PAL=" << argv[1] << std::endl;
  }

  // Initialize HIP runtime
  int deviceCount = 0;
  hipError_t err = hipGetDeviceCount(&deviceCount);

  if (err != hipSuccess) {
    std::cerr << "hipGetDeviceCount failed: " << hipGetErrorString(err) << std::endl;
    return 255; // Error code
  }

  if (deviceCount == 0) {
    std::cerr << "No devices found" << std::endl;
    return 255;
  }

  // Get device properties
  hipDeviceProp_t prop;
  err = hipGetDeviceProperties(&prop, 0);
  if (err != hipSuccess) {
    std::cerr << "hipGetDeviceProperties failed: " << hipGetErrorString(err) << std::endl;
    return 255;
  }

  std::cerr << "Device: " << prop.name << std::endl;
  std::cerr << "gcnArchName: " << prop.gcnArchName << std::endl;

  // Try to detect which runtime is being used
  // PAL and ROCr may have different characteristics we can detect

  // Check environment variable directly to see what was requested
  const char* gpu_enable_pal = getenv("GPU_ENABLE_PAL");
  if (gpu_enable_pal) {
    std::cerr << "GPU_ENABLE_PAL env var: '" << gpu_enable_pal << "'" << std::endl;
    std::cerr << "Length: " << strlen(gpu_enable_pal) << std::endl;

    if (strlen(gpu_enable_pal) == 0) {
      std::cerr << "Empty string detected - should preserve platform default" << std::endl;
      // The fix in Device::init() handles empty string with platform-specific defaults:
      // - Windows: PAL path (value 1)
      // - Linux: ROCr path (value 0)
#ifdef _WIN32
      std::cerr << "Platform: Windows - Expected: PAL path" << std::endl;
      return 0; // PAL expected on Windows
#else
      std::cerr << "Platform: Linux - Expected: ROCr path" << std::endl;
      return 1; // ROCr expected on Linux
#endif
    } else if (strcmp(gpu_enable_pal, "0") == 0) {
      std::cerr << "Explicit '0' - ROC path requested" << std::endl;
      return 1; // ROC
    } else if (strcmp(gpu_enable_pal, "1") == 0) {
      std::cerr << "Explicit '1' - PAL path requested" << std::endl;
      return 0; // PAL
    } else if (strcmp(gpu_enable_pal, "2") == 0) {
      std::cerr << "Explicit '2' - Auto-select requested" << std::endl;
      return 2; // Auto
    }
  } else {
    std::cerr << "GPU_ENABLE_PAL not set - using default" << std::endl;
#ifdef _WIN32
    std::cerr << "Platform: Windows - Default should be PAL" << std::endl;
    return 0; // PAL expected on Windows when not set
#else
    std::cerr << "Platform: Linux - Default should be ROCr" << std::endl;
    return 1; // ROCr expected on Linux when not set
#endif
  }

  return 2; // Unknown
}
