/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

#include <iostream>

namespace {

constexpr int kProbeErrorExitCode = 1;
// Stable sentinel used by the parent test to distinguish unsupported hardware
// from a probe failure.
constexpr int kVmmUnsupportedExitCode = 77;

int reportHipError(const char* operation, hipError_t error) {
  std::cerr << "hipVmmSupportProbe: " << operation << " failed: " << hipGetErrorString(error)
            << " (" << static_cast<int>(error) << ")" << std::endl;
  return kProbeErrorExitCode;
}

}  // namespace

int main() {
  hipDevice_t device;
  hipError_t error = hipDeviceGet(&device, 0);
  if (error != hipSuccess) {
    return reportHipError("hipDeviceGet", error);
  }

  int supported = 0;
  error =
      hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported, device);
  if (error != hipSuccess) {
    return reportHipError("hipDeviceGetAttribute", error);
  }

  return supported != 0 ? 0 : kVmmUnsupportedExitCode;
}
