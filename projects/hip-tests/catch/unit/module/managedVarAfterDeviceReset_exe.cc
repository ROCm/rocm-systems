/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Touches a __managed__ variable from a kernel and the host, calls hipDeviceReset(), then
// uses the variable again. Runs as its own process because a regression here is a SIGSEGV
// on the host read, and because the behaviour depends on the number of visible devices and
// on HIP_ENABLE_DEFERRED_LOADING, both of which the parent sets in the environment.

#include <hip/hip_runtime.h>

#include <iostream>

__managed__ int g_managedVal = 7;

__global__ void writeManaged(int value) { g_managedVal = value; }

namespace {

// Distinct codes so a failure says which step broke without needing the output.
enum ExitCode {
  kOk = 0,
  kNotOneDevice = 2,
  kPreResetWrongValue = 3,
  kResetFailed = 4,
  kContentsLost = 5,
  kPostResetLaunchFailed = 6,
  kPostResetWrongValue = 7,
  kUserAllocSurvived = 8,
  kSetupFailed = 9,
};

#define RETURN_IF_ERROR(expr, code)                                                                \
  do {                                                                                             \
    const hipError_t error = (expr);                                                               \
    if (error != hipSuccess) {                                                                     \
      std::cout << #expr << " failed: " << hipGetErrorString(error) << std::endl;                  \
      return (code);                                                                               \
    }                                                                                              \
  } while (0)

int run() {
  int visibleDevices = 0;
  RETURN_IF_ERROR(hipGetDeviceCount(&visibleDevices), kSetupFailed);
  if (visibleDevices != 1) {
    // With more than one visible device the managed allocation lives in a multi-device
    // context, which MemObjMap::Purge() skips, so the case under test cannot arise.
    std::cout << "expected exactly 1 visible device, got " << visibleDevices << std::endl;
    return kNotOneDevice;
  }

  // An ordinary managed allocation, which a reset is expected to release.
  int* userPtr = nullptr;
  RETURN_IF_ERROR(hipMallocManaged(&userPtr, sizeof(int)), kSetupFailed);

  writeManaged<<<1, 1>>>(11);
  RETURN_IF_ERROR(hipGetLastError(), kSetupFailed);
  RETURN_IF_ERROR(hipDeviceSynchronize(), kSetupFailed);
  if (g_managedVal != 11) {
    std::cout << "before reset: expected 11, got " << g_managedVal << std::endl;
    return kPreResetWrongValue;
  }

  RETURN_IF_ERROR(hipDeviceReset(), kResetFailed);

  // The storage behind a __managed__ variable is part of the program image, so the reset
  // must leave it both mapped and intact. This read is what faults on a regression.
  if (g_managedVal != 11) {
    std::cout << "after reset: contents lost, expected 11, got " << g_managedVal << std::endl;
    return kContentsLost;
  }

  writeManaged<<<1, 1>>>(22);
  RETURN_IF_ERROR(hipGetLastError(), kPostResetLaunchFailed);
  RETURN_IF_ERROR(hipDeviceSynchronize(), kPostResetLaunchFailed);
  if (g_managedVal != 22) {
    std::cout << "after reset: expected 22, got " << g_managedVal << std::endl;
    return kPostResetWrongValue;
  }

  // The opposite direction: a hipMallocManaged() allocation belongs to the application and
  // the reset releases it, so the runtime must no longer know this pointer.
  if (hipFree(userPtr) == hipSuccess) {
    std::cout << "hipMallocManaged() allocation outlived hipDeviceReset()" << std::endl;
    return kUserAllocSurvived;
  }

  return kOk;
}

}  // namespace

int main() {
  const int result = run();
  if (result == kOk) {
    std::cout << "OK: managed variable survived hipDeviceReset() with its contents" << std::endl;
  }
  return result;
}
