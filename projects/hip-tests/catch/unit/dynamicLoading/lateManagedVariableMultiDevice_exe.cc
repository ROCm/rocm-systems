/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

#include <cstdio>
#include <dlfcn.h>

namespace {

using LaunchLateManagedVariable = hipError_t (*)(int*, hipStream_t, int);

bool Check(hipError_t status, const char* operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
  return false;
}

bool RunLibraryOnDevice(int deviceId, int expectedValue) {
  if (!Check(hipSetDevice(deviceId), "hipSetDevice")) {
    return false;
  }

  hipStream_t stream = nullptr;
  if (!Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags")) {
    return false;
  }

  int* deviceResult = nullptr;
  if (!Check(hipMalloc(&deviceResult, sizeof(*deviceResult)), "hipMalloc")) {
    return false;
  }

  void* handle = dlopen("./libLateManagedVariable.so", RTLD_NOW);
  if (handle == nullptr) {
    std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return false;
  }

  dlerror();
  auto launch = reinterpret_cast<LaunchLateManagedVariable>(
      dlsym(handle, "launchLateManagedVariable"));
  if (const char* error = dlerror(); error != nullptr || launch == nullptr) {
    std::fprintf(stderr, "dlsym failed: %s\n", error == nullptr ? "symbol not found" : error);
    return false;
  }

  if (!Check(launch(deviceResult, stream, expectedValue), "launchLateManagedVariable") ||
      !Check(hipStreamSynchronize(stream), "hipStreamSynchronize")) {
    return false;
  }

  int result = 0;
  if (!Check(hipMemcpy(&result, deviceResult, sizeof(result), hipMemcpyDeviceToHost), "hipMemcpy") ||
      result != expectedValue) {
    std::fprintf(stderr, "managed variable mismatch: expected %d, got %d\n", expectedValue, result);
    return false;
  }

  if (!Check(hipFree(deviceResult), "hipFree") ||
      !Check(hipStreamDestroy(stream), "hipStreamDestroy")) {
    return false;
  }

  if (dlclose(handle) != 0) {
    std::fprintf(stderr, "dlclose failed: %s\n", dlerror());
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!RunLibraryOnDevice(/*deviceId=*/0, /*expectedValue=*/41)) {
    return 1;
  }
  if (!RunLibraryOnDevice(/*deviceId=*/1, /*expectedValue=*/42)) {
    return 1;
  }
  return 0;
}
