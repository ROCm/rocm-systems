/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

#include <dlfcn.h>

namespace {

constexpr int kInitialManagedValue = 41;
__managed__ int initialManagedValue = kInitialManagedValue;

using LaunchLateManagedVariable = hipError_t (*)(int*, hipStream_t, int);

__global__ void ResetInitialManagedValue() { initialManagedValue = kInitialManagedValue; }

void MakeInitialManagedVariableCurrentOnDevice(int device) {
  HIP_CHECK(hipSetDevice(device));

  ResetInitialManagedValue<<<1, 1>>>();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // The second launch observes the completed initialization marker and
  // publishes this device's fast-path sequence.
  ResetInitialManagedValue<<<1, 1>>>();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(initialManagedValue == kInitialManagedValue);
}

void LoadVerifyAndUnloadLateManagedVariable(int expectedValue) {
  void* handle = dlopen("./libLateManagedVariable.so", RTLD_NOW);
  const char* loadError = dlerror();
  INFO("dlopen failed: " << (loadError == nullptr ? "" : loadError));
  REQUIRE(handle != nullptr);

  auto launch = reinterpret_cast<LaunchLateManagedVariable>(
      dlsym(handle, "launchLateManagedVariable"));
  const char* symbolError = dlerror();
  INFO("dlsym failed: " << (symbolError == nullptr ? "" : symbolError));
  REQUIRE(symbolError == nullptr);
  REQUIRE(launch != nullptr);

  // The library writes its managed variable on the host and reads it from a
  // kernel, proving that late registration initialized its device pointer.
  LinearAllocGuard<int> deviceResult(LinearAllocs::hipMalloc, sizeof(int));
  HIP_CHECK(launch(deviceResult.ptr(), nullptr, expectedValue));

  int result = 0;
  HIP_CHECK(hipMemcpy(&result, deviceResult.ptr(), sizeof(result), hipMemcpyDeviceToHost));
  REQUIRE(result == expectedValue);
  REQUIRE(dlclose(handle) == 0);
}

void VerifyLateManagedVariableOnDevice(LaunchLateManagedVariable launch, int device,
                                       int expectedValue) {
  HIP_CHECK(hipSetDevice(device));
  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  LinearAllocGuard<int> deviceResult(LinearAllocs::hipMalloc, sizeof(int));

  HIP_CHECK(launch(deviceResult.ptr(), stream.stream(), expectedValue));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  int result = 0;
  HIP_CHECK(hipMemcpy(&result, deviceResult.ptr(), sizeof(result), hipMemcpyDeviceToHost));
  REQUIRE(result == expectedValue);
}

}  // namespace

// Loading a library after the device's managed-variable cache is current must
// initialize variables registered by that library. Reloading verifies that an
// unloaded registration does not leave stale initialization state behind.
HIP_TEST_CASE(Unit_StatCO_Positive_ManagedVariableFromRepeatedLateDlopen) {
  CHECK_MANAGED_MEMORY_SUPPORT

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  MakeInitialManagedVariableCurrentOnDevice(device);

  LoadVerifyAndUnloadLateManagedVariable(/*expectedValue=*/42);
  LoadVerifyAndUnloadLateManagedVariable(/*expectedValue=*/43);
}

// Managed-variable initialization state is per device. Loading the same library
// first on device 0 and then on device 1 must initialize its late-registered
// variable independently for both devices.
HIP_TEST_CASE(Unit_StatCO_Positive_ManagedVariableFromRepeatedLateDlopenAcrossDevices) {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
    return;
  }
  CHECK_MANAGED_MEMORY_SUPPORT_ON_DEVICE(0)
  CHECK_MANAGED_MEMORY_SUPPORT_ON_DEVICE(1)

  for (int device = 0; device < 2; ++device) {
    MakeInitialManagedVariableCurrentOnDevice(device);
  }

  void* handle = dlopen("./libLateManagedVariable.so", RTLD_NOW);
  const char* loadError = dlerror();
  INFO("dlopen failed: " << (loadError == nullptr ? "" : loadError));
  REQUIRE(handle != nullptr);

  auto launch = reinterpret_cast<LaunchLateManagedVariable>(
      dlsym(handle, "launchLateManagedVariable"));
  const char* symbolError = dlerror();
  INFO("dlsym failed: " << (symbolError == nullptr ? "" : symbolError));
  REQUIRE(symbolError == nullptr);
  REQUIRE(launch != nullptr);

  VerifyLateManagedVariableOnDevice(launch, /*device=*/0, /*expectedValue=*/41);
  // Publish device 0's completed late-registration sequence before device 1
  // first touches the same registration.
  VerifyLateManagedVariableOnDevice(launch, /*device=*/0, /*expectedValue=*/43);
  VerifyLateManagedVariableOnDevice(launch, /*device=*/1, /*expectedValue=*/42);
  REQUIRE(dlclose(handle) == 0);
}
