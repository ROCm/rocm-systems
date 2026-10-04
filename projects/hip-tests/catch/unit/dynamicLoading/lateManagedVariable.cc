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
using LateManagedVariableAddress = int* (*)();
using ReadLateManagedVariable = hipError_t (*)(int*, hipStream_t);

__global__ void IncrementInitialManagedValue() { ++initialManagedValue; }

void LoadVerifyAndUnloadLateManagedVariable() {
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
  constexpr int kExpectedValue = 42;
  LinearAllocGuard<int> deviceResult(LinearAllocs::hipMalloc, sizeof(int));
  HIP_CHECK(launch(deviceResult.ptr(), nullptr, kExpectedValue));

  int result = 0;
  HIP_CHECK(hipMemcpy(&result, deviceResult.ptr(), sizeof(result), hipMemcpyDeviceToHost));
  REQUIRE(result == kExpectedValue);
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

  // Initialize a managed variable registered with the executable before the
  // library adds another managed variable.
  IncrementInitialManagedValue<<<1, 1>>>();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(initialManagedValue == kInitialManagedValue + 1);

  LoadVerifyAndUnloadLateManagedVariable();
  LoadVerifyAndUnloadLateManagedVariable();
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
    HIP_CHECK(hipSetDevice(device));
    IncrementInitialManagedValue<<<1, 1>>>();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
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
  VerifyLateManagedVariableOnDevice(launch, /*device=*/1, /*expectedValue=*/42);
  REQUIRE(dlclose(handle) == 0);
}

// TODO Review this test.
// A memory API is the first thing to touch the library's managed variable here, so
// nothing has gone through the launch path that writes device-side pointer slots. The
// variable must already be backed by a runtime-tracked allocation when dlopen returns.
//
// hipMemset is the discriminating API: a fill is issued against a buffer object, so
// ihipMemset rejects a destination that MemObjMap cannot resolve and has no
// system-pointer path to fall back on. hipMemcpyDefault would instead degrade to a
// host-to-host copy into the same range and succeed either way.
HIP_TEST_CASE(Unit_StatCO_Positive_ManagedVariableFromLateDlopenBeforeFirstUse) {
  CHECK_MANAGED_MEMORY_SUPPORT

  void* handle = dlopen("./libLateManagedVariable.so", RTLD_NOW);
  const char* loadError = dlerror();
  INFO("dlopen failed: " << (loadError == nullptr ? "" : loadError));
  REQUIRE(handle != nullptr);

  auto variableAddress =
      reinterpret_cast<LateManagedVariableAddress>(dlsym(handle, "lateManagedVariableAddress"));
  auto readVariable =
      reinterpret_cast<ReadLateManagedVariable>(dlsym(handle, "readLateManagedVariable"));
  const char* symbolError = dlerror();
  INFO("dlsym failed: " << (symbolError == nullptr ? "" : symbolError));
  REQUIRE(symbolError == nullptr);
  REQUIRE(variableAddress != nullptr);
  REQUIRE(readVariable != nullptr);

  constexpr unsigned char kFillByte = 0x5A;
  HIP_CHECK(hipMemset(variableAddress(), kFillByte, sizeof(int)));

  // Reading through the symbol rather than the address covers the device-side half as
  // well: the kernel resolves the variable through its pointer slot, so the fill and
  // the slot have to name the same storage.
  LinearAllocGuard<int> deviceResult(LinearAllocs::hipMalloc, sizeof(int));
  HIP_CHECK(readVariable(deviceResult.ptr(), nullptr));

  int result = 0;
  HIP_CHECK(hipMemcpy(&result, deviceResult.ptr(), sizeof(result), hipMemcpyDeviceToHost));
  constexpr int kExpectedValue = 0x5A5A5A5A;
  REQUIRE(result == kExpectedValue);

  REQUIRE(dlclose(handle) == 0);
}
