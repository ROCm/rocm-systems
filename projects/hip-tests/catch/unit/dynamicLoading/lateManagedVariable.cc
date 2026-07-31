/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include <dlfcn.h>

namespace {

__global__ void InitializeRuntime() {}

}  // namespace

HIP_TEST_CASE(Unit_StatCO_Positive_ManagedVariableFromRepeatedLateDlopen) {
  CHECK_MANAGED_MEMORY_SUPPORT

  // Make the original per-device initialization cache current before the library adds a variable.
  InitializeRuntime<<<1, 1>>>();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  for (int load = 0; load < 2; ++load) {
    INFO("load iteration: " << load);
    void* handle = dlopen("./libLateManagedVariable.so", RTLD_NOW);
    const char* loadError = dlerror();
    INFO("dlopen failed: " << (loadError == nullptr ? "" : loadError));
    REQUIRE(handle != nullptr);

    using VerifyLateManagedVariable = int (*)();
    dlerror();
    auto verify = reinterpret_cast<VerifyLateManagedVariable>(
        dlsym(handle, "verifyLateManagedVariable"));
    const char* symbolError = dlerror();
    INFO("dlsym failed: " << (symbolError == nullptr ? "" : symbolError));
    REQUIRE(symbolError == nullptr);
    REQUIRE(verify != nullptr);

    REQUIRE(verify() == 1);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(dlclose(handle) == 0);
  }
}

HIP_TEST_CASE(Unit_StatCO_Positive_ManagedVariableFromRepeatedLateDlopenAcrossDevices) {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
    return;
  }
  CHECK_MANAGED_MEMORY_SUPPORT_ON_DEVICE(0)
  CHECK_MANAGED_MEMORY_SUPPORT_ON_DEVICE(1)

  hip::SpawnProc process("lateManagedVariableMultiDevice_exe", true);
  REQUIRE(process.run() == 0);
}
