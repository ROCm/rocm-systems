/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_module_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

TEST_CASE(Unit_hipModuleGetFunction_Positive_Basic) {
  auto mg = ModuleGuard::InitModule("get_function_module.code");
  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleGetFunction(&kernel, mg.module(), "GlobalKernel"));
  REQUIRE(kernel != nullptr);
}

TEST_CASE(Unit_hipModuleGetFunction_Negative_Parameters) {
  auto mg = ModuleGuard::InitModule("get_function_module.code");
  hipFunction_t kernel = nullptr;

  SECTION("function == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetFunction(nullptr, mg.module(), "GlobalKernel"),
                    hipErrorInvalidValue);
  }

// Disabled on AMD due to defect - EXSWHTEC-154
#if HT_NVIDIA
  SECTION("module == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetFunction(&kernel, nullptr, "GlobalKernel"),
                    hipErrorInvalidResourceHandle);
  }
#endif

  SECTION("kname == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetFunction(&kernel, mg.module(), nullptr), hipErrorInvalidValue);
  }

// Disabled on AMD due to defect - EXSWHTEC-155
#if HT_NVIDIA
  SECTION("kname == empty string") {
    HIP_CHECK_ERROR(hipModuleGetFunction(&kernel, mg.module(), ""), hipErrorInvalidValue);
  }
#endif

  SECTION("kname == non existent kernel") {
    HIP_CHECK_ERROR(hipModuleGetFunction(&kernel, mg.module(), "NonExistentKernel"),
                    hipErrorNotFound);
  }

  SECTION("kname == __device__ kernel") {
    HIP_CHECK_ERROR(hipModuleGetFunction(&kernel, mg.module(), "DeviceKernel"), hipErrorNotFound);
  }
}

// Test description: Loading kernel function from different device than the one on which the module
// is loaded
TEST_CASE(Unit_hipModuleGetFunction_DiffDevice) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    SUCCEED("skipped the testcase as no of devices is less than 2");
    return;
  }

  auto mg = ModuleGuard::InitModule("get_function_module.code");
  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipModuleGetFunction(&kernel, mg.module(), "GlobalKernel"));
  REQUIRE(kernel != nullptr);
}
