/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_module_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>

static hipModule_t GetModule() {
  HIP_CHECK(hipFree(nullptr));
  static const auto mg = ModuleGuard::LoadModule("get_function_module.code");
  return mg.module();
}

TEST_CASE(Unit_hipFuncGetAttribute_Positive_Basic) {
  hipFunction_t kernel = GetKernel(GetModule(), "GlobalKernel");

  int value;

  SECTION("binaryVersion") {
    HIP_CHECK(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, kernel));
    const auto major = GetDeviceAttribute(hipDeviceAttributeComputeCapabilityMajor, 0);
    const auto minor = GetDeviceAttribute(hipDeviceAttributeComputeCapabilityMinor, 0);
    REQUIRE(value == major * 10 + minor);
  }

  SECTION("cacheModeCA") {
    HIP_CHECK(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA, kernel));
    REQUIRE((value == 0 || value == 1));
  }

  SECTION("maxThreadsPerBlock") {
    HIP_CHECK(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel));
    REQUIRE(value == GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0));
  }

  SECTION("numRegs") {
    HIP_CHECK(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_NUM_REGS, kernel));
    REQUIRE(value >= 0);
  }

  SECTION("ptxVersion") {
    HIP_CHECK(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_PTX_VERSION, kernel));
    REQUIRE(value > 0);
  }

  SECTION("sharedSizeBytes") {
    HIP_CHECK(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel));
    REQUIRE(value <= GetDeviceAttribute(hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
  }
}

TEST_CASE(Unit_hipFuncGetAttribute_Negative_Parameters) {
  hipFunction_t kernel = GetKernel(GetModule(), "GlobalKernel");

  int value;

  SECTION("value == nullptr") {
    HIP_CHECK_ERROR(hipFuncGetAttribute(nullptr, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, kernel),
                    hipErrorInvalidValue);
  }

  SECTION("invalid attribute") {
    HIP_CHECK_ERROR(hipFuncGetAttribute(&value, static_cast<hipFunction_attribute>(-1), kernel),
                    hipErrorInvalidValue);
  }

  SECTION("hfunc == nullptr") {
    HIP_CHECK_ERROR(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, nullptr),
                    hipErrorInvalidResourceHandle);
  }
}