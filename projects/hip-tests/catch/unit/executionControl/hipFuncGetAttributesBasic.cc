/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>

constexpr size_t kConstSizeBytes = 128;
__constant__ char const_data[kConstSizeBytes];

__global__ void attribute_test_kernel() {}

HIP_TEST_CASE(Unit_hipFuncGetAttributes_Positive_Basic) {
  hipFuncAttributes attr;
  HIP_CHECK(hipFuncGetAttributes(&attr, reinterpret_cast<void*>(attribute_test_kernel)));

  SECTION("binaryVersion") {
    const auto major = GetDeviceAttribute(hipDeviceAttributeComputeCapabilityMajor, 0);
    const auto minor = GetDeviceAttribute(hipDeviceAttributeComputeCapabilityMinor, 0);
    REQUIRE(attr.binaryVersion == major * 10 + minor);
  }

  SECTION("cacheModeCA") { REQUIRE((attr.cacheModeCA == 0 || attr.cacheModeCA == 1)); }

  SECTION("constSizeBytes") { REQUIRE(attr.constSizeBytes == kConstSizeBytes); }

  SECTION("maxThreadsPerBlock") {
    REQUIRE(attr.maxThreadsPerBlock == GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0));
  }

  SECTION("numRegs") { REQUIRE(attr.numRegs >= 0); }

  SECTION("ptxVersion") { REQUIRE(attr.ptxVersion > 0); }

  SECTION("sharedSizeBytes") {
    REQUIRE(attr.sharedSizeBytes <=
            GetDeviceAttribute(hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
  }
}

HIP_TEST_CASE(Unit_hipFuncGetAttributes_Negative_Parameters) {
  SECTION("attr == nullptr") {
    HIP_CHECK_ERROR(hipFuncGetAttributes(nullptr, reinterpret_cast<void*>(attribute_test_kernel)),
                    hipErrorInvalidValue);
  }
  SECTION("func == nullptr") {
    hipFuncAttributes attr;
    HIP_CHECK_ERROR(hipFuncGetAttributes(&attr, nullptr), hipErrorInvalidDeviceFunction);
  }
}
