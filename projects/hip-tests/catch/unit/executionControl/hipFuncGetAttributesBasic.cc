/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>

#if HT_AMD
#include "hipFuncGetAttributesConstantFixtures.hh"

static void CheckStaticConstantSize(const void* kernel, size_t expected_size) {
  hipFunction_t function = nullptr;
  HIP_CHECK(hipGetFuncBySymbol(&function, kernel));
  // Query the static handle first: accounting must not rely on an earlier struct query.
  int scalar_size = -1;
  HIP_CHECK(hipFuncGetAttribute(&scalar_size, HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, function));
  REQUIRE(scalar_size == static_cast<int>(expected_size));

  hipFuncAttributes attributes{};
  HIP_CHECK(hipFuncGetAttributes(&attributes, kernel));
  REQUIRE(attributes.constSizeBytes == expected_size);
}
#endif

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

#if HT_AMD
  SECTION("Registered static constant allocations remain isolated by translation unit") {
    // Static attributes describe the owning program's registered __constant__ allocations,
    // including an empty kernel. Raw module/library metadata cannot provide this identity.
    CheckStaticConstantSize(ConstantAttributeTest::MultipleConstantsKernel(), 17 + 29);
    CheckStaticConstantSize(ConstantAttributeTest::NoConstantsKernel(), 0);
    CheckStaticConstantSize(reinterpret_cast<const void*>(attribute_test_kernel), kConstSizeBytes);
    CheckStaticConstantSize(ConstantAttributeTest::MultipleConstantsReaderKernel(), 17 + 29);
    CheckStaticConstantSize(ConstantAttributeTest::NoConstantsKernel(), 0);

    size_t first_size = 0;
    HIP_CHECK(ConstantAttributeTest::GetFirstConstantSize(&first_size));
    REQUIRE(first_size == 17);

    size_t inline_size = 0;
    HIP_CHECK(ConstantAttributeTest::GetInlineConstantSize(&inline_size));
    REQUIRE(inline_size == 29);
  }
#endif

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
