/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
Testcase Scenarios :

 1) Test hipHostMalloc() api with ptr as nullptr and check for return value.
 2) Test hipHostMalloc() api with size as max(size_t) and check for OOM error.
 3) Test hipHostMalloc() api with flags as max(unsigned int) and validate
 return value.
 4) Pass size as zero for hipHostMalloc() api and check ptr is reset with
 with return value success.
*/

#include <hip_test_common.hh>
#include <hip_test_helper.hh>

/**
 * Performs argument validation of hipHostMalloc api.
 */
TEST_CASE(Unit_hipHostMalloc_ArgValidation) {
#if HT_NVIDIA
  HipTest::HIP_SKIP_TEST("TODO: Need to debug");
#endif
  constexpr size_t allocSize = 1000;
  char* ptr;

  SECTION("Pass ptr as nullptr") {
    HIP_CHECK_ERROR(hipHostMalloc(static_cast<void**>(nullptr), allocSize), hipErrorInvalidValue);
  }

  SECTION("Size as max(size_t)") {
    HIP_CHECK_ERROR(hipHostMalloc(&ptr, (std::numeric_limits<std::size_t>::max)()),
                    hipErrorMemoryAllocation);
  }

  SECTION("Flags as max(uint)") {
    HIP_CHECK_ERROR(hipHostMalloc(&ptr, allocSize, (std::numeric_limits<unsigned int>::max)()),
                    hipErrorInvalidValue);
  }

  SECTION("Pass size as zero and check ptr reset") {
    HIP_CHECK(hipHostMalloc(&ptr, 0));
    REQUIRE(ptr == nullptr);
  }

  SECTION("Pass hipHostMallocCoherent and hipHostMallocNonCoherent simultaneously") {
    HIP_CHECK_ERROR(
        hipHostMalloc(&ptr, allocSize, hipHostMallocCoherent | hipHostMallocNonCoherent),
        hipErrorInvalidValue);
  }
}
