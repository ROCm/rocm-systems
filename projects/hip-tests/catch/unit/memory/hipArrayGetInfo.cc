/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipArrayGetInfo hipArrayGetInfo
 * @{
 * @ingroup MemoryTest
 * `hipArrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent, unsigned int* flags, hipArray*
 * array)` - Gets info about the specified array.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

/**
 * Test Description
 * ------------------------
 *  - Basic sanity test for `hipArrayGetInfo`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetInfo.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
TEST_CASE(Unit_hipArrayGetInfo_Positive_Basic) {
  CHECK_IMAGE_SUPPORT

  ArrayAllocGuard<float> array(make_hipExtent(1024, 4, 2));

  hipChannelFormatDesc desc;
  hipExtent extent;
  unsigned int flags = 1;

  HIP_CHECK(hipArrayGetInfo(&desc, &extent, &flags, array.ptr()));

  REQUIRE(extent.width == 1024);
  REQUIRE(extent.height == 4);
  REQUIRE(extent.depth == 2);

  REQUIRE(flags == 0);

  auto expected_desc = hipCreateChannelDesc<float>();
  REQUIRE(desc.x == expected_desc.x);
  REQUIRE(desc.y == expected_desc.y);
  REQUIRE(desc.z == expected_desc.z);
  REQUIRE(desc.w == expected_desc.w);
  REQUIRE(desc.f == expected_desc.f);
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipArrayGetInfo`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetInfo.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
TEST_CASE(Unit_hipArrayGetInfo_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT
  ArrayAllocGuard<float> array(make_hipExtent(1024, 4, 4));

  hipChannelFormatDesc desc;
  hipExtent extent;
  unsigned int flags;

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipArrayGetInfo(&desc, &extent, &flags, nullptr), hipErrorInvalidHandle);
  }

  SECTION("array is freed") {
    HIP_CHECK(hipFreeArray(array.ptr()));
    HIP_CHECK_ERROR(hipArrayGetInfo(&desc, &extent, &flags, array.ptr()), hipErrorInvalidHandle);
  }
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
