/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipArray3DGetDescriptor hipArray3DGetDescriptor
 * @{
 * @ingroup MemoryTest
 * `hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor, hipArray* array)` -
 * Gets a 3D array descriptor.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

/**
 * Test Description
 * ------------------------
 *  - Basic sanity test for `hipArray3DGetDescriptor`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DGetDescriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
HIP_TEST_CASE(Unit_hipArray3DGetDescriptor_Positive_Basic) {
  CHECK_IMAGE_SUPPORT
  DrvArrayAllocGuard<float> array(make_hipExtent(1024, 4, 2));

  HIP_ARRAY3D_DESCRIPTOR desc;
  HIP_CHECK(hipArray3DGetDescriptor(&desc, array.ptr()));

  using vec_info = vector_info<float>;
  REQUIRE(desc.Format == vec_info::format);
  REQUIRE(desc.NumChannels == vec_info::size);
  REQUIRE(desc.Width == 1024 / sizeof(float));
  REQUIRE(desc.Height == 4);
  REQUIRE(desc.Depth == 2);
  REQUIRE(desc.Flags == 0);
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipArray3DGetDescriptor`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DGetDescriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
HIP_TEST_CASE(Unit_hipArray3DGetDescriptor_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT
  DrvArrayAllocGuard<float> array(make_hipExtent(1024, 4, 2));

  HIP_ARRAY3D_DESCRIPTOR desc;

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipArray3DGetDescriptor(nullptr, array.ptr()), hipErrorInvalidValue);
  }

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipArray3DGetDescriptor(&desc, nullptr), hipErrorInvalidHandle);
  }

  SECTION("array is freed") {
    HIP_CHECK(hipArrayDestroy(array.ptr()));
    HIP_CHECK_ERROR(hipArray3DGetDescriptor(&desc, array.ptr()), hipErrorInvalidHandle);
  }
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
