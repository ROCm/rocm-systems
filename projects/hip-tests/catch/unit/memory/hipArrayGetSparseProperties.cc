/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipArrayGetSparseProperties hipArrayGetSparseProperties
 * @{
 * @ingroup MemoryTest
 * `hipArrayGetSparseProperties(hipArraySparseProperties* sparseProperties, hipArray_t array)` -
 * Returns the layout properties of a sparse HIP array.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipArrayGetSparseProperties`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetSparseProperties.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipArrayGetSparseProperties_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT

  ArrayAllocGuard<float> array(make_hipExtent(1024, 4, 2));
  hipArraySparseProperties properties{};

  SECTION("sparseProperties is nullptr") {
    HIP_CHECK_ERROR(hipArrayGetSparseProperties(nullptr, array.ptr()), hipErrorInvalidValue);
  }

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipArrayGetSparseProperties(&properties, nullptr), hipErrorInvalidHandle);
  }
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
