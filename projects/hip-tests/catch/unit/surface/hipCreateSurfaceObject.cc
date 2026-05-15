/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipCreateSurfaceObject hipCreateSurfaceObject
 * @{
 * @ingroup SurfaceTest
 */

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test for `hipCreateSurfaceObject`.
 * Test source
 * ------------------------
 *    - unit/texture/hipCreateSurfaceObject.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
HIP_TEST_CASE(Unit_hipCreateSurfaceObject_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT

  hipArray_t array;
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();

  HIP_CHECK(hipMallocArray(&array, &desc, 64, 0, hipArraySurfaceLoadStore));

  hipSurfaceObject_t surf;

  hipResourceDesc resc = {};
  resc.resType = hipResourceTypeArray;
  resc.res.array.array = array;

  SECTION("pSurfObject is nullptr") {
    HIP_CHECK_ERROR(hipCreateSurfaceObject(nullptr, &resc), hipErrorInvalidValue);
  }

  SECTION("pResDesc is nullptr") {
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, nullptr), hipErrorInvalidValue);
  }

  SECTION("invalid resource type") {
    resc.resType = hipResourceTypeLinear;
#if HT_AMD
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, &resc), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, &resc), hipErrorInvalidChannelDescriptor);
#endif
  }

  SECTION("array handle is nullptr") {
    resc.res.array.array = nullptr;
#if HT_AMD
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, &resc), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, &resc), hipErrorInvalidHandle);
#endif
  }

  HIP_CHECK(hipFreeArray(array));
}

/**
 * End doxygen group SurfaceTest.
 * @}
 */
