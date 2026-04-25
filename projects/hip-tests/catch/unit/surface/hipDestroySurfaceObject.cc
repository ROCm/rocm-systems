/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipDestroySurfaceObject hipDestroySurfaceObject
 * @{
 * @ingroup SurfaceTest
 */

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test for `hipDestroySurfaceObject`.
 * Test source
 * ------------------------
 *    - unit/texture/hipDestroySurfaceObject.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
HIP_TEST_CASE(Unit_hipDestroySurfaceObject_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT

  SECTION("surfObject is NULL") {
    HIP_CHECK(hipDestroySurfaceObject(static_cast<hipSurfaceObject_t>(0)));
  }

  SECTION("double free") {
    hipArray_t array;
    hipChannelFormatDesc desc = hipCreateChannelDesc<float>();

    HIP_CHECK(hipMallocArray(&array, &desc, 64, 0, hipArraySurfaceLoadStore));

    hipSurfaceObject_t surf;

    hipResourceDesc resc = {};
    resc.resType = hipResourceTypeArray;
    resc.res.array.array = array;

    HIP_CHECK(hipCreateSurfaceObject(&surf, &resc));

    HIP_CHECK(hipDestroySurfaceObject(surf));
#if HT_AMD
    HIP_CHECK_ERROR(hipDestroySurfaceObject(surf), hipErrorInvalidValue);
#else
    HIP_CHECK(hipDestroySurfaceObject(surf));
#endif

    HIP_CHECK(hipFreeArray(array));
  }
}

/**
 * End doxygen group SurfaceTest.
 * @}
 */
