/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipFreeMipmappedArray hipFreeMipmappedArray
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test for `hipFreeMipmappedArray`.
 * Test source
 * ------------------------
 *    - unit/texture/hipFreeMipmappedArray.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
HIP_TEST_CASE(Unit_hipFreeMipmappedArray_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT;

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipFreeMipmappedArray(nullptr), hipErrorInvalidValue);
  }

  SECTION("double free") {
    INFO("Double free cheching isn't supported. Skipped.");
    return;
    hipMipmappedArray_t array;
    hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
    hipExtent extent = make_hipExtent(4, 4, 6);
    unsigned int levels = 4;

    HIP_CHECK(hipMallocMipmappedArray(&array, &desc, extent, levels, 0));

    HIP_CHECK(hipFreeMipmappedArray(array));
    HIP_CHECK_ERROR(hipFreeMipmappedArray(array), hipErrorContextIsDestroyed);
  }
}

/**
 * End doxygen group TextureTest.
 * @}
 */
