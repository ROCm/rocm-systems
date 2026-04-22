/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipGetMipmappedArrayLevel hipGetMipmappedArrayLevel
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test for `hipGetMipmappedArrayLevel`.
 * Test source
 * ------------------------
 *    - unit/texture/hipGetMipmappedArrayLevel.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
HIP_TEST_CASE(Unit_hipGetMipmappedArrayLevel_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT;

  hipMipmappedArray_t array;
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
  hipExtent extent = make_hipExtent(4, 4, 6);
  unsigned int levels = 1 + std::log2(extent.depth);

  HIP_CHECK(hipMallocMipmappedArray(&array, &desc, extent, levels, 0));

  hipArray_t levelArray;

  SECTION("levelArray is nullptr") {
    HIP_CHECK_ERROR(hipGetMipmappedArrayLevel(nullptr, array, 2), hipErrorInvalidValue);
  }

  SECTION("mipmappedArray is nullptr") {
    HIP_CHECK_ERROR(hipGetMipmappedArrayLevel(&levelArray, nullptr, 2), hipErrorInvalidHandle);
  }

  SECTION("level index is greater than number of levels") {
    HIP_CHECK_ERROR(hipGetMipmappedArrayLevel(&levelArray, array, levels), hipErrorInvalidValue);
  }

  HIP_CHECK(hipFreeMipmappedArray(array));
}

/**
 * End doxygen group TextureTest.
 * @}
 */
