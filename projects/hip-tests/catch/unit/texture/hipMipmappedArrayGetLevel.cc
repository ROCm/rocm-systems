/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_array_common.hh>
#include <hip_test_common.hh>

/**
 * @addtogroup hipMipmappedArrayGetLevel hipMipmappedArrayGetLevel
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test for `hipMipmappedArrayGetLevel`.
 * Test source
 * ------------------------
 *    - unit/texture/hipMipmappedArrayGetLevel.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE(Unit_hipMipmappedArrayGetLevel_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT;

  hipmipmappedArray array;

  HIP_ARRAY3D_DESCRIPTOR desc = {};
  using vec_info = vector_info<float>;
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;
  desc.Width = 4;
  desc.Height = 4;
  desc.Depth = 6;
  desc.Flags = 0;

  unsigned int levels = 1 + std::log2(desc.Depth);

  HIP_CHECK(hipFree(0));
  HIP_CHECK(hipMipmappedArrayCreate(&array, &desc, levels));

  hipArray_t levelArray;

  SECTION("levelArray is nullptr") {
    HIP_CHECK_ERROR(hipMipmappedArrayGetLevel(nullptr, array, 2), hipErrorInvalidValue);
  }

  SECTION("mipmappedArray is nullptr") {
    HIP_CHECK_ERROR(hipMipmappedArrayGetLevel(&levelArray, nullptr, 2), hipErrorInvalidHandle);
  }

  SECTION("level index is greater than number of levels") {
    HIP_CHECK_ERROR(hipMipmappedArrayGetLevel(&levelArray, array, levels), hipErrorInvalidValue);
  }

  HIP_CHECK(hipMipmappedArrayDestroy(array));
  (void)hipGetLastError();
}

/**
 * End doxygen group TextureTest.
 * @}
 */
