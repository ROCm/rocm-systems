/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_array_common.hh>
#include <hip_test_common.hh>

/**
 * @addtogroup hipMipmappedArrayDestroy hipMipmappedArrayDestroy
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test for `hipMipmappedArrayDestroy`.
 * Test source
 * ------------------------
 *    - unit/texture/hipMipmappedArrayDestroy.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE(Unit_hipMipmappedArrayDestroy_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT;

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipMipmappedArrayDestroy(nullptr), hipErrorInvalidValue);
  }

  SECTION("double free") {
    INFO("Double free cheching isn't supported. Skipped.");
    return;
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

    HIP_CHECK(hipMipmappedArrayCreate(&array, &desc, levels));

    HIP_CHECK(hipMipmappedArrayDestroy(array));
    HIP_CHECK_ERROR(hipMipmappedArrayDestroy(array), hipErrorContextIsDestroyed);
  }
}

/**
 * End doxygen group TextureTest.
 * @}
 */
