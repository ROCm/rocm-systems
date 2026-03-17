/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipGetChannelDesc hipGetChannelDesc
 * @{
 * @ingroup TextureTest
 * `hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array)` -
 * Gets the channel descriptor in an array.
 */

#define R 8  // rows, height
#define C 8  // columns, width

/**
 * Test Description
 * ------------------------
 *  - Creates a regular channel description.
 *  - Creates array using previously created description.
 *  - Checks that valid description is returned.
 * Test source
 * ------------------------
 *  - unit/texture/hipGetChanDesc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipGetChannelDesc_CreateAndGet) {
  CHECK_IMAGE_SUPPORT;

  hipChannelFormatDesc chan_test, chan_desc;
  hipArray_t hip_array;

  chan_desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindSigned);
  HIP_CHECK(hipMallocArray(&hip_array, &chan_desc, C, R, 0));
  HIP_CHECK(hipGetChannelDesc(&chan_test, hip_array));

  if ((chan_test.x != 32) || (chan_test.y != 0) || (chan_test.z != 0) || (chan_test.f != 0)) {
    INFO("Mismatch observed : " << chan_test.x << chan_test.y << chan_test.z << chan_test.f);
    REQUIRE(false);
  }

  HIP_CHECK(hipFreeArray(hip_array));
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the description is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When array handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/texture/hipGetChanDesc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipGetChannelDesc_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT;

  hipChannelFormatDesc chan_test, chan_desc;
  hipArray_t hip_array;

  chan_desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindSigned);
  HIP_CHECK(hipMallocArray(&hip_array, &chan_desc, C, R, 0));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipGetChannelDesc(nullptr, hip_array), hipErrorInvalidValue);
  }

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipGetChannelDesc(&chan_test, nullptr), hipErrorInvalidHandle);
  }

  HIP_CHECK(hipFreeArray(hip_array));
}

/**
 * End doxygen group TextureTest.
 * @}
 */
