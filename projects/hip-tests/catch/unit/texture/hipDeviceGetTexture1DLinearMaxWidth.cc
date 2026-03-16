/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * Test Description
 * ------------------------
 *    - Positive test for hipDeviceGetTexture1DLinearMaxWidth
 *    - Retrieves the maximum 1D linear texture width for a valid device and channel format.
 *    - Verifies that return value is hipSuccess and the max width is non-zero.
 * Test source
 * ------------------------
 *    - unit/texture/hipDeviceGetTexture1DLinearMaxWidth.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
TEST_CASE(Unit_hipDeviceGetTexture1DLinearMaxWidth_Positive) {
  CHECK_IMAGE_SUPPORT

  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE(deviceCount > 0);

  int device = 0;
  HIP_CHECK(hipSetDevice(device));

  size_t maxWidth = 0;
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();

  HIP_CHECK(hipDeviceGetTexture1DLinearMaxWidth(&maxWidth, &desc, device));
  REQUIRE(maxWidth > 0);
}

/**
 * Test Description
 * ------------------------
 *    - Negative test for hipDeviceGetTexture1DLinearMaxWidth
 *    - Covers the following error scenarios:
 *        1. nullptr for maxWidth
 *        2. nullptr for channel format descriptor
 *        3. zero-sized format descriptor (invalid element size)
 *        4. invalid device ID
 *    - Verifies that the API returns hipErrorInvalidValue or hipErrorInvalidDevice as expected.
 * Test source
 * ------------------------
 *    - unit/texture/hipDeviceGetTexture1DLinearMaxWidth.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
TEST_CASE(Unit_hipDeviceGetTexture1DLinearMaxWidth_Negative) {
  CHECK_IMAGE_SUPPORT

  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE(deviceCount > 0);

  int device = 0;
  HIP_CHECK(hipSetDevice(device));

  size_t maxWidth = 0;
  hipChannelFormatDesc validDesc = hipCreateChannelDesc<float>();
  hipChannelFormatDesc zeroSizeDesc = {};

  SECTION("maxWidth is nullptr") {
    HIP_CHECK_ERROR(hipDeviceGetTexture1DLinearMaxWidth(nullptr, &validDesc, device),
                    hipErrorInvalidValue);
  }

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipDeviceGetTexture1DLinearMaxWidth(&maxWidth, nullptr, device),
                    hipErrorInvalidValue);
  }

  SECTION("desc has zero-sized element") {
    HIP_CHECK_ERROR(hipDeviceGetTexture1DLinearMaxWidth(&maxWidth, &zeroSizeDesc, device),
                    hipErrorInvalidValue);
  }

  SECTION("invalid device index") {
    HIP_CHECK_ERROR(hipDeviceGetTexture1DLinearMaxWidth(&maxWidth, &validDesc, deviceCount + 100),
                    hipErrorInvalidDevice);
  }
}