/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
TEST_CASE("Unit_hipDeviceGetTexture1DLinearMaxWidth_Positive") {
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
TEST_CASE("Unit_hipDeviceGetTexture1DLinearMaxWidth_Negative") {
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