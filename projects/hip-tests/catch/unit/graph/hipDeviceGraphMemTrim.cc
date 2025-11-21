/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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
 * @addtogroup hipDeviceGraphMemTrim hipDeviceGraphMemTrim
 * @{
 * @ingroup GraphTest
 * `hipDeviceGraphMemTrim(int device)` - Free unused memory on specific device used for graph back
 * to OS.
 */

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that unused memory used for graph can be freed on each device.
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGraphMemTrim.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDeviceGraphMemTrim_Positive_Default") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));

  // Check for each device
  HIP_CHECK(hipDeviceGraphMemTrim(device));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipDeviceGraphMemTrim behavior with invalid arguments:
 *    -# Device is not valid
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGraphMemTrim.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDeviceGraphMemTrim_Negative_Parameters") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  SECTION("Device is not valid") {
    HIP_CHECK_ERROR(hipDeviceGraphMemTrim(num_dev), hipErrorInvalidDevice);
  }
}

/**
 * End doxygen group GraphTest.
 * @}
 */
