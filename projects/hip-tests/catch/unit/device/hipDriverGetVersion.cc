/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

/**
 * @addtogroup hipDriverGetVersion hipDriverGetVersion
 * @{
 * @ingroup DriverTest
 * `hipDriverGetVersion(int* driverVersion)` -
 * Returns the approximate HIP driver version.
 */

/**
 * Test Description
 * ------------------------
 *  - Check that the returned driver version has valid value.
 *  - Both CUDA and HIP driver version can be returned, depending on the device.
 * Test source
 * ------------------------
 *  - unit/device/hipDriverGetVersion.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDriverGetVersion_Positive") {
  int driverVersion = -1;
  HIP_CHECK(hipDriverGetVersion(&driverVersion));
  REQUIRE(driverVersion > 0);
  INFO("Driver version " << driverVersion);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the driver version is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/device/hipDriverGetVersion.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDriverGetVersion_Negative") {
  // If initialization is attempted with nullptr, error shall be reported
  HIP_CHECK_ERROR(hipDriverGetVersion(nullptr), hipErrorInvalidValue);
}


/**
 * End doxygen group DriverTest.
 * @}
 */
