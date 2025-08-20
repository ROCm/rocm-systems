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
 * @addtogroup hipInit hipInit
 * @{
 * @ingroup DriverTest
 * `hipInit(unsigned int flags)` -
 * Explicitly initializes the HIP runtime.
 */

/**
 * Test Description
 * ------------------------
 *  - Initialize HIP runtime.
 *  - Call a HIP API and check that the runtime is initialized successfully.
 * Test source
 * ------------------------
 *  - unit/device/hipInit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipInit_Positive") {
  HIP_CHECK(hipInit(0));

  // Verify that HIP runtime is successfully initialized by calling a HIP API
  int count = -1;
  HIP_CHECK(hipGetDeviceCount(&count));
  REQUIRE(count >= 0);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When flag has invalid value equal to -1
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/device/hipInit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipInit_Negative") {
  // If initialization is attempted with invalid flag, error shall be reported
  unsigned int invalid_flag = 1;
  HIP_CHECK_ERROR(hipInit(invalid_flag), hipErrorInvalidValue);
}


/**
 * End doxygen group DriverTest.
 * @}
 */
