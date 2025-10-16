/*
Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdlib>
#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include "hip/hip_runtime_api.h"
#include <hip_test_process.hh>
#include <string>

/**
 * Test Description
 * ------------------------
 *  - Get all possible combinations of attributes between all pairs of devices.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetP2PAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */

/**
 * Test Description
 * ------------------------
 *    - Test all possible combination of attributes and devices for hipDeviceGetP2PAttribute
 * Verify that the output is within the range of acceptable values.

 * Test source
 * ------------------------
 *    - catch/unit/p2p/hipDeviceGetP2PAtomicCapabilities.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

TEST_CASE("Unit_hipDeviceGetP2PAtomicCapabilities_Negative") {

/*   int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  } */
  unsigned int* capabilities;
  const hipAtomicOperation*  operations;

  SECTION("Nullptr capabilities") {
    HIP_CHECK_ERROR(hipDeviceGetP2PAtomicCapabilities(nullptr, &operations, 2, 0, 1),
                    hipErrorInvalidValue);
  }

  SECTION("Nullptr operations") {
    HIP_CHECK_ERROR(hipDeviceGetP2PAtomicCapabilities(capabilities, nullptr, 2, 0, 1),
                    hipErrorInvalidValue);
  }

  SECTION("size zero") {
    HIP_CHECK_ERROR(hipDeviceGetP2PAtomicCapabilities(capabilities, &operations, 0, 0, 1),
                    hipErrorInvalidValue);
  }

  SECTION("src invalid") {
    HIP_CHECK_ERROR(hipDeviceGetP2PAtomicCapabilities(capabilities, &operations, 2, 2, 1),
                    hipErrorInvalidDevice);
  }

  SECTION("dst invalid") {
    HIP_CHECK_ERROR(hipDeviceGetP2PAtomicCapabilities(capabilities, &operations, 2, 0, 2),
                    hipErrorInvalidDevice);
  }

}


/**
 * End doxygen group DriverTest.
 * @}
 */
