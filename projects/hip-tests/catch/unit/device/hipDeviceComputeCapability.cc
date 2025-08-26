/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
 * @addtogroup hipDeviceComputeCapability hipDeviceComputeCapability
 * @{
 * @ingroup DriverTest
 * `hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device)` -
 * Returns the compute capability of the device.
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the major number is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When output pointer to the minor number is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When device ordinal is negative
 *      - Expected output: do not return `hipSuccess`
 *    -# When device ordinal is out of bounds
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceComputeCapability.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceComputeCapability_Negative") {
  int major, minor, numDevices;
  hipDevice_t device;

  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices > 0) {
    HIP_CHECK(hipDeviceGet(&device, 0));

    // Scenario1
    SECTION("major is nullptr") {
      REQUIRE_FALSE(hipDeviceComputeCapability(nullptr, &minor, device) == hipSuccess);
    }

    // Scenario2
    SECTION("minor is nullptr") {
      REQUIRE_FALSE(hipDeviceComputeCapability(&major, nullptr, device) == hipSuccess);
    }
    // Scenario3
    SECTION("device is -1") {
      REQUIRE_FALSE(hipDeviceComputeCapability(&major, &minor, -1) == hipSuccess);
    }
    // Scenario4
    SECTION("device is out of bounds") {
      REQUIRE_FALSE(hipDeviceComputeCapability(&major, &minor, numDevices) == hipSuccess);
    }
  } else {
    WARN("Test skipped as no gpu devices available");
  }
}

/**
 * Test Description
 * ------------------------
 *  - Checks that valid major and minor numbers are returned.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceComputeCapability.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceComputeCapability_ValidateVersion") {
  int major, minor;
  hipDevice_t device;
  int numDevices = -1;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipDeviceGet(&device, i));
    HIP_CHECK(hipDeviceComputeCapability(&major, &minor, device));
    REQUIRE(major >= 0);
    REQUIRE(minor >= 0);
  }
}

/**
 * End doxygen group DriverTest.
 * @}
 */
