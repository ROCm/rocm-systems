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
 * @addtogroup hipDeviceTotalMem hipDeviceTotalMem
 * @{
 * @ingroup DriverTest
 * `hipDeviceTotalMem(size_t* bytes, hipDevice_t device)` -
 * Returns the total amount of memory on the device.
 */

/**
 * Test Description
 * ------------------------
 *  - Validate handling of invalid arguments:
 *    -# When output pointer to the total memory is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When device ordinal is negative (-1)
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When device ordinal is out of bounds
 *      - Expected output: return `hipErrorInvalidDevice`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceTotalMem.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceTotalMem_NegTst") {
#if HT_NVIDIA
  HIP_CHECK(hipInit(0));
#endif
  size_t totMem;
  // Scenario 1
  SECTION("bytes is nullptr") {
    HIP_CHECK_ERROR(hipDeviceTotalMem(nullptr, 0), hipErrorInvalidValue);
  }

  // Scenario 2
  SECTION("device is -1") {
    HIP_CHECK_ERROR(hipDeviceTotalMem(&totMem, -1), hipErrorInvalidDevice);
  }

  // Scenario 3
  SECTION("device is out of bounds") {
    int numDevices;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    HIP_CHECK_ERROR(hipDeviceTotalMem(&totMem, numDevices), hipErrorInvalidDevice);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Check that the returned number of bytes is the same as the
 *    one from device attributes.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceTotalMem.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceTotalMem_ValidateTotalMem") {
  size_t totMem;
  int numDevices = 0;

  HIP_CHECK(hipGetDeviceCount(&numDevices));
  REQUIRE(numDevices != 0);

  hipDevice_t device;
  hipDeviceProp_t prop;
  auto devNo = GENERATE_COPY(range(0, numDevices));
  totMem = 0;
  HIP_CHECK(hipDeviceGet(&device, devNo));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  HIP_CHECK(hipDeviceTotalMem(&totMem, device));

  size_t free = 0, total = 0;
  HIP_CHECK(hipSetDevice(devNo));
  HIP_CHECK(hipMemGetInfo(&free, &total));

  REQUIRE(totMem == prop.totalGlobalMem);
  REQUIRE(total == totMem);
}

/**
 * Test Description
 * ------------------------
 *  - Check that total memory is returned when other device is
 *    set than the one in the API call.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceTotalMem.cc
 * Test requirements
 * ------------------------
 *  - Multi-device test
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceTotalMem_NonSelectedDevice") {
  auto deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Multi Device Test, will not run on single gpu systems. Skipping.");
    return;
  }

  for (int i = 1; i < deviceCount; i++) {
    HIP_CHECK(hipSetDevice(i - 1));
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, i));

    size_t totMem = 0;
    hipDeviceProp_t prop;
    HIP_CHECK(hipDeviceTotalMem(&totMem, device));
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    REQUIRE(totMem == prop.totalGlobalMem);
  }
}

/**
 * End doxygen group DriverTest.
 * @}
 */
