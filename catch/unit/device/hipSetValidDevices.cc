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
#include <hip_test_process.hh>

/**
 * @addtogroup hipSetValidDevices hipSetValidDevices
 * @{
 * @ingroup DeviceTest
 * `hipSetValidDevices(int* device_arr, int len)` -
 * Sets a list of valid devices that can be used by HIP runtime
 */

/**
 * Test Description
 * ------------------------
 *  - Validates that hipSetValidDevices API can handle invalid parameters
 *    -#  When device array passed is `nullptr` but len is not `0`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -#  When len exceeds the number of devices in the system
 *      - Expected output: return `hipErrorInvalidValue`
 *    -#  When the device Id specified in the list does not exist
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -#  When len exceeds the number of device IDs passed in the device array
 *      - Expected output: return `hipErrorInvalidDevice`
 * Test source
 * ------------------------
 *  - unit/device/hipSetValidDevices.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipSetValidDevices_Negative") {
  auto totalDevices = HipTest::getDeviceCount();
  int device_arr1[] = {0};
  int device_arr2[] = {totalDevices + 1, 1};
 
  SECTION("Devicearray - nullptr") {
    HIP_CHECK_ERROR(hipSetValidDevices(nullptr, 1), hipErrorInvalidValue);
  }
  SECTION("len > total devices") {
    HIP_CHECK_ERROR(hipSetValidDevices(device_arr1, totalDevices + 2), hipErrorInvalidValue);
  }
  SECTION("DeviceId is not valid") {
    HIP_CHECK_ERROR(hipSetValidDevices(device_arr2, 2), hipErrorInvalidDevice);
  }
  SECTION("len > size of device array") {
    HIP_CHECK_ERROR(hipSetValidDevices(device_arr1, 2), hipErrorInvalidDevice);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates the functionality of hipSetValidDevices by default and
 *    also by resetting using hipSetDevice
 * Test source
 * ------------------------
 *  - unit/device/hipSetValidDevices.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipSetValidDevices_Positive_Basic") {
  int totalDevices = HipTest::getDeviceCount();
  if (totalDevices < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 or more GPUs. Skipping.");
    return;
  }
 
  //By default, without setting any device, validate that 0th device is being used
  int device;
  HIP_CHECK(hipGetDevice(&device));
  REQUIRE(device == 0);

  //Set the devices 1 and 0 as valid ones using hipSetValidDevices
  int valid_devices1[] = {1, 0};
  HIP_CHECK(hipSetValidDevices(valid_devices1, 2));
 
  //Fetch the device and validate that the device 1 is being used currently
  //Since the device 1 is set as the first valid device earlier
  HIP_CHECK(hipGetDevice(&device));
  REQUIRE(device == 1);

  if (totalDevices > 2) {
    // Set the device 2 as the current device
    HIP_CHECK(hipSetDevice(2));
    // Fetch the device and validate that the device 2 is being used currently
    // This is to confirm that hipSetDevice sets the device (if the device exists)
    // irrespective of the valid devices set by the app
    HIP_CHECK(hipGetDevice(&device));
    REQUIRE(device == 2);
    // Set 0 as the valid device
    int valid_devices2[] = {0};
    HIP_CHECK(hipSetValidDevices(valid_devices2, 1));
    // Fetch the device and validate that the device 2 is the current device still
    // Since hipSetValidDevices doesn't take effect once hipSetDevice is set
    HIP_CHECK(hipGetDevice(&device));
    REQUIRE(device == 2);
  }
}
