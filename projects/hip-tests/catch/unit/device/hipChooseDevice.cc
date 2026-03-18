/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipChooseDevice hipChooseDevice
 * @{
 * @ingroup DeviceTest
 * `hipChooseDevice(int* device, const hipDeviceProp_t* prop)` -
 * Device which matches `hipDeviceProp_t` is returned.
 */

/**
 * Test Description
 * ------------------------
 *  - Validate chosen device against gotten device properties.
 * Test source
 * ------------------------
 *  - unit/device/hipChooseDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipChooseDevice_ValidateDevId) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = -1;
  HIP_CHECK(hipChooseDevice(&dev, &prop));
  REQUIRE_FALSE(dev < 0);
  REQUIRE_FALSE(dev >= numDevices);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When pointer to the device is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When pointer to the properties is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/device/hipChooseDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipChooseDevice_NegTst) {
  hipDeviceProp_t prop;
  int dev = -1;

  SECTION("dev is nullptr") { REQUIRE_FALSE(hipSuccess == hipChooseDevice(nullptr, &prop)); }

  SECTION("prop is nullptr") { REQUIRE_FALSE(hipSuccess == hipChooseDevice(&dev, nullptr)); }
}

/**
 * End doxygen group DeviceTest.
 * @}
 */
