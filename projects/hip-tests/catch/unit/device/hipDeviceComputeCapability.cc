/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
HIP_TEST_CASE(Unit_hipDeviceComputeCapability_Negative) {
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
HIP_TEST_CASE(Unit_hipDeviceComputeCapability_ValidateVersion) {
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
