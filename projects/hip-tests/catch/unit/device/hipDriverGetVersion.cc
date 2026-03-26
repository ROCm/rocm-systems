/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
HIP_TEST_CASE(Unit_hipDriverGetVersion_Positive) {
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
HIP_TEST_CASE(Unit_hipDriverGetVersion_Negative) {
  // If initialization is attempted with nullptr, error shall be reported
  HIP_CHECK_ERROR(hipDriverGetVersion(nullptr), hipErrorInvalidValue);
}


/**
 * End doxygen group DriverTest.
 * @}
 */
