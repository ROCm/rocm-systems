/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
TEST_CASE(Unit_hipInit_Positive) {
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
TEST_CASE(Unit_hipInit_Negative) {
  // If initialization is attempted with invalid flag, error shall be reported
  unsigned int invalid_flag = 1;
  HIP_CHECK_ERROR(hipInit(invalid_flag), hipErrorInvalidValue);
}


/**
 * End doxygen group DriverTest.
 * @}
 */
