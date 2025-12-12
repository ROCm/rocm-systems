/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipRuntimeGetVersion hipRuntimeGetVersion
 * @{
 * @ingroup DriverTest
 * `hipRuntimeGetVersion(int* runtimeVersion)` -
 * Returns the approximate HIP runtime version.
 * On HIP/HCC path this function returns HIP runtime patch version
 * (a 5 digit code) however on
 * HIP/NVCC path this function return CUDA runtime version.
 */

/**
 * Test Description
 * ------------------------
 *  - Checks that valid runtime version is returned.
 *  - Print out the runtime version.
 * Test source
 * ------------------------
 *  - unit/device/hipRuntimeGetVersion.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipRuntimeGetVersion_Positive) {
  int runtimeVersion = -1;
  HIP_CHECK(hipRuntimeGetVersion(&runtimeVersion));
  REQUIRE(runtimeVersion > 0);
  INFO("Runtime version " << runtimeVersion);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the runtime version is nullptr
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/device/hipRuntimeGetVersion.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipRuntimeGetVersion_Negative) {
  // If initialization is attempted with nullptr, error shall be reported
  CHECK_FALSE(hipRuntimeGetVersion(nullptr) == hipSuccess);
}

/**
 * End doxygen group DriverTest.
 * @}
 */
