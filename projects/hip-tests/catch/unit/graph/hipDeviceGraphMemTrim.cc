/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipDeviceGraphMemTrim hipDeviceGraphMemTrim
 * @{
 * @ingroup GraphTest
 * `hipDeviceGraphMemTrim(int device)` - Free unused memory on specific device used for graph back
 * to OS.
 */

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that unused memory used for graph can be freed on each device.
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGraphMemTrim.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE(Unit_hipDeviceGraphMemTrim_Positive_Default) {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));

  // Check for each device
  HIP_CHECK(hipDeviceGraphMemTrim(device));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipDeviceGraphMemTrim behavior with invalid arguments:
 *    -# Device is not valid
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGraphMemTrim.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE(Unit_hipDeviceGraphMemTrim_Negative_Parameters) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  SECTION("Device is not valid") {
    HIP_CHECK_ERROR(hipDeviceGraphMemTrim(num_dev), hipErrorInvalidDevice);
  }
}

/**
 * End doxygen group GraphTest.
 * @}
 */
