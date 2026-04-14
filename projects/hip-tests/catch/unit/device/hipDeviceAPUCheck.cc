/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipDeviceAPUCheck
 * @{
 * @ingroup DeviceTest
 * `hipGetDeviceProperties(const hipDeviceProp_t* prop, int device)` -
 * Device which matches `hipDeviceProp_t` is returned.
 */

/**
 * Test Description
 * ------------------------
 *  - Prints if the system is an APU or has discrete graphics card.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceAPUCheck.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE(Unit_hipDeviceAPUCheck) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  if (!prop.integrated) {
    HipTest::HIP_SKIP_TEST("This device is a Discrete Graphics card. So skipping test");
    return;
  } else {
    std::cout << "This device is an APU" << std::endl;
  }
}
