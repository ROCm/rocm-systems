/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

/**
 * @addtogroup hipGetDeviceCount hipGetDeviceCount
 * @{
 * @ingroup DeviceTest
 * `hipGetDeviceCount(int* count)` -
 * Return number of compute-capable devices.
 */

/**
 * Test Description
 * ------------------------
 *  - Passes invalid pointer as output parameter for device count - `nullptr`
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceCount.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGetDeviceCount_NegTst) {
  REQUIRE_FALSE(hipGetDeviceCount(nullptr) == hipSuccess);
}

/**
 * Test Description
 * ------------------------
 *  - Validates correct functionality when the device visibility
 *    environment variables are set. Uses unit/device/hipDeviceCount_exe.cc
 *    to set visibility.
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceCount.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGetDeviceCount_HideDevices) {
  int deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("This test requires more than 2 GPUs. Skipping.");
    return;
  }

  for (int i = deviceCount; i >= 1; i--) {
    std::string visibleStr;
    for (int j = 0; j < i; j++) {  // Generate a string which has first i devices
      visibleStr += std::to_string(j);
      if (j != (i - 1)) {
        visibleStr += ",";
      }
    }

    hip::SpawnProc proc("getDeviceCount", true);
    INFO("Output from process : " << proc.getOutput());
    REQUIRE(proc.run(visibleStr) == i);
  }
}

/**
 * End doxygen group DeviceTest.
 * @}
 */
