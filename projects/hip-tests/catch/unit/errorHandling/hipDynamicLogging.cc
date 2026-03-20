/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include "OutCapture.hh"

/**
 * @addtogroup hipDynamicLogging hipDynamicLogging
 * @{
 * @ingroup ErrorTest
 * `hipExtSetLoggingParams(size_t log_level, size_t log_size, size_t log_mask)` -
 * Sets logging parameters for HIP runtime.
 * `hipExtEnableLogging()` -
 * Enables HIP runtime logging.
 * `hipExtDisableLogging()` -
 * Disables HIP runtime logging.
 */

static bool hipDynamicLoggingTest() {
  // Create output capture instance
  OutCapture capture;
  capture.startCapture();

  // Set Logging params
  HIP_CHECK(hipExtSetLoggingParams(4, 0, -1));

  // Logging is disabled here - allocate memory
  int* dptr = nullptr;
  HIP_CHECK(hipMalloc(&dptr, sizeof(int)));

  // Stop capture after hipMalloc and check no output (logging disabled)
  std::string malloc_output = capture.stopCapture();
  if (malloc_output.size() != 0) {
    INFO("Unexpected logging output during hipMalloc (logging should be disabled): " << malloc_output);
    return false;
  }

  // Start capture before enabling logging
  capture.startCapture();

  // Enable logging and do memset
  HIP_CHECK(hipExtEnableLogging());
  HIP_CHECK(hipMemset(dptr, 0x00, sizeof(int)));

  // Disable logging
  HIP_CHECK(hipExtDisableLogging());

  // Stop capture after disabling logging and check for output
  std::string logging_output = capture.stopCapture();
  if (logging_output.size() == 0) {
    INFO("Expected logging output during enabled logging period, but got none");
    return false;
  }

  // Clean up
  HIP_CHECK(hipFree(dptr));

  INFO("Successfully captured HIP logging output (" << logging_output.size() << " bytes)");
  INFO("Logging output: " << logging_output);

  return true;
}

/**
 * Test Description
 * ------------------------
 *  - Validates that HIP dynamic logging works correctly:
 *    1. No output when logging is disabled
 *    2. Logging output is captured when logging is enabled
 *    3. hipMemset operation produces logging output during enabled period
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipDynamicLogging.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipDynamicLogging_Positive_Basic) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices <= 0) {
    HipTest::HIP_SKIP_TEST("Skipping hipDynamicLogging test - no devices available");
    return;
  }

  REQUIRE(hipDynamicLoggingTest() == true);
}

/**
 * Test Description
 * ------------------------
 *  - Validates that hipExtSetLoggingParams sets logging parameters correctly
 *    and that logging can be enabled/disabled multiple times
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipDynamicLogging.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipDynamicLogging_Positive_MultipleEnableDisable) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices <= 0) {
    HipTest::HIP_SKIP_TEST("Skipping hipDynamicLogging test - no devices available");
    return;
  }

  // Test multiple enable/disable cycles
  OutCapture capture;
  int* dptr = nullptr;
  HIP_CHECK(hipMalloc(&dptr, sizeof(int)));

  // Set different logging parameters
  HIP_CHECK(hipExtSetLoggingParams(3, 0, -1));

  for (int i = 0; i < 3; ++i) {
    // Start capture and enable logging
    capture.startCapture();
    HIP_CHECK(hipExtEnableLogging());
    HIP_CHECK(hipMemset(dptr, 0x42, sizeof(int)));
    HIP_CHECK(hipExtDisableLogging());

    // Check that we captured some output
    std::string output = capture.stopCapture();
    REQUIRE(output.size() > 0);
  }

  HIP_CHECK(hipFree(dptr));
}

/**
 * End doxygen group ErrorTest.
 * @}
 */
