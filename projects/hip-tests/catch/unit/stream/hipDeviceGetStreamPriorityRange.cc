/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipDeviceGetStreamPriorityRange_Default - Check if device stream piority range is valid
*/

#include <hip_test_common.hh>

TEST_CASE(Unit_hipDeviceGetStreamPriorityRange_Default) {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));

  REQUIRE(priority_low >= priority_high);
}
