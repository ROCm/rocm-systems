/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

bool test_host_defines();

HIP_TEST_CASE(Unit_gcc_host_defines) {
  REQUIRE(test_host_defines());
}
