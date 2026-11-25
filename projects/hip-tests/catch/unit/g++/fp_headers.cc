/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

bool test_fp8_host();
bool test_fp6_host();
bool test_fp4_host();

// fp_headers.cpp is built by g++ with -Werror; the conversions run on the host.
HIP_TEST_CASE(Unit_gcc_fp_headers) {
  REQUIRE(test_fp8_host());
  REQUIRE(test_fp6_host());
  REQUIRE(test_fp4_host());
}
