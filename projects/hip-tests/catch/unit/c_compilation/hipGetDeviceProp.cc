/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
/**
 * Test Description
 * ------------------------
 *    - Call a hip function (hipGetDeviceProperties) from a c compilation unit

 * Test source
 * ------------------------
 *    - catch/unit/c_compilation/hipGetDeviceProp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

extern "C" int hipGetDeviceProp();

HIP_TEST_CASE(Unit_hipGetDeviceProp_ctest) {
  int result = hipGetDeviceProp();
  REQUIRE(result == 1);
}
