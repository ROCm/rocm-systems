/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

extern "C" {
#include "LaunchKernel.h"
}

/**
 * Test Description
 * ------------------------
 *    - calling launchKernel which is c function from catch2
 * and compile with gcc compiler and verify the results.

 * Test source
 * ------------------------
 *    - catch/unit/gcc/gccTest.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE(Unit_LaunchKernelgccTests) {
  printf("Calling launchKernel files from here\n");
  int result = launchKernel();
  REQUIRE(result == 1);
}

/**
 * Test Description
 * ------------------------
 *    - Calling hipMalloc which is c file from catch2 and compile
 * with gcc compiler and verify the results.

 * Test source
 * ------------------------
 *    - catch/unit/gcc/gccTest.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE(Unit_hipMallocgccTests) {
  printf("Calling hipMalloc files from here\n");
  int result = hipMallocfunc();
  REQUIRE(result == 1);
}
