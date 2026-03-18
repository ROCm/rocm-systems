/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include "hipMalloc.h"
/**
 * @addtogroup hipMalloc hipMalloc
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMalloc(void** ptr, size_t size)` -
 * Allocate memory on the default accelerator.
 * @}
 */

/**
 * Test Description
 * ------------------------
 *    - Allocate memory by using hipMalloc API and verify hipSuccess is returned.

 * Test source
 * ------------------------
 *    - catch/unit/g++/hipMalloc.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE(Unit_hipMalloc_gpptest) {
  printf("calling cpp function from here\n");
  int result = MallocFunc();
  REQUIRE(result == 1);
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
