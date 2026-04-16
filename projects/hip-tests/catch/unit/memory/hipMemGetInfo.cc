/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

HIP_TEST_CASE(Unit_hipMemGetInfo_FreeLessThanTotal) {
  unsigned int* A_mem{nullptr};
  size_t freeMemInit, totalMemInit;
  size_t freeMem, totalMem;

  HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));
  REQUIRE(freeMemInit <= totalMemInit);
  HIP_CHECK(hipMalloc(&A_mem, 1024));
  HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));
  REQUIRE(freeMem < totalMem);
  REQUIRE(totalMem == totalMemInit);

  HIP_CHECK(hipFree(A_mem));
}

/**
 * Test Description
 * ------------------------
 *  - Verify that free memory never exceeds total memory.
 *  - This is a universal sanity check that catches memory reporting bugs.
 *  - Prevents regression where free=25.85 GiB > total=15.35 GiB occurred.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemGetInfo.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipMemGetInfo_ConsistencyCheck) {
  size_t free, total;
  HIP_CHECK(hipMemGetInfo(&free, &total));

  // Free memory must never exceed total memory
  REQUIRE(free <= total);
}
