/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

TEST_CASE(Unit_hipMemGetInfo_FreeLessThanTotal) {
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
