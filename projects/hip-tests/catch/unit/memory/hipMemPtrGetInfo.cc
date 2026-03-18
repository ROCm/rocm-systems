/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
This testfile verifies the basic scenario of hipMemPtrGetInfo API
*/
#include <hip_test_common.hh>
struct MemInfo {
  float a;
  int b;
  void* c;
};

/*
This testcase verifies the basic scenario of
hipMemPtrGetInfo API
1. Allocates specific size of memory for the variables
2. Gets the allocated size of that variable using hipMemPtrGetInfo API
3. Validates the initial size and allocated size
*/
TEST_CASE(Unit_hipMemPtrGetInfo_Basic) {
  int* iPtr;
  float* fPtr;
  MemInfo* sPtr;
  size_t sSetSize = 1024, sGetSize;
  HIP_CHECK(hipMalloc(&iPtr, sSetSize));
  HIP_CHECK(hipMalloc(&fPtr, sSetSize));
  HIP_CHECK(hipMalloc(&sPtr, sSetSize));
  HIP_CHECK(hipMemPtrGetInfo(iPtr, &sGetSize));
  REQUIRE(sGetSize == sSetSize);
  HIP_CHECK(hipMemPtrGetInfo(fPtr, &sGetSize));
  REQUIRE(sGetSize == sSetSize);
  HIP_CHECK(hipMemPtrGetInfo(sPtr, &sGetSize));
  REQUIRE(sGetSize == sSetSize);

  HIP_CHECK(hipFree(iPtr));
  HIP_CHECK(hipFree(fPtr));
  HIP_CHECK(hipFree(sPtr));
}

/*
This testcase verifies the scenario of
hipMemPtrGetInfo API being called on a zero-sized allocation.
*/
TEST_CASE(Unit_hipMemPtrGetInfo_SizeZeroAllocation) {
  int* iPtr;
  size_t sSetSize = 0, sGetSize;
  HIP_CHECK(hipMalloc(&iPtr, sSetSize));
  HIP_CHECK(hipMemPtrGetInfo(iPtr, &sGetSize));
  REQUIRE(sGetSize == sSetSize);

  HIP_CHECK(hipFree(iPtr));
}
