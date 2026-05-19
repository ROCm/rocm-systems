/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
HIP_TEST_CASE(Unit_hipTestDeviceLimit_Basic) {
  size_t heap;
  HIP_CHECK(hipDeviceGetLimit(&heap, hipLimitMallocHeapSize));
  REQUIRE(heap != 0);
}
