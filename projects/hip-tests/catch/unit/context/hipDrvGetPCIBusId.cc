/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

TEST_CASE(Unit_hipDeviceGetPCIBusId_Functional) {
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  char pciBusId[13];
  memset(pciBusId, 0, 13);
  HIP_CHECK(hipDeviceGetPCIBusId(pciBusId, 13, device));
  REQUIRE(pciBusId[0] != '\0');
}
