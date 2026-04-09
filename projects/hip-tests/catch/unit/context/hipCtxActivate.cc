/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * Test Description
 * ------------------------
 *  - Validates that the primary context is active when hipSetDevice is called
 *      - Expected output: is_active = 1
 * Test source
 * ------------------------
 *  - unit/context/hipCtxActivate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipSetDevice_CheckPrimaryCtxState) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  unsigned flags;
  int is_active;
  HIP_CHECK(hipDevicePrimaryCtxGetState(device_id, &flags, &is_active));

  REQUIRE(is_active == 1);
}
