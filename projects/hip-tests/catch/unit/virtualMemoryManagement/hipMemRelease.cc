/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemRelease hipMemRelease
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipMemRelease(hipMemGenericAllocationHandle_t handle)` -
 * Release a memory handle representing a memory allocation which was previously
 * allocated through hipMemCreate.
 */

#include <hip_test_common.hh>

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemRelease.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemRelease_negative) {
  CTX_CREATE();
  SECTION("Nullptr to handle") {
    REQUIRE(hipMemRelease((hipMemGenericAllocationHandle_t) nullptr) == hipErrorInvalidValue);
  }
  CTX_DESTROY();
}

HIP_TEST_CASE(Unit_hipMemRelease_Capture) {
  CTX_CREATE();

  hipMemGenericAllocationHandle_t allocation_handle;
  size_t allocation_granularity = 0;
  int device_id = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, device_id));

  hipMemAllocationProp allocation_prop{};
  allocation_prop.type = hipMemAllocationTypePinned;
  allocation_prop.location.type = hipMemLocationTypeDevice;
  allocation_prop.location.id = device;

  HIP_CHECK(hipMemGetAllocationGranularity(&allocation_granularity, &allocation_prop,
                                           hipMemAllocationGranularityMinimum));
  HIP_CHECK(hipMemCreate(&allocation_handle, allocation_granularity, &allocation_prop, 0));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemRelease(allocation_handle));
  END_CAPTURE(stream);
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
