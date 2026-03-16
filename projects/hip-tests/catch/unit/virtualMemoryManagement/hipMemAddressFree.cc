/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemAddressFree hipMemAddressFree
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemAddressFree (void* devPtr, size_t size)` -
 * Frees an address range reservation made via hipMemAddressReserve.
 */

#include <hip_test_common.hh>

#include "hip_vmm_common.hh"

#define DATA_SIZE (1 << 13)

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemAddressFree.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE(Unit_hipMemAddressFree_negative) {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;
  CTX_CREATE();
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  // Allocate virtual address range
  void* ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));

  SECTION("nullptr to devptr") {
    REQUIRE(hipMemAddressFree(nullptr, size_mem) == hipErrorInvalidValue);
  }

  SECTION("pass zero to size") { REQUIRE(hipMemAddressFree(ptrA, 0) == hipErrorInvalidValue); }

  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

TEST_CASE(Unit_hipMemAddressFree_Capture) {
  CTX_CREATE();
  size_t granularity = 0;
  size_t buffer_size = DATA_SIZE * sizeof(int);
  int device_id = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, device_id));
  checkVMMSupported(device);

  hipMemAllocationProp alloc_prop{};
  alloc_prop.type = hipMemAllocationTypePinned;
  alloc_prop.location.type = hipMemLocationTypeDevice;
  alloc_prop.location.id = device;

  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &alloc_prop,
                                           hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);

  size_t reserved_size = ((granularity + buffer_size - 1) / granularity) * granularity;

  void* reserved_ptr = nullptr;
  HIP_CHECK(hipMemAddressReserve(&reserved_ptr, reserved_size, 0, nullptr, 0));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemAddressFree(reserved_ptr, reserved_size));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
  CTX_DESTROY();
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
