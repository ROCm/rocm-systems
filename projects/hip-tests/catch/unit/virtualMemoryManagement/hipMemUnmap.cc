/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemUnmap hipMemUnmap
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemUnmap (void* ptr, size_t size)` -
 * Unmap memory allocation of a given address range.
 */


#include <hip_test_common.hh>

#include "hip_vmm_common.hh"

constexpr int N = (1 << 13);

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemUnmap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE(Unit_hipMemUnmap_negative) {
  size_t granularity = 0;
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

  hipMemGenericAllocationHandle_t handle;
  void* ptrA;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));

  SECTION("nullptr to ptrA") { REQUIRE(hipMemUnmap(nullptr, size_mem) == hipErrorInvalidValue); }

  SECTION("pass zero to size") { REQUIRE(hipMemUnmap(ptrA, 0) == hipErrorInvalidValue); }

  SECTION("unmap a smaller size") {
    REQUIRE(hipMemUnmap(ptrA, (size_mem - 1)) == hipErrorInvalidValue);
  }

  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

TEST_CASE(Unit_hipMemUnmap_Capture) {
  CTX_CREATE();
  size_t granularity = 0;
  constexpr size_t kBufferSize = N * sizeof(int);
  int device_id = 0;
  hipDevice_t device;

  HIP_CHECK(hipDeviceGet(&device, device_id));
  checkVMMSupported(device);

  hipMemAllocationProp allocation_prop{};
  allocation_prop.type = hipMemAllocationTypePinned;
  allocation_prop.location.type = hipMemLocationTypeDevice;
  allocation_prop.location.id = device;

  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &allocation_prop,
                                           hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t mem_size = ((granularity + kBufferSize - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t allocation_handle;
  void* device_ptr = nullptr;
  HIP_CHECK(hipMemCreate(&allocation_handle, mem_size, &allocation_prop, 0));
  HIP_CHECK(hipMemAddressReserve(&device_ptr, mem_size, 0, nullptr, 0));
  HIP_CHECK(hipMemMap(device_ptr, mem_size, 0, allocation_handle, 0));
  HIP_CHECK(hipMemRelease(allocation_handle));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemUnmap(device_ptr, mem_size));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemAddressFree(device_ptr, mem_size));
  CTX_DESTROY();
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
