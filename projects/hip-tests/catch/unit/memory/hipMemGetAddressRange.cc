/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include <resource_guards.hh>

/**
 * @addtogroup hipMemGetAddressRange hipMemGetAddressRange
 * @{
 * @ingroup PeerToPeerTest
 * `hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr)` -
 * Get information on memory allocations.
 */

/**
 * Test Description
 * ------------------------
 *  - Allocate memory and check if base and size match allocated memory values.
 *  - Check for various offset values from base memory address:
 *    - Host address range
 *    - Device address range
 *    - Pitch address range
 * Test source
 * ------------------------
 *  - unit/memory/hipMemGetAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipMemGetAddressRange_Positive) {
  hipDeviceptr_t base_ptr;
  size_t mem_size = 0;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const int offset = GENERATE(0, 20, 40, 60, 80);

  SECTION("Host address range") {
    using LA = LinearAllocs;
    LinearAllocGuard<int> host_alloc(LA::hipHostMalloc, allocation_size);

    HIP_CHECK(hipMemGetAddressRange(&base_ptr, &mem_size,
                                    reinterpret_cast<hipDeviceptr_t>(host_alloc.ptr() + offset)));

    REQUIRE(reinterpret_cast<hipDeviceptr_t>(host_alloc.ptr()) == base_ptr);
    REQUIRE(mem_size == allocation_size);
  }
  SECTION("Device address range") {
    using LA = LinearAllocs;
    const auto device_allocation_type = GENERATE(LA::hipMalloc, LA::hipMallocManaged);
    LinearAllocGuard<int> device_alloc(device_allocation_type, allocation_size);

    HIP_CHECK(hipMemGetAddressRange(&base_ptr, &mem_size,
                                    reinterpret_cast<hipDeviceptr_t>(device_alloc.ptr() + offset)));

    REQUIRE(reinterpret_cast<hipDeviceptr_t>(device_alloc.ptr()) == base_ptr);
    REQUIRE(mem_size == allocation_size);
  }
  SECTION("Pitch address range") {
    size_t width = 32;
    size_t height = 32;
    LinearAllocGuard2D<int> device_alloc(width, height);

    HIP_CHECK(hipMemGetAddressRange(&base_ptr, &mem_size,
                                    reinterpret_cast<hipDeviceptr_t>(device_alloc.ptr() + offset)));

    REQUIRE(reinterpret_cast<hipDeviceptr_t>(device_alloc.ptr()) == base_ptr);
    REQUIRE(mem_size == (device_alloc.pitch() * height));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When device handle is not valid
 *      - Expected output: return `hipErrorNotFound`
 *    -# When offset is greated than allocated size
 *      - Expected output: return `hipErrorNotFound`
 * Test source
 * ------------------------
 *  - unit/memory/hipMemGetAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipMemGetAddressRange_Negative) {
  hipDeviceptr_t base_ptr;
  size_t mem_size = 0;
  const auto allocation_size = kPageSize / 2;
  const int offset = kPageSize;
  LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, allocation_size);

  hipDeviceptr_t dummy_ptr = NULL;

  SECTION("Device pointer is invalid") {
    HIP_CHECK_ERROR(hipMemGetAddressRange(&base_ptr, &mem_size, dummy_ptr), hipErrorNotFound);
  }
  SECTION("Offset is greater than allocated size") {
    HIP_CHECK_ERROR(
        hipMemGetAddressRange(&base_ptr, &mem_size,
                              reinterpret_cast<hipDeviceptr_t>(dst_alloc.ptr() + offset)),
        hipErrorNotFound);
  }
}

/**
 * End doxygen group PeerToPeerTest.
 * @}
 */
