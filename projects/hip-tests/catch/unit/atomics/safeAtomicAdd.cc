/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "arithmetic_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup safeAtomicAdd safeAtomicAdd
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *    - Executes a single kernel on a single device wherein all threads will perform an atomic
 * addition on a target memory location. Each thread will add the same value to the memory location,
 * storing the return value into a separate output array slot corresponding to it. Once complete,
 * the output array and target memory is validated to contain all the expected values. Several
 * memory access patterns are tested:
 *      -# All threads add to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of safeAtomicAdd
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Shared memory
 *      - Several grid and block dimension combinations (only one block is used for shared memory).
 * Test source
 * ------------------------
 *    - unit/atomics/safeAtomicAdd.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_safeAtomicAdd_Positive, float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kSafeAdd>(1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kSafeAdd>(warp_size,
                                                                        sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kSafeAdd>(warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel two times concurrently on a single device wherein all threads will
 * perform an atomic addition on a target memory location. Each thread will add the same value to
 * the memory location, storing the return value into a separate output array slot corresponding
 * to it. Once complete, the output array and target memory is validated to contain all the
 * expected values. Several memory access patterns are tested:
 *      -# All threads add to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of safeAtomicAdd
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Several grid and block dimension combinations.
 * Test source
 * ------------------------
 *    - unit/atomics/safeAtomicAdd.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_safeAtomicAdd_Positive_Multi_Kernel, float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      SingleDeviceMultipleKernelTest<TestType, AtomicOperation::kSafeAdd>(2, 1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      SingleDeviceMultipleKernelTest<TestType, AtomicOperation::kSafeAdd>(2, warp_size,
                                                                          sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      SingleDeviceMultipleKernelTest<TestType, AtomicOperation::kSafeAdd>(2, warp_size,
                                                                          cache_line_size);
    }
  }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
