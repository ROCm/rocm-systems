/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "arithmetic_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup __hip_atomic_fetch_add __hip_atomic_fetch_add
 * @{
 * @ingroup AtomicsTest
 * ________________________
 * Test cases from other modules:
 *    - @ref Unit_AtomicBuiltins_Negative_Parameters_RTC
 */

// Helper function to run __hip_atomic_fetch_add tests with WAVEFRONT scope
template <typename TestType>
static void runHipAtomicFetchAddWavefrontTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kBuiltinAdd,
                                   __HIP_MEMORY_SCOPE_WAVEFRONT>(1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kBuiltinAdd,
                                   __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size, sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kBuiltinAdd,
                                   __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size, cache_line_size);
    }
  }
}

// Helper function to run __hip_atomic_fetch_add tests with WORKGROUP scope
template <typename TestType>
static void runHipAtomicFetchAddWorkgroupTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kBuiltinAdd,
                                   __HIP_MEMORY_SCOPE_WORKGROUP>(1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kBuiltinAdd,
                                   __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size, sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kBuiltinAdd,
                                   __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size, cache_line_size);
    }
  }
}

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
 *      - All overloads of __hip_atomic_fetch_add
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Shared memory
 *      - WAVEFRONT memory scope.
 * Test source
 * ------------------------
 *    - unit/atomics/__hip_atomic_fetch_add.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_add_Positive_Wavefront) {
  SECTION("int") { runHipAtomicFetchAddWavefrontTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchAddWavefrontTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchAddWavefrontTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchAddWavefrontTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchAddWavefrontTest<float>(); }
  SECTION("double") { runHipAtomicFetchAddWavefrontTest<double>(); }
}

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
 *      - All overloads of __hip_atomic_fetch_add
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Shared memory
 *      - WORKGROUP memory scope.
 * Test source
 * ------------------------
 *    - unit/atomics/__hip_atomic_fetch_add.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_add_Positive_Workgroup) {
  SECTION("int") { runHipAtomicFetchAddWorkgroupTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchAddWorkgroupTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchAddWorkgroupTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchAddWorkgroupTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchAddWorkgroupTest<float>(); }
  SECTION("double") { runHipAtomicFetchAddWorkgroupTest<double>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
