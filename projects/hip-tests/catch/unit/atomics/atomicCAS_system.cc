/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "arithmetic_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup atomicCAS_system atomicCAS_system
 * @{
 * @ingroup AtomicsTest
 */

#ifdef HT_NVIDIA
#define TYPES
#else
#define TYPES , float, double
#endif

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel two times concurrently on a two devices wherein all threads will perform
 * an atomic addition, implemented using an atomic CAS operation, on a target memory location. Each
 * thread will add the same value to the memory location, storing the return value into a separate
 * output array slot corresponding to it. Once complete, the output array and target memory is
 * validated to contain all the expected values. Several memory access patterns are tested:
 *      -# All threads exchange to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicCAS_system
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Several grid and block dimension combinations.
 * Test source
 * ------------------------
 *    - unit/atomics/atomicCAS_system.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_atomicCAS_system_Positive_Peer_GPUs,
                   int, unsigned int, unsigned long long,
                   unsigned short int TYPES) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  SECTION("Same address") {
    MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
        2, 2, 1, sizeof(TestType));
  }

  SECTION("Scattered addresses") {
    MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
        2, 2, warp_size, cache_line_size);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel on a single device wherein all threads will perform
 * an atomic addition, implemented using an atomic CAS operation, on a target memory location.
 * Each thread will add the same value to the memory location, storing the return value into a
 * separate output array slot corresponding to it. While the kernel is running, the host
 * performs atomic additions, in 4 threads, on the same memory location(s). Once complete, the
 * output array and target memory is validated to contain all the expected values. Several
 * memory access patterns are tested:
 *      -# All threads exchange to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicCAS_system
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Several grid and block dimension combinations.
 * Test source
 * ------------------------
 *    - unit/atomics/atomicCAS_system.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_atomicCAS_system_Positive_Host_And_GPU,
                   int, unsigned int, unsigned long long,
                   unsigned short int TYPES) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  SECTION("Same address") {
    MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
        1, 1, 1, sizeof(TestType), 4);
  }

  SECTION("Adjacent addresses") {
    MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
        1, 1, warp_size, sizeof(TestType), 4);
  }

  SECTION("Scattered addresses") {
    MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
        1, 1, warp_size, cache_line_size, 4);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel two times on two devices wherein all threads will perform
 * an atomic addition, implemented using an atomic CAS operation, on a target memory location.
 * Each thread will add the same value to the memory location, storing the return value into a
 * separate output array slot corresponding to it. While the kernel is running, the host
 * performs atomic additions, in 4 threads, on the same memory location(s). Once complete, the
 * output array and target memory is validated to contain all the expected values. Several
 * memory access patterns are tested:
 *      -# All threads exchange to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicCAS_system
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Several grid and block dimension combinations.
 * Test source
 * ------------------------
 *    - unit/atomics/atomicCAS_system.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_atomicCAS_system_Positive_Host_And_Peer_GPUs,
                   int, unsigned int, unsigned long long,
                   unsigned short int TYPES) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  SECTION("Same address") {
    MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
        2, 2, 1, sizeof(TestType), 4);
  }

  SECTION("Scattered addresses") {
    MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
        2, 2, warp_size, cache_line_size, 4);
  }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
