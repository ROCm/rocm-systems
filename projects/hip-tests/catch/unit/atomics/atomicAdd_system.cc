/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "arithmetic_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup atomicAdd_system atomicAdd_system
 * @{
 * @ingroup AtomicsTest
 */

// Helper function to run atomicAdd_system tests for peer GPUs
template <typename TestType>
static void runAtomicAddSystemPeerGPUsTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kAddSystem>(
          2, 2, 1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kAddSystem>(
          2, 2, warp_size, sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kAddSystem>(
          2, 2, warp_size, cache_line_size);
    }
  }
}

// Helper function to run atomicAdd_system tests for host and GPU
template <typename TestType>
static void runAtomicAddSystemHostAndGPUTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kAddSystem>(
          1, 1, 1, sizeof(TestType), 4);
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kAddSystem>(
          1, 1, warp_size, sizeof(TestType), 4);
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kAddSystem>(
          1, 1, warp_size, cache_line_size, 4);
    }
  }
}

// Helper function to run atomicAdd_system tests for host and peer GPUs
template <typename TestType>
static void runAtomicAddSystemHostAndPeerGPUsTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kAddSystem>(
          2, 2, 1, sizeof(TestType), 4);
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kAddSystem>(
          2, 2, warp_size, sizeof(TestType), 4);
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kAddSystem>(
          2, 2, warp_size, cache_line_size, 4);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel two times concurrently on a two devices wherein all threads will perform
 * an atomic addition on a target memory location. Each thread will add the same value to the memory
 * location, storing the return value into a separate output array slot corresponding to it. Once
 * complete, the output array and target memory is validated to contain all the expected values.
 * Several memory access patterns are tested:
 *      -# All threads add to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicAdd_system
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Several grid and block dimension combinations.
 * Test source
 * ------------------------
 *    - unit/atomics/atomicAdd_system.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_atomicAdd_system_Positive_Peer_GPUs) {
  SECTION("int") { runAtomicAddSystemPeerGPUsTest<int>(); }
  SECTION("unsigned int") { runAtomicAddSystemPeerGPUsTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicAddSystemPeerGPUsTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicAddSystemPeerGPUsTest<unsigned long long>(); }
  SECTION("float") { runAtomicAddSystemPeerGPUsTest<float>(); }
  SECTION("double") { runAtomicAddSystemPeerGPUsTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel on a single device wherein all threads will perform
 * an atomic addition on a target memory location. Each thread will add the same value to the memory
 * location, storing the return value into a separate output array slot corresponding to it. While
 * the kernel is running, the host performs atomic additions, in 4 threads, on the same memory
 * location(s). Once complete, the output array and target memory is validated to contain all the
 * expected values. Several memory access patterns are tested:
 *      -# All threads add to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicAdd_system
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Several grid and block dimension combinations.
 * Test source
 * ------------------------
 *    - unit/atomics/atomicAdd_system.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_atomicAdd_system_Positive_Host_And_GPU) {
  SECTION("int") { runAtomicAddSystemHostAndGPUTest<int>(); }
  SECTION("unsigned int") { runAtomicAddSystemHostAndGPUTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicAddSystemHostAndGPUTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicAddSystemHostAndGPUTest<unsigned long long>(); }
  SECTION("float") { runAtomicAddSystemHostAndGPUTest<float>(); }
  SECTION("double") { runAtomicAddSystemHostAndGPUTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel two times on two devices wherein all threads will perform
 * an atomic addition on a target memory location. Each thread will add the same value to the memory
 * location, storing the return value into a separate output array slot corresponding to it. While
 * the kernel is running, the host performs atomic additions, in 4 threads, on the same memory
 * location(s). Once complete, the output array and target memory is validated to contain all the
 * expected values. Several memory access patterns are tested:
 *      -# All threads add to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicAdd_system
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Several grid and block dimension combinations.
 * Test source
 * ------------------------
 *    - unit/atomics/atomicAdd_system.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_atomicAdd_system_Positive_Host_And_Peer_GPUs) {
  SECTION("int") { runAtomicAddSystemHostAndPeerGPUsTest<int>(); }
  SECTION("unsigned int") { runAtomicAddSystemHostAndPeerGPUsTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicAddSystemHostAndPeerGPUsTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicAddSystemHostAndPeerGPUsTest<unsigned long long>(); }
  SECTION("float") { runAtomicAddSystemHostAndPeerGPUsTest<float>(); }
  SECTION("double") { runAtomicAddSystemHostAndPeerGPUsTest<double>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
