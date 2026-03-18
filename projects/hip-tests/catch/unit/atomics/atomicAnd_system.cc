/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "bitwise_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup atomicAnd_system atomicAnd_system
 * @{
 * @ingroup AtomicsTest
 * `atomicAnd_system(TestType* address, TestType* val)` -
 * performs system-wide atomic bitwise AND between address and val, returns old value.
 */

// Helper function to run atomicAnd_system tests for same address
template <typename TestType>
static void runAtomicAndSystemSameAddressTest() {
  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::MultipleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kAndSystem>(
          2, 2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicAnd_system tests for adjacent addresses
template <typename TestType>
static void runAtomicAndSystemAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::MultipleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kAndSystem>(
          2, 2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicAnd_system tests for scattered addresses
template <typename TestType>
static void runAtomicAndSystemScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::MultipleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kAndSystem>(
          2, 2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicAnd_system from multiple threads on the same address.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicAnd_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicAnd_system_Positive_Peer_GPUs_Same_Address) {
  SECTION("int") { runAtomicAndSystemSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicAndSystemSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicAndSystemSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicAndSystemSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicAnd_system from multiple threads on adjacent addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicAnd_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicAnd_system_Positive_Peer_GPUs_Adjacent_Addresses) {
  SECTION("int") { runAtomicAndSystemAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicAndSystemAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicAndSystemAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicAndSystemAdjacentAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicAnd_system from multiple threads on scattered addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicAnd_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicAnd_system_Positive_Peer_GPUs_Scattered_Addresses) {
  SECTION("int") { runAtomicAndSystemScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicAndSystemScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicAndSystemScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicAndSystemScatteredAddressesTest<unsigned long long>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
