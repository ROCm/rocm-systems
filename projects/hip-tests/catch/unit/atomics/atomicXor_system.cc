/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "bitwise_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup atomicXor_system atomicXor_system
 * @{
 * @ingroup AtomicsTest
 * `atomicXor_system(TestType* address, TestType* val)` -
 * performs system-wide atomic bitwise XOR between address and val, returns old value.
 */

// Helper function to run atomicXor_system tests for same address
template <typename TestType>
static void runAtomicXorSystemSameAddressTest() {
  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::MultipleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kXorSystem>(
          2, 2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicXor_system tests for adjacent addresses
template <typename TestType>
static void runAtomicXorSystemAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::MultipleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kXorSystem>(
          2, 2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicXor_system tests for scattered addresses
template <typename TestType>
static void runAtomicXorSystemScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::MultipleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kXorSystem>(
          2, 2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicXor_system from multiple threads on the same address.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicXor_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_atomicXor_system_Positive_Peer_GPUs_Same_Address) {
  SECTION("int") { runAtomicXorSystemSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicXorSystemSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicXorSystemSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicXorSystemSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicXor_system from multiple threads on adjacent addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicXor_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_atomicXor_system_Positive_Peer_GPUs_Adjacent_Addresses) {
  SECTION("int") { runAtomicXorSystemAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicXorSystemAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicXorSystemAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicXorSystemAdjacentAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicXor_system from multiple threads on scattered addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicXor_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_atomicXor_system_Positive_Peer_GPUs_Scattered_Addresses) {
  SECTION("int") { runAtomicXorSystemScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicXorSystemScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicXorSystemScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicXorSystemScatteredAddressesTest<unsigned long long>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
