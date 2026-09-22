/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "bitwise_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup atomicOr_system atomicOr_system
 * @{
 * @ingroup AtomicsTest
 * `atomicOr_system(TestType* address, TestType* val)` -
 * performs system-wide atomic bitwise OR between address and val, returns old value.
 */

// Helper function to run atomicOr_system tests for same address
template <typename TestType>
static void runAtomicOrSystemSameAddressTest() {
  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::MultipleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kOrSystem>(
          2, 2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicOr_system tests for adjacent addresses
template <typename TestType>
static void runAtomicOrSystemAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::MultipleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kOrSystem>(
          2, 2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicOr_system tests for scattered addresses
template <typename TestType>
static void runAtomicOrSystemScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::MultipleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kOrSystem>(
          2, 2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicOr_system from multiple threads on the same address.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicOr_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_atomicOr_system_Positive_Peer_GPUs_Same_Address) {
  SECTION("int") { runAtomicOrSystemSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicOrSystemSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicOrSystemSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicOrSystemSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicOr_system from multiple threads on adjacent addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicOr_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_atomicOr_system_Positive_Peer_GPUs_Adjacent_Addresses) {
  SECTION("int") { runAtomicOrSystemAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicOrSystemAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicOrSystemAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicOrSystemAdjacentAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicOr_system from multiple threads on scattered addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicOr_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_atomicOr_system_Positive_Peer_GPUs_Scattered_Addresses) {
  SECTION("int") { runAtomicOrSystemScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicOrSystemScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicOrSystemScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicOrSystemScatteredAddressesTest<unsigned long long>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
