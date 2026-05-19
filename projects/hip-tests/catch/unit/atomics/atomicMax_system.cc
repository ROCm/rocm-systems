/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "min_max_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup atomicMax_system atomicMax_system
 * @{
 * @ingroup AtomicsTest
 * `atomicMax_system(TestType* address, TestType* val)` -
 * performs system-wide atomic maximum between address and val, returns old value.
 */

// Helper function to run atomicMax_system tests for same address
template <typename TestType>
static void runAtomicMaxSystemSameAddressTest() {
  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::MultipleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMaxSystem>(
          2, 2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicMax_system tests for adjacent addresses
template <typename TestType>
static void runAtomicMaxSystemAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::MultipleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMaxSystem>(
          2, 2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicMax_system tests for scattered addresses
template <typename TestType>
static void runAtomicMaxSystemScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::MultipleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMaxSystem>(
          2, 2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMax_system from multiple threads on the same address.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_atomicMax_system_Positive_Peer_GPUs_Same_Address) {
  SECTION("int") { runAtomicMaxSystemSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicMaxSystemSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMaxSystemSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMaxSystemSameAddressTest<unsigned long long>(); }
#ifdef HT_AMD
  SECTION("float") { runAtomicMaxSystemSameAddressTest<float>(); }
  SECTION("double") { runAtomicMaxSystemSameAddressTest<double>(); }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMax_system from multiple threads on adjacent addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_atomicMax_system_Positive_Peer_GPUs_Adjacent_Addresses) {
  SECTION("int") { runAtomicMaxSystemAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicMaxSystemAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMaxSystemAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMaxSystemAdjacentAddressesTest<unsigned long long>(); }
#ifdef HT_AMD
  SECTION("float") { runAtomicMaxSystemAdjacentAddressesTest<float>(); }
  SECTION("double") { runAtomicMaxSystemAdjacentAddressesTest<double>(); }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMax_system from multiple threads on scaterred addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_atomicMax_system_Positive_Peer_GPUs_Scattered_Addresses) {
  SECTION("int") { runAtomicMaxSystemScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicMaxSystemScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMaxSystemScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMaxSystemScatteredAddressesTest<unsigned long long>(); }
#ifdef HT_AMD
  SECTION("float") { runAtomicMaxSystemScatteredAddressesTest<float>(); }
  SECTION("double") { runAtomicMaxSystemScatteredAddressesTest<double>(); }
#endif
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
