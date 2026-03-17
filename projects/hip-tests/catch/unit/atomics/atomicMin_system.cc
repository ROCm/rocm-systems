/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "min_max_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup atomicMin_system atomicMin_system
 * @{
 * @ingroup AtomicsTest
 * `atomicMin_system(TestType* address, TestType* val)` -
 * performs system-wide atomic minimum between address and val, returns old value.
 */

// Helper function to run atomicMin_system tests for same address
template <typename TestType>
static void runAtomicMinSystemSameAddressTest() {
  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::MultipleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMinSystem>(
          2, 2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicMin_system tests for adjacent addresses
template <typename TestType>
static void runAtomicMinSystemAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::MultipleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMinSystem>(
          2, 2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicMin_system tests for scattered addresses
template <typename TestType>
static void runAtomicMinSystemScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::MultipleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMinSystem>(
          2, 2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMin_system from multiple threads on the same address.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicMin_system_Positive_Peer_GPUs_Same_Address) {
  SECTION("int") { runAtomicMinSystemSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicMinSystemSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMinSystemSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMinSystemSameAddressTest<unsigned long long>(); }
#ifdef HT_AMD
  SECTION("float") { runAtomicMinSystemSameAddressTest<float>(); }
  SECTION("double") { runAtomicMinSystemSameAddressTest<double>(); }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMin_system from multiple threads on adjacent addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicMin_system_Positive_Peer_GPUs_Adjacent_Addresses) {
  SECTION("int") { runAtomicMinSystemAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicMinSystemAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMinSystemAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMinSystemAdjacentAddressesTest<unsigned long long>(); }
#ifdef HT_AMD
  SECTION("float") { runAtomicMinSystemAdjacentAddressesTest<float>(); }
  SECTION("double") { runAtomicMinSystemAdjacentAddressesTest<double>(); }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMin_system from multiple threads on scaterred addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicMin_system_Positive_Peer_GPUs_Scattered_Addresses) {
  SECTION("int") { runAtomicMinSystemScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicMinSystemScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMinSystemScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMinSystemScatteredAddressesTest<unsigned long long>(); }
#ifdef HT_AMD
  SECTION("float") { runAtomicMinSystemScatteredAddressesTest<float>(); }
  SECTION("double") { runAtomicMinSystemScatteredAddressesTest<double>(); }
#endif
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
