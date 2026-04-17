/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "min_max_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup safeAtomicMax safeAtomicMax
 * @{
 * @ingroup AtomicsTest
 * `safeAtomicMax(TestType* address, TestType* val)` -
 * calculates maximum between address and val, returns old value.
 */

// Helper function to run safeAtomicMax tests for same address (single kernel)
template <typename TestType>
static void runSafeAtomicMaxSameAddressTest() {
  const auto iterations = TestParameterStore::instance().getIterationsForCurrentLevel();

  for (auto current = 0; current < iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kSafeMax>(
          1, sizeof(TestType));
    }
  }
}

// Helper function to run safeAtomicMax tests for adjacent addresses (single kernel)
template <typename TestType>
static void runSafeAtomicMaxAdjacentAddressesTest() {
  const auto iterations = TestParameterStore::instance().getIterationsForCurrentLevel();

  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kSafeMax>(
          warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run safeAtomicMax tests for scattered addresses (single kernel)
template <typename TestType>
static void runSafeAtomicMaxScatteredAddressesTest() {
  const auto iterations = TestParameterStore::instance().getIterationsForCurrentLevel();

  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kSafeMax>(
          warp_size, cache_line_size);
    }
  }
}

// Helper function to run safeAtomicMax tests for same address (multiple kernels)
template <typename TestType>
static void runSafeAtomicMaxMultiKernelSameAddressTest() {
  const auto iterations = TestParameterStore::instance().getIterationsForCurrentLevel();

  for (auto current = 0; current < iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kSafeMax>(
          2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run safeAtomicMax tests for adjacent addresses (multiple kernels)
template <typename TestType>
static void runSafeAtomicMaxMultiKernelAdjacentAddressesTest() {
  const auto iterations = TestParameterStore::instance().getIterationsForCurrentLevel();

  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kSafeMax>(
          2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run safeAtomicMax tests for scattered addresses (multiple kernels)
template <typename TestType>
static void runSafeAtomicMaxMultiKernelScatteredAddressesTest() {
  const auto iterations = TestParameterStore::instance().getIterationsForCurrentLevel();

  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kSafeMax>(
          2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs safeAtomicMax from multiple threads on the same address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/safeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_safeAtomicMax_Positive_SameAddress) {
  SECTION("float") { runSafeAtomicMaxSameAddressTest<float>(); }
  SECTION("double") { runSafeAtomicMaxSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs safeAtomicMax from multiple threads on adjacent addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/safeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_safeAtomicMax_Positive_Adjacent_Addresses) {
  SECTION("float") { runSafeAtomicMaxAdjacentAddressesTest<float>(); }
  SECTION("double") { runSafeAtomicMaxAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs safeAtomicMax from multiple threads on the scattered addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/safeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_safeAtomicMax_Positive_Scattered_Addresses) {
  SECTION("float") { runSafeAtomicMaxScatteredAddressesTest<float>(); }
  SECTION("double") { runSafeAtomicMaxScatteredAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs safeAtomicMax from multiple threads on the same address.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/safeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_safeAtomicMax_Positive_Multi_Kernel_Same_Address) {
  SECTION("float") { runSafeAtomicMaxMultiKernelSameAddressTest<float>(); }
  SECTION("double") { runSafeAtomicMaxMultiKernelSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs safeAtomicMax from multiple threads on adjacent addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/safeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_safeAtomicMax_Positive_Multi_Kernel_Adjacent_Addresses) {
  SECTION("float") { runSafeAtomicMaxMultiKernelAdjacentAddressesTest<float>(); }
  SECTION("double") { runSafeAtomicMaxMultiKernelAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs safeAtomicMax from multiple threads on the scattered addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/safeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_safeAtomicMax_Positive_Multi_Kernel_Scattered_Addresses) {
  SECTION("float") { runSafeAtomicMaxMultiKernelScatteredAddressesTest<float>(); }
  SECTION("double") { runSafeAtomicMaxMultiKernelScatteredAddressesTest<double>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
