/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "min_max_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup unsafeAtomicMin unsafeAtomicMin
 * @{
 * @ingroup AtomicsTest
 * `unsafeAtomicMin(TestType* address, TestType* val)` -
 * calculates minimum between address and val, returns old value.
 */

// Helper function to run unsafeAtomicMin tests for same address (single kernel)
template <typename TestType>
static void runUnsafeAtomicMinSameAddressTest() {
  const auto iterations = TestParameterStore::instance().getIterationsForCurrentLevel();

  for (auto current = 0; current < iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMin>(
          1, sizeof(TestType));
    }
  }
}

// Helper function to run unsafeAtomicMin tests for adjacent addresses (single kernel)
template <typename TestType>
static void runUnsafeAtomicMinAdjacentAddressesTest() {
  const auto iterations = TestParameterStore::instance().getIterationsForCurrentLevel();

  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMin>(
          warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run unsafeAtomicMin tests for scattered addresses (single kernel)
template <typename TestType>
static void runUnsafeAtomicMinScatteredAddressesTest() {
  const auto iterations = TestParameterStore::instance().getIterationsForCurrentLevel();

  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMin>(
          warp_size, cache_line_size);
    }
  }
}

// Helper function to run unsafeAtomicMin tests for same address (multi kernel)
template <typename TestType>
static void runUnsafeAtomicMinMultiKernelSameAddressTest() {
  const auto iterations = TestParameterStore::instance().getIterationsForCurrentLevel();

  for (auto current = 0; current < iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMin>(
          2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run unsafeAtomicMin tests for adjacent addresses (multi kernel)
template <typename TestType>
static void runUnsafeAtomicMinMultiKernelAdjacentAddressesTest() {
  const auto iterations = TestParameterStore::instance().getIterationsForCurrentLevel();

  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMin>(
          2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run unsafeAtomicMin tests for scattered addresses (multi kernel)
template <typename TestType>
static void runUnsafeAtomicMinMultiKernelScatteredAddressesTest() {
  const auto iterations = TestParameterStore::instance().getIterationsForCurrentLevel();

  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMin>(
          2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMin from multiple threads on the same address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_unsafeAtomicMin_Positive_SameAddress) {
  SECTION("float") { runUnsafeAtomicMinSameAddressTest<float>(); }
  SECTION("double") { runUnsafeAtomicMinSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMin from multiple threads on adjacent addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_unsafeAtomicMin_Positive_Adjacent_Addresses) {
  SECTION("float") { runUnsafeAtomicMinAdjacentAddressesTest<float>(); }
  SECTION("double") { runUnsafeAtomicMinAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMin from multiple threads on the scattered addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_unsafeAtomicMin_Positive_Scattered_Addresses) {
  SECTION("float") { runUnsafeAtomicMinScatteredAddressesTest<float>(); }
  SECTION("double") { runUnsafeAtomicMinScatteredAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMin from multiple threads on the same address.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_unsafeAtomicMin_Positive_Multi_Kernel_Same_Address) {
  SECTION("float") { runUnsafeAtomicMinMultiKernelSameAddressTest<float>(); }
  SECTION("double") { runUnsafeAtomicMinMultiKernelSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMin from multiple threads on adjacent addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_unsafeAtomicMin_Positive_Multi_Kernel_Adjacent_Addresses) {
  SECTION("float") { runUnsafeAtomicMinMultiKernelAdjacentAddressesTest<float>(); }
  SECTION("double") { runUnsafeAtomicMinMultiKernelAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMin from multiple threads on the scattered addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_unsafeAtomicMin_Positive_Multi_Kernel_Scattered_Addresses) {
  SECTION("float") { runUnsafeAtomicMinMultiKernelScatteredAddressesTest<float>(); }
  SECTION("double") { runUnsafeAtomicMinMultiKernelScatteredAddressesTest<double>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
