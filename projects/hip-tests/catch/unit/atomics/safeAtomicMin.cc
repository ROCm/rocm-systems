/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "min_max_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup safeAtomicMin safeAtomicMin
 * @{
 * @ingroup AtomicsTest
 * `safeAtomicMin(TestType* address, TestType* val)` -
 * calculates minimum between address and val, returns old value.
 */

// Helper function to run safeAtomicMin tests for same address (single kernel)
template <typename TestType>
static void runSafeAtomicMinSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kSafeMin>(
          1, sizeof(TestType));
    }
  }
}

// Helper function to run safeAtomicMin tests for adjacent addresses (single kernel)
template <typename TestType>
static void runSafeAtomicMinAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kSafeMin>(
          warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run safeAtomicMin tests for scattered addresses (single kernel)
template <typename TestType>
static void runSafeAtomicMinScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kSafeMin>(
          warp_size, cache_line_size);
    }
  }
}

// Helper function to run safeAtomicMin tests for same address (multiple kernels)
template <typename TestType>
static void runSafeAtomicMinMultiKernelSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kSafeMin>(
          2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run safeAtomicMin tests for adjacent addresses (multiple kernels)
template <typename TestType>
static void runSafeAtomicMinMultiKernelAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kSafeMin>(
          2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run safeAtomicMin tests for scattered addresses (multiple kernels)
template <typename TestType>
static void runSafeAtomicMinMultiKernelScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kSafeMin>(
          2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs safeAtomicMin from multiple threads on the same address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/safeAtomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_safeAtomicMin_Positive_SameAddress) {
  SECTION("float") { runSafeAtomicMinSameAddressTest<float>(); }
  SECTION("double") { runSafeAtomicMinSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs safeAtomicMin from multiple threads on adjacent addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/safeAtomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_safeAtomicMin_Positive_Adjacent_Addresses) {
  SECTION("float") { runSafeAtomicMinAdjacentAddressesTest<float>(); }
  SECTION("double") { runSafeAtomicMinAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs safeAtomicMin from multiple threads on the scattered addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/safeAtomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_safeAtomicMin_Positive_Scattered_Addresses) {
  SECTION("float") { runSafeAtomicMinScatteredAddressesTest<float>(); }
  SECTION("double") { runSafeAtomicMinScatteredAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs safeAtomicMin from multiple threads on the same address.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/safeAtomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_safeAtomicMin_Positive_Multi_Kernel_Same_Address) {
  SECTION("float") { runSafeAtomicMinMultiKernelSameAddressTest<float>(); }
  SECTION("double") { runSafeAtomicMinMultiKernelSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs safeAtomicMin from multiple threads on adjacent addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/safeAtomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_safeAtomicMin_Positive_Multi_Kernel_Adjacent_Addresses) {
  SECTION("float") { runSafeAtomicMinMultiKernelAdjacentAddressesTest<float>(); }
  SECTION("double") { runSafeAtomicMinMultiKernelAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs safeAtomicMin from multiple threads on the scattered addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/safeAtomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_safeAtomicMin_Positive_Multi_Kernel_Scattered_Addresses) {
  SECTION("float") { runSafeAtomicMinMultiKernelScatteredAddressesTest<float>(); }
  SECTION("double") { runSafeAtomicMinMultiKernelScatteredAddressesTest<double>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
