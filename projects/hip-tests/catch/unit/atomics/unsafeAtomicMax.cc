/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "min_max_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup unsafeAtomicMax unsafeAtomicMax
 * @{
 * @ingroup AtomicsTest
 * `unsafeAtomicMax(TestType* address, TestType* val)` -
 * calculates maximum between address and val, returns old value.
 */

// Helper function to run unsafeAtomicMax tests for same address (single kernel)
template <typename TestType>
static void runUnsafeAtomicMaxSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMax>(
          1, sizeof(TestType));
    }
  }
}

// Helper function to run unsafeAtomicMax tests for adjacent addresses (single kernel)
template <typename TestType>
static void runUnsafeAtomicMaxAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMax>(
          warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run unsafeAtomicMax tests for scattered addresses (single kernel)
template <typename TestType>
static void runUnsafeAtomicMaxScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMax>(
          warp_size, cache_line_size);
    }
  }
}

// Helper function to run unsafeAtomicMax tests for same address (multi kernel)
template <typename TestType>
static void runUnsafeAtomicMaxMultiKernelSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMax>(
          2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run unsafeAtomicMax tests for adjacent addresses (multi kernel)
template <typename TestType>
static void runUnsafeAtomicMaxMultiKernelAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMax>(
          2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run unsafeAtomicMax tests for scattered addresses (multi kernel)
template <typename TestType>
static void runUnsafeAtomicMaxMultiKernelScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMax>(
          2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMax from multiple threads on the same address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_unsafeAtomicMax_Positive_SameAddress) {
  SECTION("float") { runUnsafeAtomicMaxSameAddressTest<float>(); }
  SECTION("double") { runUnsafeAtomicMaxSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMax from multiple threads on adjacent addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_unsafeAtomicMax_Positive_Adjacent_Addresses) {
  SECTION("float") { runUnsafeAtomicMaxAdjacentAddressesTest<float>(); }
  SECTION("double") { runUnsafeAtomicMaxAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMax from multiple threads on the scattered addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_unsafeAtomicMax_Positive_Scattered_Addresses) {
  SECTION("float") { runUnsafeAtomicMaxScatteredAddressesTest<float>(); }
  SECTION("double") { runUnsafeAtomicMaxScatteredAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMax from multiple threads on the same address.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_unsafeAtomicMax_Positive_Multi_Kernel_Same_Address) {
  SECTION("float") { runUnsafeAtomicMaxMultiKernelSameAddressTest<float>(); }
  SECTION("double") { runUnsafeAtomicMaxMultiKernelSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMax from multiple threads on adjacent addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_unsafeAtomicMax_Positive_Multi_Kernel_Adjacent_Addresses) {
  SECTION("float") { runUnsafeAtomicMaxMultiKernelAdjacentAddressesTest<float>(); }
  SECTION("double") { runUnsafeAtomicMaxMultiKernelAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMax from multiple threads on the scattered addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_unsafeAtomicMax_Positive_Multi_Kernel_Scattered_Addresses) {
  SECTION("float") { runUnsafeAtomicMaxMultiKernelScatteredAddressesTest<float>(); }
  SECTION("double") { runUnsafeAtomicMaxMultiKernelScatteredAddressesTest<double>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
