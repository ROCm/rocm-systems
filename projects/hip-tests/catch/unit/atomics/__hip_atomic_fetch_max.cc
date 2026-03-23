/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "min_max_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup __hip_atomic_fetch_max __hip_atomic_fetch_max
 * @{
 * @ingroup AtomicsTest
 */

// Helper function to run __hip_atomic_fetch_max tests for WAVEFRONT scope, same address
template <typename TestType>
static void runHipAtomicFetchMaxWavefrontSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMax,
                                           __HIP_MEMORY_SCOPE_WAVEFRONT>(1, sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_max tests for WAVEFRONT scope, adjacent addresses
template <typename TestType>
static void runHipAtomicFetchMaxWavefrontAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMax,
                                           __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size,
                                                                         sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_max tests for WAVEFRONT scope, scattered addresses
template <typename TestType>
static void runHipAtomicFetchMaxWavefrontScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMax,
                                           __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size,
                                                                         cache_line_size);
    }
  }
}

// Helper function to run __hip_atomic_fetch_max tests for WORKGROUP scope, same address
template <typename TestType>
static void runHipAtomicFetchMaxWorkgroupSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMax,
                                           __HIP_MEMORY_SCOPE_WORKGROUP>(1, sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_max tests for WORKGROUP scope, adjacent addresses
template <typename TestType>
static void runHipAtomicFetchMaxWorkgroupAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMax,
                                           __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size,
                                                                         sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_max tests for WORKGROUP scope, scattered addresses
template <typename TestType>
static void runHipAtomicFetchMaxWorkgroupScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMax,
                                           __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size,
                                                                         cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MAX with memory scope WAVEFRONT from multiple threads on the same
 * address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_max.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_max_Positive_Wavefront_SameAddress) {
  SECTION("int") { runHipAtomicFetchMaxWavefrontSameAddressTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchMaxWavefrontSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchMaxWavefrontSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchMaxWavefrontSameAddressTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchMaxWavefrontSameAddressTest<float>(); }
  SECTION("double") { runHipAtomicFetchMaxWavefrontSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MAX with memory scope WAVEFRONT from multiple threads on adjacent
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_max.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_max_Positive_Wavefront_Adjacent_Addresses) {
  SECTION("int") { runHipAtomicFetchMaxWavefrontAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchMaxWavefrontAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchMaxWavefrontAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchMaxWavefrontAdjacentAddressesTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchMaxWavefrontAdjacentAddressesTest<float>(); }
  SECTION("double") { runHipAtomicFetchMaxWavefrontAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MAX with memory scope WAVEFRONT from multiple threads on scattered
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_max.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_max_Positive_Wavefront_Scattered_Addresses) {
  SECTION("int") { runHipAtomicFetchMaxWavefrontScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchMaxWavefrontScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchMaxWavefrontScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchMaxWavefrontScatteredAddressesTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchMaxWavefrontScatteredAddressesTest<float>(); }
  SECTION("double") { runHipAtomicFetchMaxWavefrontScatteredAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MAX with memory scope WORKGROUP from multiple threads on the same
 * address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_max.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_max_Positive_Workgroup_SameAddress) {
  SECTION("int") { runHipAtomicFetchMaxWorkgroupSameAddressTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchMaxWorkgroupSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchMaxWorkgroupSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchMaxWorkgroupSameAddressTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchMaxWorkgroupSameAddressTest<float>(); }
  SECTION("double") { runHipAtomicFetchMaxWorkgroupSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MAX with memory scope WORKGROUP from multiple threads on adjacent
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_max.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_max_Positive_Workgroup_Adjacent_Addresses) {
  SECTION("int") { runHipAtomicFetchMaxWorkgroupAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchMaxWorkgroupAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchMaxWorkgroupAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchMaxWorkgroupAdjacentAddressesTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchMaxWorkgroupAdjacentAddressesTest<float>(); }
  SECTION("double") { runHipAtomicFetchMaxWorkgroupAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MAX with memory scope WORKGROUP from multiple threads on scattered
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_max.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_max_Positive_Workgroup_Scattered_Addresses) {
  SECTION("int") { runHipAtomicFetchMaxWorkgroupScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchMaxWorkgroupScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchMaxWorkgroupScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchMaxWorkgroupScatteredAddressesTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchMaxWorkgroupScatteredAddressesTest<float>(); }
  SECTION("double") { runHipAtomicFetchMaxWorkgroupScatteredAddressesTest<double>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
