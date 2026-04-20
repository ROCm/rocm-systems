/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "min_max_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup __hip_atomic_fetch_min __hip_atomic_fetch_min
 * @{
 * @ingroup AtomicsTest
 */

// Helper function to run __hip_atomic_fetch_min tests for WAVEFRONT scope, same address
template <typename TestType>
static void runHipAtomicFetchMinWavefrontSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMin,
                                           __HIP_MEMORY_SCOPE_WAVEFRONT>(1, sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_min tests for WAVEFRONT scope, adjacent addresses
template <typename TestType>
static void runHipAtomicFetchMinWavefrontAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMin,
                                           __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size,
                                                                         sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_min tests for WAVEFRONT scope, scattered addresses
template <typename TestType>
static void runHipAtomicFetchMinWavefrontScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMin,
                                           __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size,
                                                                         cache_line_size);
    }
  }
}

// Helper function to run __hip_atomic_fetch_min tests for WORKGROUP scope, same address
template <typename TestType>
static void runHipAtomicFetchMinWorkgroupSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMin,
                                           __HIP_MEMORY_SCOPE_WORKGROUP>(1, sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_min tests for WORKGROUP scope, adjacent addresses
template <typename TestType>
static void runHipAtomicFetchMinWorkgroupAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMin,
                                           __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size,
                                                                         sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_min tests for WORKGROUP scope, scattered addresses
template <typename TestType>
static void runHipAtomicFetchMinWorkgroupScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMin,
                                           __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size,
                                                                         cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MIN with memory scope WAVEFRONT from multiple threads on the same
 * address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_min.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_min_Positive_Wavefront_SameAddress) {
  SECTION("int") { runHipAtomicFetchMinWavefrontSameAddressTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchMinWavefrontSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchMinWavefrontSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchMinWavefrontSameAddressTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchMinWavefrontSameAddressTest<float>(); }
  SECTION("double") { runHipAtomicFetchMinWavefrontSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MIN with memory scope WAVEFRONT from multiple threads on adjacent
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_min.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_min_Positive_Wavefront_Adjacent_Addresses) {
  SECTION("int") { runHipAtomicFetchMinWavefrontAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchMinWavefrontAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchMinWavefrontAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchMinWavefrontAdjacentAddressesTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchMinWavefrontAdjacentAddressesTest<float>(); }
  SECTION("double") { runHipAtomicFetchMinWavefrontAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MIN with memory scope WAVEFRONT from multiple threads on scattered
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_min.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_min_Positive_Wavefront_Scattered_Addresses) {
  SECTION("int") { runHipAtomicFetchMinWavefrontScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchMinWavefrontScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchMinWavefrontScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchMinWavefrontScatteredAddressesTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchMinWavefrontScatteredAddressesTest<float>(); }
  SECTION("double") { runHipAtomicFetchMinWavefrontScatteredAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MIN with memory scope WORKGROUP from multiple threads on the same
 * address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_min.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_min_Positive_Workgroup_SameAddress) {
  SECTION("int") { runHipAtomicFetchMinWorkgroupSameAddressTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchMinWorkgroupSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchMinWorkgroupSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchMinWorkgroupSameAddressTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchMinWorkgroupSameAddressTest<float>(); }
  SECTION("double") { runHipAtomicFetchMinWorkgroupSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MIN with memory scope WORKGROUP from multiple threads on adjacent
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_min.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_min_Positive_Workgroup_Adjacent_Addresses) {
  SECTION("int") { runHipAtomicFetchMinWorkgroupAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchMinWorkgroupAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchMinWorkgroupAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchMinWorkgroupAdjacentAddressesTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchMinWorkgroupAdjacentAddressesTest<float>(); }
  SECTION("double") { runHipAtomicFetchMinWorkgroupAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MIN with memory scope WORKGROUP from multiple threads on scattered
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_min.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_min_Positive_Workgroup_Scattered_Addresses) {
  SECTION("int") { runHipAtomicFetchMinWorkgroupScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchMinWorkgroupScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchMinWorkgroupScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchMinWorkgroupScatteredAddressesTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicFetchMinWorkgroupScatteredAddressesTest<float>(); }
  SECTION("double") { runHipAtomicFetchMinWorkgroupScatteredAddressesTest<double>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
