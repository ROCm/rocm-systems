/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "bitwise_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup __hip_atomic_fetch_or __hip_atomic_fetch_or
 * @{
 * @ingroup AtomicsTest
 */

// Helper function to run __hip_atomic_fetch_or tests for WAVEFRONT scope, same address
template <typename TestType>
static void runHipAtomicFetchOrWavefrontSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kBuiltinOr,
                                            __HIP_MEMORY_SCOPE_WAVEFRONT>(1, sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_or tests for WAVEFRONT scope, adjacent addresses
template <typename TestType>
static void runHipAtomicFetchOrWavefrontAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kBuiltinOr,
                                            __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size,
                                                                          sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_or tests for WAVEFRONT scope, scattered addresses
template <typename TestType>
static void runHipAtomicFetchOrWavefrontScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kBuiltinOr,
                                            __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size,
                                                                          cache_line_size);
    }
  }
}

// Helper function to run __hip_atomic_fetch_or tests for WORKGROUP scope, same address
template <typename TestType>
static void runHipAtomicFetchOrWorkgroupSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kBuiltinOr,
                                            __HIP_MEMORY_SCOPE_WORKGROUP>(1, sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_or tests for WORKGROUP scope, adjacent addresses
template <typename TestType>
static void runHipAtomicFetchOrWorkgroupAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kBuiltinOr,
                                            __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size,
                                                                          sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_or tests for WORKGROUP scope, scattered addresses
template <typename TestType>
static void runHipAtomicFetchOrWorkgroupScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kBuiltinOr,
                                            __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size,
                                                                          cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic OR with memory scope WAVEFRONT from multiple threads on the same
 * address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_or.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_or_Positive_Wavefront_SameAddress) {
  SECTION("int") { runHipAtomicFetchOrWavefrontSameAddressTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchOrWavefrontSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchOrWavefrontSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchOrWavefrontSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic OR with memory scope WAVEFRONT from multiple threads on adjacent
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_or.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_or_Positive_Wavefront_Adjacent_Addresses) {
  SECTION("int") { runHipAtomicFetchOrWavefrontAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchOrWavefrontAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchOrWavefrontAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchOrWavefrontAdjacentAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic OR with memory scope WAVEFRONT from multiple threads on scattered
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_or.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_or_Positive_Wavefront_Scattered_Addresses) {
  SECTION("int") { runHipAtomicFetchOrWavefrontScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchOrWavefrontScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchOrWavefrontScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchOrWavefrontScatteredAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic OR with memory scope WORKGROUP from multiple threads on the same
 * address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_or.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_or_Positive_Workgroup_SameAddress) {
  SECTION("int") { runHipAtomicFetchOrWorkgroupSameAddressTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchOrWorkgroupSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchOrWorkgroupSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchOrWorkgroupSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic OR with memory scope WORKGROUP from multiple threads on adjacent
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_or.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_or_Positive_Workgroup_Adjacent_Addresses) {
  SECTION("int") { runHipAtomicFetchOrWorkgroupAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchOrWorkgroupAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchOrWorkgroupAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchOrWorkgroupAdjacentAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic OR with memory scope WORKGROUP from multiple threads on scattered
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_or.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___hip_atomic_fetch_or_Positive_Workgroup_Scattered_Addresses) {
  SECTION("int") { runHipAtomicFetchOrWorkgroupScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchOrWorkgroupScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchOrWorkgroupScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchOrWorkgroupScatteredAddressesTest<unsigned long long>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
