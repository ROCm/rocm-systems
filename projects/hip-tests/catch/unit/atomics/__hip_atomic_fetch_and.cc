/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "bitwise_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup __hip_atomic_fetch_and __hip_atomic_fetch_and
 * @{
 * @ingroup AtomicsTest
 */

// Helper function to run __hip_atomic_fetch_and tests for WAVEFRONT scope, same address
template <typename TestType>
static void runHipAtomicFetchAndWavefrontSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kBuiltinAnd,
                                            __HIP_MEMORY_SCOPE_WAVEFRONT>(1, sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_and tests for WAVEFRONT scope, adjacent addresses
template <typename TestType>
static void runHipAtomicFetchAndWavefrontAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kBuiltinAnd,
                                            __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size,
                                                                          sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_and tests for WAVEFRONT scope, scattered addresses
template <typename TestType>
static void runHipAtomicFetchAndWavefrontScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kBuiltinAnd,
                                            __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size,
                                                                          cache_line_size);
    }
  }
}

// Helper function to run __hip_atomic_fetch_and tests for WORKGROUP scope, same address
template <typename TestType>
static void runHipAtomicFetchAndWorkgroupSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kBuiltinAnd,
                                            __HIP_MEMORY_SCOPE_WORKGROUP>(1, sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_and tests for WORKGROUP scope, adjacent addresses
template <typename TestType>
static void runHipAtomicFetchAndWorkgroupAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kBuiltinAnd,
                                            __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size,
                                                                          sizeof(TestType));
    }
  }
}

// Helper function to run __hip_atomic_fetch_and tests for WORKGROUP scope, scattered addresses
template <typename TestType>
static void runHipAtomicFetchAndWorkgroupScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kBuiltinAnd,
                                            __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size,
                                                                          cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic AND with memory scope WAVEFRONT from multiple threads on the same
 * address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_and.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___hip_atomic_fetch_and_Positive_Wavefront_SameAddress") {
  SECTION("int") { runHipAtomicFetchAndWavefrontSameAddressTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchAndWavefrontSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchAndWavefrontSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchAndWavefrontSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic AND with memory scope WAVEFRONT from multiple threads on adjacent
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_and.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___hip_atomic_fetch_and_Positive_Wavefront_Adjacent_Addresses") {
  SECTION("int") { runHipAtomicFetchAndWavefrontAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchAndWavefrontAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchAndWavefrontAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchAndWavefrontAdjacentAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic AND with memory scope WAVEFRONT from multiple threads on scattered
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_and.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___hip_atomic_fetch_and_Positive_Wavefront_Scattered_Addresses") {
  SECTION("int") { runHipAtomicFetchAndWavefrontScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchAndWavefrontScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchAndWavefrontScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchAndWavefrontScatteredAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic AND with memory scope WORKGROUP from multiple threads on the same
 * address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_and.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___hip_atomic_fetch_and_Positive_Workgroup_SameAddress") {
  SECTION("int") { runHipAtomicFetchAndWorkgroupSameAddressTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchAndWorkgroupSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchAndWorkgroupSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchAndWorkgroupSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic AND with memory scope WORKGROUP from multiple threads on adjacent
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_and.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___hip_atomic_fetch_and_Positive_Workgroup_Adjacent_Addresses") {
  SECTION("int") { runHipAtomicFetchAndWorkgroupAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchAndWorkgroupAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchAndWorkgroupAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchAndWorkgroupAdjacentAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic AND with memory scope WORKGROUP from multiple threads on scattered
 * addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_and.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___hip_atomic_fetch_and_Positive_Workgroup_Scattered_Addresses") {
  SECTION("int") { runHipAtomicFetchAndWorkgroupScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runHipAtomicFetchAndWorkgroupScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicFetchAndWorkgroupScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicFetchAndWorkgroupScatteredAddressesTest<unsigned long long>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
