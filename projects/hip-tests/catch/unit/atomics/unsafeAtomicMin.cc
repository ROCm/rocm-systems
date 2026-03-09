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
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMin>(
          1, sizeof(TestType));
    }
  }
}

// Helper function to run unsafeAtomicMin tests for adjacent addresses (single kernel)
template <typename TestType>
static void runUnsafeAtomicMinAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMin>(
          warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run unsafeAtomicMin tests for scattered addresses (single kernel)
template <typename TestType>
static void runUnsafeAtomicMinScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMin>(
          warp_size, cache_line_size);
    }
  }
}

// Helper function to run unsafeAtomicMin tests for same address (multi kernel)
template <typename TestType>
static void runUnsafeAtomicMinMultiKernelSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMin>(
          2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run unsafeAtomicMin tests for adjacent addresses (multi kernel)
template <typename TestType>
static void runUnsafeAtomicMinMultiKernelAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMin>(
          2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run unsafeAtomicMin tests for scattered addresses (multi kernel)
template <typename TestType>
static void runUnsafeAtomicMinMultiKernelScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
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
TEST_CASE(Unit_unsafeAtomicMin_Positive_SameAddress) {
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
TEST_CASE(Unit_unsafeAtomicMin_Positive_Adjacent_Addresses) {
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
TEST_CASE(Unit_unsafeAtomicMin_Positive_Scattered_Addresses) {
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
TEST_CASE(Unit_unsafeAtomicMin_Positive_Multi_Kernel_Same_Address) {
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
TEST_CASE(Unit_unsafeAtomicMin_Positive_Multi_Kernel_Adjacent_Addresses) {
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
TEST_CASE(Unit_unsafeAtomicMin_Positive_Multi_Kernel_Scattered_Addresses) {
  SECTION("float") { runUnsafeAtomicMinMultiKernelScatteredAddressesTest<float>(); }
  SECTION("double") { runUnsafeAtomicMinMultiKernelScatteredAddressesTest<double>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
