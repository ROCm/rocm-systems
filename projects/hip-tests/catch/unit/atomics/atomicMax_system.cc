/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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
 * @addtogroup atomicMax_system atomicMax_system
 * @{
 * @ingroup AtomicsTest
 * `atomicMax_system(TestType* address, TestType* val)` -
 * performs system-wide atomic maximum between address and val, returns old value.
 */

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMax_system from multiple threads on the same address.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
#if HT_AMD
TEMPLATE_TEST_CASE("Unit_atomicMax_system_Positive_Peer_GPUs_Same_Address", "", int, unsigned int,
                   unsigned long, unsigned long long, float, double) {
#else
TEMPLATE_TEST_CASE("Unit_atomicMax_system_Positive_Peer_GPUs_Same_Address", "", int, unsigned int,
                   unsigned long, unsigned long long) {
#endif
  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::MultipleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMaxSystem>(
          2, 2, 1, sizeof(TestType));
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMax_system from multiple threads on adjacent addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
#if HT_AMD
TEMPLATE_TEST_CASE("Unit_atomicMax_system_Positive_Peer_GPUs_Adjacent_Addresses", "", int,
                   unsigned int, unsigned long, unsigned long long, float, double) {
#else
TEMPLATE_TEST_CASE("Unit_atomicMax_system_Positive_Peer_GPUs_Adjacent_Addresses", "", int,
                   unsigned int, unsigned long, unsigned long long) {
#endif
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::MultipleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMaxSystem>(
          2, 2, warp_size, sizeof(TestType));
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMax_system from multiple threads on scaterred addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
#if HT_AMD
TEMPLATE_TEST_CASE("Unit_atomicMax_system_Positive_Peer_GPUs_Scattered_Addresses", "", int,
                   unsigned int, unsigned long, unsigned long long, float, double) {
#else
TEMPLATE_TEST_CASE("Unit_atomicMax_system_Positive_Peer_GPUs_Scattered_Addresses", "", int,
                   unsigned int, unsigned long, unsigned long long) {
#endif
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::MultipleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMaxSystem>(
          2, 2, warp_size, cache_line_size);
    }
  }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
