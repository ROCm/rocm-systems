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
 * @addtogroup __hip_atomic_fetch_min __hip_atomic_fetch_min
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MIN with memory scope WAVEFRONT in multiple memory patterns.
 *    -# From multiple threads on the same address;
 *    -# From multiple threads on adjacent addresses;
 *    -# From multiple threads on the scattered addresses. Addresses are spread by L1 cache line
 *        size;
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_min.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit___hip_atomic_fetch_min_Positive_Wavefront", "", int, unsigned int,
                   unsigned long, unsigned long long, float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMin,
                                           __HIP_MEMORY_SCOPE_WAVEFRONT>(1, sizeof(TestType));
    }
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMin,
                                           __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size,
                                                                         sizeof(TestType));
    }
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMin,
                                           __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size,
                                                                         cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs a builtin atomic MIN with memory scope WORKGROUP in multiple memory patterns.
 *    -# From multiple threads on the same address;
 *    -# From multiple threads on adjacent addresses;
 *    -# From multiple threads on the scattered addresses. Addresses are spread by L1 cache line
 *        size;
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/__hip_atomic_fetch_min.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit___hip_atomic_fetch_min_Positive_Workgroup", "", int, unsigned int,
                   unsigned long, unsigned long long, float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMin,
                                           __HIP_MEMORY_SCOPE_WORKGROUP>(1, sizeof(TestType));
    }
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMin,
                                           __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size,
                                                                         sizeof(TestType));
    }
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kBuiltinMin,
                                           __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size,
                                                                         cache_line_size);
    }
  }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
