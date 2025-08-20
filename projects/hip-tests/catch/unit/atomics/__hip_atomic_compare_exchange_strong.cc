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

#include "arithmetic_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup __hip_atomic_compare_exchange_strong __hip_atomic_compare_exchange_strong
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *    - Executes a single kernel on a single device wherein all threads will perform an atomic
 * addition on a target memory location. Each thread will add the same value to the memory location,
 * storing the return value into a separate output array slot corresponding to it. Once complete,
 * the output array and target memory is validated to contain all the expected values. Several
 * memory access patterns are tested:
 *      -# All threads add to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of __hip_atomic_compare_exchange_strong
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Shared memory
 *      - WAVEFRONT memory scope.
 * Test source
 * ------------------------
 *    - unit/atomics/__hip_atomic_compare_exchange_strong.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit___hip_atomic_compare_exchange_strong_Positive_Wavefront", "", int,
                   unsigned int, unsigned long, unsigned long long, float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kBuiltinCAS,
                                   __HIP_MEMORY_SCOPE_WAVEFRONT>(1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kBuiltinCAS,
                                   __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size, sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kBuiltinCAS,
                                   __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a single kernel on a single device wherein all threads will perform an atomic
 * addition on a target memory location. Each thread will add the same value to the memory location,
 * storing the return value into a separate output array slot corresponding to it. Once complete,
 * the output array and target memory is validated to contain all the expected values. Several
 * memory access patterns are tested:
 *      -# All threads add to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of __hip_atomic_compare_exchange_strong
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Shared memory
 *      - WORKGROUP memory scope.
 * Test source
 * ------------------------
 *    - unit/atomics/__hip_atomic_compare_exchange_strong.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit___hip_atomic_compare_exchange_strong_Positive_Workgroup", "", int,
                   unsigned int, unsigned long, unsigned long long, float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kBuiltinCAS,
                                   __HIP_MEMORY_SCOPE_WORKGROUP>(1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kBuiltinCAS,
                                   __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size, sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kBuiltinCAS,
                                   __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size, cache_line_size);
    }
  }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
