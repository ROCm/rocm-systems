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

#include "atomicExch_common.hh"

/**
 * @addtogroup __hip_atomic_exchange __hip_atomic_exchange
 * @{
 * @ingroup AtomicsTest
 * ________________________
 * Test cases from other modules:
 *    - @ref Unit_AtomicBuiltins_Negative_Parameters_RTC
 */

// Helper function to run __hip_atomic_exchange tests with WAVEFRONT scope
template <typename TestType> static void runHipAtomicExchangeWavefrontTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      AtomicExchSingleDeviceSingleKernelTest<TestType, AtomicScopes::builtin,
                                             __HIP_MEMORY_SCOPE_WAVEFRONT>(1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      AtomicExchSingleDeviceSingleKernelTest<TestType, AtomicScopes::builtin,
                                             __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size,
                                                                           sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      AtomicExchSingleDeviceSingleKernelTest<TestType, AtomicScopes::builtin,
                                             __HIP_MEMORY_SCOPE_WAVEFRONT>(warp_size,
                                                                           cache_line_size);
    }
  }
}

// Helper function to run __hip_atomic_exchange tests with WORKGROUP scope
template <typename TestType> static void runHipAtomicExchangeWorkgroupTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      AtomicExchSingleDeviceSingleKernelTest<TestType, AtomicScopes::builtin,
                                             __HIP_MEMORY_SCOPE_WORKGROUP>(1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      AtomicExchSingleDeviceSingleKernelTest<TestType, AtomicScopes::builtin,
                                             __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size,
                                                                           sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      AtomicExchSingleDeviceSingleKernelTest<TestType, AtomicScopes::builtin,
                                             __HIP_MEMORY_SCOPE_WORKGROUP>(warp_size,
                                                                           cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a single kernel on a single device wherein all threads will perform an atomic
 * exchange into a runtime determined memory location. Each thread will exchange its own grid wide
 * linear index + offset into the memory location, storing the return value into a separate output
 * array slot corresponding to it. Once complete, the union of output array and exchange memory is
 * validated to contain all values in the range [0, number_of_threads +
 * number_of_exchange_memory_slots). Several memory access patterns are tested:
 *      -# All threads exchange to a single memory location
 *      -# Each thread exchanges into an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the exchange elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicExch
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated exchange memory
 *      - Exchange memory located in shared memory
 *      - WAVEFRONT memory scope
 * Test source
 * ------------------------
 *    - unit/atomics/__hip_atomic_exchange.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___hip_atomic_exchange_Positive_Wavefront") {
  SECTION("int") { runHipAtomicExchangeWavefrontTest<int>(); }
  SECTION("unsigned int") { runHipAtomicExchangeWavefrontTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicExchangeWavefrontTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicExchangeWavefrontTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicExchangeWavefrontTest<float>(); }
  SECTION("double") { runHipAtomicExchangeWavefrontTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a single kernel on a single device wherein all threads will perform an atomic
 * exchange into a runtime determined memory location. Each thread will exchange its own grid wide
 * linear index + offset into the memory location, storing the return value into a separate output
 * array slot corresponding to it. Once complete, the union of output array and exchange memory is
 * validated to contain all values in the range [0, number_of_threads +
 * number_of_exchange_memory_slots). Several memory access patterns are tested:
 *      -# All threads exchange to a single memory location
 *      -# Each thread exchanges into an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the exchange elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicExch
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated exchange memory
 *      - Exchange memory located in shared memory
 *      - WORKGROUP memory scope
 * Test source
 * ------------------------
 *    - unit/atomics/__hip_atomic_exchange.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___hip_atomic_exchange_Positive_Workgroup") {
  SECTION("int") { runHipAtomicExchangeWorkgroupTest<int>(); }
  SECTION("unsigned int") { runHipAtomicExchangeWorkgroupTest<unsigned int>(); }
  SECTION("unsigned long") { runHipAtomicExchangeWorkgroupTest<unsigned long>(); }
  SECTION("unsigned long long") { runHipAtomicExchangeWorkgroupTest<unsigned long long>(); }
  SECTION("float") { runHipAtomicExchangeWorkgroupTest<float>(); }
  SECTION("double") { runHipAtomicExchangeWorkgroupTest<double>(); }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
