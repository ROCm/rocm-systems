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
 * @addtogroup atomicCAS_system atomicCAS_system
 * @{
 * @ingroup AtomicsTest
 */

#ifdef HT_NVIDIA
#define TYPES
#else
#define TYPES , float, double
#endif

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel two times concurrently on a two devices wherein all threads will perform
 * an atomic addition, implemented using an atomic CAS operation, on a target memory location. Each
 * thread will add the same value to the memory location, storing the return value into a separate
 * output array slot corresponding to it. Once complete, the output array and target memory is
 * validated to contain all the expected values. Several memory access patterns are tested:
 *      -# All threads exchange to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicCAS_system
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Several grid and block dimension combinations.
 * Test source
 * ------------------------
 *    - unit/atomics/atomicCAS_system.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_atomicCAS_system_Positive_Peer_GPUs", "", int, unsigned int,
                   unsigned long long, unsigned short int TYPES) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
          2, 2, 1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
          2, 2, warp_size, sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
          2, 2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel on a single device wherein all threads will perform
 * an atomic addition, implemented using an atomic CAS operation, on a target memory location.
 * Each thread will add the same value to the memory location, storing the return value into a
 * separate output array slot corresponding to it. While the kernel is running, the host
 * performs atomic additions, in 4 threads, on the same memory location(s). Once complete, the
 * output array and target memory is validated to contain all the expected values. Several
 * memory access patterns are tested:
 *      -# All threads exchange to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicCAS_system
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Several grid and block dimension combinations.
 * Test source
 * ------------------------
 *    - unit/atomics/atomicCAS_system.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_atomicCAS_system_Positive_Host_And_GPU", "", int, unsigned int,
                   unsigned long long, unsigned short int TYPES) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
          1, 1, 1, sizeof(TestType), 4);
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
          1, 1, warp_size, sizeof(TestType), 4);
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
          1, 1, warp_size, cache_line_size, 4);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel two times on two devices wherein all threads will perform
 * an atomic addition, implemented using an atomic CAS operation, on a target memory location.
 * Each thread will add the same value to the memory location, storing the return value into a
 * separate output array slot corresponding to it. While the kernel is running, the host
 * performs atomic additions, in 4 threads, on the same memory location(s). Once complete, the
 * output array and target memory is validated to contain all the expected values. Several
 * memory access patterns are tested:
 *      -# All threads exchange to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicCAS_system
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Several grid and block dimension combinations.
 * Test source
 * ------------------------
 *    - unit/atomics/atomicCAS_system.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_atomicCAS_system_Positive_Host_And_Peer_GPUs", "", int, unsigned int,
                   unsigned long long, unsigned short int TYPES) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < 1; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
          2, 2, 1, sizeof(TestType), 4);
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
          2, 2, warp_size, sizeof(TestType), 4);
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      MultipleDeviceMultipleKernelAndHostTest<TestType, AtomicOperation::kCASAddSystem>(
          2, 2, warp_size, cache_line_size, 4);
    }
  }
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
