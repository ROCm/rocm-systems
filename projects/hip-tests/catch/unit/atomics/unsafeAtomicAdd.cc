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

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

/**
 * @addtogroup unsafeAtomicAdd unsafeAtomicAdd
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
 *      - All overloads of unsafeAtomicAdd
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Shared memory
 *      - Several grid and block dimension combinations (only one block is used for shared memory).
 * Test source
 * ------------------------
 *    - unit/atomics/unsafeAtomicAdd.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_unsafeAtomicAdd_Positive", "", float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kUnsafeAdd>(1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kUnsafeAdd>(warp_size,
                                                                          sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      SingleDeviceSingleKernelTest<TestType, AtomicOperation::kUnsafeAdd>(warp_size,
                                                                          cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel two times concurrently on a single device wherein all threads will
 * perform an atomic addition on a target memory location. Each thread will add the same value to
 * the memory location, storing the return value into a separate output array slot corresponding
 * to it. Once complete, the output array and target memory is validated to contain all the
 * expected values. Several memory access patterns are tested:
 *      -# All threads add to a single, compile time deducible, memory location
 *      -# Each thread targets an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of unsafeAtomicAdd
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated memory
 *      - Several grid and block dimension combinations.
 * Test source
 * ------------------------
 *    - unit/atomics/unsafeAtomicAdd.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_unsafeAtomicAdd_Positive_Multi_Kernel", "", float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      SingleDeviceMultipleKernelTest<TestType, AtomicOperation::kUnsafeAdd>(2, 1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      SingleDeviceMultipleKernelTest<TestType, AtomicOperation::kUnsafeAdd>(2, warp_size,
                                                                            sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      SingleDeviceMultipleKernelTest<TestType, AtomicOperation::kUnsafeAdd>(2, warp_size,
                                                                            cache_line_size);
    }
  }
}

template <typename Type,
          std::enable_if_t<
              std::is_same<Type, __half2>::value || std::is_same<Type, __hip_bfloat162>::value ||
                  std::is_same<Type, __hip_bfloat16>::value || std::is_same<Type, __half>::value,
              bool> = true>
__global__ void unsafe_add_kernel(Type* ptr, Type val) {
  (void)unsafeAtomicAdd(ptr, val);
}

TEMPLATE_TEST_CASE("Unit_unsafe_atomic_add_half_and_bfloat", "", __half2, __hip_bfloat162, __half,
                   __hip_bfloat16) {
  auto kernel = unsafe_add_kernel<TestType>;
  TestType val;
  if constexpr (std::is_same<TestType, __half2>::value) {
    val = __float22half2_rn(float2{1.0f, 2.0f});
  } else if constexpr (std::is_same<TestType, __hip_bfloat162>::value) {
    val = __float22bfloat162_rn(float2{1.0f, 2.0f});
  } else if constexpr (std::is_same<TestType, __half>::value) {
    val = __float2half(float{2.0f});
  } else {
    val = __float2bfloat16(float{2.0f});
  }

  TestType* out;
  HIP_CHECK(hipMalloc(&out, sizeof(TestType)));
  HIP_CHECK(hipMemset(out, 0, sizeof(TestType)));
  kernel<<<1, 32>>>(out, val);

  TestType dout;
  HIP_CHECK(hipMemcpy(&dout, out, sizeof(TestType), hipMemcpyDeviceToHost));

  float2 hout;
  if constexpr (std::is_same<TestType, __half2>::value) {
    hout = __half22float2(dout);
  } else if constexpr (std::is_same<TestType, __hip_bfloat162>::value) {
    hout = __bfloat1622float2(dout);
  } else if constexpr (std::is_same<TestType, __half>::value) {
    hout.x = 32.0f;
    hout.y = __half2float(dout);
  } else {
    hout.x = 32.0f;
    hout.y = __bfloat162float(dout);
  }

  REQUIRE(hout.x == 32.0f);
  REQUIRE(hout.y == 64.0f);
  HIP_CHECK(hipFree(out));
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
