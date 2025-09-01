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

#include <hip_test_common.hh>
#include <resource_guards.hh>

__global__ void __brev_kernel(unsigned int* y, unsigned int x) { y[0] = __brev(x); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__brev(x)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___brev_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  __brev_kernel<<<1, 1>>>(y.ptr(), 0xAAAAAAAA);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 0x55555555);
}

__global__ void __brevll_kernel(unsigned long long int* y, unsigned long long int x) {
  y[0] = __brevll(x);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__brevll(x)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___brevll_Sanity_Positive") {
  LinearAllocGuard<unsigned long long int> y(LinearAllocs::hipMallocManaged,
                                             sizeof(unsigned long long int));

  __brevll_kernel<<<1, 1>>>(y.ptr(), 0xAAAAAAAAAAAAAAAA);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 0x5555555555555555);
}

template <typename T> __global__ void __clz_kernel(T* y, T x) { y[0] = __clz(x); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__clz(x)`. Run for `int` and `unsigned int` overloads.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device___clz_Sanity_Positive", "", int, unsigned int) {
  LinearAllocGuard<TestType> y(LinearAllocs::hipMallocManaged, sizeof(TestType));

  __clz_kernel<<<1, 1>>>(y.ptr(), static_cast<TestType>(0));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 32);

  TestType x = 1;
  for (int i = 0; i < 32; ++i) {
    __clz_kernel<<<1, 1>>>(y.ptr(), x << i);
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(y.ptr()[0] == 31 - i);
  }
}

template <typename T> __global__ void __clzll_kernel(T* y, T x) { y[0] = __clzll(x); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__clzll(x)`. Run for `long long int` and `unsigned long long int`
 * overloads.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device___clzll_Sanity_Positive", "", long long int,
                   unsigned long long int) {
  LinearAllocGuard<TestType> y(LinearAllocs::hipMallocManaged, sizeof(TestType));

  __clzll_kernel<<<1, 1>>>(y.ptr(), static_cast<TestType>(0));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 64);

  TestType x = 1;
  for (int i = 0; i < 64; ++i) {
    __clzll_kernel<<<1, 1>>>(y.ptr(), x << i);
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(y.ptr()[0] == 63 - i);
  }
}

template <typename T> __global__ void __ffs_kernel(T* y, T x) { y[0] = __ffs(x); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__ffs(x)`. Run for `int` and `unsigned int` overloads.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device___ffs_Sanity_Positive", "", int, unsigned int) {
  LinearAllocGuard<TestType> y(LinearAllocs::hipMallocManaged, sizeof(TestType));

  __ffs_kernel<<<1, 1>>>(y.ptr(), static_cast<TestType>(0));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 0);

  TestType x = 1;
  for (int i = 0; i < 32; ++i) {
    __ffs_kernel<<<1, 1>>>(y.ptr(), x << i);
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(y.ptr()[0] == i + 1);
  }
}

template <typename T> __global__ void __ffsll_kernel(T* y, T x) { y[0] = __ffsll(x); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__ffsll(x)`. Run for `long long int` and `unsigned long long int`
 * overloads.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device___ffsll_Sanity_Positive", "", long long int,
                   unsigned long long int) {
  LinearAllocGuard<TestType> y(LinearAllocs::hipMallocManaged, sizeof(TestType));

  __ffsll_kernel<<<1, 1>>>(y.ptr(), static_cast<TestType>(0));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 0);

  TestType x = 1;
  for (int i = 0; i < 64; ++i) {
    __ffsll_kernel<<<1, 1>>>(y.ptr(), x << i);
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(y.ptr()[0] == i + 1);
  }
}

__global__ void __popc_kernel(unsigned int* y, unsigned int x) { y[0] = __popc(x); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__popc(x)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___popc_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  __popc_kernel<<<1, 1>>>(y.ptr(), 0);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 0);

  unsigned int x = 0;
  for (int i = 0; i < 32; ++i) {
    __popc_kernel<<<1, 1>>>(y.ptr(), x |= (1u << i));
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(y.ptr()[0] == i + 1);
  }
}

__global__ void __popcll_kernel(unsigned long long int* y, unsigned long long int x) {
  y[0] = __popcll(x);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__popcll(x)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___popcll_Sanity_Positive") {
  LinearAllocGuard<unsigned long long int> y(LinearAllocs::hipMallocManaged,
                                             sizeof(unsigned long long int));

  __popcll_kernel<<<1, 1>>>(y.ptr(), 0);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 0);

  unsigned long long int x = 0;
  for (int i = 0; i < 64; ++i) {
    __popcll_kernel<<<1, 1>>>(y.ptr(), x |= (1ull << i));
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(y.ptr()[0] == i + 1);
  }
}

__global__ void __mul24_kernel(int* y, int x1, int x2) { y[0] = __mul24(x1, x2); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__mul24(x,y)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___mul24_Sanity_Positive") {
  LinearAllocGuard<int> y(LinearAllocs::hipMallocManaged, sizeof(int));

  int x1 = GENERATE(0, -42, 42, 0xFFFFFFFF);
  int x2 = GENERATE(0, -42, 42, 0xFFFFFFFF);

  __mul24_kernel<<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == x1 * x2);
}

__global__ void __umul24_kernel(unsigned int* y, unsigned int x1, unsigned int x2) {
  y[0] = __umul24(x1, x2);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__umul24(x,y)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___umul24_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  unsigned int x1 = GENERATE(0, 42, 0xFFFFFF);
  unsigned int x2 = GENERATE(0, 42, 0xFFFFFF);

  __umul24_kernel<<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == x1 * x2);
}

__global__ void __funnelshift_l_kernel(unsigned int* y, unsigned int lo, unsigned int hi,
                                       unsigned int shift) {
  y[0] = __funnelshift_l(lo, hi, shift);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__funnelshift_l(lo,hi,shift)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___funnelshift_l_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  const unsigned int lo = 0xAAAAAAAA, hi = 0xBBBBBBBB;
  const unsigned long long hi_lo = (static_cast<unsigned long long>(hi) << 32) | lo;

  for (unsigned int shift = 0; shift < 64; ++shift) {
    __funnelshift_l_kernel<<<1, 1>>>(y.ptr(), lo, hi, shift);
    HIP_CHECK(hipDeviceSynchronize());

    INFO("shift: " << shift);
    REQUIRE(y.ptr()[0] == static_cast<unsigned int>((hi_lo << (shift & 31)) >> 32));
  }
}

__global__ void __funnelshift_lc_kernel(unsigned int* y, unsigned int lo, unsigned int hi,
                                        unsigned int shift) {
  y[0] = __funnelshift_lc(lo, hi, shift);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__funnelshift_lc(lo,hi,shift)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___funnelshift_lc_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  const unsigned int lo = 0xAAAAAAAA, hi = 0xBBBBBBBB;
  const unsigned long long hi_lo = (static_cast<unsigned long long>(hi) << 32) | lo;

  for (unsigned int shift = 0; shift < 64; ++shift) {
    __funnelshift_lc_kernel<<<1, 1>>>(y.ptr(), lo, hi, shift);
    HIP_CHECK(hipDeviceSynchronize());

    INFO("shift: " << shift);
    REQUIRE(y.ptr()[0] == static_cast<unsigned int>((hi_lo << std::min(shift, 32u)) >> 32));
  }
}

__global__ void __funnelshift_r_kernel(unsigned int* y, unsigned int lo, unsigned int hi,
                                       unsigned int shift) {
  y[0] = __funnelshift_r(lo, hi, shift);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__funnelshift_r(lo,hi,shift)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___funnelshift_r_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  const unsigned int lo = 0xAAAAAAAA, hi = 0xBBBBBBBB;
  const unsigned long long hi_lo = (static_cast<unsigned long long>(hi) << 32) | lo;

  for (unsigned int shift = 0; shift < 64; ++shift) {
    __funnelshift_r_kernel<<<1, 1>>>(y.ptr(), lo, hi, shift);
    HIP_CHECK(hipDeviceSynchronize());

    INFO("shift: " << shift);
    REQUIRE(y.ptr()[0] == static_cast<unsigned int>(hi_lo >> (shift & 31)));
  }
}

__global__ void __funnelshift_rc_kernel(unsigned int* y, unsigned int lo, unsigned int hi,
                                        unsigned int shift) {
  y[0] = __funnelshift_rc(lo, hi, shift);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__funnelshift_rc(lo,hi,shift)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___funnelshift_rc_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  const unsigned int lo = 0xAAAAAAAA, hi = 0xBBBBBBBB;
  const unsigned long long hi_lo = (static_cast<unsigned long long>(hi) << 32) | lo;

  for (unsigned int shift = 0; shift < 64; ++shift) {
    __funnelshift_rc_kernel<<<1, 1>>>(y.ptr(), lo, hi, shift);
    HIP_CHECK(hipDeviceSynchronize());

    INFO("shift: " << shift);
    REQUIRE(y.ptr()[0] == static_cast<unsigned int>(hi_lo >> std::min(shift, 32u)));
  }
}

__global__ void __hadd_kernel(int* y, int x1, int x2) { y[0] = __hadd(x1, x2); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__hadd(x,y)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___hadd_Sanity_Positive") {
  LinearAllocGuard<int> y(LinearAllocs::hipMallocManaged, sizeof(int));

  int x1 = GENERATE(0, -42, 42, 0xFFFFFFFF);
  int x2 = GENERATE(0, -42, 42, 0xFFFFFFFF);

  __hadd_kernel<<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  INFO("x1: " << x1);
  INFO("x2: " << x2);
  REQUIRE(y.ptr()[0] == static_cast<int>((static_cast<long long>(x1) + x2) >> 1));
}

__global__ void __uhadd_kernel(unsigned int* y, unsigned int x1, unsigned int x2) {
  y[0] = __uhadd(x1, x2);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__uhadd(x,y)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___uhadd_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  unsigned int x1 = GENERATE(0, 42, 0xFFFFFFFF);
  unsigned int x2 = GENERATE(0, 42, 0xFFFFFFFF);

  __uhadd_kernel<<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  INFO("x1: " << x1);
  INFO("x2: " << x2);
  REQUIRE(y.ptr()[0] == static_cast<unsigned int>((static_cast<unsigned long long>(x1) + x2) >> 1));
}

__global__ void __rhadd_kernel(int* y, int x1, int x2) { y[0] = __rhadd(x1, x2); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__rhadd(x,y)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___rhadd_Sanity_Positive") {
  LinearAllocGuard<int> y(LinearAllocs::hipMallocManaged, sizeof(int));

  int x1 = GENERATE(0, -42, 42, 0xFFFFFFFF);
  int x2 = GENERATE(0, -42, 42, 0xFFFFFFFF);

  __rhadd_kernel<<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  INFO("x1: " << x1);
  INFO("x2: " << x2);
  REQUIRE(y.ptr()[0] == static_cast<int>((static_cast<long long>(x1) + x2 + 1) >> 1));
}

__global__ void __urhadd_kernel(unsigned int* y, unsigned int x1, unsigned int x2) {
  y[0] = __urhadd(x1, x2);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__urhadd(x,y)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___urhadd_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  unsigned int x1 = GENERATE(0, 42, 0xFFFFFFFF);
  unsigned int x2 = GENERATE(0, 42, 0xFFFFFFFF);

  __urhadd_kernel<<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  INFO("x1: " << x1);
  INFO("x2: " << x2);
  REQUIRE(y.ptr()[0] ==
          static_cast<unsigned int>((static_cast<unsigned long long>(x1) + x2 + 1) >> 1));
}

__global__ void __mulhi_kernel(int* y, int x1, int x2) { y[0] = __mulhi(x1, x2); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__mulhi(x,y)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___mulhi_Sanity_Positive") {
  LinearAllocGuard<int> y(LinearAllocs::hipMallocManaged, sizeof(int));

  int x1 = GENERATE(0, -42, 42, 0xFFFFFFFF);
  int x2 = GENERATE(0, -42, 42, 0xFFFFFFFF);

  __mulhi_kernel<<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  INFO("x1: " << x1);
  INFO("x2: " << x2);
  REQUIRE(y.ptr()[0] ==
          static_cast<int>((static_cast<long long>(x1) * static_cast<long long>(x2)) >> 32));
}

__global__ void __umulhi_kernel(unsigned int* y, unsigned int x1, unsigned int x2) {
  y[0] = __umulhi(x1, x2);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__umulhi(x,y)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___umulhi_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  unsigned int x1 = GENERATE(0, 42, 0xFFFFFFFF);
  unsigned int x2 = GENERATE(0, 42, 0xFFFFFFFF);

  __umulhi_kernel<<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  INFO("x1: " << x1);
  INFO("x2: " << x2);
  REQUIRE(y.ptr()[0] ==
          static_cast<unsigned int>((static_cast<unsigned long long>(x1) * x2) >> 32));
}

__global__ void __mul64hi_kernel(long long* y, long long x1, long long x2) {
  y[0] = __mul64hi(x1, x2);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__mul64hi(x,y)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___mul64hi_Sanity_Positive") {
  LinearAllocGuard<long long> y(LinearAllocs::hipMallocManaged, sizeof(long long));

  long long x1 = GENERATE(0, -42, 42, 0xFFFFFFFF);
  long long x2 = GENERATE(0, -42, 42, 0xFFFFFFFF);

  __mul64hi_kernel<<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  INFO("x1: " << x1);
  INFO("x2: " << x2);
  REQUIRE(y.ptr()[0] == static_cast<long long>(
                            (static_cast<__int128_t>(x1) * static_cast<__int128_t>(x2)) >> 64));
}

__global__ void __umul64hi_kernel(unsigned long long* y, unsigned long long x1,
                                  unsigned long long x2) {
  y[0] = __umul64hi(x1, x2);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__umul64hi(x,y)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___umul64hi_Sanity_Positive") {
  LinearAllocGuard<unsigned long long> y(LinearAllocs::hipMallocManaged,
                                         sizeof(unsigned long long));

  unsigned long long x1 = GENERATE(0, 42, 0xFFFFFFFF);
  unsigned long long x2 = GENERATE(0, 42, 0xFFFFFFFF);

  __umul64hi_kernel<<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  INFO("x1: " << x1);
  INFO("x2: " << x2);
  REQUIRE(y.ptr()[0] == static_cast<unsigned long long>(
                            (static_cast<__uint128_t>(x1) * static_cast<__uint128_t>(x2)) >> 64));
}

__global__ void __sad_kernel(unsigned int* y, int x1, int x2, unsigned int x3) {
  y[0] = __sad(x1, x2, x3);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__sad(x,y,z)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___sad_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  int x1 = GENERATE(0, -42, 42, 0xFFFFFFFF);
  int x2 = GENERATE(0, -42, 42, 0xFFFFFFFF);
  unsigned int x3 = GENERATE(0, 42, 0xFFFFFFFF);

  __sad_kernel<<<1, 1>>>(y.ptr(), x1, x2, x3);
  HIP_CHECK(hipDeviceSynchronize());

  INFO("x1: " << x1);
  INFO("x2: " << x2);
  REQUIRE(y.ptr()[0] == (static_cast<unsigned int>(std::abs(x1 - x2)) + x3));
}

__global__ void __usad_kernel(unsigned int* y, unsigned int x1, unsigned int x2, unsigned int x3) {
  y[0] = __usad(x1, x2, x3);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__usad(x,y,z)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___usad_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  unsigned int x1 = GENERATE(0, 42, 0xFFFFFFFF);
  unsigned int x2 = GENERATE(0, 42, 0xFFFFFFFF);
  unsigned int x3 = GENERATE(0, 42, 0xFFFFFFFF);

  __usad_kernel<<<1, 1>>>(y.ptr(), x1, x2, x3);
  HIP_CHECK(hipDeviceSynchronize());

  INFO("x1: " << x1);
  INFO("x2: " << x2);
  REQUIRE(y.ptr()[0] == (static_cast<unsigned int>(
                             std::abs(static_cast<long long>(x1) - static_cast<long long>(x2))) +
                         x3));
}

__global__ void __byte_perm(unsigned int* y, unsigned int x1, unsigned int x2, unsigned int s) {
  y[0] = __byte_perm(x1, x2, s);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__byte_perm(x,y,s)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___byte_perm_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  unsigned int bytes[] = {0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};

  unsigned int x1 = (bytes[3] << 24) | (bytes[2] << 16) | (bytes[1] << 8) | bytes[0];
  unsigned int x2 = (bytes[7] << 24) | (bytes[6] << 16) | (bytes[5] << 8) | bytes[4];

  unsigned int s0 = GENERATE(0, 1);
  unsigned int s1 = GENERATE(2, 3);
  unsigned int s2 = GENERATE(4, 5);
  unsigned int s3 = GENERATE(6, 7);

  unsigned int s = (s3 << 12) | (s2 << 8) | (s1 << 4) | s0;

  __byte_perm<<<1, 1>>>(y.ptr(), x1, x2, s);
  HIP_CHECK(hipDeviceSynchronize());

  unsigned int expected = (bytes[s3] << 24) | (bytes[s2] << 16) | (bytes[s1] << 8) | bytes[s0];
  REQUIRE(y.ptr()[0] == expected);
}