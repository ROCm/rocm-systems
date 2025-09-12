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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip/hip_fp16.h>
#include <hip_test_common.hh>
#include <type_traits>

__global__ void __halfTest(bool* result, __half a) {
  // Construction
  result[0] &= std::is_default_constructible<__half>{};
  result[0] &= std::is_copy_constructible<__half>{};
  result[0] &= std::is_move_constructible<__half>{};
  result[0] &= std::is_constructible<__half, float>{};
  result[0] &= std::is_constructible<__half, double>{};
  result[0] &= std::is_constructible<__half, uint32_t>{};
  result[0] &= std::is_constructible<__half, int32_t>{};
  result[0] &= std::is_constructible<__half, uint32_t>{};
  result[0] &= std::is_constructible<__half, int>{};
  result[0] &= std::is_constructible<__half, uint64_t>{};
  result[0] &= std::is_constructible<__half, int64_t>{};
  result[0] &= std::is_constructible<__half, int64_t>{};
  result[0] &= std::is_constructible<__half, uint64_t>{};
  result[0] &= std::is_constructible<__half, __half_raw>{};

  // Assignment
  result[0] &= std::is_copy_assignable<__half>{};
  result[0] &= std::is_move_assignable<__half>{};
  result[0] &= std::is_assignable<__half, float>{};
  result[0] &= std::is_assignable<__half, double>{};
  result[0] &= std::is_assignable<__half, uint32_t>{};
  result[0] &= std::is_assignable<__half, int32_t>{};
  result[0] &= std::is_assignable<__half, uint32_t>{};
  result[0] &= std::is_assignable<__half, int>{};
  result[0] &= std::is_assignable<__half, uint64_t>{};
  result[0] &= std::is_assignable<__half, int64_t>{};
  result[0] &= std::is_assignable<__half, int64_t>{};
  result[0] &= std::is_assignable<__half, uint64_t>{};
  result[0] &= std::is_assignable<__half, __half_raw>{};
  result[0] &= std::is_assignable<__half, volatile __half_raw&>{};
  result[0] &= std::is_assignable<__half, volatile __half_raw&&>{};

  // Conversion
  result[0] &= std::is_convertible<__half, float>{};
  result[0] &= std::is_convertible<__half, uint32_t>{};
  result[0] &= std::is_convertible<__half, int32_t>{};
  result[0] &= std::is_convertible<__half, uint32_t>{};
  result[0] &= std::is_convertible<__half, int>{};
  result[0] &= std::is_convertible<__half, uint64_t>{};
  result[0] &= std::is_convertible<__half, int64_t>{};
  result[0] &= std::is_convertible<__half, int64_t>{};
  result[0] &= std::is_convertible<__half, bool>{};
  result[0] &= std::is_convertible<__half, uint64_t>{};
  result[0] &= std::is_convertible<__half, __half_raw>{};
  result[0] &= std::is_convertible<__half, volatile __half_raw>{};

  // Nullary
  result[0] &= __heq(a, +a) && result[0];
  result[0] &= __heq(__hneg(a), -a) && result[0];

  // Unary arithmetic
  result[0] &= __heq(a += 0, a) && result[0];
  result[0] &= __heq(a -= 0, a) && result[0];
  result[0] &= __heq(a *= 1, a) && result[0];
  result[0] &= __heq(a /= 1, a) && result[0];

  // Binary arithmetic
  result[0] &= __heq((a + a), __hadd(a, a)) && result[0];
  result[0] &= __heq((a - a), __hsub(a, a)) && result[0];
  result[0] &= __heq((a * a), __hmul(a, a)) && result[0];
  result[0] &= __heq((a / a), __hdiv(a, a)) && result[0];

  // Relations
  result[0] &= (a == a) && result[0];
  result[0] &= !(a != a) && result[0];
  result[0] &= (a <= a) && result[0];
  result[0] &= (a >= a) && result[0];
  result[0] &= !(a < a) && result[0];
  result[0] &= !(a > a) && result[0];
}

__device__ static bool to_bool(const __half2& x) {
  auto r = static_cast<const __half2_raw&>(x);
  return r.data.x != 0 && r.data.y != 0;
}

__global__ void __half2Test(bool* result, __half2 a) {
  // Construction
  result[0] &= std::is_default_constructible<__half2>{};
  result[0] &= std::is_copy_constructible<__half2>{};
  result[0] &= std::is_move_constructible<__half2>{};
  result[0] &= std::is_constructible<__half2, __half, __half>{};
  result[0] &= std::is_constructible<__half2, __half2_raw>{};

  // Assignment
  result[0] &= std::is_copy_assignable<__half2>{};
  result[0] &= std::is_move_assignable<__half2>{};
  result[0] &= std::is_assignable<__half2, __half2_raw>{};

  // Conversion
  result[0] &= std::is_convertible<__half2, __half2_raw>{};

  // Nullary
  result[0] &= to_bool(__heq2(a, +a)) && result[0];
  result[0] &= to_bool(__heq2(__hneg2(a), -a)) && result[0];

  // Unary arithmetic
  result[0] &= to_bool(__heq2(a += 0, a)) && result[0];
  result[0] &= to_bool(__heq2(a -= 0, a)) && result[0];
  result[0] &= to_bool(__heq2(a *= 1, a)) && result[0];
  result[0] &= to_bool(__heq2(a /= 1, a)) && result[0];

  // Binary arithmetic
  result[0] &= to_bool(__heq2((a + a), __hadd2(a, a))) && result[0];
  result[0] &= to_bool(__heq2((a - a), __hsub2(a, a))) && result[0];
  result[0] &= to_bool(__heq2((a * a), __hmul2(a, a))) && result[0];
  result[0] &= to_bool(__heq2((a / a), __h2div(a, a))) && result[0];

  // Relations
  result[0] &= (a == a) && result[0];
  result[0] &= !(a != a) && result[0];
  result[0] &= (a <= a) && result[0];
  result[0] &= (a >= a) && result[0];
  result[0] &= !(a < a) && result[0];
  result[0] &= !(a > a) && result[0];

  // Dot Functions
  result[0] &= amd_mixed_dot(a, a, 1, 1) && result[0];
}

TEST_CASE("Unit_hipTestNativeHalf") {
  bool* result{nullptr};
  HIP_CHECK(hipHostMalloc(&result, 1));
  SECTION("Half Test") {
    result[0] = true;
    hipLaunchKernelGGL(__halfTest, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result, __half{1});
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(result[0] == true);
  }
  SECTION("Half2 Test") {
    result[0] = true;
    hipLaunchKernelGGL(__half2Test, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result, __half2{1, 1});
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(result[0] == true);
  }
  HIP_CHECK(hipHostFree(result));
}
