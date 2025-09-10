/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_fp6.h>
#include <hip_test_common.hh>

#include <bitset>
#include <type_traits>
#include <vector>

std::vector<__hip_fp6_storage_t> get_all_fp6_ocp_nums() {
  std::vector<__hip_fp6_storage_t> ret;
  constexpr unsigned short max_fp6_num = 0b0011'1111;
  ret.reserve(max_fp6_num + 1);

  for (unsigned short i = 0; i <= max_fp6_num; i++) {
    ret.push_back(static_cast<__hip_fp6_storage_t>(i));
  }
  return ret;
}

template <bool is_e2m3, typename T>
__global__ void Type_to_fp6(T* f, __hip_fp6_storage_t* res, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    if constexpr (is_e2m3) {
      __hip_fp6_e2m3 tmp(f[i]);
      res[i] = tmp.__x;
    } else {
      __hip_fp6_e3m2 tmp(f[i]);
      res[i] = tmp.__x;
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to convert given interger values to FP6 type in the host
 * with E2M3 and E3M2 formats.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/fp6_ocp.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEMPLATE_TEST_CASE("Unit_all_fp6_ocp_vector_cvt_interger_data", "", int, long int, long long int,
                   short int) {
  SECTION("Fp6 with e2m3") {
    std::vector<TestType> input = {0, 1, 2, 3, 4, 5, 6, 7, -0, -1, -2, -3, -4, -5, -6, -7};
    for (const auto val : input) {
      __hip_fp6_e2m3 fp6(val);
      float ret = fp6;
      INFO("In: " << val);
      INFO("Out: " << ret);
      REQUIRE(ret == val);
    }
  }
  SECTION("Fp6 with e3m2 ") {
    std::vector<TestType> input = {0,  1,  2,  3,   4,   5,   6,   7,   8,   10, 12,
                                   14, 16, 20, 24,  28,  -0,  -1,  -2,  -3,  -4, -5,
                                   -6, -7, -8, -10, -12, -14, -16, -20, -24, -28};
    for (const auto val : input) {
      __hip_fp6_e3m2 fp6(val);
      float ret = fp6;
      INFO("In: " << val);
      INFO("Out: " << ret);
      REQUIRE(ret == val);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to convert given unsigned interger values to FP6 type in the host
 * with E2M3 and E3M2 formats.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/fp6_ocp.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEMPLATE_TEST_CASE("Unit_all_fp6_ocp_vector_cvt_unsigned_interger_data", "", int, long int,
                   long long int, short int) {
  SECTION("Fp6 with e2m3") {
    std::vector<TestType> input = {0, 1, 2, 3, 4, 5, 6, 7};
    for (const auto val : input) {
      __hip_fp6_e2m3 fp6(val);
      float ret = fp6;
      INFO("In: " << val);
      INFO("Out: " << ret);
      REQUIRE(ret == val);
    }
  }
  SECTION("Fp6 with e3m2 ") {
    std::vector<TestType> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28};
    for (const auto val : input) {
      __hip_fp6_e3m2 fp6(val);
      float ret = fp6;
      INFO("In: " << val);
      INFO("Out: " << ret);
      REQUIRE(ret == val);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to convert given unsigned interger values to FP6 type in the device
 * with E2M3 and E3M2 formats.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/fp6_ocp.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEMPLATE_TEST_CASE("Unit_all_fp6_ocp_vector_cvt_unsigned_integer_device", "", unsigned int,
                   unsigned long int, unsigned long long int, unsigned short int) {
  bool is_e2m3 = GENERATE(true, false);
  std::vector<TestType> f_vals;
  std::vector<__hip_fp6_storage_t> all_vals;
  constexpr TestType lhs = 0;
  constexpr TestType rhs = 64;
  constexpr TestType step = 1;

  f_vals.reserve(500);
  all_vals.reserve(500);

  for (TestType fval = lhs; fval <= rhs; fval += step) {
    if (is_e2m3) {
      __hip_fp6_e2m3 tmp(fval);
      all_vals.push_back(tmp.__x);
    } else {
      __hip_fp6_e3m2 tmp(fval);
      all_vals.push_back(tmp.__x);
    }
    f_vals.push_back(fval);
  }

  TestType* d_f_vals;
  __hip_fp6_storage_t* d_res;

  HIP_CHECK(hipMalloc(&d_f_vals, sizeof(TestType) * f_vals.size()));
  HIP_CHECK(hipMalloc(&d_res, sizeof(__hip_fp6_storage_t) * f_vals.size()));

  HIP_CHECK(
      hipMemcpy(d_f_vals, f_vals.data(), sizeof(TestType) * f_vals.size(), hipMemcpyHostToDevice));

  auto fp6_kernel = is_e2m3 ? Type_to_fp6<true, TestType> : Type_to_fp6<false, TestType>;
  fp6_kernel<<<(f_vals.size() / 64) + 1, 64>>>(d_f_vals, d_res, f_vals.size());

  std::vector<__hip_fp6_storage_t> final_res(f_vals.size(), static_cast<__hip_fp6_storage_t>(0));

  HIP_CHECK(hipMemcpy(final_res.data(), d_res, sizeof(__hip_fp6_storage_t) * final_res.size(),
                      hipMemcpyDeviceToHost));

  for (size_t i = 0; i < final_res.size(); i++) {
    INFO("Checking: " << f_vals[i] << " for: " << (is_e2m3 ? "e2m3" : "e3m2")
                      << " original: " << (int)all_vals[i] << " convert back: " << (int)final_res[i]
                      << " Idx : " << i);
    TestType gpu_cvt_res = 0, cpu_cvt_res = 0;
    float gpu_cvt_res_ = 0.0, cpu_cvt_res_ = 0.0;
    if (is_e2m3) {
      __hip_fp6_e2m3 gtmp;
      gtmp.__x = final_res[i];
      gpu_cvt_res_ = gtmp;
      gpu_cvt_res = gpu_cvt_res_;
      __hip_fp6_e2m3 ctmp;
      ctmp.__x = all_vals[i];
      cpu_cvt_res_ = ctmp;
      cpu_cvt_res = cpu_cvt_res_;
    } else {
      __hip_fp6_e3m2 gtmp;
      gtmp.__x = final_res[i];
      gpu_cvt_res_ = gtmp;
      gpu_cvt_res = gpu_cvt_res_;
      __hip_fp6_e3m2 ctmp;
      ctmp.__x = all_vals[i];
      cpu_cvt_res_ = ctmp;
      cpu_cvt_res = cpu_cvt_res_;
    }
    INFO("cpu cvt val: " << cpu_cvt_res << " gpu cvt val: " << gpu_cvt_res);
    REQUIRE(cpu_cvt_res == gpu_cvt_res);
  }

  HIP_CHECK(hipFree(d_f_vals));
  HIP_CHECK(hipFree(d_res));
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to convert given interger values to FP6 type in device with
 *  E2M3 and E3M2 formats.
 *  Test source
 * ------------------------
 *  - /unit/deviceLib/fp6_ocp.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */

TEMPLATE_TEST_CASE("Unit_all_fp6_ocp_vector_cvt_interger_data_device", "", int, long int,
                   long long int, short int) {
  bool is_e2m3 = GENERATE(true, false);
  std::vector<TestType> f_vals;
  std::vector<__hip_fp6_storage_t> all_vals;
  SECTION("all representable numbers") {
    all_vals = get_all_fp6_ocp_nums();
    f_vals.reserve(all_vals.size());

    for (const auto& fp6 : all_vals) {
      TestType f = 0;
      float f_ = 0.0;
      if (is_e2m3) {
        __hip_fp6_e2m3 tmp;
        tmp.__x = fp6;
        f_ = tmp;
        f = f_;
      } else {
        __hip_fp6_e3m2 tmp;
        tmp.__x = fp6;
        f_ = tmp;
        f = f_;
      }
      f_vals.push_back(f);
    }
  }
  SECTION("Range stepped numbers") {
    constexpr TestType lhs = -30;
    constexpr TestType rhs = 30;
    constexpr TestType step = 1;

    f_vals.reserve(500);
    all_vals.reserve(500);

    for (TestType fval = lhs; fval <= rhs; fval += step) {
      if (is_e2m3) {
        __hip_fp6_e2m3 tmp(fval);
        all_vals.push_back(tmp.__x);
      } else {
        __hip_fp6_e3m2 tmp(fval);
        all_vals.push_back(tmp.__x);
      }
      f_vals.push_back(fval);
    }
  }

  TestType* d_f_vals;
  __hip_fp6_storage_t* d_res;

  HIP_CHECK(hipMalloc(&d_f_vals, sizeof(TestType) * f_vals.size()));
  HIP_CHECK(hipMalloc(&d_res, sizeof(__hip_fp6_storage_t) * f_vals.size()));

  HIP_CHECK(
      hipMemcpy(d_f_vals, f_vals.data(), sizeof(TestType) * f_vals.size(), hipMemcpyHostToDevice));

  auto fp6_kernel = is_e2m3 ? Type_to_fp6<true, TestType> : Type_to_fp6<false, TestType>;
  fp6_kernel<<<(f_vals.size() / 64) + 1, 64>>>(d_f_vals, d_res, f_vals.size());

  std::vector<__hip_fp6_storage_t> final_res(f_vals.size(), static_cast<__hip_fp6_storage_t>(0));

  HIP_CHECK(hipMemcpy(final_res.data(), d_res, sizeof(__hip_fp6_storage_t) * final_res.size(),
                      hipMemcpyDeviceToHost));

  for (size_t i = 0; i < final_res.size(); i++) {
    INFO("Checking: " << f_vals[i] << " for: " << (is_e2m3 ? "e2m3" : "e3m2")
                      << " original: " << (int)all_vals[i] << " convert back: " << (int)final_res[i]
                      << " Idx : " << i);
    TestType gpu_cvt_res = 0.0f, cpu_cvt_res = 0.0f;
    float gpu_cvt_res_ = 0.0, cpu_cvt_res_ = 0.0;
    if (is_e2m3) {
      __hip_fp6_e2m3 gtmp;
      gtmp.__x = final_res[i];
      gpu_cvt_res_ = gtmp;
      gpu_cvt_res = gpu_cvt_res_;
      __hip_fp6_e2m3 ctmp;
      ctmp.__x = all_vals[i];
      cpu_cvt_res_ = ctmp;
      cpu_cvt_res = cpu_cvt_res_;
    } else {
      __hip_fp6_e3m2 gtmp;
      gtmp.__x = final_res[i];
      gpu_cvt_res_ = gtmp;
      gpu_cvt_res = gpu_cvt_res_;
      __hip_fp6_e3m2 ctmp;
      ctmp.__x = all_vals[i];
      cpu_cvt_res_ = ctmp;
      cpu_cvt_res = cpu_cvt_res_;
    }

    INFO("cpu cvt val: " << cpu_cvt_res << " gpu cvt val: " << gpu_cvt_res);
    REQUIRE(cpu_cvt_res == gpu_cvt_res);
  }

  HIP_CHECK(hipFree(d_f_vals));
  HIP_CHECK(hipFree(d_res));
}
