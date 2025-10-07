/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_ext_ocp.h>
#include <hip/hip_fp16.h>
#include <hip_test_common.hh>

#include <iostream>
#include <vector>

TEST_CASE("Unit_amd_ocp_type_to_hip_type") {
  SECTION("fp16") {
    float in = 10.0f;
    __half hf = in;
    auto fp16 = __amd_cvt_half_to_fp16(hf);
    auto hf_back = __amd_cvt_fp16_to_half(fp16);
    float out = fp16;
    REQUIRE(out == in);
    REQUIRE(float(hf_back) == float(hf));
  }

  SECTION("fp16x2") {
    float a = -10.0f, b = 10.0f;
    __half2 hf = {a, b};
    auto fp16x2 = __amd_cvt_half2_to_fp16x2(hf);
    auto hf_back = __amd_cvt_fp16x2_to_half2(fp16x2);
    float o_a = fp16x2[0], o_b = fp16x2[1];
    REQUIRE(o_a == a);
    REQUIRE(o_b == b);
    REQUIRE(__hbeq2(hf_back, hf));
  }

  SECTION("floatx2 to float2") {
    __amd_floatx2_storage_t in = {-10.0f, 10.0f};
    auto f2 = __amd_cvt_floatx2_to_float2(in);
    REQUIRE(f2.x == in[0]);
    REQUIRE(f2.y == in[1]);
  }

  SECTION("bf16") {
    float in = 10.0f;
    __hip_bfloat16 bf = in;
    auto bf16 = __amd_cvt_hipbf16_to_bf16(bf);
    auto bf_back = __amd_cvt_bf16_to_hipbf16(bf16);
    float out = bf16;
    REQUIRE(out == in);
    REQUIRE(float(bf_back) == float(bf));
  }

  SECTION("bf16x2") {
    float a = -10.0f, b = 10.0f;
    __hip_bfloat162 bf = {a, b};
    auto bf16x2 = __amd_cvt_hipbf162_to_bf16x2(bf);
    auto bf_back = __amd_cvt_bf16x2_to_hipbf162(bf16x2);
    float o_a = bf16x2[0], o_b = bf16x2[1];
    REQUIRE(o_a == a);
    REQUIRE(o_b == b);
    REQUIRE(__hbeq2(bf_back, bf));
  }
}

template <typename Kernel, typename... Args>
static __global__ void t_lambda_launch(Kernel k, Args... args) {
  int i = threadIdx.x;
  if (i == 0) {
    k(args...);
  }
}

TEST_CASE("Unit_amd_ocp_extract_tests") {
  SECTION("fp8x2 host") {
    constexpr auto interpret = __AMD_OCP_E4M3;
    __amd_floatx2_storage_t in{-10.0f, 10.0f};
    auto fp8x2 = __amd_cvt_floatx2_to_fp8x2(in, interpret);
    REQUIRE(__amd_cvt_fp8_to_float(__amd_extract_fp8(fp8x2, 0), interpret) == -10.0f);
    REQUIRE(__amd_cvt_fp8_to_float(__amd_extract_fp8(fp8x2, 1), interpret) == 10.0f);
  }

#if __AVX__
  SECTION("fp8x8 host") {
    constexpr auto interpret = __AMD_OCP_E4M3;
    __amd_floatx8_storage_t in{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto fp8x8 = __amd_cvt_floatx8_to_fp8x8_scale(in, interpret, 0);
    REQUIRE(__amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 0), 0),
                                   interpret) == 1.0f);
    REQUIRE(__amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 0), 1),
                                   interpret) == 2.0f);
    REQUIRE(__amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 1), 0),
                                   interpret) == 3.0f);
    REQUIRE(__amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 1), 1),
                                   interpret) == 4.0f);
    REQUIRE(__amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 2), 0),
                                   interpret) == 5.0f);
    REQUIRE(__amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 2), 1),
                                   interpret) == 6.0f);
    REQUIRE(__amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 3), 0),
                                   interpret) == 7.0f);
    REQUIRE(__amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 3), 1),
                                   interpret) == 8.0f);
  }
#endif

  SECTION("fp8x2 device") {
    auto l = [] __device__(float a, float b, float* o_a, float* o_b) {
      constexpr auto interpret = __AMD_OCP_E4M3;
      __amd_floatx2_storage_t in{a, b};
      auto fp8x2 = __amd_cvt_floatx2_to_fp8x2(in, interpret);
      *o_a = __amd_cvt_fp8_to_float(__amd_extract_fp8(fp8x2, 0), interpret);
      *o_b = __amd_cvt_fp8_to_float(__amd_extract_fp8(fp8x2, 1), interpret);
    };

    float a = -10.0f, b = 10.0f, *res_a, *res_b;

    HIP_CHECK(hipMallocManaged(&res_a, sizeof(float)));
    HIP_CHECK(hipMallocManaged(&res_b, sizeof(float)));
    t_lambda_launch<<<1, 32>>>(l, a, b, res_a, res_b);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(*res_a == a);
    REQUIRE(*res_b == b);

    HIP_CHECK(hipFree(res_a));
    HIP_CHECK(hipFree(res_b));
  }

  SECTION("fp8x8 device") {
    auto l = [] __device__(float* res) {
      constexpr auto interpret = __AMD_OCP_E4M3;
      __amd_floatx8_storage_t in{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
      auto fp8x8 = __amd_cvt_floatx8_to_fp8x8_scale(in, interpret, 0);
      res[0] =
          __amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 0), 0), interpret);
      res[1] =
          __amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 0), 1), interpret);
      res[2] =
          __amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 1), 0), interpret);
      res[3] =
          __amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 1), 1), interpret);
      res[4] =
          __amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 2), 0), interpret);
      res[5] =
          __amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 2), 1), interpret);
      res[6] =
          __amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 3), 0), interpret);
      res[7] =
          __amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(fp8x8, 3), 1), interpret);
    };

    float* res;

    HIP_CHECK(hipMallocManaged(&res, sizeof(float)));
    t_lambda_launch<<<1, 32>>>(l, res);
    HIP_CHECK(hipDeviceSynchronize());

    for (size_t i = 0; i < 8; i++) {
      INFO("Index: " << i << " res: " << res[i]);
      REQUIRE(res[i] == static_cast<float>(i + 1));
    }

    HIP_CHECK(hipFree(res));
  }

  SECTION("fp4x2 host") {
    constexpr auto interpret = __AMD_OCP_E2M1;
    __amd_floatx2_storage_t in{-1.0f, 1.0f};
    auto fp4x2 = __amd_cvt_floatx2_to_fp4x2_scale(in, interpret, 0 /* scale*/);
    REQUIRE((__amd_extract_fp4(fp4x2, 0) & 0b1000) != 0);
    REQUIRE((__amd_extract_fp4(fp4x2, 1) & 0b1000) == 0);
  }

#if __AVX__
  SECTION("fp4x8 host") {
    constexpr auto interpret = __AMD_OCP_E2M1;
    __amd_floatx8_storage_t in{0, 1, 1, 0, 1, 0, 0, 1};
    auto fp4x8 = __amd_cvt_floatx8_to_fp4x8_scale(in, interpret, 0);
    for (size_t i = 0; i < 4; i++) {
      auto r1 = __amd_cvt_fp4x2_to_floatx2_scale(__amd_extract_fp4x2(fp4x8, i), interpret, 0);
      INFO("Index: " << i << " vals: " << in[i * 2] << ", " << in[i * 2 + 1]);
      CHECK(r1[0] == in[i * 2]);
      CHECK(r1[1] == in[i * 2 + 1]);
    }
  }
#endif

  SECTION("fp4x2 device") {
    auto l = [] __device__(__amd_fp4x2_storage_t * res) {
      constexpr auto interpret = __AMD_OCP_E2M1;
      __amd_floatx2_storage_t in{-1.0f, 1.0f};
      *res = __amd_cvt_floatx2_to_fp4x2_scale(in, interpret, 0 /* scale*/);
    };

    __amd_fp4x2_storage_t* fp4x2;
    HIP_CHECK(hipMallocManaged(&fp4x2, sizeof(__amd_fp4x2_storage_t)));
    t_lambda_launch<<<1, 32>>>(l, fp4x2);
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE((__amd_extract_fp4(*fp4x2, 0) & 0b1000) != 0);
    REQUIRE((__amd_extract_fp4(*fp4x2, 1) & 0b1000) == 0);
    HIP_CHECK(hipFree(fp4x2));
  }

  SECTION("fp4x8 device") {
    auto l = [] __device__(__amd_fp4x2_storage_t * res) {
      __amd_floatx8_storage_t in{0, 1, 1, 0, 1, 0, 0, 1};
      auto fp4x8 = __amd_cvt_floatx8_to_fp4x8_scale(in, __AMD_OCP_E2M1, 0);
      for (size_t i = 0; i < 4; i++) {
        res[i] = __amd_extract_fp4x2(fp4x8, i);
      }
    };

    __amd_fp4x2_storage_t* fp4x2;
    HIP_CHECK(hipMallocManaged(&fp4x2, sizeof(__amd_fp4x2_storage_t) * 4));
    t_lambda_launch<<<1, 32>>>(l, fp4x2);
    HIP_CHECK(hipDeviceSynchronize());

    __amd_floatx8_storage_t in{0, 1, 1, 0, 1, 0, 0, 1};
    for (size_t i = 0; i < 4; i++) {
      auto r1 = __amd_cvt_fp4x2_to_floatx2_scale(fp4x2[i], __AMD_OCP_E2M1, 0);
      INFO("Index: " << i << " vals: " << in[i * 2] << ", " << in[i * 2 + 1]);
      CHECK(r1[0] == in[i * 2]);
      CHECK(r1[1] == in[i * 2 + 1]);
    }
    HIP_CHECK(hipFree(fp4x2));
  }
}
