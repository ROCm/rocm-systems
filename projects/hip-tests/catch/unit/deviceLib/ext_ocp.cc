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

static __global__ void float_to_fp8_sr(float* in, __amd_fp8_storage_t* out,
                                       __amd_fp8_interpretation_t interpret, int size,
                                       unsigned int rng = 0) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = __amd_cvt_float_to_fp8_sr(in[i], interpret, rng);
  }
}

static __global__ void float_to_fp8_sr_scale(float* in, __amd_fp8_storage_t* out,
                                             __amd_fp8_interpretation_t interpret, int size,
                                             unsigned int rng, __amd_scale_t scale) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = __amd_cvt_float_to_fp8_sr_scale(in[i], interpret, rng, scale);
  }
}

static __global__ void fp8_to_float(__amd_fp8_storage_t* in, __amd_fp8_interpretation_t interpret,
                                    float* out, int size) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = __amd_cvt_fp8_to_float(in[i], interpret);
  }
}

static __global__ void fp8_to_float_scale(__amd_fp8_storage_t* in,
                                          __amd_fp8_interpretation_t interpret, float* out,
                                          int size, __amd_scale_t scale) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = __amd_cvt_fp8_to_float_scale(in[i], interpret, scale);
  }
}

static __global__ void floatx2_to_fp8x2(__amd_floatx2_storage_t* in, __amd_fp8x2_storage_t* out,
                                        __amd_fp8_interpretation_t interpret, int size) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = __amd_cvt_floatx2_to_fp8x2(in[i], interpret);
  }
}

static __global__ void fp8x2_to_floatx2(__amd_fp8x2_storage_t* in,
                                        __amd_fp8_interpretation_t interpret,
                                        __amd_floatx2_storage_t* out, int size) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = __amd_cvt_fp8x2_to_floatx2(in[i], interpret);
  }
}

static __global__ void floatx2_to_fp8x2_scale(__amd_floatx2_storage_t* in,
                                              __amd_fp8x2_storage_t* out,
                                              __amd_fp8_interpretation_t interpret, int size,
                                              __amd_scale_t scale) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = __amd_cvt_floatx2_to_fp8x2_scale(in[i], interpret, scale);
  }
}

static __global__ void fp8x2_to_floatx2_scale(__amd_fp8x2_storage_t* in,
                                              __amd_fp8_interpretation_t interpret,
                                              __amd_floatx2_storage_t* out, int size,
                                              __amd_scale_t scale) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = __amd_cvt_fp8x2_to_floatx2_scale(in[i], interpret, scale);
  }
}

static __global__ void cxx_fp8_to_float_e4m3(float* res1, float* res2, float* res3, float* res4,
                                             float* res5, float aa, __amd_scale_t scale,
                                             unsigned int seed) {
  int i = threadIdx.x;
  float a = aa + i;
  __half hf = a;
  __hip_bfloat16 bf16 = a;
  auto fp8_e4m3 = __hipext_ocp_fp8_e4m3(a);
  auto fp8_e4m3_seed = __hipext_ocp_fp8_e4m3(a, seed);
  auto fp8_e4m3_scale_seed = __hipext_ocp_fp8_e4m3(a, seed, scale);
  auto fp8_from_half = __hipext_ocp_fp8_e4m3(__amd_cvt_half_to_fp16(hf), seed, scale);
  auto fp8_from_bf16 = __hipext_ocp_fp8_e4m3(__amd_cvt_hipbf16_to_bf16(bf16), seed, scale);
  res1[i] = fp8_e4m3;
  res2[i] = fp8_e4m3_seed;
  res3[i] = fp8_e4m3_scale_seed.get_scaled_float(scale);
  res4[i] = fp8_from_half.get_scaled_float(scale);
  res5[i] = fp8_from_bf16.get_scaled_float(scale);
}

static __global__ void cxx_fp8_to_float_e5m2(float* res1, float* res2, float* res3, float* res4,
                                             float* res5, float aa, __amd_scale_t scale,
                                             unsigned int seed) {
  int i = threadIdx.x;
  float a = aa + i;
  __half hf = a;
  __hip_bfloat16 bf16 = a;
  auto fp8_e5m2 = __hipext_ocp_fp8_e5m2(a);
  auto fp8_e5m2_seed = __hipext_ocp_fp8_e5m2(a, seed);
  auto fp8_e5m2_scale_seed = __hipext_ocp_fp8_e5m2(a, seed, scale);
  auto fp8_from_half = __hipext_ocp_fp8_e5m2(__amd_cvt_half_to_fp16(hf), seed, scale);
  auto fp8_from_bf16 = __hipext_ocp_fp8_e5m2(__amd_cvt_hipbf16_to_bf16(bf16), seed, scale);
  res1[i] = fp8_e5m2;
  res2[i] = fp8_e5m2_seed;
  res3[i] = fp8_e5m2_scale_seed.get_scaled_float(scale);
  res4[i] = fp8_from_half.get_scaled_float(scale);
  res5[i] = fp8_from_bf16.get_scaled_float(scale);
}

static __global__ void cxx_fp8x2_to_floatx2_e4m3(
    __amd_floatx2_storage_t* res1, __amd_floatx2_storage_t* res2, __amd_floatx2_storage_t* res3,
    __amd_floatx2_storage_t* res4, __amd_floatx2_storage_t* res5, __amd_floatx2_storage_t* res6,
    float aa, float bb, __amd_scale_t scale) {
  int i = threadIdx.x;
  float a = aa + i;
  float b = bb + i;
  __amd_floatx2_storage_t fpx2{a, b};
  __half2 fp16x2{a, b};
  __hip_bfloat162 bf16x2{a, b};
  auto fp8x2_e4m3_from_float = __hipext_ocp_fp8x2_e4m3(a, b);
  auto fp8x2_e4m3_from_floatx2 = __hipext_ocp_fp8x2_e4m3(fpx2);
  auto fp8x2_e4m3_scale = __hipext_ocp_fp8x2_e4m3(a, b, scale);
  auto fp8x2_e4m3_from_floatx2_scale = __hipext_ocp_fp8x2_e4m3(fpx2, scale);
  auto fp8x2_from_half = __hipext_ocp_fp8x2_e4m3(__amd_cvt_half2_to_fp16x2(fp16x2), scale);
  auto fp8x2_from_bf16 = __hipext_ocp_fp8x2_e4m3(__amd_cvt_hipbf162_to_bf16x2(bf16x2), scale);
  res1[i] = fp8x2_e4m3_from_float;
  res2[i] = fp8x2_e4m3_from_floatx2;
  res3[i] = fp8x2_e4m3_scale.get_scaled_floatx2(scale);
  res4[i] = fp8x2_e4m3_from_floatx2_scale.get_scaled_floatx2(scale);
  res5[i] = fp8x2_from_half.get_scaled_floatx2(scale);
  res6[i] = fp8x2_from_bf16.get_scaled_floatx2(scale);
}

static __global__ void cxx_fp8x2_to_floatx2_e5m2(
    __amd_floatx2_storage_t* res1, __amd_floatx2_storage_t* res2, __amd_floatx2_storage_t* res3,
    __amd_floatx2_storage_t* res4, __amd_floatx2_storage_t* res5, __amd_floatx2_storage_t* res6,
    float aa, float bb, __amd_scale_t scale) {
  int i = threadIdx.x;
  float a = aa + i;
  float b = bb + i;
  __amd_floatx2_storage_t fpx2{a, b};
  __half2 fp16x2{a, b};
  __hip_bfloat162 bf16x2{a, b};
  auto fp8x2_e5m2_from_float = __hipext_ocp_fp8x2_e5m2(a, b);
  auto fp8x2_e5m2_from_floatx2 = __hipext_ocp_fp8x2_e5m2(fpx2);
  auto fp8x2_e5m2_scale = __hipext_ocp_fp8x2_e5m2(a, b, scale);
  auto fp8x2_e5m2_from_floatx2_scale = __hipext_ocp_fp8x2_e5m2(fpx2, scale);
  auto fp8x2_from_half = __hipext_ocp_fp8x2_e5m2(__amd_cvt_half2_to_fp16x2(fp16x2), scale);
  auto fp8x2_from_bf16 = __hipext_ocp_fp8x2_e5m2(__amd_cvt_hipbf162_to_bf16x2(bf16x2), scale);
  res1[i] = fp8x2_e5m2_from_float;
  res2[i] = fp8x2_e5m2_from_floatx2;
  res3[i] = fp8x2_e5m2_scale.get_scaled_floatx2(scale);
  res4[i] = fp8x2_e5m2_from_floatx2_scale.get_scaled_floatx2(scale);
  res5[i] = fp8x2_from_half.get_scaled_floatx2(scale);
  res6[i] = fp8x2_from_bf16.get_scaled_floatx2(scale);
}

static __global__ void cxx_fp6x32_to_floatx32_e2m3(__amd_floatx32_storage_t* res,
                                                   unsigned int round = 0,
                                                   __amd_scale_t scale = 0) {
  if (threadIdx.x == 0) {
    __amd_floatx32_storage_t in;
    for (int i = 0; i < 32; i++) {
      in[i] = static_cast<float>(i % 8);
    }
    __hipext_ocp_fp6x32_e2m3 fp6(in, round, scale);
    *res = fp6.get_scaled_floatx32(scale);
  }
}

static __global__ void cxx_fp6x32_to_floatx32_e3m2(__amd_floatx32_storage_t* res,
                                                   unsigned int round = 0,
                                                   __amd_scale_t scale = 0) {
  if (threadIdx.x == 0) {
    __amd_floatx32_storage_t in;
    for (int i = 0; i < 32; i++) {
      in[i] = static_cast<float>(i % 8);
    }
    __hipext_ocp_fp6x32_e3m2 fp6(in, round, scale);
    *res = fp6.get_scaled_floatx32(scale);
  }
}

static __global__ void cxx_fp4x2_to_floatx2_e2m1(__amd_floatx2_storage_t* res1,
                                                 __amd_floatx2_storage_t* res2,
                                                 __amd_floatx2_storage_t* res3,
                                                 __amd_floatx2_storage_t* res4, float a, float b,
                                                 __amd_scale_t scale) {
  int i = threadIdx.x;
  __amd_floatx2_storage_t fpx2{a, b};
  __half2 fp16x2{a, b};
  __hip_bfloat162 bf16x2{a, b};
  auto fp4x2_scale = __hipext_ocp_fp4x2_e2m1(a, b, scale);
  auto fp4x2_from_floatx2_scale = __hipext_ocp_fp4x2_e2m1(fpx2, scale);
  auto fp4x2_from_half = __hipext_ocp_fp4x2_e2m1(__amd_cvt_half2_to_fp16x2(fp16x2), scale);
  auto fp4x2_from_bf16 = __hipext_ocp_fp4x2_e2m1(__amd_cvt_hipbf162_to_bf16x2(bf16x2), scale);
  res1[i] = fp4x2_scale.get_scaled_floatx2(scale);
  res2[i] = fp4x2_from_floatx2_scale.get_scaled_floatx2(scale);
  res3[i] = fp4x2_from_half.get_scaled_floatx2(scale);
  res4[i] = fp4x2_from_bf16.get_scaled_floatx2(scale);
}

static __global__ void pack_and_unpack_fp8x4(float* a) {
  int i = threadIdx.x;
  if (i == 0) {
    auto fp8_a = __amd_cvt_float_to_fp8_sr(a[0], __AMD_OCP_E4M3, 0 /* seed */);
    auto fp8_b = __amd_cvt_float_to_fp8_sr(a[1], __AMD_OCP_E4M3, 0 /* seed */);
    auto fp8_c = __amd_cvt_float_to_fp8_sr(a[2], __AMD_OCP_E4M3, 0 /* seed */);
    auto fp8_d = __amd_cvt_float_to_fp8_sr(a[3], __AMD_OCP_E4M3, 0 /* seed */);
    auto packed =
        __amd_create_fp8x8(__amd_create_fp8x2(fp8_a, fp8_b), __amd_create_fp8x2(fp8_c, fp8_d),
                           __amd_create_fp8x2(fp8_a, fp8_b), __amd_create_fp8x2(fp8_c, fp8_d));
    a[0] = __amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(packed, 0), 0),
                                  __AMD_OCP_E4M3);
    a[1] = __amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(packed, 0), 1),
                                  __AMD_OCP_E4M3);
    a[2] = __amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(packed, 1), 0),
                                  __AMD_OCP_E4M3);
    a[3] = __amd_cvt_fp8_to_float(__amd_extract_fp8(__amd_extract_fp8x2(packed, 1), 1),
                                  __AMD_OCP_E4M3);
  }
}

HIP_TEST_CASE(Unit_amd_ocp_fp8) {
  constexpr int size = 32;
  SECTION("E4M3") {
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<float> in;
    in.reserve(size);
    for (int i = -15; i <= (size / 2); i++) {
      in.push_back(i * 1.0f);
    }
    float* d_ptr;
    __amd_fp8_storage_t* d_out;
    HIP_CHECK(hipMalloc(&d_ptr, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp8_storage_t) * size));
    HIP_CHECK(hipMemcpy(d_ptr, in.data(), sizeof(float) * in.size(), hipMemcpyHostToDevice));
    float_to_fp8_sr<<<1, size>>>(d_ptr, d_out, interpret, size);
    // d_out is populated, cvt back and populate in d_ptr
    HIP_CHECK(hipMemset(d_ptr, 0, sizeof(float) * size));
    fp8_to_float<<<1, size>>>(d_out, interpret, d_ptr, size);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_ptr, sizeof(float) * res.size(), hipMemcpyDeviceToHost));
    for (int i = 0; i < size; i++) {
      INFO("Result: " << res[i] << " input: " << in[i]);
      REQUIRE(std::fabs(res[i] - in[i]) <= 2.0f);
    }
    HIP_CHECK(hipFree(d_ptr));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("E5M2") {
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<float> in;
    in.reserve(size);
    for (int i = -15; i <= (size / 2); i++) {
      in.push_back(i * 1.0f);
    }
    float* d_ptr;
    __amd_fp8_storage_t* d_out;
    HIP_CHECK(hipMalloc(&d_ptr, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp8_storage_t) * size));
    HIP_CHECK(hipMemcpy(d_ptr, in.data(), sizeof(float) * in.size(), hipMemcpyHostToDevice));
    float_to_fp8_sr<<<1, size>>>(d_ptr, d_out, interpret, size);
    // d_out is populated, cvt back and populate in d_ptr
    HIP_CHECK(hipMemset(d_ptr, 0, sizeof(float) * size));
    fp8_to_float<<<1, size>>>(d_out, interpret, d_ptr, size);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_ptr, sizeof(float) * res.size(), hipMemcpyDeviceToHost));
    for (int i = 0; i < size; i++) {
      INFO("Result: " << res[i] << " input: " << in[i]);
      REQUIRE(std::fabs(res[i] - in[i]) <= 2.0f);
    }
    HIP_CHECK(hipFree(d_ptr));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("E4M3x2") {
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_floatx2_storage_t> in;
    for (int i = 0; i < size; i++) {
      __amd_floatx2_storage_t tmp{i + 1.0f, i * 1.0f};
      in.push_back(tmp);
    }
    __amd_floatx2_storage_t* d_in;
    __amd_fp8x2_storage_t* d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp8x2_storage_t) * size));
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(__amd_floatx2_storage_t) * in.size(),
                        hipMemcpyHostToDevice));
    floatx2_to_fp8x2<<<1, size>>>(d_in, d_out, interpret, size);
    HIP_CHECK(hipMemset(d_in, 0, sizeof(__amd_floatx2_storage_t) * size));
    fp8x2_to_floatx2<<<1, size>>>(d_out, interpret, d_in, size);
    std::vector<__amd_floatx2_storage_t> out(size);
    HIP_CHECK(hipMemcpy(out.data(), d_in, sizeof(__amd_floatx2_storage_t) * out.size(),
                        hipMemcpyDeviceToHost));
    for (int i = 0; i < size; i++) {
      auto in1 = in[i][0];
      auto in2 = in[i][1];
      auto out1 = out[i][0];
      auto out2 = out[i][1];
      INFO("Input: " << in1 << ", " << in2);
      INFO("Output: " << out1 << ", " << out2);
      REQUIRE(std::fabs(in1 - out1) <= 2.0f);
      REQUIRE(std::fabs(in2 - out2) <= 2.0f);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("E5M2x2") {
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_floatx2_storage_t> in;
    for (int i = 0; i < size; i++) {
      __amd_floatx2_storage_t tmp{i + 1.0f, i * 1.0f};
      in.push_back(tmp);
    }
    __amd_floatx2_storage_t* d_in;
    __amd_fp8x2_storage_t* d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp8x2_storage_t) * size));
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(__amd_floatx2_storage_t) * in.size(),
                        hipMemcpyHostToDevice));
    floatx2_to_fp8x2<<<1, size>>>(d_in, d_out, interpret, size);
    HIP_CHECK(hipMemset(d_in, 0, sizeof(__amd_floatx2_storage_t) * size));
    fp8x2_to_floatx2<<<1, size>>>(d_out, interpret, d_in, size);
    std::vector<__amd_floatx2_storage_t> out(size);
    HIP_CHECK(hipMemcpy(out.data(), d_in, sizeof(__amd_floatx2_storage_t) * out.size(),
                        hipMemcpyDeviceToHost));
    for (int i = 0; i < size; i++) {
      auto in1 = in[i][0];
      auto in2 = in[i][1];
      auto out1 = out[i][0];
      auto out2 = out[i][1];
      INFO("Input: " << in1 << ", " << in2);
      INFO("Output: " << out1 << ", " << out2);
      REQUIRE(std::fabs(in1 - out1) <= 2.0f);
      REQUIRE(std::fabs(in2 - out2) <= 2.0f);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("E4M3x2 scale") {
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_floatx2_storage_t> in;
    for (int i = 0; i < size; i++) {
      __amd_floatx2_storage_t tmp{i + 1.0f, i * 1.0f};
      in.push_back(tmp);
    }
    __amd_floatx2_storage_t* d_in;
    __amd_fp8x2_storage_t* d_out;
    __amd_scale_t scale = 1;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp8x2_storage_t) * size));
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(__amd_floatx2_storage_t) * in.size(),
                        hipMemcpyHostToDevice));
    floatx2_to_fp8x2_scale<<<1, size>>>(d_in, d_out, interpret, size, scale);
    HIP_CHECK(hipMemset(d_in, 0, sizeof(__amd_floatx2_storage_t) * size));
    fp8x2_to_floatx2_scale<<<1, size>>>(d_out, interpret, d_in, size, scale);
    std::vector<__amd_floatx2_storage_t> out(size);
    HIP_CHECK(hipMemcpy(out.data(), d_in, sizeof(__amd_floatx2_storage_t) * out.size(),
                        hipMemcpyDeviceToHost));
    for (int i = 0; i < size; i++) {
      auto in1 = in[i][0];
      auto in2 = in[i][1];
      auto out1 = out[i][0];
      auto out2 = out[i][1];
      INFO("Input: " << in1 << ", " << in2);
      INFO("Output: " << out1 << ", " << out2);
      INFO("Scale: " << (int)scale);
      REQUIRE(std::fabs(in1 - out1) <= 2.0f);
      REQUIRE(std::fabs(in2 - out2) <= 2.0f);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("E5M2x2 scale") {
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_floatx2_storage_t> in;
    for (int i = 0; i < size; i++) {
      __amd_floatx2_storage_t tmp{i + 1.0f, i * 1.0f};
      in.push_back(tmp);
    }
    __amd_floatx2_storage_t* d_in;
    __amd_fp8x2_storage_t* d_out;
    __amd_scale_t scale = 1;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp8x2_storage_t) * size));
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(__amd_floatx2_storage_t) * in.size(),
                        hipMemcpyHostToDevice));
    floatx2_to_fp8x2_scale<<<1, size>>>(d_in, d_out, interpret, size, scale);
    HIP_CHECK(hipMemset(d_in, 0, sizeof(__amd_floatx2_storage_t) * size));
    fp8x2_to_floatx2_scale<<<1, size>>>(d_out, interpret, d_in, size, scale);
    std::vector<__amd_floatx2_storage_t> out(size);
    HIP_CHECK(hipMemcpy(out.data(), d_in, sizeof(__amd_floatx2_storage_t) * out.size(),
                        hipMemcpyDeviceToHost));
    for (int i = 0; i < size; i++) {
      auto in1 = in[i][0];
      auto in2 = in[i][1];
      auto out1 = out[i][0];
      auto out2 = out[i][1];
      INFO("Input: " << in1 << ", " << in2);
      INFO("Output: " << out1 << ", " << out2);
      INFO("Scale: " << scale);
      REQUIRE(std::fabs(in1 - out1) <= 2.0f);
      REQUIRE(std::fabs(in2 - out2) <= 2.0f);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("E4M3 sr scale") {
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<float> in;
    in.reserve(size);
    for (int i = -15; i <= (size / 2); i++) {
      in.push_back(i * 1.0f);
    }
    float* d_ptr;
    __amd_fp8_storage_t* d_out;
    HIP_CHECK(hipMalloc(&d_ptr, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp8_storage_t) * size));
    HIP_CHECK(hipMemcpy(d_ptr, in.data(), sizeof(float) * in.size(), hipMemcpyHostToDevice));
    __amd_scale_t scale = 1;
    float_to_fp8_sr_scale<<<1, size>>>(d_ptr, d_out, interpret, size, 0 /* seed */, scale);
    // d_out is populated, cvt back and populate in d_ptr
    HIP_CHECK(hipMemset(d_ptr, 0, sizeof(float) * size));
    fp8_to_float_scale<<<1, size>>>(d_out, interpret, d_ptr, size, scale);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_ptr, sizeof(float) * res.size(), hipMemcpyDeviceToHost));
    for (int i = 0; i < size; i++) {
      INFO("Result: " << res[i] << " input: " << in[i]);
      REQUIRE(std::fabs(res[i] - in[i]) <= 2.0f);
    }
    HIP_CHECK(hipFree(d_ptr));
    HIP_CHECK(hipFree(d_out));
  }
}

HIP_TEST_CASE(Unit_fp8_pack_unpack) {
  float* d_a;
  HIP_CHECK(hipMalloc(&d_a, sizeof(float) * 4));
  std::vector<float> a(4, 0.0f);
  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  a[3] = 4.0f;
  HIP_CHECK(hipMemcpy(d_a, a.data(), sizeof(float) * 4, hipMemcpyHostToDevice));
  pack_and_unpack_fp8x4<<<1, 32>>>(d_a);
  std::vector<float> res(4, 0.0f);
  HIP_CHECK(hipMemcpy(res.data(), d_a, sizeof(float) * 4, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < a.size(); i++) {
    REQUIRE(a[i] == res[i]);
  }
  HIP_CHECK(hipFree(d_a));
}

static __global__ void float_to_fp6(__amd_floatx32_storage_t* in, __amd_floatx32_storage_t* out,
                                    __amd_fp6_interpretation_t interpret, unsigned int round = 0,
                                    __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    auto fp6 = __amd_cvt_floatx32_to_fp6x32_sr_scale(*in, interpret, round, scale);
    *out = __amd_cvt_fp6x32_to_floatx32_scale(fp6, interpret, scale);
  }
}

static __global__ void float_to_fp6_sr(__amd_floatx32_storage_t* in, __amd_floatx32_storage_t* out,
                                       __amd_fp6_interpretation_t interpret, unsigned int round,
                                       __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    auto fp6 = __amd_cvt_floatx32_to_fp6x32_sr_scale(*in, interpret, round, scale);
    *out = __amd_cvt_fp6x32_to_floatx32_scale(fp6, interpret, scale);
  }
}

static __global__ void bf16_to_fp6(__amd_floatx32_storage_t* in, __amd_floatx32_storage_t* out,
                                   __amd_fp6_interpretation_t interpret, __amd_scale_t scale = 0) {
  if (threadIdx.x == 0) {
    __amd_bf16x32_storage_t bf16_in, bf16_out;
    for (size_t i = 0; i < 32; i++) {
      bf16_in[i] = (*in)[i];
    }
    auto fp6 = __amd_cvt_bf16x32_to_fp6x32_scale(bf16_in, interpret, scale);
    bf16_out = __amd_cvt_fp6x32_to_bf16x32_scale(fp6, interpret, scale);
    for (size_t i = 0; i < 32; i++) {
      (*out)[i] = bf16_out[i];
    }
  }
}

static __global__ void fp16_to_fp6(__amd_floatx32_storage_t* in, __amd_floatx32_storage_t* out,
                                   __amd_fp6_interpretation_t interpret, __amd_scale_t scale = 0) {
  if (threadIdx.x == 0) {
    __amd_fp16x32_storage_t fp16_in, fp16_out;
    for (size_t i = 0; i < 32; i++) {
      fp16_in[i] = (*in)[i];
    }
    auto fp6 = __amd_cvt_fp16x32_to_fp6x32_scale(fp16_in, interpret, scale);
    fp16_out = __amd_cvt_fp6x32_to_fp16x32_scale(fp6, interpret, scale);
    for (size_t i = 0; i < 32; i++) {
      (*out)[i] = fp16_out[i];
    }
  }
}

static __global__ void float_halves_to_fp6(__amd_floatx32_storage_t* in,
                                           __amd_floatx32_storage_t* out,
                                           __amd_fp6_interpretation_t interpret,
                                           __amd_scale_t scale = 0) {
  if (threadIdx.x == 0) {
    __amd_floatx16_storage_t fpx16_1, fpx16_2;
    for (size_t i = 0; i < 16; i++) {
      fpx16_1[i] = (*in)[i];
      fpx16_2[i] = (*in)[i + 16];
    }
    auto fp6 = __amd_cvt_floatx16_floatx16_to_fp6x32_scale(fpx16_1, fpx16_2, interpret, scale);
    *out = __amd_cvt_fp6x32_to_floatx32_scale(fp6, interpret, scale);
  }
}

static __global__ void floatx32_to_fp6(__amd_floatx32_storage_t* in, __amd_floatx32_storage_t* out,
                                       __amd_fp6_interpretation_t interpret,
                                       __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    auto fp6 = __amd_cvt_floatx32_to_fp6x32_scale(*in, interpret, scale);
    *out = __amd_cvt_fp6x32_to_floatx32_scale(fp6, interpret, scale);
  }
}

static __global__ void bf16_to_fp6_sr(__amd_floatx32_storage_t* in, __amd_floatx32_storage_t* out,
                                      __amd_fp6_interpretation_t interpret, unsigned int round,
                                      __amd_scale_t scale = 0) {
  if (threadIdx.x == 0) {
    __amd_bf16x32_storage_t bf16_in, bf16_out;
    for (size_t i = 0; i < 32; i++) {
      bf16_in[i] = (*in)[i];
    }
    auto fp6 = __amd_cvt_bf16x32_to_fp6x32_sr_scale(bf16_in, interpret, round, scale);
    bf16_out = __amd_cvt_fp6x32_to_bf16x32_scale(fp6, interpret, scale);
    for (size_t i = 0; i < 32; i++) {
      (*out)[i] = bf16_out[i];
    }
  }
}

static __global__ void fp16_to_fp6_sr(__amd_floatx32_storage_t* in, __amd_floatx32_storage_t* out,
                                      __amd_fp6_interpretation_t interpret, unsigned int round,
                                      __amd_scale_t scale = 0) {
  if (threadIdx.x == 0) {
    __amd_fp16x32_storage_t fp16_in, fp16_out;
    for (size_t i = 0; i < 32; i++) {
      fp16_in[i] = (*in)[i];
    }
    auto fp6 = __amd_cvt_fp16x32_to_fp6x32_sr_scale(fp16_in, interpret, round, scale);
    fp16_out = __amd_cvt_fp6x32_to_fp16x32_scale(fp6, interpret, scale);
    for (size_t i = 0; i < 32; i++) {
      (*out)[i] = fp16_out[i];
    }
  }
}

static __global__ void floatx16_to_fp6(__amd_floatx32_storage_t* in, __amd_floatx32_storage_t* out,
                                       __amd_fp6_interpretation_t interpret,
                                       __amd_scale_t scale = 0) {
  if (threadIdx.x == 0) {
    union floatx_union {
      __amd_floatx32_storage_t fpx32;
      __amd_floatx16_storage_t fpx16[2];
    };
    floatx_union u_in;
    u_in.fpx32 = *in;

    auto fp6 =
        __amd_cvt_floatx16_floatx16_to_fp6x32_scale(u_in.fpx16[0], u_in.fpx16[1], interpret, scale);
    *out = __amd_cvt_fp6x32_to_floatx32_scale(fp6, interpret, scale);
  }
}

static __global__ void floatx16_to_fp6_sr(__amd_floatx32_storage_t* in,
                                          __amd_floatx32_storage_t* out,
                                          __amd_fp6_interpretation_t interpret, unsigned int round,
                                          __amd_scale_t scale = 0) {
  if (threadIdx.x == 0) {
    auto fp6 = __amd_cvt_floatx32_to_fp6x32_sr_scale(*in, interpret, round, scale);
    *out = __amd_cvt_fp6x32_to_floatx32_scale(fp6, interpret, scale);
  }
}

HIP_TEST_CASE(Unit_amd_ocp_fp6) {
  __amd_floatx32_storage_t fpx32, out;
  __amd_floatx32_storage_t *d_in = nullptr, *d_out = nullptr;
  float iter = 0.0f;
  for (size_t i = 0; i < 32; i++) {
    iter++;
    fpx32[i] = iter;
    if (iter >= 8) {
      iter = 0.0f;
    }
  }

  HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx32_storage_t)));
  HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx32_storage_t)));
  HIP_CHECK(hipMemcpy(d_in, &fpx32, sizeof(__amd_floatx32_storage_t), hipMemcpyHostToDevice));

  SECTION("float to fp6 E2M3") {
    float_to_fp6<<<1, 32>>>(d_in, d_out, __AMD_OCP_E2M3);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 1.0f);
    }
  }

  SECTION("bf16 to fp6 E2M3") {
    bf16_to_fp6<<<1, 32>>>(d_in, d_out, __AMD_OCP_E2M3);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 1.0f);
    }
  }

  SECTION("fp16 to fp6 E2M3") {
    fp16_to_fp6<<<1, 32>>>(d_in, d_out, __AMD_OCP_E2M3);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 1.0f);
    }
  }

  SECTION("float halves to fp6 E2M3") {
    float_halves_to_fp6<<<1, 32>>>(d_in, d_out, __AMD_OCP_E2M3);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 1.0f);
    }
  }

  SECTION("floatx32 to fp6 E2M3") {
    floatx32_to_fp6<<<1, 32>>>(d_in, d_out, __AMD_OCP_E2M3);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 1.0f);
    }
  }

  SECTION("floatx16 to fp6 E2M3") {
    floatx16_to_fp6<<<1, 32>>>(d_in, d_out, __AMD_OCP_E2M3);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 1.0f);
    }
  }

  SECTION("float to fp6 E2M3 sr") {
    __amd_scale_t scale = 1;
    float_to_fp6_sr<<<1, 32>>>(d_in, d_out, __AMD_OCP_E2M3, 0, scale);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i] << " idx:" << i);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 3.0f);
    }
  }

  SECTION("bf16 to fp6 E2M3 sr") {
    __amd_scale_t scale = 1;
    bf16_to_fp6_sr<<<1, 32>>>(d_in, d_out, __AMD_OCP_E2M3, 0, scale);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 3.0f);
    }
  }

  SECTION("fp16 to fp6 E2M3 sr") {
    __amd_scale_t scale = 1;
    fp16_to_fp6_sr<<<1, 32>>>(d_in, d_out, __AMD_OCP_E2M3, 0, scale);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 3.0f);
    }
  }

  SECTION("floatx16 to fp6 E2M3 sr") {
    __amd_scale_t scale = 1;
    floatx16_to_fp6_sr<<<1, 32>>>(d_in, d_out, __AMD_OCP_E2M3, 0, scale);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 3.0f);
    }
  }

  SECTION("float to fp6 E3M2") {
    float_to_fp6<<<1, 32>>>(d_in, d_out, __AMD_OCP_E3M2);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 1.0f);
    }
  }

  SECTION("bf16 to fp6 E3M2") {
    bf16_to_fp6<<<1, 32>>>(d_in, d_out, __AMD_OCP_E3M2);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 1.0f);
    }
  }

  SECTION("fp16 to fp6 E3M2") {
    fp16_to_fp6<<<1, 32>>>(d_in, d_out, __AMD_OCP_E3M2);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 1.0f);
    }
  }

  SECTION("float halves to fp6 E3M2") {
    float_halves_to_fp6<<<1, 32>>>(d_in, d_out, __AMD_OCP_E3M2);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 1.0f);
    }
  }

  SECTION("floatx32 to fp6 E3M2") {
    floatx32_to_fp6<<<1, 32>>>(d_in, d_out, __AMD_OCP_E3M2);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 1.0f);
    }
  }

  SECTION("floatx16 to fp6 E3M2") {
    floatx16_to_fp6<<<1, 32>>>(d_in, d_out, __AMD_OCP_E3M2);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 1.0f);
    }
  }

  SECTION("float to fp6 E3M2 sr") {
    __amd_scale_t scale = 1;
    float_to_fp6_sr<<<1, 32>>>(d_in, d_out, __AMD_OCP_E3M2, 0, scale);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 3.0f);
    }
  }

  SECTION("bf16 to fp6 E3M2 sr") {
    __amd_scale_t scale = 1;
    bf16_to_fp6_sr<<<1, 32>>>(d_in, d_out, __AMD_OCP_E3M2, 0, scale);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 3.0f);
    }
  }

  SECTION("fp16 to fp6 E3M2 sr") {
    __amd_scale_t scale = 1;
    fp16_to_fp6_sr<<<1, 32>>>(d_in, d_out, __AMD_OCP_E3M2, 0, scale);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 3.0f);
    }
  }

  SECTION("floatx16 to fp6 E3M2 sr") {
    __amd_scale_t scale = 1;
    floatx16_to_fp6_sr<<<1, 32>>>(d_in, d_out, __AMD_OCP_E3M2, 0, scale);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 32; i++) {
      INFO("In: " << fpx32[i] << " out: " << out[i]);
      CHECK(std::fabs(fpx32[i] - out[i]) <= 3.0f);
    }
  }

  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_in));
}

static __global__ void float_to_fp4(__amd_floatx2_storage_t* in, __amd_floatx2_storage_t* out,
                                    const __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    auto fp4 = __amd_cvt_floatx2_to_fp4x2_scale(*in, __AMD_OCP_E2M1, scale);
    *out = __amd_cvt_fp4x2_to_floatx2_scale(fp4, __AMD_OCP_E2M1, scale);
  }
}

static __global__ void fp16_to_fp4(__amd_floatx2_storage_t* in, __amd_floatx2_storage_t* out,
                                   const __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    __amd_fp16x2_storage_t fp16;
    fp16[0] = (*in)[0];
    fp16[1] = (*in)[1];
    auto fp4 = __amd_cvt_fp16x2_to_fp4x2_scale(fp16, __AMD_OCP_E2M1, scale);
    auto fp16_cvt = __amd_cvt_fp4x2_to_fp16x2_scale(fp4, __AMD_OCP_E2M1, scale);
    (*out)[0] = fp16_cvt[0];
    (*out)[1] = fp16_cvt[1];
  }
}

static __global__ void bf16_to_fp4(__amd_floatx2_storage_t* in, __amd_floatx2_storage_t* out,
                                   const __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    __amd_bf16x2_storage_t bf16;
    bf16[0] = (*in)[0];
    bf16[1] = (*in)[1];
    auto fp4 = __amd_cvt_bf16x2_to_fp4x2_scale(bf16, __AMD_OCP_E2M1, scale);
    auto bf16_cvt = __amd_cvt_fp4x2_to_bf16x2_scale(fp4, __AMD_OCP_E2M1, scale);
    (*out)[0] = bf16_cvt[0];
    (*out)[1] = bf16_cvt[1];
  }
}

static __global__ void float_to_fp4_sr(__amd_floatx2_storage_t* in, __amd_floatx2_storage_t* out,
                                       unsigned int round, const __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    auto fp4 = __amd_cvt_floatx2_to_fp4x2_sr_scale(*in, __AMD_OCP_E2M1, round, scale);
    *out = __amd_cvt_fp4x2_to_floatx2_scale(fp4, __AMD_OCP_E2M1, scale);
  }
}

static __global__ void fp16_to_fp4_sr(__amd_floatx2_storage_t* in, __amd_floatx2_storage_t* out,
                                      unsigned int round, const __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    __amd_fp16x2_storage_t fp16;
    fp16[0] = (*in)[0];
    fp16[1] = (*in)[1];
    auto fp4 = __amd_cvt_fp16x2_to_fp4x2_sr_scale(fp16, __AMD_OCP_E2M1, round, scale);
    auto fp16_cvt = __amd_cvt_fp4x2_to_fp16x2_scale(fp4, __AMD_OCP_E2M1, scale);
    (*out)[0] = fp16_cvt[0];
    (*out)[1] = fp16_cvt[1];
  }
}

static __global__ void bf16_to_fp4_sr(__amd_floatx2_storage_t* in, __amd_floatx2_storage_t* out,
                                      unsigned int round, const __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    __amd_bf16x2_storage_t bf16;
    bf16[0] = (*in)[0];
    bf16[1] = (*in)[1];
    auto fp4 = __amd_cvt_bf16x2_to_fp4x2_sr_scale(bf16, __AMD_OCP_E2M1, round, scale);
    auto bf16_cvt = __amd_cvt_fp4x2_to_bf16x2_scale(fp4, __AMD_OCP_E2M1, scale);
    (*out)[0] = bf16_cvt[0];
    (*out)[1] = bf16_cvt[1];
  }
}

HIP_TEST_CASE(Unit_amd_ocp_fp4) {
  __amd_floatx2_storage_t fpx2{4.0f, 2.0f}, *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t)));
  HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx2_storage_t)));
  HIP_CHECK(hipMemcpy(d_in, &fpx2, sizeof(__amd_floatx2_storage_t), hipMemcpyHostToDevice));

  SECTION("float to fp4") {
    float_to_fp4<<<1, 32>>>(d_in, d_out);
    __amd_floatx2_storage_t out;
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
    INFO("In: " << fpx2[0] << ", " << fpx2[1]);
    INFO("Out: " << out[0] << ", " << out[1]);
    CHECK(fpx2[0] == out[0]);
    CHECK(fpx2[1] == out[1]);
  }

  SECTION("float to fp4 scale") {
    __amd_scale_t scale = 1;
    float_to_fp4<<<1, 32>>>(d_in, d_out, scale);
    __amd_floatx2_storage_t out;
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
    INFO("In: " << fpx2[0] << ", " << fpx2[1]);
    INFO("Out: " << out[0] << ", " << out[1]);
    CHECK(fpx2[0] == out[0]);
    CHECK(fpx2[1] == out[1]);
  }

  SECTION("fp16 to fp4") {
    fp16_to_fp4<<<1, 32>>>(d_in, d_out);
    __amd_floatx2_storage_t out;
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
    INFO("In: " << fpx2[0] << ", " << fpx2[1]);
    INFO("Out: " << out[0] << ", " << out[1]);
    CHECK(fpx2[0] == out[0]);
    CHECK(fpx2[1] == out[1]);
  }

  SECTION("fp16 to fp4 scale") {
    __amd_scale_t scale = 1;
    fp16_to_fp4<<<1, 32>>>(d_in, d_out, scale);
    __amd_floatx2_storage_t out;
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
    INFO("In: " << fpx2[0] << ", " << fpx2[1]);
    INFO("Out: " << out[0] << ", " << out[1]);
    CHECK(fpx2[0] == out[0]);
    CHECK(fpx2[1] == out[1]);
  }

  SECTION("bf16 to fp4") {
    bf16_to_fp4<<<1, 32>>>(d_in, d_out);
    __amd_floatx2_storage_t out;
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
    INFO("In: " << fpx2[0] << ", " << fpx2[1]);
    INFO("Out: " << out[0] << ", " << out[1]);
    CHECK(fpx2[0] == out[0]);
    CHECK(fpx2[1] == out[1]);
  }

  SECTION("bf16 to fp4 scale") {
    __amd_scale_t scale = 1;
    bf16_to_fp4<<<1, 32>>>(d_in, d_out, scale);
    __amd_floatx2_storage_t out;
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
    INFO("In: " << fpx2[0] << ", " << fpx2[1]);
    INFO("Out: " << out[0] << ", " << out[1]);
    CHECK(fpx2[0] == out[0]);
    CHECK(fpx2[1] == out[1]);
  }

  SECTION("float to fp4 sr scale") {
    __amd_scale_t scale = 0;
    unsigned int round = 1;
    float_to_fp4_sr<<<1, 32>>>(d_in, d_out, round, scale);
    __amd_floatx2_storage_t out;
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
    INFO("In: " << fpx2[0] << ", " << fpx2[1]);
    INFO("Out: " << out[0] << ", " << out[1]);
    CHECK(fpx2[0] == out[0]);
    CHECK(fpx2[1] == out[1]);
  }

  SECTION("fp16 to fp4 sr scale") {
    __amd_scale_t scale = 0;
    unsigned int round = 1;
    fp16_to_fp4_sr<<<1, 32>>>(d_in, d_out, round, scale);
    __amd_floatx2_storage_t out;
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
    INFO("In: " << fpx2[0] << ", " << fpx2[1]);
    INFO("Out: " << out[0] << ", " << out[1]);
    CHECK(fpx2[0] == out[0]);
    CHECK(fpx2[1] == out[1]);
  }

  SECTION("bf16 to fp4 sr scale") {
    __amd_scale_t scale = 0;
    unsigned int round = 1;
    bf16_to_fp4_sr<<<1, 32>>>(d_in, d_out, round, scale);
    __amd_floatx2_storage_t out;
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
    INFO("In: " << fpx2[0] << ", " << fpx2[1]);
    INFO("Out: " << out[0] << ", " << out[1]);
    CHECK(fpx2[0] == out[0]);
    CHECK(fpx2[1] == out[1]);
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}

static __global__ void fp16x8_to_fp4x8_sr_scale(__amd_fp16x8_storage_t* in,
                                                __amd_fp16x8_storage_t* out, unsigned int round,
                                                const __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    auto fp4 = __amd_cvt_fp16x8_to_fp4x8_sr_scale(in[i], __AMD_OCP_E2M1, round, scale);
    out[i] = __amd_cvt_fp4x8_to_fp16x8_scale(fp4, __AMD_OCP_E2M1, scale);
  }
}

static __global__ void fp16x8_to_fp4x8_scale(__amd_fp16x8_storage_t* in,
                                             __amd_fp16x8_storage_t* out,
                                             const __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    auto fp4 = __amd_cvt_fp16x8_to_fp4x8_scale(in[i], __AMD_OCP_E2M1, scale);
    out[i] = __amd_cvt_fp4x8_to_fp16x8_scale(fp4, __AMD_OCP_E2M1, scale);
  }
}


static __global__ void bf16x8_to_fp4x8_sr_scale(__amd_bf16x8_storage_t* in,
                                                __amd_bf16x8_storage_t* out, unsigned int round,
                                                const __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    auto fp4 = __amd_cvt_bf16x8_to_fp4x8_sr_scale(in[i], __AMD_OCP_E2M1, round, scale);
    out[i] = __amd_cvt_fp4x8_to_bf16x8_scale(fp4, __AMD_OCP_E2M1, scale);
  }
}


static __global__ void bf16x8_to_fp4x8_scale(__amd_bf16x8_storage_t* in,
                                             __amd_bf16x8_storage_t* out,
                                             const __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    auto fp4 = __amd_cvt_bf16x8_to_fp4x8_scale(in[i], __AMD_OCP_E2M1, scale);
    out[i] = __amd_cvt_fp4x8_to_bf16x8_scale(fp4, __AMD_OCP_E2M1, scale);
  }
}

#if __AVX512F__
static __global__ void floatx8_to_fp4x8_sr_scale(__amd_floatx8_storage_t* in,
                                                 __amd_floatx8_storage_t* out, unsigned int round,
                                                 const __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    auto fp4 = __amd_cvt_floatx8_to_fp4x8_sr_scale(in[i], __AMD_OCP_E2M1, round, scale);
    out[i] = __amd_cvt_fp4x8_to_floatx8_scale(fp4, __AMD_OCP_E2M1, scale);
  }
}

static __global__ void floatx8_to_fp4x8_scale(__amd_floatx8_storage_t* in,
                                              __amd_floatx8_storage_t* out,
                                              const __amd_scale_t scale = 0) {
  int i = threadIdx.x;
  if (i == 0) {
    auto fp4 = __amd_cvt_floatx8_to_fp4x8_scale(in[i], __AMD_OCP_E2M1, scale);
    out[i] = __amd_cvt_fp4x8_to_floatx8_scale(fp4, __AMD_OCP_E2M1, scale);
  }
}
#endif

HIP_TEST_CASE(Unit_amd_ocp_fp4x8) {
  __amd_fp4x8_storage_t* d_tmp;
  __amd_floatx8_storage_t in;

  HIP_CHECK(hipMalloc(&d_tmp, sizeof(__amd_fp4x8_storage_t)));

  for (size_t i = 0; i < 8; i++) {
    in[i] = int(i) - 4;
  }

  SECTION("fp16x8 sr scale") {
    __amd_fp16x8_storage_t tmp_in, *d_in, *d_out, tmp_out;
    for (size_t i = 0; i < 8; i++) {
      tmp_in[i] = in[i];
    }

    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16x8_storage_t)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16x8_storage_t)));

    HIP_CHECK(hipMemcpy(d_in, &tmp_in, sizeof(__amd_fp16x8_storage_t), hipMemcpyHostToDevice));
    fp16x8_to_fp4x8_sr_scale<<<1, 32>>>(d_in, d_out, 0 /* round */, 0 /* scale */);
    HIP_CHECK(hipMemcpy(&tmp_out, d_out, sizeof(__amd_fp16x8_storage_t), hipMemcpyDeviceToHost));

    auto cpu_fp4 =
        __amd_cvt_fp16x8_to_fp4x8_sr_scale(tmp_in, __AMD_OCP_E2M1, 0 /* round */, 0 /* scale */);
    auto cpu_fp16 = __amd_cvt_fp4x8_to_fp16x8_scale(cpu_fp4, __AMD_OCP_E2M1, 0 /* scale */);

    for (size_t i = 0; i < 8; i++) {
      INFO("index: " << i << " in: " << float(tmp_in[i]) << " cpu: " << float(cpu_fp16[i])
                     << " gpu: " << float(tmp_out[i]));
      REQUIRE(float(cpu_fp16[i]) == float(tmp_out[i]));
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("fp16x8 scale") {
    __amd_fp16x8_storage_t tmp_in, *d_in, *d_out, tmp_out;
    for (size_t i = 0; i < 8; i++) {
      tmp_in[i] = in[i];
    }

    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16x8_storage_t)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16x8_storage_t)));

    HIP_CHECK(hipMemcpy(d_in, &tmp_in, sizeof(__amd_fp16x8_storage_t), hipMemcpyHostToDevice));
    fp16x8_to_fp4x8_scale<<<1, 32>>>(d_in, d_out, 0 /* scale */);
    HIP_CHECK(hipMemcpy(&tmp_out, d_out, sizeof(__amd_fp16x8_storage_t), hipMemcpyDeviceToHost));

    auto cpu_fp4 = __amd_cvt_fp16x8_to_fp4x8_scale(tmp_in, __AMD_OCP_E2M1, 0 /* scale */);
    auto cpu_fp16 = __amd_cvt_fp4x8_to_fp16x8_scale(cpu_fp4, __AMD_OCP_E2M1, 0 /* scale */);

    for (size_t i = 0; i < 8; i++) {
      INFO("index: " << i << " in: " << float(tmp_in[i]) << " cpu: " << float(cpu_fp16[i])
                     << " gpu: " << float(tmp_out[i]));
      REQUIRE(float(cpu_fp16[i]) == float(tmp_out[i]));
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("bf16x8 sr scale") {
    __amd_bf16x8_storage_t tmp_in, *d_in, *d_out, tmp_out;
    for (size_t i = 0; i < 8; i++) {
      tmp_in[i] = in[i];
    }

    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_bf16x8_storage_t)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_bf16x8_storage_t)));

    HIP_CHECK(hipMemcpy(d_in, &tmp_in, sizeof(__amd_bf16x8_storage_t), hipMemcpyHostToDevice));
    bf16x8_to_fp4x8_sr_scale<<<1, 32>>>(d_in, d_out, 0 /* round */, 0 /* scale */);
    HIP_CHECK(hipMemcpy(&tmp_out, d_out, sizeof(__amd_bf16x8_storage_t), hipMemcpyDeviceToHost));

    auto cpu_fp4 =
        __amd_cvt_bf16x8_to_fp4x8_sr_scale(tmp_in, __AMD_OCP_E2M1, 0 /* round */, 0 /* scale */);
    auto cpu_bf16 = __amd_cvt_fp4x8_to_bf16x8_scale(cpu_fp4, __AMD_OCP_E2M1, 0 /* scale */);

    for (size_t i = 0; i < 8; i++) {
      INFO("index: " << i << " in: " << float(tmp_in[i]) << " cpu: " << float(cpu_bf16[i])
                     << " gpu: " << float(tmp_out[i]));
      REQUIRE(float(cpu_bf16[i]) == float(tmp_out[i]));
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("bf16x8 scale") {
    __amd_bf16x8_storage_t tmp_in, *d_in, *d_out, tmp_out;
    for (size_t i = 0; i < 8; i++) {
      tmp_in[i] = in[i];
    }

    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_bf16x8_storage_t)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_bf16x8_storage_t)));

    HIP_CHECK(hipMemcpy(d_in, &tmp_in, sizeof(__amd_bf16x8_storage_t), hipMemcpyHostToDevice));
    bf16x8_to_fp4x8_scale<<<1, 32>>>(d_in, d_out, 0 /* scale */);
    HIP_CHECK(hipMemcpy(&tmp_out, d_out, sizeof(__amd_bf16x8_storage_t), hipMemcpyDeviceToHost));

    auto cpu_fp4 = __amd_cvt_bf16x8_to_fp4x8_scale(tmp_in, __AMD_OCP_E2M1, 0 /* scale */);
    auto cpu_bf16 = __amd_cvt_fp4x8_to_bf16x8_scale(cpu_fp4, __AMD_OCP_E2M1, 0 /* scale */);

    for (size_t i = 0; i < 8; i++) {
      INFO("index: " << i << " in: " << float(tmp_in[i]) << " cpu: " << float(cpu_bf16[i])
                     << " gpu: " << float(tmp_out[i]));
      REQUIRE(float(cpu_bf16[i]) == float(tmp_out[i]));
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

#if __AVX512F__
  SECTION("floatx8 sr scale") {
    __amd_floatx8_storage_t *d_in, *d_out, tmp_out;

    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx8_storage_t)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx8_storage_t)));

    HIP_CHECK(hipMemcpy(d_in, &in, sizeof(__amd_floatx8_storage_t), hipMemcpyHostToDevice));
    floatx8_to_fp4x8_sr_scale<<<1, 32>>>(d_in, d_out, 0 /* round */, 0 /* scale */);
    HIP_CHECK(hipMemcpy(&tmp_out, d_out, sizeof(__amd_floatx8_storage_t), hipMemcpyDeviceToHost));

    auto cpu_fp4 =
        __amd_cvt_floatx8_to_fp4x8_sr_scale(in, __AMD_OCP_E2M1, 0 /* round */, 0 /* scale */);
    auto cpu_out = __amd_cvt_fp4x8_to_floatx8_scale(cpu_fp4, __AMD_OCP_E2M1, 0 /* scale */);

    for (size_t i = 0; i < 8; i++) {
      INFO("index: " << i << " in: " << float(in[i]) << " cpu: " << float(cpu_out[i])
                     << " gpu: " << float(tmp_out[i]));
      REQUIRE(cpu_out[i] == tmp_out[i]);
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("floatx8 scale") {
    __amd_floatx8_storage_t *d_in, *d_out, tmp_out;

    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx8_storage_t)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx8_storage_t)));

    HIP_CHECK(hipMemcpy(d_in, &in, sizeof(__amd_floatx8_storage_t), hipMemcpyHostToDevice));
    floatx8_to_fp4x8_scale<<<1, 32>>>(d_in, d_out, 0 /* scale */);
    HIP_CHECK(hipMemcpy(&tmp_out, d_out, sizeof(__amd_floatx8_storage_t), hipMemcpyDeviceToHost));

    auto cpu_fp4 = __amd_cvt_floatx8_to_fp4x8_scale(in, __AMD_OCP_E2M1, 0 /* scale */);
    auto cpu_out = __amd_cvt_fp4x8_to_floatx8_scale(cpu_fp4, __AMD_OCP_E2M1, 0 /* scale */);

    for (size_t i = 0; i < 8; i++) {
      INFO("index: " << i << " in: " << float(in[i]) << " cpu: " << float(cpu_out[i])
                     << " gpu: " << float(tmp_out[i]));
      REQUIRE(cpu_out[i] == tmp_out[i]);
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }
#endif

  HIP_CHECK(hipFree(d_tmp));
}

HIP_TEST_CASE(Unit_amd_ocp_cpp_types) {
  SECTION("fp8 to float e4m3") {
    constexpr size_t size = 32;
    float *d_res1, *d_res2, *d_res3, *d_res4, *d_res5;
    HIP_CHECK(hipMalloc(&d_res1, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res2, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res3, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res4, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res5, sizeof(float) * size));
    float a = -15.0f;
    __amd_scale_t scale = 0;
    unsigned int seed = 10;
    cxx_fp8_to_float_e4m3<<<1, size>>>(d_res1, d_res2, d_res3, d_res4, d_res5, a, scale, seed);
    std::vector<float> res1(size, 0.0f), res2(size, 0.0f), res3(size, 0.0f), res4(size, 0.0f),
        res5(size, 0.0f);
    HIP_CHECK(hipMemcpy(res1.data(), d_res1, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res2.data(), d_res2, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res3.data(), d_res3, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res4.data(), d_res4, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res5.data(), d_res5, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      auto input_val = a + i;
      INFO("Input: " << input_val);
      INFO("Output: " << res1[i] << ", " << res2[i] << ", " << res3[i] << ", " << res4[i] << ", "
                      << res5[i]);
      REQUIRE(std::fabs(res1[i] - input_val) <= 2.0f);
      REQUIRE(std::fabs(res2[i] - input_val) <= 2.0f);
      REQUIRE(std::fabs(res3[i] - input_val) <= 2.0f);
      REQUIRE(std::fabs(res4[i] - input_val) <= 2.0f);
      REQUIRE(std::fabs(res5[i] - input_val) <= 2.0f);
    }
    HIP_CHECK(hipFree(d_res1));
    HIP_CHECK(hipFree(d_res2));
    HIP_CHECK(hipFree(d_res3));
    HIP_CHECK(hipFree(d_res4));
    HIP_CHECK(hipFree(d_res5));
  }

  SECTION("fp8 to float e5m2") {
    constexpr size_t size = 32;
    float *d_res1, *d_res2, *d_res3, *d_res4, *d_res5;
    HIP_CHECK(hipMalloc(&d_res1, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res2, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res3, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res4, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res5, sizeof(float) * size));
    float a = -15.0f;
    __amd_scale_t scale = 0;
    unsigned int seed = 10;
    cxx_fp8_to_float_e5m2<<<1, size>>>(d_res1, d_res2, d_res3, d_res4, d_res5, a, scale, seed);
    std::vector<float> res1(size, 0.0f), res2(size, 0.0f), res3(size, 0.0f), res4(size, 0.0f),
        res5(size, 0.0f);
    HIP_CHECK(hipMemcpy(res1.data(), d_res1, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res2.data(), d_res2, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res3.data(), d_res3, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res4.data(), d_res4, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res5.data(), d_res5, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      auto input_val = a + i;
      INFO("Input: " << input_val);
      INFO("Output: " << res1[i] << ", " << res2[i] << ", " << res3[i] << ", " << res4[i] << ", "
                      << res5[i]);
      REQUIRE(std::fabs(res1[i] - input_val) <= 2.0f);
      REQUIRE(std::fabs(res2[i] - input_val) <= 2.0f);
      REQUIRE(std::fabs(res3[i] - input_val) <= 2.0f);
      REQUIRE(std::fabs(res4[i] - input_val) <= 2.0f);
      REQUIRE(std::fabs(res5[i] - input_val) <= 2.0f);
    }
    HIP_CHECK(hipFree(d_res1));
    HIP_CHECK(hipFree(d_res2));
    HIP_CHECK(hipFree(d_res3));
    HIP_CHECK(hipFree(d_res4));
    HIP_CHECK(hipFree(d_res5));
  }

  SECTION("fp8x2 to floatx2 e4m3") {
    constexpr size_t size = 32;
    __amd_floatx2_storage_t *d_res1, *d_res2, *d_res3, *d_res4, *d_res5, *d_res6;
    HIP_CHECK(hipMalloc(&d_res1, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res2, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res3, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res4, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res5, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res6, sizeof(__amd_floatx2_storage_t) * size));
    float a = -15.0f, b = -14.0f;
    __amd_scale_t scale = 0;
    cxx_fp8x2_to_floatx2_e4m3<<<1, size>>>(d_res1, d_res2, d_res3, d_res4, d_res5, d_res6, a, b,
                                           scale);
    std::vector<__amd_floatx2_storage_t> res1(size), res2(size), res3(size), res4(size), res5(size),
        res6(size);
    HIP_CHECK(hipMemcpy(res1.data(), d_res1, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res2.data(), d_res2, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res3.data(), d_res3, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res4.data(), d_res4, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res5.data(), d_res5, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res6.data(), d_res6, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      auto input_val1 = a + i;
      auto input_val2 = b + i;
      INFO("Input val: " << input_val1 << ", " << input_val2);
      INFO("Output1: " << res1[i][0] << ", " << res1[i][1]);
      INFO("Output2: " << res2[i][0] << ", " << res2[i][1]);
      INFO("Output3: " << res3[i][0] << ", " << res3[i][1]);
      INFO("Output4: " << res4[i][0] << ", " << res4[i][1]);
      INFO("Output5: " << res5[i][0] << ", " << res5[i][1]);
      INFO("Output6: " << res6[i][0] << ", " << res6[i][1]);
      REQUIRE(std::fabs(res1[i][0] - input_val1) <= 2.0f);
      REQUIRE(std::fabs(res1[i][1] - input_val2) <= 2.0f);
      REQUIRE(std::fabs(res2[i][0] - input_val1) <= 2.0f);
      REQUIRE(std::fabs(res2[i][1] - input_val2) <= 2.0f);
      REQUIRE(std::fabs(res3[i][0] - input_val1) <= 2.0f);
      REQUIRE(std::fabs(res3[i][1] - input_val2) <= 2.0f);
      REQUIRE(std::fabs(res4[i][0] - input_val1) <= 2.0f);
      REQUIRE(std::fabs(res4[i][1] - input_val2) <= 2.0f);
      REQUIRE(std::fabs(res5[i][0] - input_val1) <= 2.0f);
      REQUIRE(std::fabs(res5[i][1] - input_val2) <= 2.0f);
      REQUIRE(std::fabs(res6[i][0] - input_val1) <= 2.0f);
      REQUIRE(std::fabs(res6[i][1] - input_val2) <= 2.0f);
    }
    HIP_CHECK(hipFree(d_res1));
    HIP_CHECK(hipFree(d_res2));
    HIP_CHECK(hipFree(d_res3));
    HIP_CHECK(hipFree(d_res4));
    HIP_CHECK(hipFree(d_res5));
    HIP_CHECK(hipFree(d_res6));
  }

  SECTION("fp8x2 to floatx2 e5m2") {
    constexpr size_t size = 32;
    __amd_floatx2_storage_t *d_res1, *d_res2, *d_res3, *d_res4, *d_res5, *d_res6;
    HIP_CHECK(hipMalloc(&d_res1, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res2, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res3, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res4, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res5, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res6, sizeof(__amd_floatx2_storage_t) * size));
    float a = -15.0f, b = -14.0f;
    __amd_scale_t scale = 0;
    cxx_fp8x2_to_floatx2_e5m2<<<1, size>>>(d_res1, d_res2, d_res3, d_res4, d_res5, d_res6, a, b,
                                           scale);
    std::vector<__amd_floatx2_storage_t> res1(size), res2(size), res3(size), res4(size), res5(size),
        res6(size);
    HIP_CHECK(hipMemcpy(res1.data(), d_res1, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res2.data(), d_res2, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res3.data(), d_res3, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res4.data(), d_res4, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res5.data(), d_res5, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res6.data(), d_res6, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      auto input_val1 = a + i;
      auto input_val2 = b + i;
      INFO("Input val: " << input_val1 << ", " << input_val2);
      INFO("Output1: " << res1[i][0] << ", " << res1[i][1]);
      INFO("Output2: " << res2[i][0] << ", " << res2[i][1]);
      INFO("Output3: " << res3[i][0] << ", " << res3[i][1]);
      INFO("Output4: " << res4[i][0] << ", " << res4[i][1]);
      INFO("Output5: " << res5[i][0] << ", " << res5[i][1]);
      INFO("Output6: " << res6[i][0] << ", " << res6[i][1]);
      REQUIRE(std::fabs(res1[i][0] - input_val1) <= 2.0f);
      REQUIRE(std::fabs(res1[i][1] - input_val2) <= 2.0f);
      REQUIRE(std::fabs(res2[i][0] - input_val1) <= 2.0f);
      REQUIRE(std::fabs(res2[i][1] - input_val2) <= 2.0f);
      REQUIRE(std::fabs(res3[i][0] - input_val1) <= 2.0f);
      REQUIRE(std::fabs(res3[i][1] - input_val2) <= 2.0f);
      REQUIRE(std::fabs(res4[i][0] - input_val1) <= 2.0f);
      REQUIRE(std::fabs(res4[i][1] - input_val2) <= 2.0f);
      REQUIRE(std::fabs(res5[i][0] - input_val1) <= 2.0f);
      REQUIRE(std::fabs(res5[i][1] - input_val2) <= 2.0f);
      REQUIRE(std::fabs(res6[i][0] - input_val1) <= 2.0f);
      REQUIRE(std::fabs(res6[i][1] - input_val2) <= 2.0f);
    }
    HIP_CHECK(hipFree(d_res1));
    HIP_CHECK(hipFree(d_res2));
    HIP_CHECK(hipFree(d_res3));
    HIP_CHECK(hipFree(d_res4));
    HIP_CHECK(hipFree(d_res5));
    HIP_CHECK(hipFree(d_res6));
  }

  SECTION("fp6 to float e2m3") {
    __amd_floatx32_storage_t* d_res;
    __amd_floatx32_storage_t res;
    HIP_CHECK(hipMalloc(&d_res, sizeof(__amd_floatx32_storage_t)));
    const __amd_scale_t scale = 0;
    cxx_fp6x32_to_floatx32_e2m3<<<1, 32>>>(d_res, scale);
    HIP_CHECK(hipMemcpy(&res, d_res, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < 32; i++) {
      INFO("Index: " << i << " res: " << res[i]);
      CHECK(res[i] == (i % 8));
    }
    HIP_CHECK(hipFree(d_res));
  }

  SECTION("fp6 to float e3m2") {
    __amd_floatx32_storage_t* d_res;
    __amd_floatx32_storage_t res;
    HIP_CHECK(hipMalloc(&d_res, sizeof(__amd_floatx32_storage_t)));
    const __amd_scale_t scale = 1;
    cxx_fp6x32_to_floatx32_e3m2<<<1, 32>>>(d_res, scale);
    HIP_CHECK(hipMemcpy(&res, d_res, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < 32; i++) {
      INFO("Index: " << i << " res: " << res[i]);
      CHECK(res[i] == (i % 8));
    }
    HIP_CHECK(hipFree(d_res));
  }

  SECTION("fp4 to float e2m1") {
    constexpr size_t size = 32;
    __amd_floatx2_storage_t *d_res1, *d_res2, *d_res3, *d_res4;
    HIP_CHECK(hipMalloc(&d_res1, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res2, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res3, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_res4, sizeof(__amd_floatx2_storage_t) * size));
    float a = 1.0f, b = 2.0f;
    __amd_scale_t scale = 0;
    cxx_fp4x2_to_floatx2_e2m1<<<1, size>>>(d_res1, d_res2, d_res3, d_res4, a, b, scale);
    std::vector<__amd_floatx2_storage_t> res1(size), res2(size), res3(size), res4(size);
    HIP_CHECK(hipMemcpy(res1.data(), d_res1, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res2.data(), d_res2, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res3.data(), d_res3, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res4.data(), d_res4, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      CHECK(res1[i][0] == a);
      CHECK(res1[i][1] == b);
      CHECK(res2[i][0] == a);
      CHECK(res2[i][1] == b);
      CHECK(res3[i][0] == a);
      CHECK(res3[i][1] == b);
      CHECK(res4[i][0] == a);
      CHECK(res4[i][1] == b);
    }
    HIP_CHECK(hipFree(d_res1));
    HIP_CHECK(hipFree(d_res2));
    HIP_CHECK(hipFree(d_res3));
    HIP_CHECK(hipFree(d_res4));
  }
}

HIP_TEST_CASE(Unit_amd_ocp_hip_to_compiler_types) {
  SECTION("bf16") {
    const float f_in = 1.5f;
    __amd_bf16_storage_t in = f_in;
    auto hip_bf = __amd_cvt_bf16_to_hipbf16(in);
    float cvt_back = hip_bf;
    REQUIRE(f_in == cvt_back);
    auto bf_cvt = __amd_cvt_hipbf16_to_bf16(hip_bf);
    float f_res = bf_cvt;
    REQUIRE(f_res == f_in);
  }

  SECTION("bf16x2") {
    const float f_in1 = 1.5f, f_in2 = 2.5f;
    __amd_bf16x2_storage_t in{static_cast<__bf16>(f_in1), static_cast<__bf16>(f_in2)};
    auto hip_bf = __amd_cvt_bf16x2_to_hipbf162(in);
    float cvt_back1 = hip_bf.x;
    float cvt_back2 = hip_bf.y;
    REQUIRE(f_in1 == cvt_back1);
    REQUIRE(f_in2 == cvt_back2);
    auto bf162_cvt = __amd_cvt_hipbf162_to_bf16x2(hip_bf);
    float f_out1 = bf162_cvt[0];
    float f_out2 = bf162_cvt[1];
    REQUIRE(f_in1 == f_out1);
    REQUIRE(f_in2 == f_out2);
  }

  SECTION("half") {
    const float f_in = 1.5f;
    __amd_fp16_storage_t in = f_in;
    auto hip_half = __amd_cvt_fp16_to_half(in);
    float cvt_back = hip_half;
    REQUIRE(f_in == cvt_back);
    auto fp16_cvt = __amd_cvt_half_to_fp16(hip_half);
    float f_res = fp16_cvt;
    REQUIRE(f_in == f_res);
  }

  SECTION("halfx2") {
    const float f_in1 = 1.5f, f_in2 = 2.5f;
    __amd_fp16x2_storage_t in{static_cast<_Float16>(f_in1), static_cast<_Float16>(f_in2)};
    auto hip_half = __amd_cvt_fp16x2_to_half2(in);
    float cvt_back1 = hip_half.x;
    float cvt_back2 = hip_half.y;
    REQUIRE(f_in1 == cvt_back1);
    REQUIRE(f_in2 == cvt_back2);
    auto fp16_cvt = __amd_cvt_half2_to_fp16x2(hip_half);
    float f_res1 = fp16_cvt[0];
    float f_res2 = fp16_cvt[1];
    REQUIRE(f_in1 == f_res1);
    REQUIRE(f_in2 == f_res2);
  }
}

__global__ void fp8_device_cvt(float* in, float* out, __amd_fp8_interpretation_t interpret,
                               size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_float_to_fp8_sr(in[i], interpret, 0 /*seed*/);
    out[i] = __amd_cvt_fp8_to_float(tmp, interpret);
  }
}

__global__ void fp8_sr_scale_device_cvt(float* in, float* out, __amd_fp8_interpretation_t interpret,
                                        size_t size, __amd_scale_t scale) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_float_to_fp8_sr_scale(in[i], interpret, 1 /*seed*/, scale);
    out[i] = __amd_cvt_fp8_to_float_scale(tmp, interpret, scale);
  }
}

__global__ void fp8_fp16_sr_scale_device_cvt(__amd_fp16_storage_t* in, __amd_fp16_storage_t* out,
                                             __amd_fp8_interpretation_t interpret, size_t size,
                                             __amd_scale_t scale) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_fp16_to_fp8_sr_scale(in[i], interpret, 0 /*seed*/, scale);
    out[i] = __amd_cvt_fp8_to_fp16_scale(tmp, interpret, scale);
  }
}

__global__ void fp8_fp16_sr_device_cvt(__amd_fp16_storage_t* in, __amd_fp16_storage_t* out,
                                       __amd_fp8_interpretation_t interpret, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_fp16_to_fp8_sr(in[i], interpret, 0 /*seed*/);
    out[i] = __amd_cvt_fp8_to_fp16(tmp, interpret);
  }
}

__global__ void fp8x2_device_cvt(__amd_floatx2_storage_t* in, __amd_floatx2_storage_t* out,
                                 __amd_fp8_interpretation_t interpret, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_floatx2_to_fp8x2(in[i], interpret);
    out[i] = __amd_cvt_fp8x2_to_floatx2(tmp, interpret);
  }
}

__global__ void fp8x2_device_cvt_scale(__amd_floatx2_storage_t* in, __amd_floatx2_storage_t* out,
                                       __amd_fp8_interpretation_t interpret, __amd_scale_t scale,
                                       size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_floatx2_to_fp8x2_scale(in[i], interpret, scale);
    out[i] = __amd_cvt_fp8x2_to_floatx2_scale(tmp, interpret, scale);
  }
}

__global__ void fp8x2_fp16x2_device_cvt_scale(__amd_fp16x2_storage_t* in,
                                              __amd_fp16x2_storage_t* out,
                                              __amd_fp8_interpretation_t interpret,
                                              __amd_scale_t scale, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_fp16x2_to_fp8x2_scale(in[i], interpret, scale);
    out[i] = __amd_cvt_fp8x2_to_fp16x2_scale(tmp, interpret, scale);
  }
}

__global__ void fp8x2_fp16x2_device_cvt(__amd_fp16x2_storage_t* in, __amd_fp16x2_storage_t* out,
                                        __amd_fp8_interpretation_t interpret, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_fp16x2_to_fp8x2(in[i], interpret);
    out[i] = __amd_cvt_fp8x2_to_fp16x2(tmp, interpret);
  }
}

__global__ void fp8_bf16_device_cvt_sr_scale(__amd_bf16_storage_t* in, __amd_bf16_storage_t* out,
                                             __amd_fp8_interpretation_t interpret,
                                             unsigned int round, __amd_scale_t scale, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_bf16_to_fp8_sr_scale(in[i], interpret, round, scale);
    out[i] = __amd_cvt_fp8_to_bf16_scale(tmp, interpret, scale);
  }
}

__global__ void fp8x2_bf16x2_device_cvt_scale(__amd_bf16x2_storage_t* in,
                                              __amd_bf16x2_storage_t* out,
                                              __amd_fp8_interpretation_t interpret,
                                              __amd_scale_t scale, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_bf16x2_to_fp8x2_scale(in[i], interpret, scale);
    out[i] = __amd_cvt_fp8x2_to_bf16x2_scale(tmp, interpret, scale);
  }
}

__global__ void fp8x8_bf16x8_device_cvt_scale(__amd_bf16x8_storage_t* in,
                                              __amd_bf16x8_storage_t* out,
                                              __amd_fp8_interpretation_t interpret,
                                              __amd_scale_t scale, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_bf16x8_to_fp8x8_scale(in[i], interpret, scale);
    out[i] = __amd_cvt_fp8x8_to_bf16x8_scale(tmp, interpret, scale);
  }
}

__global__ void fp8x8_floatx8_device_cvt_sr_scale(__amd_floatx8_storage_t* in,
                                                  __amd_floatx8_storage_t* out,
                                                  __amd_fp8_interpretation_t interpret,
                                                  unsigned int round, __amd_scale_t scale,
                                                  size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_floatx8_to_fp8x8_sr_scale(in[i], interpret, round, scale);
    out[i] = __amd_cvt_fp8x8_to_floatx8_scale(tmp, interpret, scale);
  }
}

__global__ void fp8x8_floatx8_device_cvt_scale(__amd_floatx8_storage_t* in,
                                               __amd_floatx8_storage_t* out,
                                               __amd_fp8_interpretation_t interpret,
                                               __amd_scale_t scale, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_floatx8_to_fp8x8_scale(in[i], interpret, scale);
    out[i] = __amd_cvt_fp8x8_to_floatx8_scale(tmp, interpret, scale);
  }
}

__global__ void fp8x8_fp16x8_device_cvt_sr_scale(__amd_fp16x8_storage_t* in,
                                                 __amd_fp16x8_storage_t* out,
                                                 __amd_fp8_interpretation_t interpret,
                                                 unsigned int round, __amd_scale_t scale,
                                                 size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_fp16x8_to_fp8x8_sr_scale(in[i], interpret, round, scale);
    out[i] = __amd_cvt_fp8x8_to_fp16x8_scale(tmp, interpret, scale);
  }
}

__global__ void fp8x8_bf16x8_device_cvt_sr_scale(__amd_bf16x8_storage_t* in,
                                                 __amd_bf16x8_storage_t* out,
                                                 __amd_fp8_interpretation_t interpret,
                                                 unsigned int round, __amd_scale_t scale,
                                                 size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_bf16x8_to_fp8x8_sr_scale(in[i], interpret, round, scale);
    out[i] = __amd_cvt_fp8x8_to_bf16x8_scale(tmp, interpret, scale);
  }
}

__global__ void fp8x8_fp16x8_device_cvt_scale(__amd_fp16x8_storage_t* in,
                                              __amd_fp16x8_storage_t* out,
                                              __amd_fp8_interpretation_t interpret,
                                              __amd_scale_t scale, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_fp16x8_to_fp8x8_scale(in[i], interpret, scale);
    out[i] = __amd_cvt_fp8x8_to_fp16x8_scale(tmp, interpret, scale);
  }
}

__global__ void fp6x32_convert(__amd_floatx32_storage_t* in, __amd_floatx32_storage_t* out,
                               __amd_fp6_interpretation_t interpret, unsigned int seed,
                               unsigned int scale) {
  int i = threadIdx.x;
  if (i == 0) {
    auto tmp = __amd_cvt_floatx32_to_fp6x32_sr_scale(*in, interpret, seed, scale);
    *out = __amd_cvt_fp6x32_to_floatx32_scale(tmp, interpret, scale);
  }
}

__global__ void fp4x2_convert(__amd_floatx2_storage_t* in, __amd_floatx2_storage_t* out,
                              __amd_scale_t scale, size_t size,
                              __amd_fp4x2_storage_t* tmp_out = nullptr) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_floatx2_to_fp4x2_scale(in[i], __AMD_OCP_E2M1, scale);
    out[i] = __amd_cvt_fp4x2_to_floatx2_scale(tmp, __AMD_OCP_E2M1, scale);
    if (tmp_out != nullptr) {
      tmp_out[i] = tmp;
    }
  }
}

__global__ void fp4x2_sr_scale_convert(__amd_floatx2_storage_t* in, __amd_floatx2_storage_t* out,
                                       const unsigned int seed, __amd_scale_t scale, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    auto tmp = __amd_cvt_floatx2_to_fp4x2_sr_scale(in[i], __AMD_OCP_E2M1, seed, scale);
    out[i] = __amd_cvt_fp4x2_to_floatx2_scale(tmp, __AMD_OCP_E2M1, scale);
  }
}

HIP_TEST_CASE(Unit_ocp_host_fp8_device_compare) {
  SECTION("e4m3") {
    constexpr size_t size = 447 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<float> in;
    in.reserve(size);
    for (int i = -447; i <= 447; i++) {
      in.push_back(static_cast<float>(i));
    }
    REQUIRE(in.size() == size);
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));
    fp8_device_cvt<<<1, size>>>(d_in, d_out, interpret, size);
    // CPU calc
    std::vector<float> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      auto tmp = __amd_cvt_float_to_fp8_sr(in[i], interpret, 0 /*seed*/);
      cpu_res[i] = __amd_cvt_fp8_to_float(tmp, interpret);
    }
    std::vector<float> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " in: " << in[i] << " cpu: " << cpu_res[i] << " gpu: " << gpu_res[i]);
      REQUIRE(cpu_res[i] == gpu_res[i]);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e5m2") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<float> in;
    in.reserve(size);
    for (int i = -511; i <= 511; i++) {
      in.push_back(static_cast<float>(i));
    }
    REQUIRE(in.size() == size);
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));
    fp8_device_cvt<<<1, size>>>(d_in, d_out, interpret, size);
    // CPU calc
    std::vector<float> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      auto tmp = __amd_cvt_float_to_fp8_sr(in[i], interpret, 0 /*seed*/);
      cpu_res[i] = __amd_cvt_fp8_to_float(tmp, interpret);
    }
    std::vector<float> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " in: " << in[i] << " cpu: " << cpu_res[i] << " gpu: " << gpu_res[i]);
      REQUIRE(cpu_res[i] == gpu_res[i]);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e4m3x2") {
    constexpr size_t size = 448 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_floatx2_storage_t> in;
    in.reserve(size);
    for (int i = -448, j = 448; i <= 448; i++, j--) {
      __amd_floatx2_storage_t tmp{static_cast<float>(i), static_cast<float>(j)};
      in.push_back(tmp);
    }
    REQUIRE(in.size() == size);
    __amd_floatx2_storage_t *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(
        hipMemcpy(d_in, in.data(), sizeof(__amd_floatx2_storage_t) * size, hipMemcpyHostToDevice));
    fp8x2_device_cvt<<<1, size>>>(d_in, d_out, interpret, size);
    // CPU calc
    std::vector<__amd_floatx2_storage_t> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      auto tmp = __amd_cvt_floatx2_to_fp8x2(in[i], interpret);
      cpu_res[i] = __amd_cvt_fp8x2_to_floatx2(tmp, interpret);
    }
    std::vector<__amd_floatx2_storage_t> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << "\n\tin:  a: " << in[i][0] << " b: " << in[i][1]
                     << "\n\tcpu: a: " << cpu_res[i][0] << " b: " << cpu_res[i][1]
                     << "\n\tgpu: a: " << gpu_res[i][0] << " b: " << gpu_res[i][1]);
      REQUIRE(float(cpu_res[i][0]) == float(gpu_res[i][0]));
      REQUIRE(float(cpu_res[i][1]) == float(gpu_res[i][1]));
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e5m2x2") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_floatx2_storage_t> in;
    in.reserve(size);
    for (int i = -511, j = 511; i <= 511; i++, j--) {
      __amd_floatx2_storage_t tmp{static_cast<float>(i), static_cast<float>(j)};
      in.push_back(tmp);
    }
    REQUIRE(in.size() == size);
    __amd_floatx2_storage_t *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(
        hipMemcpy(d_in, in.data(), sizeof(__amd_floatx2_storage_t) * size, hipMemcpyHostToDevice));
    fp8x2_device_cvt<<<1, size>>>(d_in, d_out, interpret, size);
    // CPU calc
    std::vector<__amd_floatx2_storage_t> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      auto tmp = __amd_cvt_floatx2_to_fp8x2(in[i], interpret);
      cpu_res[i] = __amd_cvt_fp8x2_to_floatx2(tmp, interpret);
    }
    std::vector<__amd_floatx2_storage_t> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << "\n\tin:  a: " << in[i][0] << " b: " << in[i][1]
                     << "\n\tcpu: a: " << cpu_res[i][0] << " b: " << cpu_res[i][1]
                     << "\n\tgpu: a: " << gpu_res[i][0] << " b: " << gpu_res[i][1]);
      REQUIRE(float(cpu_res[i][0]) == float(gpu_res[i][0]));
      REQUIRE(float(cpu_res[i][1]) == float(gpu_res[i][1]));
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("fp4x2 to float") {
    constexpr __amd_fp4_interpretation_t interpret = __AMD_OCP_E2M1;
    const std::vector<__amd_scale_t> scales = {0, 1, 2};
    std::vector<__amd_floatx2_storage_t> in_vals = {
        __amd_floatx2_storage_t{-3.0f, 3.0f}, __amd_floatx2_storage_t{-2.0f, 2.0f},
        __amd_floatx2_storage_t{-1.0f, 1.0f}, __amd_floatx2_storage_t{-0.0f, 0.0f},
        __amd_floatx2_storage_t{1.0f, -1.0f}, __amd_floatx2_storage_t{2.0f, -2.0f},
        __amd_floatx2_storage_t{3.0f, -3.0f}};
    const size_t size = in_vals.size();
    for (const auto scale : scales) {
      __amd_floatx2_storage_t *d_in, *d_out;
      __amd_fp4x2_storage_t* d_tmp_out;
      std::vector<__amd_fp4x2_storage_t> gpu_tmp_out(size);
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx2_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_tmp_out, sizeof(__amd_fp4x2_storage_t) * size));
      HIP_CHECK(hipMemcpy(d_in, in_vals.data(), sizeof(__amd_floatx2_storage_t) * size,
                          hipMemcpyHostToDevice));
      fp4x2_convert<<<1, 32>>>(d_in, d_out, scale, size, d_tmp_out);
      HIP_CHECK(hipMemcpy(gpu_tmp_out.data(), d_tmp_out, sizeof(__amd_fp4x2_storage_t) * size,
                          hipMemcpyDeviceToHost));
      std::vector<__amd_floatx2_storage_t> gpu_out(size);
      HIP_CHECK(hipMemcpy(gpu_out.data(), d_out, sizeof(__amd_floatx2_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        auto gpu_res = gpu_out[i];
        auto cpu_fp4_temp = __amd_cvt_floatx2_to_fp4x2_scale(in_vals[i], interpret, scale);
        auto cpu_res = __amd_cvt_fp4x2_to_floatx2_scale(cpu_fp4_temp, interpret, scale);
        INFO("Index: " << i << " Scale: " << scale << "\n  Input l: " << in_vals[i][0]
                       << " r: " << in_vals[i][1] << "\n  cpu l: " << cpu_res[0]
                       << " r: " << cpu_res[1] << "\n  gpu l: " << gpu_res[0]
                       << " r: " << gpu_res[1] << "\n  cpu_tmp: " << std::hex
                       << (unsigned)cpu_fp4_temp << " gpu_tmp: " << (unsigned)gpu_tmp_out[i]);
        CHECK(cpu_res[0] == gpu_res[0]);
        CHECK(cpu_res[1] == gpu_res[1]);
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
      HIP_CHECK(hipFree(d_tmp_out));
    }
  }

  SECTION("e4m3x2_scale") {
    constexpr size_t size = 448 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_floatx2_storage_t> in;
      in.reserve(size);
      for (int i = -448, j = 448; i <= 448; i++, j--) {
        __amd_floatx2_storage_t tmp{static_cast<float>(i), static_cast<float>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_floatx2_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx2_storage_t) * size));
      HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(__amd_floatx2_storage_t) * size,
                          hipMemcpyHostToDevice));
      fp8x2_device_cvt_scale<<<1, size>>>(d_in, d_out, interpret, scale, size);
      // CPU calc
      std::vector<__amd_floatx2_storage_t> cpu_res(size, 0.0f);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_floatx2_to_fp8x2_scale(in[i], interpret, scale);
        cpu_res[i] = __amd_cvt_fp8x2_to_floatx2_scale(tmp, interpret, scale);
      }
      std::vector<__amd_floatx2_storage_t> gpu_res(size, 0.0f);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_floatx2_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        INFO("Scale: " << scale << " Index: " << i << "\n\tin:  a: " << in[i][0] << " b: "
                       << in[i][1] << "\n\tcpu: a: " << cpu_res[i][0] << " b: " << cpu_res[i][1]
                       << "\n\tgpu: a: " << gpu_res[i][0] << " b: " << gpu_res[i][1]);
        REQUIRE(float(cpu_res[i][0]) == float(gpu_res[i][0]));
        REQUIRE(float(cpu_res[i][1]) == float(gpu_res[i][1]));
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e4m3x2_fp16x2_scale") {
    constexpr size_t size = 448 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_fp16x2_storage_t> in;
      in.reserve(size);
      for (int i = -448, j = 448; i <= 448; i++, j--) {
        __amd_fp16x2_storage_t tmp{static_cast<__amd_fp16_storage_t>(i),
                                   static_cast<__amd_fp16_storage_t>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_fp16x2_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16x2_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16x2_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_fp16x2_storage_t) * size, hipMemcpyHostToDevice));
      fp8x2_fp16x2_device_cvt_scale<<<1, size>>>(d_in, d_out, interpret, scale, size);
      // CPU calc
      std::vector<__amd_fp16x2_storage_t> cpu_res(size, 0.0f);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_fp16x2_to_fp8x2_scale(in[i], interpret, scale);
        cpu_res[i] = __amd_cvt_fp8x2_to_fp16x2_scale(tmp, interpret, scale);
      }
      std::vector<__amd_fp16x2_storage_t> gpu_res(size, 0.0f);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_fp16x2_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        INFO("Scale: " << int(scale) << " Index: " << i << "\n\tin:  a: " << float(in[i][0])
                       << " b: " << float(in[i][1]) << "\n\tcpu: a: " << float(cpu_res[i][0])
                       << " b: " << float(cpu_res[i][1]) << "\n\tgpu: a: " << float(gpu_res[i][0])
                       << " b: " << float(gpu_res[i][1]));
        REQUIRE(float(cpu_res[i][0]) == float(gpu_res[i][0]));
        REQUIRE(float(cpu_res[i][1]) == float(gpu_res[i][1]));
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e4m3x2_fp16x2") {
    constexpr size_t size = 448 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_fp16x2_storage_t> in;
    in.reserve(size);
    for (int i = -448, j = 448; i <= 448; i++, j--) {
      __amd_fp16x2_storage_t tmp{static_cast<__amd_fp16_storage_t>(i),
                                 static_cast<__amd_fp16_storage_t>(j)};
      in.push_back(tmp);
    }
    REQUIRE(in.size() == size);
    __amd_fp16x2_storage_t *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16x2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16x2_storage_t) * size));
    HIP_CHECK(
        hipMemcpy(d_in, in.data(), sizeof(__amd_fp16x2_storage_t) * size, hipMemcpyHostToDevice));
    fp8x2_fp16x2_device_cvt<<<1, size>>>(d_in, d_out, interpret, size);
    // CPU calc
    std::vector<__amd_fp16x2_storage_t> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      auto tmp = __amd_cvt_fp16x2_to_fp8x2(in[i], interpret);
      cpu_res[i] = __amd_cvt_fp8x2_to_fp16x2(tmp, interpret);
    }
    std::vector<__amd_fp16x2_storage_t> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_fp16x2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << "\n\tin:  a: " << float(in[i][0]) << " b: " << float(in[i][1])
                     << "\n\tcpu: a: " << float(cpu_res[i][0]) << " b: " << float(cpu_res[i][1])
                     << "\n\tgpu: a: " << float(gpu_res[i][0]) << " b: " << float(gpu_res[i][1]));
      REQUIRE(float(cpu_res[i][0]) == float(gpu_res[i][0]));
      REQUIRE(float(cpu_res[i][1]) == float(gpu_res[i][1]));
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e4m3x2_bf16x2_scale") {
    constexpr size_t size = 448 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_bf16x2_storage_t> in;
      in.reserve(size);
      for (int i = -448, j = 448; i <= 448; i++, j--) {
        __amd_bf16x2_storage_t tmp{static_cast<__amd_bf16_storage_t>(i),
                                   static_cast<__amd_bf16_storage_t>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_bf16x2_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_bf16x2_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_bf16x2_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_bf16x2_storage_t) * size, hipMemcpyHostToDevice));
      fp8x2_bf16x2_device_cvt_scale<<<1, size>>>(d_in, d_out, interpret, scale, size);
      // CPU calc
      std::vector<__amd_bf16x2_storage_t> cpu_res(size, 0.0f);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_bf16x2_to_fp8x2_scale(in[i], interpret, scale);
        cpu_res[i] = __amd_cvt_fp8x2_to_bf16x2_scale(tmp, interpret, scale);
      }
      std::vector<__amd_bf16x2_storage_t> gpu_res(size, 0.0f);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_bf16x2_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        INFO("Scale: " << scale << " Index: " << i << "\n\tin:  a: " << float(in[i][0])
                       << " b: " << float(in[i][1]) << "\n\tcpu: a: " << float(cpu_res[i][0])
                       << " b: " << float(cpu_res[i][1]) << "\n\tgpu: a: " << float(gpu_res[i][0])
                       << " b: " << float(gpu_res[i][1]));
        REQUIRE(float(cpu_res[i][0]) == float(gpu_res[i][0]));
        REQUIRE(float(cpu_res[i][1]) == float(gpu_res[i][1]));
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e4m3_bf16_sr_scale") {
    constexpr size_t size = 448 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_bf16_storage_t> in;
      in.reserve(size);
      for (int i = -448; i <= 448; i++) {
        in.push_back(static_cast<__amd_bf16_storage_t>(i));
      }
      REQUIRE(in.size() == size);
      __amd_bf16_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_bf16_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_bf16_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_bf16_storage_t) * size, hipMemcpyHostToDevice));
      fp8_bf16_device_cvt_sr_scale<<<1, size>>>(d_in, d_out, interpret, 0, scale, size);
      // CPU calc
      std::vector<__amd_bf16_storage_t> cpu_res(size, 0.0f);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_bf16_to_fp8_sr_scale(in[i], interpret, 0, scale);
        cpu_res[i] = __amd_cvt_fp8_to_bf16_scale(tmp, interpret, scale);
      }
      std::vector<__amd_bf16_storage_t> gpu_res(size, 0.0f);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_bf16_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        INFO("Scale: " << scale << " Index: " << i << "\n\tin:  a: " << float(in[i])
                       << "\n\tcpu: a: " << float(cpu_res[i])
                       << "\n\tgpu: a: " << float(gpu_res[i]));
        REQUIRE(float(cpu_res[i]) == float(gpu_res[i]));
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e5m2x2_scale") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_floatx2_storage_t> in;
      in.reserve(size);
      for (int i = -511, j = 511; i <= 511; i++, j--) {
        __amd_floatx2_storage_t tmp{static_cast<float>(i), static_cast<float>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_floatx2_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx2_storage_t) * size));
      HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(__amd_floatx2_storage_t) * size,
                          hipMemcpyHostToDevice));
      fp8x2_device_cvt_scale<<<1, size>>>(d_in, d_out, interpret, scale, size);
      // CPU calc
      std::vector<__amd_floatx2_storage_t> cpu_res(size, 0.0f);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_floatx2_to_fp8x2_scale(in[i], interpret, scale);
        cpu_res[i] = __amd_cvt_fp8x2_to_floatx2_scale(tmp, interpret, scale);
      }
      std::vector<__amd_floatx2_storage_t> gpu_res(size, 0.0f);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_floatx2_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        INFO("Scale: " << scale << " Index: " << i << "\n\tin:  a: " << in[i][0] << " b: "
                       << in[i][1] << "\n\tcpu: a: " << cpu_res[i][0] << " b: " << cpu_res[i][1]
                       << "\n\tgpu: a: " << gpu_res[i][0] << " b: " << gpu_res[i][1]);
        REQUIRE(float(cpu_res[i][0]) == float(gpu_res[i][0]));
        REQUIRE(float(cpu_res[i][1]) == float(gpu_res[i][1]));
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e5m2x2_fp16x2_scale") {
    constexpr size_t size = 400 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_fp16x2_storage_t> in;
      in.reserve(size);
      for (int i = -400, j = 400; i <= 400; i++, j--) {
        __amd_fp16x2_storage_t tmp{static_cast<__amd_fp16_storage_t>(i),
                                   static_cast<__amd_fp16_storage_t>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_fp16x2_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16x2_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16x2_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_fp16x2_storage_t) * size, hipMemcpyHostToDevice));
      fp8x2_fp16x2_device_cvt_scale<<<1, size>>>(d_in, d_out, interpret, scale, size);
      // CPU calc
      std::vector<__amd_fp16x2_storage_t> cpu_res(size, 0.0f);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_fp16x2_to_fp8x2_scale(in[i], interpret, scale);
        cpu_res[i] = __amd_cvt_fp8x2_to_fp16x2_scale(tmp, interpret, scale);
      }
      std::vector<__amd_fp16x2_storage_t> gpu_res(size, 0.0f);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_fp16x2_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        INFO("Scale: " << scale << " Index: " << i << "\n\tin:  a: " << float(in[i][0])
                       << " b: " << float(in[i][1]) << "\n\tcpu: a: " << float(cpu_res[i][0])
                       << " b: " << float(cpu_res[i][1]) << "\n\tgpu: a: " << float(gpu_res[i][0])
                       << " b: " << float(gpu_res[i][1]));
        REQUIRE(float(cpu_res[i][0]) == float(gpu_res[i][0]));
        REQUIRE(float(cpu_res[i][1]) == float(gpu_res[i][1]));
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e5m2x2_bf16x2_scale") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_bf16x2_storage_t> in;
      in.reserve(size);
      for (int i = -511, j = 511; i <= 511; i++, j--) {
        __amd_bf16x2_storage_t tmp{static_cast<__amd_bf16_storage_t>(i),
                                   static_cast<__amd_bf16_storage_t>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_bf16x2_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_bf16x2_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_bf16x2_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_bf16x2_storage_t) * size, hipMemcpyHostToDevice));
      fp8x2_bf16x2_device_cvt_scale<<<1, size>>>(d_in, d_out, interpret, scale, size);
      // CPU calc
      std::vector<__amd_bf16x2_storage_t> cpu_res(size, 0.0f);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_bf16x2_to_fp8x2_scale(in[i], interpret, scale);
        cpu_res[i] = __amd_cvt_fp8x2_to_bf16x2_scale(tmp, interpret, scale);
      }
      std::vector<__amd_bf16x2_storage_t> gpu_res(size, 0.0f);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_bf16x2_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        INFO("Scale: " << scale << " Index: " << i << "\n\tin:  a: " << float(in[i][0])
                       << " b: " << float(in[i][1]) << "\n\tcpu: a: " << float(cpu_res[i][0])
                       << " b: " << float(cpu_res[i][1]) << "\n\tgpu: a: " << float(gpu_res[i][0])
                       << " b: " << float(gpu_res[i][1]));
        REQUIRE(float(cpu_res[i][0]) == float(gpu_res[i][0]));
        REQUIRE(float(cpu_res[i][1]) == float(gpu_res[i][1]));
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

// To enable these tests we need to pass -mavx512f since we pass floatx8 as return types
#if __AVX512F__
  SECTION("e4m3x8_floatx8_scale") {
    constexpr size_t size = 448 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_floatx8_storage_t> in;
      in.reserve(size);
      for (int i = -448, j = 448; i <= 448; i++, j--) {
        __amd_floatx8_storage_t tmp{static_cast<float>(i), static_cast<float>(j),
                                    static_cast<float>(j), static_cast<float>(i),
                                    static_cast<float>(i), static_cast<float>(i),
                                    static_cast<float>(j), static_cast<float>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_floatx8_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx8_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx8_storage_t) * size));
      HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(__amd_floatx8_storage_t) * size,
                          hipMemcpyHostToDevice));
      fp8x8_floatx8_device_cvt_scale<<<1, size>>>(d_in, d_out, interpret, scale, size);
      // CPU calc
      std::vector<__amd_floatx8_storage_t> cpu_res(size);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_floatx8_to_fp8x8_scale(in[i], interpret, scale);
        cpu_res[i] = __amd_cvt_fp8x8_to_floatx8_scale(tmp, interpret, scale);
      }
      std::vector<__amd_floatx8_storage_t> gpu_res(size);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_floatx8_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 8; j++) {
          INFO("Scale: " << int(scale) << "Index: i: " << i << " subindex : " << j
                         << "In: " << float(in[i][j]) << " cpu res: " << float(cpu_res[i][j])
                         << " gpu res: " << float(gpu_res[i][j]));
          REQUIRE(float(cpu_res[i][j]) == float(gpu_res[i][j]));
        }
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e5m2x8_floatx8_scale") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_floatx8_storage_t> in;
      in.reserve(size);
      for (int i = -511, j = 511; i <= 511; i++, j--) {
        __amd_floatx8_storage_t tmp{static_cast<float>(i), static_cast<float>(j),
                                    static_cast<float>(j), static_cast<float>(i),
                                    static_cast<float>(i), static_cast<float>(i),
                                    static_cast<float>(j), static_cast<float>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_floatx8_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx8_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx8_storage_t) * size));
      HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(__amd_floatx8_storage_t) * size,
                          hipMemcpyHostToDevice));
      fp8x8_floatx8_device_cvt_scale<<<1, size>>>(d_in, d_out, interpret, scale, size);
      // CPU calc
      std::vector<__amd_floatx8_storage_t> cpu_res(size);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_floatx8_to_fp8x8_scale(in[i], interpret, scale);
        cpu_res[i] = __amd_cvt_fp8x8_to_floatx8_scale(tmp, interpret, scale);
      }
      std::vector<__amd_floatx8_storage_t> gpu_res(size);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_floatx8_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 8; j++) {
          INFO("Scale: " << int(scale) << "Index: i: " << i << " subindex : " << j
                         << "In: " << float(in[i][j]) << " cpu res: " << float(cpu_res[i][j])
                         << " gpu res: " << float(gpu_res[i][j]));
          REQUIRE(float(cpu_res[i][j]) == float(gpu_res[i][j]));
        }
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e4m3x8_floatx8_sr_scale") {
    constexpr size_t size = 448 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_floatx8_storage_t> in;
      in.reserve(size);
      for (int i = -448, j = 448; i <= 448; i++, j--) {
        __amd_floatx8_storage_t tmp{static_cast<float>(i), static_cast<float>(j),
                                    static_cast<float>(j), static_cast<float>(i),
                                    static_cast<float>(i), static_cast<float>(i),
                                    static_cast<float>(j), static_cast<float>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_floatx8_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx8_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx8_storage_t) * size));
      HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(__amd_floatx8_storage_t) * size,
                          hipMemcpyHostToDevice));
      fp8x8_floatx8_device_cvt_sr_scale<<<1, size>>>(d_in, d_out, interpret, 0 /* round */, scale,
                                                     size);
      // CPU calc
      std::vector<__amd_floatx8_storage_t> cpu_res(size);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_floatx8_to_fp8x8_sr_scale(in[i], interpret, 0, scale);
        cpu_res[i] = __amd_cvt_fp8x8_to_floatx8_scale(tmp, interpret, scale);
      }
      std::vector<__amd_floatx8_storage_t> gpu_res(size);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_floatx8_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 8; j++) {
          INFO("Scale: " << int(scale) << " Index: " << i << " subindex : " << j
                         << " In: " << float(in[i][j]) << " cpu res: " << float(cpu_res[i][j])
                         << " gpu res: " << float(gpu_res[i][j]));
          CHECK(cpu_res[i][j] == gpu_res[i][j]);
        }
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e5m2x8_floatx8_sr_scale") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_floatx8_storage_t> in;
      in.reserve(size);
      for (int i = -511, j = 511; i <= 511; i++, j--) {
        __amd_floatx8_storage_t tmp{static_cast<float>(i), static_cast<float>(j),
                                    static_cast<float>(j), static_cast<float>(i),
                                    static_cast<float>(i), static_cast<float>(i),
                                    static_cast<float>(j), static_cast<float>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_floatx8_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx8_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx8_storage_t) * size));
      HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(__amd_floatx8_storage_t) * size,
                          hipMemcpyHostToDevice));
      fp8x8_floatx8_device_cvt_sr_scale<<<1, size>>>(d_in, d_out, interpret, 0 /* round*/, scale,
                                                     size);
      // CPU calc
      std::vector<__amd_floatx8_storage_t> cpu_res(size);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_floatx8_to_fp8x8_sr_scale(in[i], interpret, 0, scale);
        cpu_res[i] = __amd_cvt_fp8x8_to_floatx8_scale(tmp, interpret, scale);
      }
      std::vector<__amd_floatx8_storage_t> gpu_res(size);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_floatx8_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 8; j++) {
          INFO("Scale: " << int(scale) << " Index: " << i << " subindex : " << j
                         << " In: " << float(in[i][j]) << " cpu res: " << float(cpu_res[i][j])
                         << " gpu res: " << float(gpu_res[i][j]));
          CHECK(cpu_res[i][j] == gpu_res[i][j]);
        }
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }
#endif

  SECTION("e4m3x8_bf16x8_scale") {
    constexpr size_t size = 400 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_bf16x8_storage_t> in;
      in.reserve(size);
      for (int i = -400, j = 400; i <= 400; i++, j--) {
        __amd_bf16x8_storage_t tmp{
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j),
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j),
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j),
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_bf16x8_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_bf16x8_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_bf16x8_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_bf16x8_storage_t) * size, hipMemcpyHostToDevice));
      fp8x8_bf16x8_device_cvt_scale<<<1, size>>>(d_in, d_out, interpret, scale, size);
      // CPU calc
      std::vector<__amd_bf16x8_storage_t> cpu_res(size);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_bf16x8_to_fp8x8_scale(in[i], interpret, scale);
        cpu_res[i] = __amd_cvt_fp8x8_to_bf16x8_scale(tmp, interpret, scale);
      }
      std::vector<__amd_bf16x8_storage_t> gpu_res(size);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_bf16x8_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 8; j++) {
          INFO("Scale: " << int(scale) << " Index: i: " << i << " subindex : " << j
                         << " In: " << float(in[i][j]) << " cpu res: " << float(cpu_res[i][j])
                         << " gpu res: " << float(gpu_res[i][j]));
          REQUIRE(float(cpu_res[i][j]) == float(gpu_res[i][j]));
        }
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e5m2x8_bf16x8_scale") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_bf16x8_storage_t> in;
      in.reserve(size);
      for (int i = -511, j = 511; i <= 511; i++, j--) {
        __amd_bf16x8_storage_t tmp{
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j),
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j),
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j),
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_bf16x8_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_bf16x8_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_bf16x8_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_bf16x8_storage_t) * size, hipMemcpyHostToDevice));
      fp8x8_bf16x8_device_cvt_scale<<<1, size>>>(d_in, d_out, interpret, scale, size);
      // CPU calc
      std::vector<__amd_bf16x8_storage_t> cpu_res(size);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_bf16x8_to_fp8x8_scale(in[i], interpret, scale);
        cpu_res[i] = __amd_cvt_fp8x8_to_bf16x8_scale(tmp, interpret, scale);
      }
      std::vector<__amd_bf16x8_storage_t> gpu_res(size);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_bf16x8_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 8; j++) {
          INFO("Scale: " << int(scale) << "Index: i: " << i << " subindex : " << j
                         << "In: " << float(in[i][j]) << " cpu res: " << float(cpu_res[i][j])
                         << " gpu res: " << float(gpu_res[i][j]));
          REQUIRE(float(cpu_res[i][j]) == float(gpu_res[i][j]));
        }
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e4m3x8_fp16x8_scale") {
    constexpr size_t size = 448 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_fp16x8_storage_t> in;
      in.reserve(size);
      for (int i = -448, j = 448; i <= 448; i++, j--) {
        __amd_fp16x8_storage_t tmp{
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j),
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j),
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j),
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_fp16x8_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16x8_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16x8_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_fp16x8_storage_t) * size, hipMemcpyHostToDevice));
      fp8x8_fp16x8_device_cvt_scale<<<1, size>>>(d_in, d_out, interpret, scale, size);
      // CPU calc
      std::vector<__amd_fp16x8_storage_t> cpu_res(size);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_fp16x8_to_fp8x8_scale(in[i], interpret, scale);
        cpu_res[i] = __amd_cvt_fp8x8_to_fp16x8_scale(tmp, interpret, scale);
      }
      std::vector<__amd_fp16x8_storage_t> gpu_res(size);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_fp16x8_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 8; j++) {
          INFO("Scale: " << int(scale) << "Index: i: " << i << " subindex : " << j
                         << "In: " << float(in[i][j]) << " cpu res: " << float(cpu_res[i][j])
                         << " gpu res: " << float(gpu_res[i][j]));
          REQUIRE(float(cpu_res[i][j]) == float(gpu_res[i][j]));
        }
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e5m2x8_fp16x8_scale") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_fp16x8_storage_t> in;
      in.reserve(size);
      for (int i = -511, j = 511; i <= 511; i++, j--) {
        __amd_fp16x8_storage_t tmp{
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j),
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j),
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j),
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_fp16x8_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16x8_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16x8_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_fp16x8_storage_t) * size, hipMemcpyHostToDevice));
      fp8x8_fp16x8_device_cvt_scale<<<1, size>>>(d_in, d_out, interpret, scale, size);
      // CPU calc
      std::vector<__amd_fp16x8_storage_t> cpu_res(size);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_fp16x8_to_fp8x8_scale(in[i], interpret, scale);
        cpu_res[i] = __amd_cvt_fp8x8_to_fp16x8_scale(tmp, interpret, scale);
      }
      std::vector<__amd_fp16x8_storage_t> gpu_res(size);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_fp16x8_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 8; j++) {
          INFO("Scale: " << int(scale) << "Index: i: " << i << " subindex : " << j
                         << "In: " << float(in[i][j]) << " cpu res: " << float(cpu_res[i][j])
                         << " gpu res: " << float(gpu_res[i][j]));
          REQUIRE(float(cpu_res[i][j]) == float(gpu_res[i][j]));
        }
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e4m3x8_fp16x8_sr_scale") {
    constexpr size_t size = 448 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_fp16x8_storage_t> in;
      in.reserve(size);
      for (int i = -448, j = 448; i <= 448; i++, j--) {
        __amd_fp16x8_storage_t tmp{
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j),
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j),
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j),
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_fp16x8_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16x8_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16x8_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_fp16x8_storage_t) * size, hipMemcpyHostToDevice));
      fp8x8_fp16x8_device_cvt_sr_scale<<<1, size>>>(d_in, d_out, interpret, 0, scale, size);
      // CPU calc
      std::vector<__amd_fp16x8_storage_t> cpu_res(size);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_fp16x8_to_fp8x8_sr_scale(in[i], interpret, 0, scale);
        cpu_res[i] = __amd_cvt_fp8x8_to_fp16x8_scale(tmp, interpret, scale);
      }
      std::vector<__amd_fp16x8_storage_t> gpu_res(size);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_fp16x8_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 8; j++) {
          INFO("Scale: " << int(scale) << "Index: i: " << i << " subindex : " << j
                         << "In: " << float(in[i][j]) << " cpu res: " << float(cpu_res[i][j])
                         << " gpu res: " << float(gpu_res[i][j]));
          REQUIRE(float(cpu_res[i][j]) == float(gpu_res[i][j]));
        }
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e5m2x8_fp16x8_sr_scale") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_fp16x8_storage_t> in;
      in.reserve(size);
      for (int i = -511, j = 511; i <= 511; i++, j--) {
        __amd_fp16x8_storage_t tmp{
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j),
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j),
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j),
            static_cast<__amd_fp16_storage_t>(i), static_cast<__amd_fp16_storage_t>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_fp16x8_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16x8_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16x8_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_fp16x8_storage_t) * size, hipMemcpyHostToDevice));
      fp8x8_fp16x8_device_cvt_sr_scale<<<1, size>>>(d_in, d_out, interpret, 0, scale, size);
      // CPU calc
      std::vector<__amd_fp16x8_storage_t> cpu_res(size);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_fp16x8_to_fp8x8_sr_scale(in[i], interpret, 0, scale);
        cpu_res[i] = __amd_cvt_fp8x8_to_fp16x8_scale(tmp, interpret, scale);
      }
      std::vector<__amd_fp16x8_storage_t> gpu_res(size);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_fp16x8_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 8; j++) {
          INFO("Scale: " << int(scale) << "Index: i: " << i << " subindex : " << j
                         << "In: " << float(in[i][j]) << " cpu res: " << float(cpu_res[i][j])
                         << " gpu res: " << float(gpu_res[i][j]));
          REQUIRE(float(cpu_res[i][j]) == float(gpu_res[i][j]));
        }
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e4m3x8_bf16x8_sr_scale") {
    constexpr size_t size = 448 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_bf16x8_storage_t> in;
      in.reserve(size);
      for (int i = -448, j = 448; i <= 448; i++, j--) {
        __amd_bf16x8_storage_t tmp{
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j),
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j),
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j),
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_bf16x8_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_bf16x8_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_bf16x8_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_bf16x8_storage_t) * size, hipMemcpyHostToDevice));
      fp8x8_bf16x8_device_cvt_sr_scale<<<1, size>>>(d_in, d_out, interpret, 0, scale, size);
      // CPU calc
      std::vector<__amd_bf16x8_storage_t> cpu_res(size);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_bf16x8_to_fp8x8_sr_scale(in[i], interpret, 0, scale);
        cpu_res[i] = __amd_cvt_fp8x8_to_bf16x8_scale(tmp, interpret, scale);
      }
      std::vector<__amd_bf16x8_storage_t> gpu_res(size);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_bf16x8_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 8; j++) {
          INFO("Scale: " << int(scale) << "Index: i: " << i << " subindex : " << j
                         << "In: " << float(in[i][j]) << " cpu res: " << float(cpu_res[i][j])
                         << " gpu res: " << float(gpu_res[i][j]));
          REQUIRE(float(cpu_res[i][j]) == float(gpu_res[i][j]));
        }
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e5m2x8_bf16x8_sr_scale") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      std::vector<__amd_bf16x8_storage_t> in;
      in.reserve(size);
      for (int i = -511, j = 511; i <= 511; i++, j--) {
        __amd_bf16x8_storage_t tmp{
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j),
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j),
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j),
            static_cast<__amd_bf16_storage_t>(i), static_cast<__amd_bf16_storage_t>(j)};
        in.push_back(tmp);
      }
      REQUIRE(in.size() == size);
      __amd_bf16x8_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_bf16x8_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_bf16x8_storage_t) * size));
      HIP_CHECK(
          hipMemcpy(d_in, in.data(), sizeof(__amd_bf16x8_storage_t) * size, hipMemcpyHostToDevice));
      fp8x8_bf16x8_device_cvt_sr_scale<<<1, size>>>(d_in, d_out, interpret, 0, scale, size);
      // CPU calc
      std::vector<__amd_bf16x8_storage_t> cpu_res(size);
      for (size_t i = 0; i < size; i++) {
        auto tmp = __amd_cvt_bf16x8_to_fp8x8_sr_scale(in[i], interpret, 0, scale);
        cpu_res[i] = __amd_cvt_fp8x8_to_bf16x8_scale(tmp, interpret, scale);
      }
      std::vector<__amd_bf16x8_storage_t> gpu_res(size);
      HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_bf16x8_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 8; j++) {
          INFO("Scale: " << int(scale) << "Index: i: " << i << " subindex : " << j
                         << "In: " << float(in[i][j]) << " cpu res: " << float(cpu_res[i][j])
                         << " gpu res: " << float(gpu_res[i][j]));
          REQUIRE(float(cpu_res[i][j]) == float(gpu_res[i][j]));
        }
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }

  SECTION("e4m3-sr") {
    constexpr size_t size = 449 * 2 + 1;
    constexpr __amd_scale_t scale = 2;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<float> in;
    in.reserve(size);
    for (int i = -449; i <= 449; i++) {
      in.push_back(static_cast<float>(i));
    }
    REQUIRE(in.size() == size);
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));
    fp8_sr_scale_device_cvt<<<1, size>>>(d_in, d_out, interpret, size, scale);
    // CPU calc
    std::vector<float> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      auto tmp = __amd_cvt_float_to_fp8_sr_scale(in[i], interpret, 1 /*seed*/, scale);
      cpu_res[i] = __amd_cvt_fp8_to_float_scale(tmp, interpret, scale);
    }
    std::vector<float> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " in: " << in[i] << " cpu: " << cpu_res[i] << " gpu: " << gpu_res[i]);
      REQUIRE(cpu_res[i] == gpu_res[i]);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e5m2-sr") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_scale_t scale = 2;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<float> in;
    in.reserve(size);
    for (int i = -511; i <= 511; i++) {
      in.push_back(static_cast<float>(i));
    }
    REQUIRE(in.size() == size);
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));
    fp8_sr_scale_device_cvt<<<1, size>>>(d_in, d_out, interpret, size, scale);
    // CPU calc
    std::vector<float> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      auto tmp = __amd_cvt_float_to_fp8_sr_scale(in[i], interpret, 1 /*seed*/, scale);
      cpu_res[i] = __amd_cvt_fp8_to_float_scale(tmp, interpret, scale);
    }
    std::vector<float> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " in: " << in[i] << " cpu: " << cpu_res[i] << " gpu: " << gpu_res[i]);
      REQUIRE(cpu_res[i] == gpu_res[i]);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e4m3-fp16-sr-scale") {
    constexpr size_t size = 449 * 2 + 1;
    constexpr __amd_scale_t scale = 2;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_fp16_storage_t> in;
    in.reserve(size);
    for (int i = -449; i <= 449; i++) {
      in.push_back(static_cast<__amd_fp16_storage_t>(i));
    }
    REQUIRE(in.size() == size);
    __amd_fp16_storage_t *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16_storage_t) * size));
    HIP_CHECK(
        hipMemcpy(d_in, in.data(), sizeof(__amd_fp16_storage_t) * size, hipMemcpyHostToDevice));
    fp8_fp16_sr_scale_device_cvt<<<1, size>>>(d_in, d_out, interpret, size, scale);
    // CPU calc
    std::vector<__amd_fp16_storage_t> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      auto tmp = __amd_cvt_fp16_to_fp8_sr_scale(in[i], interpret, 0 /*seed*/, scale);
      cpu_res[i] = __amd_cvt_fp8_to_fp16_scale(tmp, interpret, scale);
    }
    std::vector<__amd_fp16_storage_t> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_fp16_storage_t) * size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " in: " << float(in[i]) << " cpu: " << float(cpu_res[i])
                     << " gpu: " << float(gpu_res[i]));
      REQUIRE(float(cpu_res[i]) == float(gpu_res[i]));
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e5m2-fp16-sr-scale") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_scale_t scale = 2;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_fp16_storage_t> in;
    in.reserve(size);
    for (int i = -511; i <= 511; i++) {
      in.push_back(static_cast<__amd_fp16_storage_t>(i));
    }
    REQUIRE(in.size() == size);
    __amd_fp16_storage_t *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16_storage_t) * size));
    HIP_CHECK(
        hipMemcpy(d_in, in.data(), sizeof(__amd_fp16_storage_t) * size, hipMemcpyHostToDevice));
    fp8_fp16_sr_scale_device_cvt<<<1, size>>>(d_in, d_out, interpret, size, scale);
    // CPU calc
    std::vector<__amd_fp16_storage_t> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      auto tmp = __amd_cvt_fp16_to_fp8_sr_scale(in[i], interpret, 0 /*seed*/, scale);
      cpu_res[i] = __amd_cvt_fp8_to_fp16_scale(tmp, interpret, scale);
    }
    std::vector<__amd_fp16_storage_t> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_fp16_storage_t) * size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " in: " << float(in[i]) << " cpu: " << float(cpu_res[i])
                     << " gpu: " << float(gpu_res[i]));
      REQUIRE(float(cpu_res[i]) == float(gpu_res[i]));
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e4m3-fp16-sr") {
    constexpr size_t size = 449 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E4M3;
    std::vector<__amd_fp16_storage_t> in;
    in.reserve(size);
    for (int i = -449; i <= 449; i++) {
      in.push_back(static_cast<__amd_fp16_storage_t>(i));
    }
    REQUIRE(in.size() == size);
    __amd_fp16_storage_t *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16_storage_t) * size));
    HIP_CHECK(
        hipMemcpy(d_in, in.data(), sizeof(__amd_fp16_storage_t) * size, hipMemcpyHostToDevice));
    fp8_fp16_sr_device_cvt<<<1, size>>>(d_in, d_out, interpret, size);
    // CPU calc
    std::vector<__amd_fp16_storage_t> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      auto tmp = __amd_cvt_fp16_to_fp8_sr(in[i], interpret, 0 /*seed*/);
      cpu_res[i] = __amd_cvt_fp8_to_fp16(tmp, interpret);
    }
    std::vector<__amd_fp16_storage_t> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_fp16_storage_t) * size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " in: " << float(in[i]) << " cpu: " << float(cpu_res[i])
                     << " gpu: " << float(gpu_res[i]));
      REQUIRE(float(cpu_res[i]) == float(gpu_res[i]));
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e5m2-fp16-sr") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr __amd_fp8_interpretation_t interpret = __AMD_OCP_E5M2;
    std::vector<__amd_fp16_storage_t> in;
    in.reserve(size);
    for (int i = -511; i <= 511; i++) {
      in.push_back(static_cast<__amd_fp16_storage_t>(i));
    }
    REQUIRE(in.size() == size);
    __amd_fp16_storage_t *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16_storage_t) * size));
    HIP_CHECK(
        hipMemcpy(d_in, in.data(), sizeof(__amd_fp16_storage_t) * size, hipMemcpyHostToDevice));
    fp8_fp16_sr_device_cvt<<<1, size>>>(d_in, d_out, interpret, size);
    // CPU calc
    std::vector<__amd_fp16_storage_t> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      auto tmp = __amd_cvt_fp16_to_fp8_sr(in[i], interpret, 0 /*seed*/);
      cpu_res[i] = __amd_cvt_fp8_to_fp16(tmp, interpret);
    }
    std::vector<__amd_fp16_storage_t> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_fp16_storage_t) * size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " in: " << float(in[i]) << " cpu: " << float(cpu_res[i])
                     << " gpu: " << float(gpu_res[i]));
      REQUIRE(float(cpu_res[i]) == float(gpu_res[i]));
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

#if __AVX512F__
  SECTION("fp6x32 cvt e2m3") {
    constexpr __amd_fp6_interpretation_t interpret = __AMD_OCP_E2M3;
    constexpr unsigned int seed = 1;
    __amd_scale_t scale = 0;
    __amd_floatx32_storage_t in, *d_in, *d_out, gpu_out, cpu_out;
    float counter = -7.5f;
    for (size_t i = 0; i < 31; i++, counter += 0.5f) {
      in[i] = counter;
    }
    in[31] = -0.0f;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx32_storage_t)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx32_storage_t)));
    HIP_CHECK(hipMemcpy(d_in, &in, sizeof(__amd_floatx32_storage_t), hipMemcpyHostToDevice));
    fp6x32_convert<<<1, 32>>>(d_in, d_out, interpret, seed, scale);
    HIP_CHECK(hipMemcpy(&gpu_out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    auto tmp = __amd_cvt_floatx32_to_fp6x32_sr_scale(in, interpret, seed, scale);
    cpu_out = __amd_cvt_fp6x32_to_floatx32_scale(tmp, interpret, scale);
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    for (size_t i = 0; i < 32; i++) {
      INFO("Index: " << i << " In: " << in[i] << " cpu: " << cpu_out[i] << " gpu: " << gpu_out[i]);
      CHECK(cpu_out[i] == gpu_out[i]);
    }
  }

  SECTION("fp6x32 cvt e3m2") {
    constexpr __amd_fp6_interpretation_t interpret = __AMD_OCP_E3M2;
    constexpr unsigned int seed = 1;
    __amd_scale_t scale = 1;
    __amd_floatx32_storage_t in, *d_in, *d_out, gpu_out, cpu_out;
    float counter = -28.0f;
    for (size_t i = 0; i < 32; i++, counter += 1.0f) {
      in[i] = counter;
    }
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx32_storage_t)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx32_storage_t)));
    HIP_CHECK(hipMemcpy(d_in, &in, sizeof(__amd_floatx32_storage_t), hipMemcpyHostToDevice));
    fp6x32_convert<<<1, 32>>>(d_in, d_out, interpret, seed, scale);
    HIP_CHECK(hipMemcpy(&gpu_out, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
    auto tmp = __amd_cvt_floatx32_to_fp6x32_sr_scale(in, interpret, seed, scale);
    cpu_out = __amd_cvt_fp6x32_to_floatx32_scale(tmp, interpret, scale);
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    for (size_t i = 0; i < 32; i++) {
      INFO("Index: " << i << " In: " << in[i] << " cpu: " << cpu_out[i] << " gpu: " << gpu_out[i]);
      CHECK(cpu_out[i] == gpu_out[i]);
    }
  }
#endif

  SECTION("fp4x2 sr") {
    constexpr __amd_fp4_interpretation_t interpret = __AMD_OCP_E2M1;
    std::vector<__amd_scale_t> scales{0, 1, 2};
    std::vector<__amd_floatx2_storage_t> in_vals = {
        __amd_floatx2_storage_t{-3.0f, 3.0f}, __amd_floatx2_storage_t{-2.5f, 2.5f},
        __amd_floatx2_storage_t{-2.0f, 2.0f}, __amd_floatx2_storage_t{-1.5f, 1.5f},
        __amd_floatx2_storage_t{-1.0f, 1.0f}, __amd_floatx2_storage_t{-0.5f, 0.5f},
        __amd_floatx2_storage_t{-0.0f, 0.0f}, __amd_floatx2_storage_t{0.5f, -0.5f},
        __amd_floatx2_storage_t{1.0f, -1.0f}, __amd_floatx2_storage_t{1.5f, -1.5f},
        __amd_floatx2_storage_t{2.0f, -2.0f}, __amd_floatx2_storage_t{2.5f, -2.5f},
        __amd_floatx2_storage_t{3.0f, -3.0f}};
    const size_t size = in_vals.size();
    for (const auto scale : scales) {
      __amd_floatx2_storage_t *d_in, *d_out;
      HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t) * size));
      HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx2_storage_t) * size));
      HIP_CHECK(hipMemcpy(d_in, in_vals.data(), sizeof(__amd_floatx2_storage_t) * size,
                          hipMemcpyHostToDevice));
      fp4x2_sr_scale_convert<<<1, 32>>>(d_in, d_out, 1 /*seed*/, scale, size);
      std::vector<__amd_floatx2_storage_t> gpu_out(size);
      HIP_CHECK(hipMemcpy(gpu_out.data(), d_out, sizeof(__amd_floatx2_storage_t) * size,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < size; i++) {
        auto gpu_res = gpu_out[i];
        auto cpu_fp4_temp = __amd_cvt_floatx2_to_fp4x2_sr_scale(in_vals[i], interpret, 1, scale);
        auto cpu_res = __amd_cvt_fp4x2_to_floatx2_scale(cpu_fp4_temp, interpret, scale);
        INFO("Scale: " << scale << " Input l: " << in_vals[i][0] << " r: " << in_vals[i][1]
                       << "\n  cpu l: " << cpu_res[0] << " r: " << cpu_res[1]
                       << "\n  gpu l: " << gpu_res[0] << " r: " << gpu_res[1]);
        CHECK(cpu_res[0] == gpu_res[0]);
        CHECK(cpu_res[1] == gpu_res[1]);
      }
      HIP_CHECK(hipFree(d_in));
      HIP_CHECK(hipFree(d_out));
    }
  }
}
