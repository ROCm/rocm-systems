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

#include <hip_test_common.hh>
#include <hip/hip_fp8.h>
#include <cmath>
#include <type_traits>

void host_cvt_bfloat16raw_to_e8m0(const std::vector<__hip_bfloat16>& in,
                                  std::vector<unsigned char>& out, __hip_saturation_t sat,
                                  hipRoundMode round) {
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = __hip_cvt_bfloat16raw_to_e8m0(in[i], sat, round);
  }
}

__global__ void bfloat16raw_to_e8m0(const __hip_bfloat16* in, unsigned char* out, size_t size,
                                    __hip_saturation_t sat, hipRoundMode round) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < size) {
    out[tid] = __hip_cvt_bfloat16raw_to_e8m0(in[tid], sat, round);
  }
}
void device_cvt_bfloat16raw_to_e8m0(const std::vector<__hip_bfloat16>& in,
                                    std::vector<unsigned char>& out, __hip_saturation_t sat,
                                    hipRoundMode round) {
  __hip_bfloat16* in_d = nullptr;
  unsigned char* out_d = nullptr;
  REQUIRE(in.size() < 1024);

  HIP_CHECK(hipMalloc(&in_d, sizeof(__hip_bfloat16) * in.size()));
  HIP_CHECK(hipMalloc(&out_d, sizeof(unsigned char) * out.size()));

  HIP_CHECK(hipMemcpy(in_d, in.data(), sizeof(__hip_bfloat16) * in.size(), hipMemcpyHostToDevice));

  bfloat16raw_to_e8m0<<<1, 1024>>>(in_d, out_d, in.size(), sat, round);

  HIP_CHECK(
      hipMemcpy(out.data(), out_d, sizeof(unsigned char) * out.size(), hipMemcpyDeviceToHost));
}

TEST_CASE("Unit__hip_cvt_bfloat16raw_to_e8m0") {
  bool run_on_host = GENERATE(true, false);
  __hip_saturation_t saturation = GENERATE(__HIP_NOSAT, __HIP_SATFINITE);
  hipRoundMode rounding = GENERATE(hipRoundZero, hipRoundPosInf);

  std::vector<__hip_bfloat16> in = {0.0f,
                                    0.5f,
                                    0.6f,
                                    4,
                                    5,
                                    8,
                                    -0.5f,
                                    -0.6f,
                                    -4,
                                    -5,
                                    -8,
                                    1e38,
                                    -1e38,
                                    std::nanf("1"),
                                    -std::nanf("1"),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()};
  std::vector<unsigned char> exp(in.size());
  std::vector<unsigned char> out(exp.size());

  if (rounding == hipRoundPosInf) {
    exp = {0x00U, 0x7EU, 0x7FU, 0x81U, 0x82U, 0x82U, 0x7EU, 0x7FU, 0x81U, 0x82U, 0x82U, 0xFE, 0xFE};
  } else {
    exp = {0x00U, 0x7EU, 0x7EU, 0x81U, 0x81U, 0x82U, 0x7EU, 0x7EU, 0x81U, 0x81U, 0x82U, 0xFD, 0xFD};
  }
  if (saturation == __HIP_NOSAT) {
    std::vector<unsigned char> exp_append = {0xFF, 0xFF, 0xFF, 0xFF};
    exp.insert(exp.end(), exp_append.begin(), exp_append.end());
  } else {
    std::vector<unsigned char> exp_append = {0xFF, 0xFF, 0xFE, 0xFE};
    exp.insert(exp.end(), exp_append.begin(), exp_append.end());
  }

  REQUIRE(exp.size() == in.size());

  if (run_on_host) {
    host_cvt_bfloat16raw_to_e8m0(in, out, saturation, rounding);
  } else {
    device_cvt_bfloat16raw_to_e8m0(in, out, saturation, rounding);
  }

  for (size_t i = 0; i < in.size(); ++i) {
    INFO("out:" << out[i] << " exp:" << exp[i] << " for index:" << i);
    REQUIRE(out[i] == exp[i]);
  }
}

////

void host_cvt_float_to_e8m0(const std::vector<float>& in, std::vector<unsigned char>& out,
                            __hip_saturation_t sat, hipRoundMode round) {
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = __hip_cvt_float_to_e8m0(in[i], sat, round);
  }
}

__global__ void float_to_e8m0_kernel(const float* in, unsigned char* out, size_t size,
                                     __hip_saturation_t sat, hipRoundMode round) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < size) {
    out[tid] = __hip_cvt_float_to_e8m0(in[tid], sat, round);
  }
}
void device_cvt_float_to_e8m0(const std::vector<float>& in, std::vector<unsigned char>& out,
                              __hip_saturation_t sat, hipRoundMode round) {
  float* in_d = nullptr;
  unsigned char* out_d = nullptr;
  REQUIRE(in.size() < 1024);

  HIP_CHECK(hipMalloc(&in_d, sizeof(float) * in.size()));
  HIP_CHECK(hipMalloc(&out_d, sizeof(unsigned char) * out.size()));

  HIP_CHECK(hipMemcpy(in_d, in.data(), sizeof(float) * in.size(), hipMemcpyHostToDevice));

  float_to_e8m0_kernel<<<1, 1024>>>(in_d, out_d, in.size(), sat, round);

  HIP_CHECK(
      hipMemcpy(out.data(), out_d, sizeof(unsigned char) * out.size(), hipMemcpyDeviceToHost));
}

TEST_CASE("Unit__hip_cvt_float_to_e8m0") {
  bool run_on_host = GENERATE(true, false);
  __hip_saturation_t saturation = GENERATE(__HIP_NOSAT, __HIP_SATFINITE);
  hipRoundMode rounding = GENERATE(hipRoundZero, hipRoundPosInf);

  std::vector<float> in = {0.0f,
                           0.5f,
                           0.6f,
                           4.0f,
                           5.0f,
                           8.0f,
                           -0.5f,
                           -0.6f,
                           -4.0f,
                           -5.0f,
                           -8.0f,
                           1e38,
                           -1e38,
                           std::nanf("1"),
                           -std::nanf("1"),
                           std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity()};
  std::vector<unsigned char> exp(in.size());
  std::vector<unsigned char> out(exp.size());

  if (rounding == hipRoundPosInf) {
    exp = {0x00U, 0x7EU, 0x7FU, 0x81U, 0x82U, 0x82U, 0x7EU, 0x7FU, 0x81U, 0x82U, 0x82U, 0xFE, 0xFE};
  } else {
    exp = {0x00U, 0x7EU, 0x7EU, 0x81U, 0x81U, 0x82U, 0x7EU, 0x7EU, 0x81U, 0x81U, 0x82U, 0xFD, 0xFD};
  }
  if (saturation == __HIP_NOSAT) {
    std::vector<unsigned char> exp_append = {0xFF, 0xFF, 0xFF, 0xFF};
    exp.insert(exp.end(), exp_append.begin(), exp_append.end());
  } else {
    std::vector<unsigned char> exp_append = {0xFF, 0xFF, 0xFE, 0xFE};
    exp.insert(exp.end(), exp_append.begin(), exp_append.end());
  }

  REQUIRE(exp.size() == in.size());

  if (run_on_host) {
    host_cvt_float_to_e8m0(in, out, saturation, rounding);
  } else {
    device_cvt_float_to_e8m0(in, out, saturation, rounding);
  }

  for (size_t i = 0; i < in.size(); ++i) {
    INFO("out:" << out[i] << " exp:" << exp[i] << " for index:" << i);
    REQUIRE(out[i] == exp[i]);
  }
}

////

void host_cvt_double_to_e8m0(const std::vector<double>& in, std::vector<unsigned char>& out,
                             __hip_saturation_t sat, hipRoundMode round) {
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = __hip_cvt_double_to_e8m0(in[i], sat, round);
  }
}

__global__ void double_to_e8m0_kernel(const double* in, unsigned char* out, size_t size,
                                      __hip_saturation_t sat, hipRoundMode round) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < size) {
    out[tid] = __hip_cvt_double_to_e8m0(in[tid], sat, round);
  }
}
void device_cvt_double_to_e8m0(const std::vector<double>& in, std::vector<unsigned char>& out,
                               __hip_saturation_t sat, hipRoundMode round) {
  double* in_d = nullptr;
  unsigned char* out_d = nullptr;
  REQUIRE(in.size() < 1024);

  HIP_CHECK(hipMalloc(&in_d, sizeof(double) * in.size()));
  HIP_CHECK(hipMalloc(&out_d, sizeof(unsigned char) * out.size()));

  HIP_CHECK(hipMemcpy(in_d, in.data(), sizeof(double) * in.size(), hipMemcpyHostToDevice));

  double_to_e8m0_kernel<<<1, 1024>>>(in_d, out_d, in.size(), sat, round);

  HIP_CHECK(
      hipMemcpy(out.data(), out_d, sizeof(unsigned char) * out.size(), hipMemcpyDeviceToHost));
}


TEST_CASE("Unit__hip_cvt_double_to_e8m0") {
  bool run_on_host = GENERATE(true, false);
  __hip_saturation_t saturation = GENERATE(__HIP_NOSAT, __HIP_SATFINITE);
  hipRoundMode rounding = GENERATE(hipRoundZero, hipRoundPosInf);

  std::vector<double> in = {0.0,
                            0.5,
                            0.6,
                            4.0,
                            5.0,
                            8.0,
                            -0.5,
                            -0.6,
                            -4.0,
                            -5.0,
                            -8.0,
                            1e38,
                            -1e38,
                            std::nan("1"),
                            -std::nan("1"),
                            std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity(),
                            1e50,
                            -1e50};
  std::vector<unsigned char> exp(in.size());
  std::vector<unsigned char> out(exp.size());

  if (rounding == hipRoundPosInf) {
    exp = {0x00U, 0x7EU, 0x7FU, 0x81U, 0x82U, 0x82U, 0x7EU, 0x7FU, 0x81U, 0x82U, 0x82U, 0xFE, 0xFE};
  } else {
    exp = {0x00U, 0x7EU, 0x7EU, 0x81U, 0x81U, 0x82U, 0x7EU, 0x7EU, 0x81U, 0x81U, 0x82U, 0xFD, 0xFD};
  }
  if (saturation == __HIP_NOSAT) {
    std::vector<unsigned char> exp_append = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    exp.insert(exp.end(), exp_append.begin(), exp_append.end());
  } else {
    std::vector<unsigned char> exp_append = {0xFF, 0xFF, 0xFE, 0xFE, 0xFE, 0xFE};
    exp.insert(exp.end(), exp_append.begin(), exp_append.end());
  }

  REQUIRE(exp.size() == in.size());

  if (run_on_host) {
    host_cvt_double_to_e8m0(in, out, saturation, rounding);
  } else {
    device_cvt_double_to_e8m0(in, out, saturation, rounding);
  }

  for (size_t i = 0; i < in.size(); ++i) {
    INFO("out:" << out[i] << " exp:" << exp[i] << " for index:" << i);
    REQUIRE(out[i] == exp[i]);
  }
}

////

void host_cvt_e8m0_to_bf16raw(const std::vector<unsigned char>& in, std::vector<float>& out) {
  for (size_t i = 0; i < in.size(); ++i) {
    __hip_bfloat16 temp = __hip_cvt_e8m0_to_bf16raw(in[i]);
    out[i] = static_cast<float>(temp);
  }
}

__global__ void e8m0_to_bf16raw_kernel(const unsigned char* in, float* out, size_t size) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < size) {
    __hip_bfloat16 temp = __hip_cvt_e8m0_to_bf16raw(in[tid]);
    out[tid] = static_cast<float>(temp);
  }
}
void device_cvt_e8m0_to_bf16raw(const std::vector<unsigned char>& in, std::vector<float>& out) {
  unsigned char* in_d = nullptr;
  float* out_d = nullptr;
  REQUIRE(in.size() < 1024);

  HIP_CHECK(hipMalloc(&in_d, sizeof(unsigned char) * in.size()));
  HIP_CHECK(hipMalloc(&out_d, sizeof(float) * out.size()));

  HIP_CHECK(hipMemcpy(in_d, in.data(), sizeof(unsigned char) * in.size(), hipMemcpyHostToDevice));

  e8m0_to_bf16raw_kernel<<<1, 1024>>>(in_d, out_d, in.size());

  HIP_CHECK(hipMemcpy(out.data(), out_d, sizeof(float) * out.size(), hipMemcpyDeviceToHost));
}

TEST_CASE("Unit__hip_cvt_e8m0_to_bf16raw") {
  bool run_on_host = GENERATE(true, false);

  std::vector<unsigned char> in = {0x00u, 0x7EU, 0x81U, 0x82U, 0xFF};
  std::vector<float> exp = {0.0f, 0.5f, 4.0f, 8.0f, std::nanf("0")};
  std::vector<float> out(exp.size());

  REQUIRE(exp.size() == in.size());

  if (run_on_host) {
    host_cvt_e8m0_to_bf16raw(in, out);
  } else {
    device_cvt_e8m0_to_bf16raw(in, out);
  }

  for (size_t i = 0; i < in.size(); ++i) {
    INFO("out:" << out[i] << " exp:" << exp[i] << " for index:" << i);
    if (std::isnan(exp[i])) {
      REQUIRE(std::isnan(out[i]));
    } else {
      REQUIRE_THAT(out[i], Catch::WithinAbs(exp[i], 1e-6f) || Catch::WithinRel(exp[i], 1e-3f));
    }
  }
}

template <typename T_in, typename T_out>
void host_e8m0_constructors(const std::vector<T_in>& in, std::vector<T_out>& out) {
  for (size_t i = 0; i < in.size(); ++i) {
    __hip_fp8_e8m0 fp8(in[i]);
    out[i] = static_cast<T_out>(fp8);
  }
}

template <typename T_in, typename T_out>
__global__ void e8m0_constructors_kernel(const T_in* in, T_out* out, size_t size) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < size) {
    __hip_fp8_e8m0 fp8(in[tid]);
    out[tid] = static_cast<T_out>(fp8);
  }
}

template <typename T_in, typename T_out>
void device_e8m0_constructors(const std::vector<T_in>& in, std::vector<T_out>& out) {
  T_in* in_d = nullptr;
  T_out* out_d = nullptr;
  REQUIRE(in.size() < 1024);

  HIP_CHECK(hipMalloc(&in_d, sizeof(T_in) * in.size()));
  HIP_CHECK(hipMalloc(&out_d, sizeof(T_out) * out.size()));

  HIP_CHECK(hipMemcpy(in_d, in.data(), sizeof(T_in) * in.size(), hipMemcpyHostToDevice));

  e8m0_constructors_kernel<T_in, T_out><<<1, 1024>>>(in_d, out_d, in.size());

  HIP_CHECK(hipMemcpy(out.data(), out_d, sizeof(T_out) * out.size(), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(in_d));
  HIP_CHECK(hipFree(out_d));
}

template <typename T>
void compare_e8m0_results(const std::vector<T>& out, const std::vector<T>& exp) {
  for (size_t i = 0; i < out.size(); ++i) {
    INFO("out:" << out[i] << " exp:" << exp[i] << " for index:" << i);
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(exp[i])) {
        REQUIRE(std::isnan(out[i]));
      } else {
        REQUIRE_THAT(out[i],
                     Catch::WithinAbs(exp[i], (T)1e-6) || Catch::WithinRel(exp[i], (T)1e-3));
      }
    } else {
      REQUIRE(out[i] == exp[i]);
    }
  }
}

TEMPLATE_TEST_CASE("Unit_e8m0_float_constructors", , half, __hip_bfloat16, float, double) {
  bool run_on_host = GENERATE(true, false);

  std::vector<TestType> in = {0, 10, 16, -10, -16, std::nanf("0")};
  std::array<float, 6> exp_values = {0, 16, 16, 16, 16, std::nanf("0")};

  if constexpr (std::is_same_v<TestType, half> || std::is_same_v<TestType, __hip_bfloat16>) {
    std::vector<float> exp(exp_values.begin(), exp_values.end());
    std::vector<float> out(exp.size());

    if (run_on_host) {
      host_e8m0_constructors<TestType, float>(in, out);
    } else {
      device_e8m0_constructors<TestType, float>(in, out);
    }
    compare_e8m0_results(out, exp);
  } else {
    std::vector<TestType> exp(exp_values.begin(), exp_values.end());
    std::vector<TestType> out(exp.size());

    if (run_on_host) {
      host_e8m0_constructors<TestType, TestType>(in, out);
    } else {
      device_e8m0_constructors<TestType, TestType>(in, out);
    }
    compare_e8m0_results(out, exp);
  }
}

TEMPLATE_TEST_CASE("Unit_e8m0_int_constructors", , int, long int, long long int, short int,
                   unsigned int, unsigned long int, unsigned long long int, unsigned short int) {
  bool run_on_host = GENERATE(true, false);

  std::vector<TestType> in = {
      static_cast<TestType>(0),  static_cast<TestType>(1),  static_cast<TestType>(2),
      static_cast<TestType>(4),  static_cast<TestType>(8),  static_cast<TestType>(15),
      static_cast<TestType>(16), static_cast<TestType>(32), static_cast<TestType>(64)};
  std::vector<TestType> exp = {
      static_cast<TestType>(0),  static_cast<TestType>(1),  static_cast<TestType>(2),
      static_cast<TestType>(4),  static_cast<TestType>(8),  static_cast<TestType>(16),
      static_cast<TestType>(16), static_cast<TestType>(32), static_cast<TestType>(64)};

  if constexpr (std::is_signed_v<TestType>) {
    in.insert(in.end(),
              {static_cast<TestType>(-1), static_cast<TestType>(-2), static_cast<TestType>(-4),
               static_cast<TestType>(-8), static_cast<TestType>(-15), static_cast<TestType>(-16),
               static_cast<TestType>(-32), static_cast<TestType>(-64)});
    exp.insert(exp.end(),
               {static_cast<TestType>(1), static_cast<TestType>(2), static_cast<TestType>(4),
                static_cast<TestType>(8), static_cast<TestType>(16), static_cast<TestType>(16),
                static_cast<TestType>(32), static_cast<TestType>(64)});
  }

  std::vector<TestType> out(exp.size());

  if (run_on_host) {
    host_e8m0_constructors<TestType, TestType>(in, out);
  } else {
    device_e8m0_constructors<TestType, TestType>(in, out);
  }
  compare_e8m0_results(out, exp);
}

template <typename T_out>
void host_e8m0_float_conversions(const std::vector<float>& in, std::vector<T_out>& out) {
  for (size_t i = 0; i < in.size(); ++i) {
    __hip_fp8_e8m0 fp8(in[i]);
    out[i] = static_cast<T_out>(fp8);
  }
}

template <typename T_out>
__global__ void e8m0_float_conversions_kernel(const float* in, T_out* out, size_t size) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < size) {
    __hip_fp8_e8m0 fp8(in[tid]);
    out[tid] = static_cast<T_out>(fp8);
  }
}

template <typename T_out>
void device_e8m0_float_conversions(const std::vector<float>& in, std::vector<T_out>& out) {
  float* in_d = nullptr;
  T_out* out_d = nullptr;
  REQUIRE(in.size() < 1024);

  HIP_CHECK(hipMalloc(&in_d, sizeof(float) * in.size()));
  HIP_CHECK(hipMalloc(&out_d, sizeof(T_out) * out.size()));

  HIP_CHECK(hipMemcpy(in_d, in.data(), sizeof(float) * in.size(), hipMemcpyHostToDevice));

  e8m0_float_conversions_kernel<T_out><<<1, 1024>>>(in_d, out_d, in.size());

  HIP_CHECK(hipMemcpy(out.data(), out_d, sizeof(T_out) * out.size(), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(in_d));
  HIP_CHECK(hipFree(out_d));
}

TEMPLATE_TEST_CASE("Unit_e8m0_float_conversions", , half, __hip_bfloat16, float, double) {
  bool run_on_host = GENERATE(true, false);

  std::vector<float> in = {0.0f, 10.0f, 16.0f, -10.0f, -16.0f, std::nanf("0")};
  std::array<float, 6> exp_values = {0.0f, 16.0f, 16.0f, 16.0f, 16.0f, std::nanf("0")};

  if constexpr (std::is_same_v<TestType, half> || std::is_same_v<TestType, __hip_bfloat16>) {
    std::vector<float> exp(exp_values.begin(), exp_values.end());
    std::vector<float> out(exp.size());

    if (run_on_host) {
      host_e8m0_float_conversions<float>(in, out);
    } else {
      device_e8m0_float_conversions<float>(in, out);
    }
    compare_e8m0_results(out, exp);
  } else {
    std::vector<TestType> exp(exp_values.begin(), exp_values.end());
    std::vector<TestType> out(exp.size());

    if (run_on_host) {
      host_e8m0_float_conversions<TestType>(in, out);
    } else {
      device_e8m0_float_conversions<TestType>(in, out);
    }
    compare_e8m0_results(out, exp);
  }
}

TEMPLATE_TEST_CASE("Unit_e8m0_int_conversions", , char, int, long int, long long int, short int,
                   signed char, unsigned char, unsigned int, unsigned long int,
                   unsigned long long int, unsigned short int) {
  bool run_on_host = GENERATE(true, false);

  std::vector<float> in = {0.0f, 1.0f, 2.0f, 4.0f, 8.0f, 15.0f, 16.0f, 32.0f, 64.0f};
  std::vector<TestType> exp = {
      static_cast<TestType>(0),  static_cast<TestType>(1),  static_cast<TestType>(2),
      static_cast<TestType>(4),  static_cast<TestType>(8),  static_cast<TestType>(16),
      static_cast<TestType>(16), static_cast<TestType>(32), static_cast<TestType>(64)};
  if constexpr (std::is_signed_v<TestType>) {
    // Test negative values if signed
    in.insert(in.end(), {-1.0f, -2.0f, -4.0f, -8.0f, -15.0f, -16.0f, -32.0f, -64.0f});
    exp.insert(exp.end(),
               {static_cast<TestType>(1), static_cast<TestType>(2), static_cast<TestType>(4),
                static_cast<TestType>(8), static_cast<TestType>(16), static_cast<TestType>(16),
                static_cast<TestType>(32), static_cast<TestType>(64)});
  }

  std::vector<TestType> out(exp.size());

  if (run_on_host) {
    host_e8m0_float_conversions<TestType>(in, out);
  } else {
    device_e8m0_float_conversions<TestType>(in, out);
  }
  compare_e8m0_results(out, exp);
}

template <typename T_out>
void host_e8m0_saturation_conversions(const std::vector<unsigned char>& in,
                                      std::vector<T_out>& out) {
  for (size_t i = 0; i < in.size(); ++i) {
    __hip_fp8_e8m0 fp8;
    fp8.__x = in[i];
    out[i] = static_cast<T_out>(fp8);
  }
}

template <typename T_out>
__global__ void e8m0_saturation_conversions_kernel(const unsigned char* in, T_out* out,
                                                   size_t size) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < size) {
    __hip_fp8_e8m0 fp8;
    fp8.__x = in[tid];
    out[tid] = static_cast<T_out>(fp8);
  }
}

template <typename T_out>
void device_e8m0_saturation_conversions(const std::vector<unsigned char>& in,
                                        std::vector<T_out>& out) {
  unsigned char* in_d = nullptr;
  T_out* out_d = nullptr;
  REQUIRE(in.size() < 1024);

  HIP_CHECK(hipMalloc(&in_d, sizeof(unsigned char) * in.size()));
  HIP_CHECK(hipMalloc(&out_d, sizeof(T_out) * out.size()));

  HIP_CHECK(hipMemcpy(in_d, in.data(), sizeof(unsigned char) * in.size(), hipMemcpyHostToDevice));

  e8m0_saturation_conversions_kernel<T_out><<<1, 1024>>>(in_d, out_d, in.size());

  HIP_CHECK(hipMemcpy(out.data(), out_d, sizeof(T_out) * out.size(), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(in_d));
  HIP_CHECK(hipFree(out_d));
}

TEMPLATE_TEST_CASE("Unit_e8m0_int_conversions_saturation", , char, int, long int, long long int,
                   short int, signed char, unsigned char, unsigned int, unsigned long int,
                   unsigned long long int, unsigned short int) {
  bool run_on_host = GENERATE(true, false);

  // Test with maximum non-NAN e8m0 value
  std::vector<unsigned char> in = {0xFE};
  std::vector<TestType> out(in.size());

  if (run_on_host) {
    host_e8m0_saturation_conversions<TestType>(in, out);
  } else {
    device_e8m0_saturation_conversions<TestType>(in, out);
  }

  // Verify that the output is clamped to the maximum value of TestType
  TestType expected_max = std::numeric_limits<TestType>::max();

  for (size_t i = 0; i < out.size(); ++i) {
    INFO("out:" << out[i] << " expected_max:" << expected_max << " for index:" << i);
    REQUIRE(out[i] == expected_max);
  }
}
