/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_fp8.h>
#include <cmath>
#include <limits>
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
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(
      hipMemcpy(out.data(), out_d, sizeof(unsigned char) * out.size(), hipMemcpyDeviceToHost));
}

HIP_TEST_CASE(Unit__hip_cvt_bfloat16raw_to_e8m0) {
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
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(
      hipMemcpy(out.data(), out_d, sizeof(unsigned char) * out.size(), hipMemcpyDeviceToHost));
}

HIP_TEST_CASE(Unit__hip_cvt_float_to_e8m0) {
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
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(
      hipMemcpy(out.data(), out_d, sizeof(unsigned char) * out.size(), hipMemcpyDeviceToHost));
}


HIP_TEST_CASE(Unit__hip_cvt_double_to_e8m0) {
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
                            // tiny underflow example -> underflows to 0x00
                            // rounding-to-+Inf promotes to 0x01
                            1e-50,
                            -1e-50,
                            // smallest double-subnormal (denorm_min) -> underflows to 0x00;
                            // rounding-to-+Inf does not promote (magnitude below half-step)
                            std::numeric_limits<double>::denorm_min(),
                            -std::numeric_limits<double>::denorm_min(),
                            // smallest double-normal -> underflows to 0x00;
                            // rounding-to-+Inf does not promote (fraction == 0)
                            std::numeric_limits<double>::min(),
                            -std::numeric_limits<double>::min(),
                            // smallest float-normal (cast to double) -> E8M0 code 0x01;
                            // rounding-to-+Inf does not promote (fraction == 0)
                            static_cast<double>(std::numeric_limits<float>::min()),
                            -static_cast<double>(std::numeric_limits<float>::min()),
                            std::nan("1"),
                            -std::nan("1"),
                            std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity(),
                            1e50,
                            -1e50};
  std::vector<unsigned char> exp(in.size());
  std::vector<unsigned char> out(exp.size());

  if (rounding == hipRoundPosInf) {
    exp = {0x00U, 0x7EU, 0x7FU, 0x81U, 0x82U, 0x82U, 0x7EU, 0x7FU, 0x81U, 0x82U, 0x82U, 0xFE, 0xFE,
           0x01U, 0x01U, 0x00U, 0x00U, 0x00U, 0x00U, 0x01U, 0x01U};
  } else {
    exp = {0x00U, 0x7EU, 0x7EU, 0x81U, 0x81U, 0x82U, 0x7EU, 0x7EU, 0x81U, 0x81U, 0x82U, 0xFD, 0xFD,
           0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x01U, 0x01U};
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
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(out.data(), out_d, sizeof(float) * out.size(), hipMemcpyDeviceToHost));
}

HIP_TEST_CASE(Unit__hip_cvt_e8m0_to_bf16raw) {
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
      REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs(exp[i], 1e-6f) || Catch::Matchers::WithinRel(exp[i], 1e-3f));
    }
  }
}

template <typename T_in, typename T_out>
void host_e8m0_constructors(const std::vector<T_in>& in, std::vector<T_out>& out) {
  REQUIRE(in.size() == out.size());
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
  REQUIRE(in.size() <= 1024);

  HIP_CHECK(hipMalloc(&in_d, sizeof(T_in) * in.size()));
  HIP_CHECK(hipMalloc(&out_d, sizeof(T_out) * out.size()));

  HIP_CHECK(hipMemcpy(in_d, in.data(), sizeof(T_in) * in.size(), hipMemcpyHostToDevice));

  e8m0_constructors_kernel<T_in, T_out><<<1, 1024>>>(in_d, out_d, in.size());
  HIP_CHECK(hipGetLastError());

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
                     Catch::Matchers::WithinAbs(exp[i], (T)1e-6) || Catch::Matchers::WithinRel(exp[i], (T)1e-3));
      }
    } else {
      REQUIRE(out[i] == exp[i]);
    }
  }
}

HIP_TEMPLATE_TEST_CASE(Unit_e8m0_float_constructors, half, __hip_bfloat16, float, double) {
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

HIP_TEMPLATE_TEST_CASE(Unit_e8m0_int_constructors, int, long int, long long int, short int,
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
  REQUIRE(in.size() <= 1024);

  HIP_CHECK(hipMalloc(&in_d, sizeof(float) * in.size()));
  HIP_CHECK(hipMalloc(&out_d, sizeof(T_out) * out.size()));

  HIP_CHECK(hipMemcpy(in_d, in.data(), sizeof(float) * in.size(), hipMemcpyHostToDevice));

  e8m0_float_conversions_kernel<T_out><<<1, 1024>>>(in_d, out_d, in.size());
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(out.data(), out_d, sizeof(T_out) * out.size(), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(in_d));
  HIP_CHECK(hipFree(out_d));
}

HIP_TEMPLATE_TEST_CASE(Unit_e8m0_float_conversions, half, __hip_bfloat16, float, double) {
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

HIP_TEMPLATE_TEST_CASE(Unit_e8m0_int_conversions, char, int, long int, long long int, short int,
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
  REQUIRE(in.size() <= 1024);

  HIP_CHECK(hipMalloc(&in_d, sizeof(unsigned char) * in.size()));
  HIP_CHECK(hipMalloc(&out_d, sizeof(T_out) * out.size()));

  HIP_CHECK(hipMemcpy(in_d, in.data(), sizeof(unsigned char) * in.size(), hipMemcpyHostToDevice));

  e8m0_saturation_conversions_kernel<T_out><<<1, 1024>>>(in_d, out_d, in.size());
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(out.data(), out_d, sizeof(T_out) * out.size(), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(in_d));
  HIP_CHECK(hipFree(out_d));
}

// Exhaustively verify that the e8m0 -> double host conversion agrees with the
// e8m0 -> float host conversion (upcast to double) for every one of the 256
// encodings. The broader-than-pointwise coverage is intentional: a prior bug
// produced 2^-1023 instead of 2^-127 at encoding 0x00 because the float
// subnormal-mantissa trick was copy-pasted into the double path even though
// double can represent 2^-127 as a normal number. Sweeping every encoding
// against the float oracle catches that class of copy-paste drift wherever it
// reappears, not just at the originally reported point.
HIP_TEST_CASE(Unit__hip_cvt_e8m0_to_double_exhaustive_host) {
  for (unsigned i = 0; i < 256u; ++i) {
    unsigned char enc = static_cast<unsigned char>(i);
    __hip_fp8_e8m0 fp8;
    fp8.__x = enc;
    double d = static_cast<double>(fp8);

    if (enc == 0xFFu) {
      INFO("encoding: 0x" << std::hex << static_cast<unsigned>(enc));
      REQUIRE(std::isnan(d));
    } else {
      __hip_fp8_e8m0 fp8f;
      fp8f.__x = enc;
      float f = static_cast<float>(fp8f);
      double expected = static_cast<double>(f);
      INFO("encoding: 0x" << std::hex << static_cast<unsigned>(enc)
                          << " out:" << d << " exp:" << expected);
      REQUIRE(d == expected);
    }
  }
}

HIP_TEMPLATE_TEST_CASE(Unit_e8m0_int_conversions_saturation, char, int, long int, long long int,
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

// Boundary-encoding saturation test for 32/64-bit integer types.
//
// For these widths, (float)numeric_limits<T>::max() rounds UP to 2^N (e.g.
// (float)INT32_MAX rounds to 2^31). The e8m0 encoding (127 + N) decodes to
// exactly 2^N as well, so the saturation comparison `f > max()` evaluated to
// false at that boundary and the subsequent integer cast was UB per
// [conv.fpint]/1. The boundary encodings:
//   int32_t   -> enc 158 (= 127 + 31)
//   uint32_t  -> enc 159 (= 127 + 32)
//   int64_t   -> enc 190 (= 127 + 63)
//   uint64_t  -> enc 191 (= 127 + 64)
// Verify both the exact-boundary encoding and one step above it saturate to
// numeric_limits<T>::max().
HIP_TEMPLATE_TEST_CASE(Unit_e8m0_int_conversions_saturation_boundary, int, unsigned int,
                       long long int, unsigned long long int) {
  bool run_on_host = GENERATE(true, false);

  constexpr unsigned int bit_width = sizeof(TestType) * 8u;
  constexpr unsigned char boundary_enc = static_cast<unsigned char>(
      127u + bit_width - (std::is_signed_v<TestType> ? 1u : 0u));

  std::vector<unsigned char> in = {boundary_enc, static_cast<unsigned char>(boundary_enc + 1)};
  std::vector<TestType> out(in.size());

  if (run_on_host) {
    host_e8m0_saturation_conversions<TestType>(in, out);
  } else {
    device_e8m0_saturation_conversions<TestType>(in, out);
  }

  TestType expected_max = std::numeric_limits<TestType>::max();
  for (size_t i = 0; i < out.size(); ++i) {
    INFO("encoding:0x" << std::hex << static_cast<unsigned>(in[i]) << std::dec
                       << " out:" << out[i] << " expected_max:" << expected_max);
    REQUIRE(out[i] == expected_max);
  }
}
