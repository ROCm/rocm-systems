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

#include "trig_negative_kernels_rtc.hh"

#include "unary_common.hh"
#include "binary_common.hh"

#include <boost/math/special_functions.hpp>


MATH_UNARY_WITHIN_ULP_TEST_DEF(sin, std::sin, 2, 2);
TEST_CASE("Unit_Device_sin_sinf_Negative_RTC") { NegativeTestRTCWrapper<4>(kSin); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(cos, std::cos, 2, 2)
TEST_CASE("Unit_Device_cos_cosf_Negative_RTC") { NegativeTestRTCWrapper<4>(kCos); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(tan, std::tan, 4, 2)
TEST_CASE("Unit_Device_tan_tanf_Negative_RTC") { NegativeTestRTCWrapper<4>(kTan); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(asin, std::asin, 2, 2)
TEST_CASE("Unit_Device_asin_asinf_Negative_RTC") { NegativeTestRTCWrapper<4>(kAsin); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(acos, std::acos, 2, 2)
TEST_CASE("Unit_Device_acos_acosf_Negative_RTC") { NegativeTestRTCWrapper<4>(kAcos); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(atan, std::atan, 2, 2)
TEST_CASE("Unit_Device_atan_atanf_Negative_RTC") { NegativeTestRTCWrapper<4>(kAtan); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(sinh, std::sinh, 3, 2)
TEST_CASE("Unit_Device_sinh_sinhf_Negative_RTC") { NegativeTestRTCWrapper<4>(kSinh); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(cosh, std::cosh, 2, 1)
TEST_CASE("Unit_Device_cosh_coshf_Negative_RTC") { NegativeTestRTCWrapper<4>(kCosh); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(tanh, std::tanh, 2, 1)
TEST_CASE("Unit_Device_tanh_tanhf_Negative_RTC") { NegativeTestRTCWrapper<4>(kTanh); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(asinh, std::asinh, 3, 2)
TEST_CASE("Unit_Device_asinh_asinhf_Negative_RTC") { NegativeTestRTCWrapper<4>(kAsinh); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(acosh, std::acosh, 4, 2)
TEST_CASE("Unit_Device_acosh_acoshf_Negative_RTC") { NegativeTestRTCWrapper<4>(kAcosh); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(atanh, std::atanh, 3, 2)
TEST_CASE("Unit_Device_atanh_atanhf_Negative_RTC") { NegativeTestRTCWrapper<4>(kAtanh); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(sinpi, boost::math::sin_pi, 2, 2);
TEST_CASE("Unit_Device_sinpi_sinpif_Negative_RTC") { NegativeTestRTCWrapper<4>(kSinpi); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(cospi, boost::math::cos_pi, 2, 2);
TEST_CASE("Unit_Device_cospi_cospif_Negative_RTC") { NegativeTestRTCWrapper<4>(kCospi); }

MATH_BINARY_WITHIN_ULP_TEST_DEF(atan2, std::atan2, 3, 2);
TEST_CASE("Unit_Device_atan2_atan2f_Negative_RTC") { NegativeTestRTCWrapper<8>(kAtan2); }


template <typename T>
__global__ void sincos_kernel(std::pair<T, T>* const ys, const size_t num_xs, T* const xs) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (auto i = tid; i < num_xs; i += stride) {
    if constexpr (std::is_same_v<float, T>) {
      sincosf(xs[i], &ys[i].first, &ys[i].second);
    } else if constexpr (std::is_same_v<double, T>) {
      sincos(xs[i], &ys[i].first, &ys[i].second);
    }
  }
}

template <typename T> std::pair<T, T> sincos(T x) { return {std::sin(x), std::cos(x)}; }

TEST_CASE("Unit_Device_sincos_Accuracy_Positive - float") {
  UnarySinglePrecisionTest(
      sincos_kernel<float>, sincos<double>,
      PairValidatorBuilderFactory<float>(ULPValidatorBuilderFactory<float>(2)));
}

TEST_CASE("Unit_Device_sincos_Accuracy_Positive - double") {
  const auto validator_builder =
      PairValidatorBuilderFactory<double>(ULPValidatorBuilderFactory<double>(2));
  UnaryDoublePrecisionTest(sincos_kernel<double>, sincos<long double>, validator_builder);
}

TEST_CASE("Unit_Device_sincos_sincosf_Negative_RTC") { NegativeTestRTCWrapper<36>(kSincos); }


template <typename T>
__global__ void sincospi_kernel(std::pair<T, T>* const ys, const size_t num_xs, T* const xs) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (auto i = tid; i < num_xs; i += stride) {
    if constexpr (std::is_same_v<float, T>) {
      sincospif(xs[i], &ys[i].first, &ys[i].second);
    } else if constexpr (std::is_same_v<double, T>) {
      sincospi(xs[i], &ys[i].first, &ys[i].second);
    }
  }
}

template <typename T> std::pair<T, T> sincospi(T x) {
  return {boost::math::sin_pi(x), boost::math::cos_pi(x)};
}

TEST_CASE("Unit_Device_sincospi_Accuracy_Positive - float") {
  UnarySinglePrecisionTest(
      sincospi_kernel<float>, sincospi<double>,
      PairValidatorBuilderFactory<float>(ULPValidatorBuilderFactory<float>(2)));
}

TEST_CASE("Unit_Device_sincospi_Accuracy_Positive - double") {
  const auto validator_builder =
      PairValidatorBuilderFactory<double>(ULPValidatorBuilderFactory<double>(2));
  UnaryDoublePrecisionTest(sincospi_kernel<double>, sincospi<long double>, validator_builder);
}

TEST_CASE("Unit_Device_sincospi_sincospif_Negative_RTC") { NegativeTestRTCWrapper<36>(kSincospi); }