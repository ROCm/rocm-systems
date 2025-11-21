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

#include "misc_negative_kernels_rtc.hh"

#include "unary_common.hh"
#include "binary_common.hh"
#include "ternary_common.hh"

MATH_UNARY_WITHIN_ULP_TEST_DEF(fabs, std::fabs, 0, 0)
TEST_CASE("Unit_Device_fabs_fabsf_Negative_RTC") { NegativeTestRTCWrapper<4>(kFabs); }

MATH_BINARY_WITHIN_ULP_TEST_DEF(copysign, std::copysign, 0, 0)
TEST_CASE("Unit_Device_copysign_copysignf_Negative_RTC") { NegativeTestRTCWrapper<8>(kCopySign); }

MATH_BINARY_WITHIN_ULP_TEST_DEF(fmax, std::fmax, 0, 0)
TEST_CASE("Unit_Device_fmax_fmaxf_Negative_RTC") { NegativeTestRTCWrapper<8>(kFmax); }

MATH_BINARY_WITHIN_ULP_TEST_DEF(fmin, std::fmin, 0, 0)
TEST_CASE("Unit_Device_fmin_fminf_Negative_RTC") { NegativeTestRTCWrapper<8>(kFmin); }

MATH_BINARY_WITHIN_ULP_TEST_DEF(nextafter, std::nextafter, 0, 0)
TEST_CASE("Unit_Device_nextafter_nextafterf_Negative_RTC") {
  NegativeTestRTCWrapper<8>(kNextAfter);
}

MATH_TERNARY_WITHIN_ULP_TEST_DEF(fma, std::fma, 0, 0)
TEST_CASE("Unit_Device_fma_fmaf_Negative_RTC") { NegativeTestRTCWrapper<12>(kFma); }

__global__ void fdividef_kernel(float* const ys, const size_t num_xs, float* const x1s,
                                float* const x2s) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (auto i = tid; i < num_xs; i += stride) {
    ys[i] = fdividef(x1s[i], x2s[i]);
  }
}

TEST_CASE("Unit_Device_fdividef_Accuracy_Positive") {
  double (*ref)(double, double) = [](double x1, double x2) { return x1 / x2; };
  BinaryFloatingPointTest(fdividef_kernel, ref, ULPValidatorBuilderFactory<float>(0));
}

TEST_CASE("Unit_Device_fdividef_Negative_RTC") { NegativeTestRTCWrapper<4>(kFdividef); }

#define MATH_BOOL_RETURNING_FUNCTION_TEST_DEF(kern_name, ref_func)                                 \
  template <typename T>                                                                            \
  __global__ void kern_name##_kernel(bool* const ys, const size_t num_xs, T* const xs) {           \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = kern_name(xs[i]);                                                                    \
    }                                                                                              \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive - float") {                              \
    bool (*ref)(double) = ref_func;                                                                \
    UnarySinglePrecisionTest(kern_name##_kernel<float>, ref, EqValidatorBuilderFactory<bool>());   \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive - double") {                             \
    bool (*ref)(long double) = ref_func;                                                           \
    UnaryDoublePrecisionTest(kern_name##_kernel<double>, ref, EqValidatorBuilderFactory<bool>());  \
  }

MATH_BOOL_RETURNING_FUNCTION_TEST_DEF(isfinite, std::isfinite)
TEST_CASE("Unit_Device_isfinite_Negative_RTC") { NegativeTestRTCWrapper<4>(kIsFinite); }

MATH_BOOL_RETURNING_FUNCTION_TEST_DEF(isinf, std::isinf)
TEST_CASE("Unit_Device_isinf_Negative_RTC") { NegativeTestRTCWrapper<4>(kIsInf); }

MATH_BOOL_RETURNING_FUNCTION_TEST_DEF(isnan, std::isnan)
TEST_CASE("Unit_Device_isnan_Negative_RTC") { NegativeTestRTCWrapper<4>(kIsNan); }

MATH_BOOL_RETURNING_FUNCTION_TEST_DEF(signbit, std::signbit)
TEST_CASE("Unit_Device_signbit_Negative_RTC") { NegativeTestRTCWrapper<4>(kSignBit); }