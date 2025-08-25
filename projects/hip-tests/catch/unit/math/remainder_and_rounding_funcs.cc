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

#include "unary_common.hh"
#include "binary_common.hh"
#include "math_remainder_rounding_negative_kernels_rtc.hh"

MATH_BINARY_WITHIN_ULP_TEST_DEF(fmod, std::fmod, 0, 0)
TEST_CASE("Unit_Device_fmod_fmodf_Negative_RTC") { NegativeTestRTCWrapper<8>(kFmod); }

MATH_BINARY_WITHIN_ULP_TEST_DEF(remainder, std::remainder, 0, 0)
TEST_CASE("Unit_Device_remainder_remainder_Negative_RTC") { NegativeTestRTCWrapper<8>(kRemainder); }

MATH_BINARY_WITHIN_ULP_TEST_DEF(fdim, std::fdim, 0, 0)
TEST_CASE("Unit_Device_fdim_fdimf_Negative_RTC") { NegativeTestRTCWrapper<8>(kFdim); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(trunc, std::trunc, 0, 0)
TEST_CASE("Unit_Device_trunc_truncf_Negative_RTC") { NegativeTestRTCWrapper<4>(kTrunc); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(round, std::round, 0, 0)
TEST_CASE("Unit_Device_round_roundf_Negative_RTC") { NegativeTestRTCWrapper<4>(kRound); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(rint, std::rint, 0, 0)
TEST_CASE("Unit_Device_rint_rintf_Negative_RTC") { NegativeTestRTCWrapper<4>(kRint); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(nearbyint, std::nearbyint, 0, 0)
TEST_CASE("Unit_Device_nearbyint_nearbyintf_Negative_RTC") {
  NegativeTestRTCWrapper<4>(kNearbyint);
}

MATH_UNARY_WITHIN_ULP_TEST_DEF(ceil, std::ceil, 0, 0)
TEST_CASE("Unit_Device_ceil_ceilf_Negative_RTC") { NegativeTestRTCWrapper<4>(kCeil); }

MATH_UNARY_WITHIN_ULP_TEST_DEF(floor, std::floor, 0, 0)
TEST_CASE("Unit_Device_floor_floorf_Negative_RTC") { NegativeTestRTCWrapper<4>(kFloor); }


#define LONG_CONVERSION_FUNCTION_TEST_DEF(kern_name, ref_func, lt)                                 \
  MATH_UNARY_KERNEL_DEF(kern_name)                                                                 \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive - float") {                              \
    lt (*ref)(double) = ref_func;                                                                  \
    UnarySinglePrecisionRangeTest(kern_name##_kernel<float, lt>, ref,                              \
                                  EqValidatorBuilderFactory<lt>(),                                 \
                                  static_cast<float>(std::numeric_limits<lt>::lowest()),           \
                                  static_cast<float>(std::numeric_limits<lt>::max()));             \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive - double") {                             \
    lt (*ref)(long double) = ref_func;                                                             \
    UnaryDoublePrecisionBruteForceTest(kern_name##_kernel<double, lt>, ref,                        \
                                       EqValidatorBuilderFactory<lt>(),                            \
                                       static_cast<double>(std::numeric_limits<lt>::lowest()),     \
                                       static_cast<double>(std::numeric_limits<lt>::max()));       \
  }

LONG_CONVERSION_FUNCTION_TEST_DEF(lrint, std::lrint, long)
TEST_CASE("Unit_Device_lrint_lrintf_Negative_RTC") { NegativeTestRTCWrapper<4>(kLrint); }

LONG_CONVERSION_FUNCTION_TEST_DEF(lround, std::lround, long)
TEST_CASE("Unit_Device_lround_lroundf_Negative_RTC") { NegativeTestRTCWrapper<4>(kLround); }

LONG_CONVERSION_FUNCTION_TEST_DEF(llrint, std::llrint, long long)
TEST_CASE("Unit_Device_llrint_llrintf_Negative_RTC") { NegativeTestRTCWrapper<4>(kLlrint); }

LONG_CONVERSION_FUNCTION_TEST_DEF(llround, std::llround, long long)
TEST_CASE("Unit_Device_llround_llroundf_Negative_RTC") { NegativeTestRTCWrapper<4>(kLlround); }


template <typename T> __global__ void remquo_kernel(std::pair<T, int>* const ys,
                                                    const size_t num_xs, T* const x1s,
                                                    T* const x2s) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (auto i = tid; i < num_xs; i += stride) {
    if constexpr (std::is_same_v<float, T>) {
      ys[i].first = remquof(x1s[i], x2s[i], &ys[i].second);
    } else if constexpr (std::is_same_v<double, T>) {
      ys[i].first = remquo(x1s[i], x2s[i], &ys[i].second);
    }
  }
}

template <typename T> std::pair<T, int> remquo_wrapper(T x1, T x2) {
  std::pair<T, int> ret;
  ret.first = std::remquo(x1, x2, &ret.second);
  return ret;
}

TEMPLATE_TEST_CASE("Unit_Device_remquo_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  std::pair<RT, int> (*ref)(RT, RT) = remquo_wrapper;
  const auto ulp_builder = ULPValidatorBuilderFactory<TestType>(0);
  const auto eq_builder = EqValidatorBuilderFactory<int>();

  BinaryFloatingPointTest(remquo_kernel<TestType>, ref,
                          PairValidatorBuilderFactory<TestType, int>(ulp_builder, eq_builder));
}

TEST_CASE("Unit_Device_remquo_remquof_Negative_RTC") { NegativeTestRTCWrapper<24>(kRemquo); }

template <typename T>
__global__ void modf_kernel(std::pair<T, T>* const ys, const size_t num_xs, T* const xs) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (auto i = tid; i < num_xs; i += stride) {
    if constexpr (std::is_same_v<float, T>) {
      ys[i].first = modff(xs[i], &ys[i].second);
    } else if constexpr (std::is_same_v<double, T>) {
      ys[i].first = modf(xs[i], &ys[i].second);
    }
  }
}

template <typename T> std::pair<T, T> modf_wrapper(T x) {
  std::pair<T, T> ret;
  ret.first = std::modf(x, &ret.second);
  return ret;
}

TEST_CASE("Unit_Device_modf_Accuracy_Positive - float") {
  UnarySinglePrecisionTest(
      modf_kernel<float>, modf_wrapper<double>,
      PairValidatorBuilderFactory<float>(ULPValidatorBuilderFactory<float>(0)));
}

TEST_CASE("Unit_Device_modf_Accuracy_Positive - double") {
  UnaryDoublePrecisionTest(
      modf_kernel<double>, modf_wrapper<long double>,
      PairValidatorBuilderFactory<double>(ULPValidatorBuilderFactory<double>(0)));
}

TEST_CASE("Unit_Device_modf_modff_Negative_RTC") { NegativeTestRTCWrapper<20>(kModf); }
