/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "misc_negative_kernels_rtc.hh"

#include "unary_common.hh"
#include "binary_common.hh"
#include "ternary_common.hh"

MATH_UNARY_WITHIN_ULP_TEST_DEF(fabs, std::fabs, 0, 0)
HIP_TEST_CASE(Unit_Device_fabs_fabsf_Negative_RTC) { NegativeTestRTCWrapper<4>(kFabs); }

MATH_BINARY_WITHIN_ULP_TEST_DEF(copysign, std::copysign, 0, 0)
HIP_TEST_CASE(Unit_Device_copysign_copysignf_Negative_RTC) { NegativeTestRTCWrapper<8>(kCopySign); }

MATH_BINARY_WITHIN_ULP_TEST_DEF(fmax, std::fmax, 0, 0)
HIP_TEST_CASE(Unit_Device_fmax_fmaxf_Negative_RTC) { NegativeTestRTCWrapper<8>(kFmax); }

MATH_BINARY_WITHIN_ULP_TEST_DEF(fmin, std::fmin, 0, 0)
HIP_TEST_CASE(Unit_Device_fmin_fminf_Negative_RTC) { NegativeTestRTCWrapper<8>(kFmin); }

MATH_BINARY_WITHIN_ULP_TEST_DEF(nextafter, std::nextafter, 1, 1)
HIP_TEST_CASE(Unit_Device_nextafter_nextafterf_Negative_RTC) {
  NegativeTestRTCWrapper<8>(kNextAfter);
}

MATH_TERNARY_WITHIN_ULP_TEST_DEF(fma, std::fma, 0, 1)
HIP_TEST_CASE(Unit_Device_fma_fmaf_Negative_RTC) { NegativeTestRTCWrapper<12>(kFma); }

__global__ void fdividef_kernel(float* const ys, const size_t num_xs, float* const x1s,
                                float* const x2s) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (size_t i = tid; i < num_xs; i += stride) {
    ys[i] = fdividef(x1s[i], x2s[i]);
  }
}

HIP_TEST_CASE(Unit_Device_fdividef_Accuracy_Positive) {
  double (*ref)(double, double) = [](double x1, double x2) { return x1 / x2; };
  BinaryFloatingPointTest(fdividef_kernel, ref, ULPValidatorBuilderFactory<float>(0));
}

HIP_TEST_CASE(Unit_Device_fdividef_Negative_RTC) { NegativeTestRTCWrapper<4>(kFdividef); }

#define MATH_BOOL_RETURNING_FUNCTION_TEST_DEF(kern_name, ref_func)                                 \
  template <typename T>                                                                            \
  __global__ void kern_name##_kernel(bool* const ys, const size_t num_xs, T* const xs) {           \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (size_t i = tid; i < num_xs; i += stride) {                                                \
      ys[i] = kern_name(xs[i]);                                                                    \
    }                                                                                              \
  }                                                                                                \
                                                                                                   \
  HIP_TEST_CASE(Unit_Device_##kern_name##_Accuracy_Positive_float) {                               \
    bool (*ref)(double) = ref_func;                                                                \
    UnarySinglePrecisionTest(kern_name##_kernel<float>, ref, EqValidatorBuilderFactory<bool>());   \
  }                                                                                                \
                                                                                                   \
  HIP_TEST_CASE(Unit_Device_##kern_name##_Accuracy_Positive_double) {                              \
    bool (*ref)(long double) = ref_func;                                                           \
    UnaryDoublePrecisionTest(kern_name##_kernel<double>, ref, EqValidatorBuilderFactory<bool>());  \
  }

MATH_BOOL_RETURNING_FUNCTION_TEST_DEF(isfinite, std::isfinite)
HIP_TEST_CASE(Unit_Device_isfinite_Negative_RTC) { NegativeTestRTCWrapper<4>(kIsFinite); }

MATH_BOOL_RETURNING_FUNCTION_TEST_DEF(isinf, std::isinf)
HIP_TEST_CASE(Unit_Device_isinf_Negative_RTC) { NegativeTestRTCWrapper<4>(kIsInf); }

MATH_BOOL_RETURNING_FUNCTION_TEST_DEF(isnan, std::isnan)
HIP_TEST_CASE(Unit_Device_isnan_Negative_RTC) { NegativeTestRTCWrapper<4>(kIsNan); }

MATH_BOOL_RETURNING_FUNCTION_TEST_DEF(signbit, std::signbit)
HIP_TEST_CASE(Unit_Device_signbit_Negative_RTC) { NegativeTestRTCWrapper<4>(kSignBit); }
