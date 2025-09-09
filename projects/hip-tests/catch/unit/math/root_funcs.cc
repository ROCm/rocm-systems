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
#include "ternary_common.hh"
#include "quaternary_common.hh"
#include "math_root_negative_kernels_rtc.hh"

/**
 * @addtogroup RootMathFuncs RootMathFuncs
 * @{
 * @ingroup MathTest
 */

/********** Unary Functions **********/

MATH_UNARY_KERNEL_DEF(sqrt)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `sqrtf(x)` for all possible inputs. The results are
 * compared against reference function `float std::exp(float)`. The maximum ulp error is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_sqrtf_Accuracy_Positive") {
  float (*ref)(float) = std::sqrt;
  UnarySinglePrecisionTest(sqrt_kernel<float>, ref, ULPValidatorBuilderFactory<float>(1));
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `sqrt(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are
 * compared against reference function `double std::sqrt(double)`. The error bounds are
 * IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_sqrt_Accuracy_Positive") {
  double (*ref)(double) = std::sqrt;
  UnaryDoublePrecisionTest<double>(sqrt_kernel<double>, ref, ULPValidatorBuilderFactory<double>(0));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for sqrtf and sqrt.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_sqrt_sqrtf_Negative_RTC") { NegativeTestRTCWrapper<4>(kSqrt); }

MATH_UNARY_KERNEL_DEF(rsqrt)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `rsqrtf(x)` for all possible inputs. The maximum ulp error
 * is 2.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_rsqrtf_Accuracy_Positive") {
  auto rsqrt_ref = [](double arg) -> double { return 1. / std::sqrt(arg); };
  double (*ref)(double) = rsqrt_ref;
  UnarySinglePrecisionTest(rsqrt_kernel<float>, ref, ULPValidatorBuilderFactory<float>(2));
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `rsqrt(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The maximum ulp error is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_rsqrt_Accuracy_Positive") {
  auto rsqrt_ref = [](long double arg) -> long double { return 1.L / std::sqrt(arg); };
  long double (*ref)(long double) = rsqrt_ref;
  UnaryDoublePrecisionTest(rsqrt_kernel<double>, ref, ULPValidatorBuilderFactory<double>(1));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for rsqrtf and rsqrt.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_rsqrt_rsqrtf_Negative_RTC") { NegativeTestRTCWrapper<4>(kRsqrt); }

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `cbrtf(x)` for all possible inputs and `cbrt(x)` against a
 * table of difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::cbrt(T)`. The maximum ulp error is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_WITHIN_ULP_TEST_DEF(cbrt, std::cbrt, 1, 1)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for cbrtf and cbrt.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_cbrt_cbrtf_Negative_RTC") { NegativeTestRTCWrapper<4>(kCbrt); }

MATH_UNARY_KERNEL_DEF(rcbrt)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `rcbrtf(x)` for all possible inputs. The maximum ulp error
 * is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_rcbrtf_Accuracy_Positive") {
  auto rcbrt_ref = [](double arg) -> double { return 1. / std::cbrt(arg); };
  double (*ref)(double) = rcbrt_ref;
  UnarySinglePrecisionTest(rcbrt_kernel<float>, ref, ULPValidatorBuilderFactory<float>(1));
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `rcbrt(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The maximum ulp error is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_rcbrt_Accuracy_Positive") {
  auto rcbrt_ref = [](long double arg) -> long double { return 1. / std::cbrt(arg); };
  long double (*ref)(long double) = rcbrt_ref;
  UnaryDoublePrecisionTest(rcbrt_kernel<double>, ref, ULPValidatorBuilderFactory<double>(1));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for rcbrtf and rcbrt.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_rcbrt_rcbrtf_Negative_RTC") { NegativeTestRTCWrapper<4>(kRcbrt); }

/********** Binary Functions **********/

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hypotf(x, y)` and  `hypot(x, y)` against a table of
 * difficult values, followed by a large number of randomly generated values. The results are
 * compared against reference function `T std::hypot(T, T)`. The maximum ulp error for single
 * precision is 3 and for double precision is 2.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_WITHIN_ULP_TEST_DEF(hypot, std::hypot, 3, 2)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for hypotf and hypot.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_hypot_hypotf_Negative_RTC") { NegativeTestRTCWrapper<8>(kHypot); }

MATH_BINARY_KERNEL_DEF(rhypot)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `rhypotf(x, y)` and `rhypot(x, y)`against a table of
 * difficult values, followed by a large number of randomly generated values. The maximum ulp error
 * for single precision is 2 and for double precision is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_rhypot_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto rhypot_ref = [](RT arg1, RT arg2) -> RT { return 1. / std::hypot(arg1, arg2); };
  RT (*ref)(RT, RT) = rhypot_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 2 : 1;
  BinaryFloatingPointTest(rhypot_kernel<TestType>, ref, ULPValidatorBuilderFactory<TestType>(ulp));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for rhypotf and rhypot.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_rhypot_rhypotf_Negative_RTC") { NegativeTestRTCWrapper<8>(kRhypot); }

/********** Ternary Functions **********/

MATH_TERNARY_KERNEL_DEF(norm3d)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `norm3df(x, y, z)` and  `norm3d(x, y, z)` against a table of
 * difficult values, followed by a large number of randomly generated values. The maximum ulp error
 * for single precision is 3 and for double precision is 2.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_norm3d_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto norm3d_ref = [](RT arg1, RT arg2, RT arg3) -> RT {
    if (std::isinf(arg1) || std::isinf(arg2) || std::isinf(arg3)) {
      return std::numeric_limits<RT>::infinity();
    }
    return std::sqrt(arg1 * arg1 + arg2 * arg2 + arg3 * arg3);
  };
  RT (*ref)(RT, RT, RT) = norm3d_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 3 : 2;
  TernaryFloatingPointTest(norm3d_kernel<TestType>, ref, ULPValidatorBuilderFactory<TestType>(ulp));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for norm3df and norm3d.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_norm3d_norm3df_Negative_RTC") { NegativeTestRTCWrapper<12>(kNorm3D); }

MATH_TERNARY_KERNEL_DEF(rnorm3d)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `rnorm3df(x, y, z)` and `rnorm3d(x, y, z)`against a table of
 * difficult values, followed by a large number of randomly generated values. The maximum ulp error
 * for single precision is 2 and for double precision is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_rnorm3d_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto rnorm3d_ref = [](RT arg1, RT arg2, RT arg3) -> RT {
    if (std::isinf(arg1) || std::isinf(arg2) || std::isinf(arg3)) {
      return 0;
    }
    return 1. / std::sqrt(arg1 * arg1 + arg2 * arg2 + arg3 * arg3);
  };
  RT (*ref)(RT, RT, RT) = rnorm3d_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 2 : 1;
  TernaryFloatingPointTest(rnorm3d_kernel<TestType>, ref,
                           ULPValidatorBuilderFactory<TestType>(ulp));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for rnorm3df and rnorm3d.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_rnorm3d_rnorm3df_Negative_RTC") { NegativeTestRTCWrapper<12>(kRnorm3D); }

/********** Quaternary Functions **********/

MATH_QUATERNARY_KERNEL_DEF(norm4d)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `norm4df(x, y, z, t)` and  `norm4d(x, y, z, t)` against a
 * table of difficult values, followed by a large number of randomly generated values. The maximum
 * ulp error for single precision is 3 and for double precision is 2.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_norm4d_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto norm4d_ref = [](RT arg1, RT arg2, RT arg3, RT arg4) -> RT {
    if (std::isinf(arg1) || std::isinf(arg2) || std::isinf(arg3) || std::isinf(arg4)) {
      return std::numeric_limits<RT>::infinity();
    }
    return std::sqrt(arg1 * arg1 + arg2 * arg2 + arg3 * arg3 + arg4 * arg4);
  };
  RT (*ref)(RT, RT, RT, RT) = norm4d_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 3 : 2;
  QuaternaryFloatingPointTest(norm4d_kernel<TestType>, ref,
                              ULPValidatorBuilderFactory<TestType>(ulp));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for norm4df and norm4d.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_norm4d_norm4df_Negative_RTC") { NegativeTestRTCWrapper<16>(kNorm4D); }

MATH_QUATERNARY_KERNEL_DEF(rnorm4d)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `rnorm4df(x, y, z, t)` and `rnorm4d(x, y, z, t)`against a
 * table of difficult values, followed by a large number of randomly generated values. The maximum
 * ulp error for single precision is 2 and for double precision is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_rnorm4d_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto rnorm4d_ref = [](RT arg1, RT arg2, RT arg3, RT arg4) -> RT {
    if (std::isinf(arg1) || std::isinf(arg2) || std::isinf(arg3) || std::isinf(arg4)) {
      return 0;
    }
    return 1. / std::sqrt(arg1 * arg1 + arg2 * arg2 + arg3 * arg3 + arg4 * arg4);
  };
  RT (*ref)(RT, RT, RT, RT) = rnorm4d_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 2 : 1;
  QuaternaryFloatingPointTest(rnorm4d_kernel<TestType>, ref,
                              ULPValidatorBuilderFactory<TestType>(ulp));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for rnorm4df and rnorm4d.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_rnorm4d_rnorm4df_Negative_RTC") { NegativeTestRTCWrapper<16>(kRnorm4D); }

/********** norm Function **********/

#define MATH_NORM_KERNEL_DEF(func_name)                                                            \
  template <typename T> __global__ void func_name##_kernel(T* const ys, int dim, T* const x1s) {   \
    if constexpr (std::is_same_v<float, T>) {                                                      \
      *ys = func_name##f(dim, x1s);                                                                \
    } else if constexpr (std::is_same_v<double, T>) {                                              \
      *ys = func_name(dim, x1s);                                                                   \
    }                                                                                              \
  }

template <typename T, typename F, typename RF, typename ValidatorBuilder>
void NormSimpleTest(F kernel, RF ref_func, const ValidatorBuilder& validator_builder) {
  const auto max_dim = 10000;

  LinearAllocGuard<T> x{LinearAllocs::hipHostMalloc, max_dim * sizeof(T)};
  LinearAllocGuard<T> x_dev{LinearAllocs::hipMalloc, max_dim * sizeof(T)};
  LinearAllocGuard<T> y{LinearAllocs::hipHostMalloc, sizeof(T)};
  LinearAllocGuard<T> y_dev{LinearAllocs::hipMalloc, sizeof(T)};

  std::fill_n(x.ptr(), max_dim, 1);
  HIP_CHECK(hipMemcpy(x_dev.ptr(), x.ptr(), max_dim * sizeof(T), hipMemcpyHostToDevice));

  for (uint64_t i = 1u; i < max_dim; i++) {
    kernel<<<1, 1>>>(y_dev.ptr(), i, x_dev.ptr());
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(y.ptr(), y_dev.ptr(), sizeof(T), hipMemcpyDeviceToHost));
    const auto actual_val = *y.ptr();
    const auto ref_val = static_cast<T>(ref_func(i, x.ptr()));
    const auto validator = validator_builder(ref_val);

    if (!validator->match(actual_val)) {
      std::stringstream ss;
      ss << std::scientific << std::setprecision(std::numeric_limits<T>::max_digits10 - 1);
      ss << "Validation fails for dim: " << i << " " << actual_val << " " << ref_val;
      INFO(ss.str());
      REQUIRE(false);
    }
  }
}

MATH_NORM_KERNEL_DEF(norm)

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `normf(dim, arr)` and `norm(dim, arr)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_norm_Sanity_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto norm_ref = [](int dim, TestType* args) -> RT {
    RT sum = 0;
    for (int i = 0; i < dim; i++) {
      if (std::isinf(args[i])) return std::numeric_limits<RT>::infinity();
      sum += static_cast<RT>(args[i]) * static_cast<RT>(args[i]);
    }
    return std::sqrt(sum);
  };
  RT (*ref)(int, TestType*) = norm_ref;

  NormSimpleTest<TestType>(norm_kernel<TestType>, ref, ULPValidatorBuilderFactory<TestType>(10));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for normf and norm.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_norm_normf_Negative_RTC") { NegativeTestRTCWrapper<18>(kNorm); }

MATH_NORM_KERNEL_DEF(rnorm)

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `rnormf(dim, arr)` and `rnorm(dim, arr)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_rnorm_Sanity_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto rnorm_ref = [](int dim, TestType* args) -> RT {
    RT sum = 0;
    for (int i = 0; i < dim; i++) {
      if (std::isinf(args[i])) return std::numeric_limits<RT>::infinity();
      sum += static_cast<RT>(args[i]) * static_cast<RT>(args[i]);
    }
    return 1. / std::sqrt(sum);
  };
  RT (*ref)(int, TestType*) = rnorm_ref;

  NormSimpleTest<TestType>(rnorm_kernel<TestType>, ref, ULPValidatorBuilderFactory<TestType>(10));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for rnormf and rnorm.
 *
 * Test source
 * ------------------------
 *    - unit/math/root_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_rnorm_rnormf_Negative_RTC") { NegativeTestRTCWrapper<18>(kRnorm); }

/**
 * End doxygen group MathTest.
 * @}
 */
