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
#include "pow_common.hh"
#include "math_pow_negative_kernels_rtc.hh"

/**
 * @addtogroup PowMathFuncs PowMathFuncs
 * @{
 * @ingroup MathTest
 */

/********** Unary Functions **********/

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `expf(x)` for all possible inputs and `exp(x)` against a
 * table of difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::exp(T)`. The maximum ulp error for single
 * precision is 2 and for double precision is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(exp, 2, 1)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for expf and exp.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_exp_expf_Negative_RTC") { NegativeTestRTCWrapper<4>(kExp); }

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `exp2f(x)` for all possible inputs and `exp2(x)` against a
 * table of difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::exp2(T)`. The maximum ulp error for single
 * precision is 2 and for double precision is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(exp2, 2, 1)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for exp2f and exp2.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_exp2_exp2f_Negative_RTC") { NegativeTestRTCWrapper<4>(kExp2); }

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `expm1f(x)` for all possible inputs and `expm1(x)` against a
 * table of difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::exp(T)`. The maximum ulp error is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(expm1, 1, 1)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for expm1f and expm1.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_expm1_expm1f_Negative_RTC") { NegativeTestRTCWrapper<4>(kExpm1); }

MATH_UNARY_KERNEL_DEF(exp10)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `exp10f(x)` for all possible inputs. The maximum ulp error
 * is 2.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_exp10f_Accuracy_Positive") {
  auto exp10_ref = [](double arg) -> double { return std::pow(10, arg); };
  double (*ref)(double) = exp10_ref;
  UnarySinglePrecisionTest(exp10_kernel<float>, ref, ULPValidatorBuilderFactory<float>(2));
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `exp10(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The maximum ulp error is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_exp10_Accuracy_Positive") {
  auto exp10_ref = [](long double arg) -> long double { return std::pow(10, arg); };
  long double (*ref)(long double) = exp10_ref;
  UnaryDoublePrecisionTest(exp10_kernel<double>, ref, ULPValidatorBuilderFactory<double>(1));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for exp10f and exp10.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_exp10_exp10f_Negative_RTC") { NegativeTestRTCWrapper<4>(kExp10); }

template <typename T>
__global__ void frexp_kernel(std::pair<T, int>* const ys, const size_t num_xs, T* const xs) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (auto i = tid; i < num_xs; i += stride) {
    if constexpr (std::is_same_v<float, T>) {
      ys[i].first = frexpf(xs[i], &ys[i].second);
    } else if constexpr (std::is_same_v<double, T>) {
      ys[i].first = frexp(xs[i], &ys[i].second);
    }
  }
}

template <typename T> std::pair<T, int> frexp_ref(T arg) {
  int exp_v;
  T res = std::frexp(arg, &exp_v);
  return {res, exp_v};
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `frexpf(x, exp)` for all possible inputs. The results are
 * compared against reference function `double std::frexp(double, int*)`. The maximum ulp error is
 * 0.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_frexpf_Accuracy_Positive") {
  UnarySinglePrecisionTest(
      frexp_kernel<float>, frexp_ref<double>,
      PairValidatorBuilderFactory<float, int>(ULPValidatorBuilderFactory<float>(0),
                                              EqValidatorBuilderFactory<int>()));
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `frexp(x, exp)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are
 * compared against reference function `long double std::frexp(long double, int*)`. The maximum ulp
 * error is 0.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_frexp_Accuracy_Positive") {
  UnaryDoublePrecisionTest(
      frexp_kernel<double>, frexp_ref<long double>,
      PairValidatorBuilderFactory<double, int>(ULPValidatorBuilderFactory<double>(0),
                                               EqValidatorBuilderFactory<int>()));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for frexpf and frexp.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_frexp_frexpf_Negative_RTC") { NegativeTestRTCWrapper<20>(kFrexp); }


/********** Binary Functions **********/

MATH_BINARY_KERNEL_DEF(pow)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `powf(x, y)` and `pow(x, y)`against a table of
 * difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::pow(T, T)`. The maximum ulp error
 * for single precision is 4 and for double precision is 2.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_pow_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto pow_ref = [](RT arg1, RT arg2) -> RT {
    if (std::isinf(arg1) && arg2 < 0) return 0;
    return std::pow(arg1, arg2);
  };
  RT (*ref)(RT, RT) = pow_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 4 : 2;
  BinaryFloatingPointTest(pow_kernel<TestType>, ref, ULPValidatorBuilderFactory<TestType>(ulp));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for powf and pow.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_pow_powf_Negative_RTC") { NegativeTestRTCWrapper<8>(kPow); }

MATH_POW_INT_KERNEL_DEF(ldexp)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `ldexpf(x, exp)` and `ldexp(x, exp)`against a table of
 * difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::ldexp(T, int)`. The maximum ulp error is 0.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_ldexp_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  RT (*ref)(RT, int) = std::ldexp;
  PowIntFloatingPointTest(ldexp_kernel<TestType, int>, ref,
                          ULPValidatorBuilderFactory<TestType>(0));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for ldexpf and ldexp.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_ldexp_ldexpf_Negative_RTC") { NegativeTestRTCWrapper<8>(kLdexp); }

MATH_POW_INT_KERNEL_DEF(powi)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `powi(x, exp)` and `powi(x, exp)`against a table of
 * difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::pow(T, T)`. The maximum ulp error
 * for single precision is 4 and for double precision is 2.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_powi_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto pow_ref = [](RT arg1, int arg2) -> RT {
    if (std::isinf(arg1) && arg2 < 0) return 0;
    return std::pow(arg1, static_cast<RT>(arg2));
  };
  RT (*ref)(RT, int) = pow_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 4 : 2;
  PowIntFloatingPointTest(powi_kernel<TestType, int>, ref,
                          ULPValidatorBuilderFactory<TestType>(ulp));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for powif and powi.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_powi_powif_Negative_RTC") { NegativeTestRTCWrapper<8>(kPowi); }

MATH_POW_INT_KERNEL_DEF(scalbn)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `scalbnf(x, n)` and `scalbn(x, n)`against a table of
 * difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::scalbn(T, int)`. The maximum ulp error is 0.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_scalbn_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  RT (*ref)(RT, int) = std::scalbn;
  PowIntFloatingPointTest(scalbn_kernel<TestType, int>, ref,
                          ULPValidatorBuilderFactory<TestType>(0));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for scalbnf and scalbn.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_scalbn_scalbnf_Negative_RTC") { NegativeTestRTCWrapper<8>(kScalbn); }

MATH_POW_INT_KERNEL_DEF(scalbln)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `scalblnf(x, l)` and `scalbln(x, l)`against a table of
 * difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::scalbn(T, long int)`. The maximum ulp error is 0.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_scalbln_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  RT (*ref)(RT, long int) = std::scalbln;
  PowIntFloatingPointTest(scalbln_kernel<TestType, long int>, ref,
                          ULPValidatorBuilderFactory<TestType>(0));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for scalblnf and scalbln.
 *
 * Test source
 * ------------------------
 *    - unit/math/pow_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_scalbln_scalblnf_Negative_RTC") { NegativeTestRTCWrapper<8>(kScalbln); }

/**
 * End doxygen group MathTest.
 * @}
 */
