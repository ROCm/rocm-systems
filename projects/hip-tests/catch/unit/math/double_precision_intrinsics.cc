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

#include <hip_test_common.hh>

#include "unary_common.hh"
#include "binary_common.hh"
#include "ternary_common.hh"

/********** Unary Functions **********/

#define MATH_UNARY_DP_KERNEL_DEF(func_name)                                                        \
  __global__ void func_name##_kernel(double* const ys, const size_t num_xs, double* const xs) {    \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(xs[i]);                                                                    \
    }                                                                                              \
  }

#define MATH_UNARY_DP_TEST_DEF_IMPL(func_name, ref_func, validator_builder)                        \
  TEST_CASE("Unit_Device_" #func_name "_Accuracy_Positive") {                                      \
    UnaryDoublePrecisionTest(func_name##_kernel, ref_func, validator_builder);                     \
  }

#define MATH_UNARY_DP_TEST_DEF(func_name, ref_func)                                                \
  MATH_UNARY_DP_TEST_DEF_IMPL(func_name, ref_func, func_name##_validator_builder)

#define MATH_UNARY_DP_VALIDATOR_BUILDER_DEF(func_name)                                             \
  static std::unique_ptr<MatcherBase<double>> func_name##_validator_builder(double target, double x)


static double __drcp_rn_ref(double x) { return 1.0 / x; }

MATH_UNARY_DP_KERNEL_DEF(__drcp_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__drcp_rn(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are
 * IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/double_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_DP_TEST_DEF_IMPL(__drcp_rn, __drcp_rn_ref, EqValidatorBuilderFactory<double>());


MATH_UNARY_DP_KERNEL_DEF(__dsqrt_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__dsqrt_rn(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are
 * compared against reference function `double std::sqrt(double)`. The error bounds are
 * IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/double_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_DP_TEST_DEF_IMPL(__dsqrt_rn, static_cast<double (*)(double)>(std::sqrt),
                            EqValidatorBuilderFactory<double>());


/********** Binary Functions **********/

#define MATH_BINARY_DP_KERNEL_DEF(func_name)                                                       \
  __global__ void func_name##_kernel(double* const ys, const size_t num_xs, double* const x1s,     \
                                     double* const x2s) {                                          \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(x1s[i], x2s[i]);                                                           \
    }                                                                                              \
  }

#define MATH_BINARY_DP_TEST_DEF_IMPL(func_name, ref_func, validator_builder)                       \
  TEST_CASE("Unit_Device_" #func_name "_Accuracy_Positive") {                                      \
    BinaryFloatingPointTest(func_name##_kernel, ref_func, validator_builder);                      \
  }

#define MATH_BINARY_DP_TEST_DEF(func_name, ref_func)                                               \
  MATH_BINARY_DP_TEST_IMPL(func_name, ref_func, func_name##_validator_builder)

#define MATH_BINARY_DP_VALIDATOR_BUILDER_DEF(func_name)                                            \
  static std::unique_ptr<MatcherBase<double>> func_name##_validator_builder(double target,         \
                                                                            double x1, double x2)


static double __dadd_rn_ref(double x1, double x2) { return x1 + x2; }

MATH_BINARY_DP_KERNEL_DEF(__dadd_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__dadd_rn(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/double_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_DP_TEST_DEF_IMPL(__dadd_rn, __dadd_rn_ref, EqValidatorBuilderFactory<double>());


static double __dsub_rn_ref(double x1, double x2) { return x1 - x2; }

MATH_BINARY_DP_KERNEL_DEF(__dsub_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__dsub_rn(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/double_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_DP_TEST_DEF_IMPL(__dsub_rn, __dsub_rn_ref, EqValidatorBuilderFactory<double>());


static double __dmul_rn_ref(double x1, double x2) { return x1 * x2; }

MATH_BINARY_DP_KERNEL_DEF(__dmul_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__dmul_rn(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/double_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_DP_TEST_DEF_IMPL(__dmul_rn, __dmul_rn_ref, EqValidatorBuilderFactory<double>());


static double __ddiv_rn_ref(double x1, double x2) { return x1 / x2; }

MATH_BINARY_DP_KERNEL_DEF(__ddiv_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__ddiv_rn(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/double_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_DP_TEST_DEF_IMPL(__ddiv_rn, __ddiv_rn_ref, EqValidatorBuilderFactory<double>());


/********** Ternary Functions **********/

#define MATH_TERNARY_DP_KERNEL_DEF(func_name)                                                      \
  __global__ void func_name##_kernel(double* const ys, const size_t num_xs, double* const x1s,     \
                                     double* const x2s, double* const x3s) {                       \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(x1s[i], x2s[i], x3s[i]);                                                   \
    }                                                                                              \
  }

#define MATH_TERNARY_DP_TEST_DEF_IMPL(func_name, ref_func, validator_builder)                      \
  TEST_CASE("Unit_Device_" #func_name "_Accuracy_Positive") {                                      \
    TernaryFloatingPointTest(func_name##_kernel, ref_func, validator_builder);                     \
  }

#define MATH_TERNARY_DP_TEST_DEF(func_name, ref_func, validator_builder)                           \
  MATH_TERNARY_DP_TEST_DEF_IMPL(func_name, ref_func, func_name##_validator_builder)

#define MATH_TERNARY_DP_VALIDATOR_BUILDER_DEF(func_name)                                           \
  static std::unique_ptr<MatcherBase<double>> func_name##_validator_builder(                       \
      double target, double x1, double x2, double x3)


MATH_TERNARY_DP_KERNEL_DEF(__fma_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fma(x,y,z)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/double_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_TERNARY_DP_TEST_DEF_IMPL(__fma_rn, static_cast<double (*)(double, double, double)>(std::fma),
                              EqValidatorBuilderFactory<double>());