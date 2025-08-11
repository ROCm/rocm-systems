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

#define MATH_UNARY_SP_KERNEL_DEF(func_name)                                                        \
  __global__ void func_name##_kernel(float* const ys, const size_t num_xs, float* const xs) {      \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(xs[i]);                                                                    \
    }                                                                                              \
  }

#define MATH_UNARY_SP_TEST_DEF_IMPL(func_name, ref_func, validator_builder)                        \
  TEST_CASE("Unit_Device_" #func_name "_Accuracy_Positive") {                                      \
    UnarySinglePrecisionTest(func_name##_kernel, ref_func, validator_builder);                     \
  }

#define MATH_UNARY_SP_TEST_DEF(func_name, ref_func)                                                \
  MATH_UNARY_SP_TEST_DEF_IMPL(func_name, ref_func, func_name##_validator_builder)

#define MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(func_name)                                             \
  static std::unique_ptr<MatcherBase<float>> func_name##_validator_builder(float target, float x)


static float __frcp_rn_ref(float x) { return 1.0f / x; }

MATH_UNARY_SP_KERNEL_DEF(__frcp_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__frcp_rn(x)` for all possible inputs. The error bounds are
 * IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF_IMPL(__frcp_rn, __frcp_rn_ref, EqValidatorBuilderFactory<float>());


MATH_UNARY_SP_KERNEL_DEF(__fsqrt_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fsqrt_rn(x)` for all possible inputs. The results are
 * compared against reference function `float std::sqrt(float)`. The error bounds are
 * IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF_IMPL(__fsqrt_rn, static_cast<float (*)(float)>(std::sqrt),
                            EqValidatorBuilderFactory<float>());


static float __frsqrt_rn_ref(float x) { return 1.0f / std::sqrt(x); }

MATH_UNARY_SP_KERNEL_DEF(__frsqrt_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__frsqrt_rn(x)` for all possible inputs. The results are
 * compared against reference function `float std::sqrt(float)`. The error bounds are
 * IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF_IMPL(__frsqrt_rn, __frsqrt_rn_ref, EqValidatorBuilderFactory<float>());


MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(__expf) {
  const int64_t ulp_err = 2 + static_cast<int64_t>(std::floor(std::abs(1.16f * x)));
  return ULPValidatorBuilderFactory<float>(ulp_err)(target);
}

MATH_UNARY_SP_KERNEL_DEF(__expf);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__expf(x)` for all possible inputs. The results are
 * compared against reference function `double std::exp(double)`. The maximum ulp error is `2 +
 * floor(abs(1.16 * x))`.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF(__expf, static_cast<double (*)(double)>(std::exp));


MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(__exp10f) {
  const int64_t ulp_err = 2 + static_cast<int64_t>(std::floor(std::abs(2.95f * x)));
  return ULPValidatorBuilderFactory<float>(ulp_err)(target);
}

MATH_UNARY_SP_KERNEL_DEF(__exp10f);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__exp10f(x)` for all possible inputs. The results are
 * compared against reference function `double exp10(double)`. The maximum ulp error is `2 +
 * floor(abs(2.95 * x))`.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF(__exp10f, static_cast<double (*)(double)>(exp10));


MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(__logf) {
  if (0.5f <= x && x <= 2.0f) {
    const auto abs_err = std::pow(2.0, -21.41);
    return AbsValidatorBuilderFactory<float>(abs_err)(target);
  } else {
    return ULPValidatorBuilderFactory<float>(3)(target);
  }
}

MATH_UNARY_SP_KERNEL_DEF(__logf);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__logf(x)` for all possible inputs. The results are
 * compared against reference function `double std::log(double)`. For `x` in [0.5, 2], the maximum
 * absolute error is 2^-21.41, otherwise, the maximum ulp error is 3.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF(__logf, static_cast<double (*)(double)>(std::log));


MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(__log2f) {
  if (0.5f <= x && x <= 2.0f) {
    const auto abs_err = std::pow(2.0, -22.0);
    return AbsValidatorBuilderFactory<float>(abs_err)(target);
  } else {
    return ULPValidatorBuilderFactory<float>(2)(target);
  }
}

MATH_UNARY_SP_KERNEL_DEF(__log2f);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__log2f(x)` for all possible inputs. The results are
 * compared against reference function `double std::log2(double)`. For `x` in [0.5, 2], the maximum
 * absolute error is 2^-22, otherwise, the maximum ulp error is 2.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF(__log2f, static_cast<double (*)(double)>(std::log2));


MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(__log10f) {
  if (0.5f <= x && x <= 2.0f) {
    const auto abs_err = std::pow(2.0, -24.0);
    return AbsValidatorBuilderFactory<float>(abs_err)(target);
  } else {
    return ULPValidatorBuilderFactory<float>(3)(target);
  }
}

MATH_UNARY_SP_KERNEL_DEF(__log10f);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__log10f(x)` for all possible inputs. The results are
 * compared against reference function `double std::log10(double)`. For `x` in [0.5, 2], the maximum
 * absolute error is 2^-24, otherwise, the maximum ulp error is 3.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF(__log10f, static_cast<double (*)(double)>(std::log10));


MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(__sinf) {
  if (-M_PI <= x && x <= M_PI) {
    const auto abs_err = std::pow(2.0, -21.41);
    return AbsValidatorBuilderFactory<float>(abs_err)(target);
  } else {
    return NopValidatorBuilderFactory<float>()();
  }
}

MATH_UNARY_SP_KERNEL_DEF(__sinf);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__sinf(x)` for all possible inputs. The results are
 * compared against reference function `double std::sin(double)`. For `x` in [-PI, PI], the maximum
 * absolute error is 2^-21.41, and larger otherwise.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF(__sinf, static_cast<double (*)(double)>(std::sin));


__device__ float __sincosf_sin(float x) {
  float sin, cos;
  __sincosf(x, &sin, &cos);
  return sin;
}

MATH_UNARY_SP_KERNEL_DEF(__sincosf_sin);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__sincosf(x, sptr, cptr)` for all possible inputs. The
 * results in `sptr` are compared against reference function `double std::sin(double)`. For `x` in
 * [-PI, PI], the maximum absolute error is 2^-21.41, and larger otherwise.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF_IMPL(__sincosf_sin, static_cast<double (*)(double)>(std::sin),
                            __sinf_validator_builder);


MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(__cosf) {
  if (-M_PI <= x && x <= M_PI) {
    const auto abs_err = std::pow(2.0, -21.19);
    return AbsValidatorBuilderFactory<float>(abs_err)(target);
  } else {
    return NopValidatorBuilderFactory<float>()();
  }
}

MATH_UNARY_SP_KERNEL_DEF(__cosf);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__cosf(x)` for all possible inputs. The results are
 * compared against reference function `double std::cos(double)`. For `x` in [-PI, PI], the maximum
 * absolute error is 2^-21.19, and larger otherwise.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF(__cosf, static_cast<double (*)(double)>(std::cos));


__device__ float __sincosf_cos(float x) {
  float sin, cos;
  __sincosf(x, &sin, &cos);
  return cos;
}

MATH_UNARY_SP_KERNEL_DEF(__sincosf_cos);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__sincosf(x, sptr, cptr)` for all possible inputs. The
 * results in `cptr` are compared against reference function `double std::cos(double)`. For `x` in
 * [-PI, PI], the maximum absolute error is 2^-21.19, and larger otherwise.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF_IMPL(__sincosf_cos, static_cast<double (*)(double)>(std::cos),
                            __cosf_validator_builder);


/********** Binary Functions **********/

#define MATH_BINARY_SP_KERNEL_DEF(func_name)                                                       \
  __global__ void func_name##_kernel(float* const ys, const size_t num_xs, float* const x1s,       \
                                     float* const x2s) {                                           \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(x1s[i], x2s[i]);                                                           \
    }                                                                                              \
  }

#define MATH_BINARY_SP_TEST_DEF_IMPL(func_name, ref_func, validator_builder)                       \
  TEST_CASE("Unit_Device_" #func_name "_Accuracy_Positive") {                                      \
    BinaryFloatingPointTest(func_name##_kernel, ref_func, validator_builder);                      \
  }

#define MATH_BINARY_SP_TEST_DEF(func_name, ref_func)                                               \
  MATH_BINARY_SP_TEST_DEF_IMPL(func_name, ref_func, func_name##_validator_builder)

#define MATH_BINARY_SP_VALIDATOR_BUILDER_DEF(func_name)                                            \
  static std::unique_ptr<MatcherBase<float>> func_name##_validator_builder(float target, float x1, \
                                                                           float x2)

static float __fadd_rn_ref(float x1, float x2) { return x1 + x2; }

MATH_BINARY_SP_KERNEL_DEF(__fadd_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fadd_rn(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_SP_TEST_DEF_IMPL(__fadd_rn, __fadd_rn_ref, EqValidatorBuilderFactory<float>());


static float __fsub_rn_ref(float x1, float x2) { return x1 - x2; }

MATH_BINARY_SP_KERNEL_DEF(__fsub_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fsub_rn(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_SP_TEST_DEF_IMPL(__fsub_rn, __fsub_rn_ref, EqValidatorBuilderFactory<float>());


static float __fmul_rn_ref(float x1, float x2) { return x1 * x2; }

MATH_BINARY_SP_KERNEL_DEF(__fmul_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fmul_rn(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_SP_TEST_DEF_IMPL(__fmul_rn, __fmul_rn_ref, EqValidatorBuilderFactory<float>());


static float __fdiv_rn_ref(float x1, float x2) { return x1 / x2; }

MATH_BINARY_SP_KERNEL_DEF(__fdiv_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fdiv_rn(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_SP_TEST_DEF_IMPL(__fdiv_rn, __fdiv_rn_ref, EqValidatorBuilderFactory<float>());


MATH_BINARY_SP_VALIDATOR_BUILDER_DEF(__fdividef) {
  x1 = 2.0f;
  const auto abs_x2 = std::abs(x2);
  if (std::pow(x1, -126.0f) <= abs_x2 && abs_x2 <= std::pow(x1, 126.0f)) {
    return ULPValidatorBuilderFactory<float>(2)(target);
  } else {
    return NopValidatorBuilderFactory<float>()();
  }
}

MATH_BINARY_SP_KERNEL_DEF(__fdividef);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fdividef(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. For `|y|` in [2^-126, 2^126], the
 * maximum ulp error is 2.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_SP_TEST_DEF(__fdividef, __fdiv_rn_ref);


/********** Ternary Functions **********/

#define MATH_TERNARY_SP_KERNEL_DEF(func_name)                                                      \
  __global__ void func_name##_kernel(float* const ys, const size_t num_xs, float* const x1s,       \
                                     float* const x2s, float* const x3s) {                         \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(x1s[i], x2s[i], x3s[i]);                                                   \
    }                                                                                              \
  }

#define MATH_TERNARY_SP_TEST_DEF_IMPL(func_name, ref_func, validator_builder)                      \
  TEST_CASE("Unit_Device_" #func_name "_Accuracy_Positive") {                                      \
    TernaryFloatingPointTest(func_name##_kernel, ref_func, validator_builder);                     \
  }

#define MATH_TERNARY_SP_TEST_DEF(func_name, ref_func, validator_builder)                           \
  MATH_TERNARY_SP_TEST_DEF_IMPL(func_name, ref_func, func_name##_validator_builder)

#define MATH_TERNARY_SP_VALIDATOR_BUILDER_DEF(func_name)                                           \
  static std::unique_ptr<MatcherBase<float>> func_name##_validator_builder(float target, float x1, \
                                                                           float x2, float x3)


MATH_TERNARY_SP_KERNEL_DEF(__fmaf_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fmaf(x,y,z)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_TERNARY_SP_TEST_DEF_IMPL(__fmaf_rn, static_cast<float (*)(float, float, float)>(std::fma),
                              EqValidatorBuilderFactory<float>());