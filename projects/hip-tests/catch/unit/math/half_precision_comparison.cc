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

#include "half_precision_common.hh"

/**
 * @addtogroup HalfPrecisionComparison HalfPrecisionComparison
 * @{
 * @ingroup MathTest
 */

/********** Unary Functions **********/

#define MATH_BOOL_UNARY_HP_TEST_DEF(func_name, ref_func)                                           \
  __global__ void func_name##_kernel(bool* const ys, const size_t num_xs, Float16* const xs) {     \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(xs[i]);                                                                    \
    }                                                                                              \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #func_name "_Accuracy_Positive") {                                      \
    UnaryHalfPrecisionTest(func_name##_kernel, ref_func, EqValidatorBuilderFactory<bool>());       \
  }

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hisinf(x)` for all possible inputs. The results are
 * compared against reference function `bool std::isinf(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BOOL_UNARY_HP_TEST_DEF(__hisinf, static_cast<bool (*)(float)>(std::isinf))

static float __hisinf2_ref(float x) { return static_cast<float>(std::isinf(x)); }

MATH_UNARY_HP_KERNEL_DEF(__hisinf2)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hisinf2(x)` for all possible inputs. The results are
 * compared against reference function `float std::isinf(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(__hisinf2, __hisinf2_ref, EqValidatorBuilderFactory<float>());

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hisnan(x)` for all possible inputs. The results are
 * compared against reference function `bool std::isnan(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BOOL_UNARY_HP_TEST_DEF(__hisnan, static_cast<bool (*)(float)>(std::isnan))

static float __hisnan2_ref(float x) { return static_cast<float>(std::isnan(x)); }

MATH_UNARY_HP_KERNEL_DEF(__hisnan2)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hisnan2(x)` for all possible inputs. The results are
 * compared against reference function `float std::isnan(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(__hisnan2, __hisnan2_ref, EqValidatorBuilderFactory<float>());

/********** Binary Functions **********/

#define MATH_COMPARISON_HP_TEST_DEF(func_name, ref_func, T, RT, nan_value)                         \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, Float16* const x1s,         \
                                     Float16* const x2s) {                                         \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(x1s[i], x2s[i]);                                                           \
    }                                                                                              \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #func_name "_Accuracy_Positive") {                                      \
    BinaryFloatingPointTest(func_name##_kernel, ref_func<nan_value, RT>,                           \
                            EqValidatorBuilderFactory<RT>());                                      \
  }


template <bool nan_value, typename T> static T __heq_ref(float x1, float x2) {
  if (std::isnan(x1) || std::isnan(x2)) {
    return static_cast<T>(nan_value);
  }
  return x1 == x2;
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__heq(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'equal
 * to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__heq, __heq_ref, bool, bool, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hbeq2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'equal
 * to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hbeq2, __heq_ref, bool, bool, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hequ(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'equal
 * to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hequ, __heq_ref, bool, bool, true)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hbequ2(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are compared against result
 * of 'equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hbequ2, __heq_ref, bool, bool, true)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__heq2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'equal
 * to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__heq2, __heq_ref, Float16, float, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hequ2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'equal
 * to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hequ2, __heq_ref, Float16, float, true)


template <bool nan_value, typename T> static T __hne_ref(float x1, float x2) {
  if (std::isnan(x1) || std::isnan(x2)) {
    return static_cast<T>(nan_value);
  }
  return x1 != x2;
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hne(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'not
 * equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hne, __hne_ref, bool, bool, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hbne2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'not
 * equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hbne2, __hne_ref, bool, bool, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hneu(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'not
 * equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hneu, __hne_ref, bool, bool, true)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hbneu2(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are compared against result
 * of 'not equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hbneu2, __hne_ref, bool, bool, true)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hne2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'not
 * equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hne2, __hne_ref, Float16, float, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hneu2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'not
 * equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hneu2, __hne_ref, Float16, float, true)


template <bool nan_value, typename T> static T __hge_ref(float x1, float x2) {
  if (std::isnan(x1) || std::isnan(x2)) {
    return static_cast<T>(nan_value);
  }
  return x1 >= x2;
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hge(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of
 * 'greater than equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hge, __hge_ref, bool, bool, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hbge2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of
 * 'greater than equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hbge2, __hge_ref, bool, bool, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hgeu(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of
 * 'greater than equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hgeu, __hge_ref, bool, bool, true)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hbgeu2(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are compared against result
 * of 'greater than equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hbgeu2, __hge_ref, bool, bool, true)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hge2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of
 * 'greater than equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hge2, __hge_ref, Float16, float, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hgeu2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of
 * 'greater than equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hgeu2, __hge_ref, Float16, float, true)


template <bool nan_value, typename T> static T __hgt_ref(float x1, float x2) {
  if (std::isnan(x1) || std::isnan(x2)) {
    return static_cast<T>(nan_value);
  }
  return x1 > x2;
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hgt(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of
 * 'greater than' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hgt, __hgt_ref, bool, bool, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hbgt2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of
 * 'greater than' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hbgt2, __hgt_ref, bool, bool, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hgtu(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of
 * 'greater than' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hgtu, __hgt_ref, bool, bool, true)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hbgtu2(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are compared against result
 * of 'greater than' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hbgtu2, __hgt_ref, bool, bool, true)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hgt2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of
 * 'greater than' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hgt2, __hgt_ref, Float16, float, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hgtu2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of
 * 'greater than' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hgtu2, __hgt_ref, Float16, float, true)


template <bool nan_value, typename T> static T __hle_ref(float x1, float x2) {
  if (std::isnan(x1) || std::isnan(x2)) {
    return static_cast<T>(nan_value);
  }
  return x1 <= x2;
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hle(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'less
 * than equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hle, __hle_ref, bool, bool, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hble2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'less
 * than equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hble2, __hle_ref, bool, bool, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hleu(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'less
 * than equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hleu, __hle_ref, bool, bool, true)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hbleu2(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are compared against result
 * of 'less than equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hbleu2, __hle_ref, bool, bool, true)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hle2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'less
 * than equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hle2, __hle_ref, Float16, float, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hleu2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'less
 * than equal to' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hleu2, __hle_ref, Float16, float, true)


template <bool nan_value, typename T> static T __hlt_ref(float x1, float x2) {
  if (std::isnan(x1) || std::isnan(x2)) {
    return static_cast<T>(nan_value);
  }
  return x1 < x2;
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hlt(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'less
 * than' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hlt, __hlt_ref, bool, bool, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hblt2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'less
 * than' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hblt2, __hlt_ref, bool, bool, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hltu(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'less
 * than' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hltu, __hlt_ref, bool, bool, true)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hbltu2(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are compared against result
 * of 'less than' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hbltu2, __hlt_ref, bool, bool, true)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hlt2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'less
 * than' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hlt2, __hlt_ref, Float16, float, false)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hltu2(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against result of 'less
 * than' relational operator for float operands.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_COMPARISON_HP_TEST_DEF(__hltu2, __hlt_ref, Float16, float, true)

MATH_BINARY_HP_KERNEL_DEF(__hmax)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hmax(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against reference
 * function `float std::fmax(float, float)`
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hmax, static_cast<float (*)(float, float)>(std::fmax),
                             EqValidatorBuilderFactory<float>())

MATH_BINARY_HP_KERNEL_DEF(__hmin)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hmin(x,y)` against a table of difficult values, followed
 * by a large number of randomly generated values. The results are compared against reference
 * function `float std::fmin(float, float)`
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hmin, static_cast<float (*)(float, float)>(std::fmin),
                             EqValidatorBuilderFactory<float>())

static float __hmax_nan_ref(float x1, float x2) {
  if (std::isnan(x1))
    return x1;
  else if (std::isnan(x2))
    return x2;
  else
    return std::fmax(x1, x2);
}

MATH_BINARY_HP_KERNEL_DEF(__hmax_nan)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hmax_nan(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are compared against
 * reference function `float std::fmax(float, float)` with modified result when an operand is nan.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hmax_nan, __hmax_nan_ref, EqValidatorBuilderFactory<float>())

static float __hmin_nan_ref(float x1, float x2) {
  if (std::isnan(x1))
    return x1;
  else if (std::isnan(x2))
    return x2;
  else
    return std::fmin(x1, x2);
}

MATH_BINARY_HP_KERNEL_DEF(__hmin_nan)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hmin_nan(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are compared against
 * reference function `float std::fmin(float, float)` with modified result when an operand is nan.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_comparison.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hmin_nan, __hmin_nan_ref, EqValidatorBuilderFactory<float>())

/**
 * End doxygen group MathTest.
 * @}
 */
