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
 * @addtogroup HalfPrecisionArithmetic HalfPrecisionArithmetic
 * @{
 * @ingroup MathTest
 */


MATH_UNARY_HP_KERNEL_DEF(__habs);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__habs(x)` for all possible inputs. The results are
 * compared against reference function `float std::abs(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(__habs, static_cast<float (*)(float)>(std::abs),
                            EqValidatorBuilderFactory<float>());

MATH_UNARY_HP_KERNEL_DEF(__habs2);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__habs2(x)` for all possible inputs. The results are
 * compared against reference function `float std::abs(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(__habs2, static_cast<float (*)(float)>(std::abs),
                            EqValidatorBuilderFactory<float>());


static float __hneg_ref(float x) { return -x; }

MATH_UNARY_HP_KERNEL_DEF(__hneg);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hneg(x)` for all possible inputs. The error bounds are
 * IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(__hneg, __hneg_ref, EqValidatorBuilderFactory<float>());

MATH_UNARY_HP_KERNEL_DEF(__hneg2);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hneg2(x)` for all possible inputs. The error bounds are
 * IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(__hneg2, __hneg_ref, EqValidatorBuilderFactory<float>());


// Wrapper to avoid ambiguity error with __hadd(int, int)
__device__ __half __hadd_wrapper(__half x1, __half x2) { return __hadd(x1, x2); }

static float __hadd_ref(float x1, float x2) { return x1 + x2; }

MATH_BINARY_HP_KERNEL_DEF(__hadd_wrapper);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hadd(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hadd_wrapper, __hadd_ref, EqValidatorBuilderFactory<float>());

MATH_BINARY_HP_KERNEL_DEF(__hadd2);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hadd2(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hadd2, __hadd_ref, EqValidatorBuilderFactory<float>());


static float __hadd_sat_ref(float x1, float x2) { return std::clamp(x1 + x2, 0.0f, 1.0f); }

MATH_BINARY_HP_KERNEL_DEF(__hadd_sat);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hadd_sat(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hadd_sat, __hadd_sat_ref, EqValidatorBuilderFactory<float>());

MATH_BINARY_HP_KERNEL_DEF(__hadd2_sat);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hadd2_sat(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hadd2_sat, __hadd_sat_ref, EqValidatorBuilderFactory<float>());


static float __hsub_ref(float x1, float x2) { return x1 - x2; }

MATH_BINARY_HP_KERNEL_DEF(__hsub);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hsub(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hsub, __hsub_ref, EqValidatorBuilderFactory<float>());

MATH_BINARY_HP_KERNEL_DEF(__hsub2);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hsub2(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hsub2, __hsub_ref, EqValidatorBuilderFactory<float>());


static float __hsub_sat_ref(float x1, float x2) { return std::clamp(x1 - x2, 0.0f, 1.0f); }

MATH_BINARY_HP_KERNEL_DEF(__hsub_sat);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hsub_sat(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hsub_sat, __hsub_sat_ref, EqValidatorBuilderFactory<float>());

MATH_BINARY_HP_KERNEL_DEF(__hsub2_sat);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hsub2_sat(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hsub2_sat, __hsub_sat_ref, EqValidatorBuilderFactory<float>());


static float __hmul_ref(float x1, float x2) { return x1 * x2; }

MATH_BINARY_HP_KERNEL_DEF(__hmul);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hmul(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hmul, __hmul_ref, EqValidatorBuilderFactory<float>());

MATH_BINARY_HP_KERNEL_DEF(__hmul2);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hmul2(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hmul2, __hmul_ref, EqValidatorBuilderFactory<float>());


static float __hmul_sat_ref(float x1, float x2) { return std::clamp(x1 * x2, 0.0f, 1.0f); }

MATH_BINARY_HP_KERNEL_DEF(__hmul_sat);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hmul_sat(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hmul_sat, __hmul_sat_ref, EqValidatorBuilderFactory<float>());

MATH_BINARY_HP_KERNEL_DEF(__hmul2_sat);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hmul2_sat(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hmul2_sat, __hmul_sat_ref, EqValidatorBuilderFactory<float>());


static float __hdiv_ref(float x1, float x2) { return x1 / x2; }

MATH_BINARY_HP_KERNEL_DEF(__hdiv);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hdiv(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__hdiv, __hdiv_ref, EqValidatorBuilderFactory<float>());

MATH_BINARY_HP_KERNEL_DEF(__h2div);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__h2div(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_HP_TEST_DEF_IMPL(__h2div, __hdiv_ref, EqValidatorBuilderFactory<float>());


MATH_TERNARY_HP_KERNEL_DEF(__hfma);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hfma(x,y,z)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_TERNARY_HP_TEST_DEF_IMPL(__hfma, static_cast<float (*)(float, float, float)>(std::fma),
                              EqValidatorBuilderFactory<float>());

MATH_TERNARY_HP_KERNEL_DEF(__hfma2);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hfma2(x,y,z)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_TERNARY_HP_TEST_DEF_IMPL(__hfma2, static_cast<float (*)(float, float, float)>(std::fma),
                              EqValidatorBuilderFactory<float>());


static float __hfma_sat_ref(float x1, float x2, float x3) {
  return std::clamp(std::fma(x1, x2, x3), 0.0f, 1.0f);
}

MATH_TERNARY_HP_KERNEL_DEF(__hfma_sat);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hfma_sat(x,y,z)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_TERNARY_HP_TEST_DEF_IMPL(__hfma_sat, __hfma_sat_ref, EqValidatorBuilderFactory<float>());

MATH_TERNARY_HP_KERNEL_DEF(__hfma2_sat);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__hfma2_sat(x,y,z)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_arithmetic.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_TERNARY_HP_TEST_DEF_IMPL(__hfma2_sat, __hfma_sat_ref, EqValidatorBuilderFactory<float>());

/**
 * End doxygen group MathTest.
 * @}
 */
