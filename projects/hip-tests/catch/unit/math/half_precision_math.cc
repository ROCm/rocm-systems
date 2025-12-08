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
 * @addtogroup HalfPrecisionMath HalfPrecisionMath
 * @{
 * @ingroup MathTest
 */


MATH_UNARY_HP_KERNEL_DEF(hcos);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hcos(x)` for all possible inputs. The results are
 * compared against reference function `float std::cos(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hcos, static_cast<float (*)(float)>(std::cos),
                            ULPValidatorBuilderFactory<float>(2));

MATH_UNARY_HP_KERNEL_DEF(h2cos);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2cos(x)` for all possible inputs. The results are
 * compared against reference function `float std::cos(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2cos, static_cast<float (*)(float)>(std::cos),
                            ULPValidatorBuilderFactory<float>(2));


MATH_UNARY_HP_KERNEL_DEF(hsin);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hsin(x)` for all possible inputs. The results are
 * compared against reference function `float std::sin(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hsin, static_cast<float (*)(float)>(std::sin),
                            ULPValidatorBuilderFactory<float>(2));

MATH_UNARY_HP_KERNEL_DEF(h2sin);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2sin(x)` for all possible inputs. The results are
 * compared against reference function `float std::sin(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2sin, static_cast<float (*)(float)>(std::sin),
                            ULPValidatorBuilderFactory<float>(2));


MATH_UNARY_HP_KERNEL_DEF(hexp);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hexp(x)` for all possible inputs. The results are
 * compared against reference function `float std::exp(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hexp, static_cast<float (*)(float)>(std::exp),
                            ULPValidatorBuilderFactory<float>(2));

MATH_UNARY_HP_KERNEL_DEF(h2exp);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2exp(x)` for all possible inputs. The results are
 * compared against reference function `float std::exp(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2exp, static_cast<float (*)(float)>(std::exp),
                            ULPValidatorBuilderFactory<float>(2));


MATH_UNARY_HP_KERNEL_DEF(hexp10);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hexp10(x)` for all possible inputs. The results are
 * compared against reference function `float exp10(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hexp10, static_cast<float (*)(float)>(exp10f),
                            ULPValidatorBuilderFactory<float>(2));

MATH_UNARY_HP_KERNEL_DEF(h2exp10);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2exp10(x)` for all possible inputs. The results are
 * compared against reference function `float exp10(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2exp10, static_cast<float (*)(float)>(exp10f),
                            ULPValidatorBuilderFactory<float>(2));


MATH_UNARY_HP_KERNEL_DEF(hexp2);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hexp2(x)` for all possible inputs. The results are
 * compared against reference function `float std::exp2(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hexp2, static_cast<float (*)(float)>(std::exp2),
                            ULPValidatorBuilderFactory<float>(2));

MATH_UNARY_HP_KERNEL_DEF(h2exp2);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2exp2(x)` for all possible inputs. The results are
 * compared against reference function `float std::exp2(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2exp2, static_cast<float (*)(float)>(std::exp2),
                            ULPValidatorBuilderFactory<float>(2));


MATH_UNARY_HP_KERNEL_DEF(hlog);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hlog(x)` for all possible inputs. The results are
 * compared against reference function `float std::log(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hlog, static_cast<float (*)(float)>(std::log),
                            ULPValidatorBuilderFactory<float>(1));

MATH_UNARY_HP_KERNEL_DEF(h2log);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2log(x)` for all possible inputs. The results are
 * compared against reference function `float std::log(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2log, static_cast<float (*)(float)>(std::log),
                            ULPValidatorBuilderFactory<float>(1));


MATH_UNARY_HP_KERNEL_DEF(hlog10);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hlog10(x)` for all possible inputs. The results are
 * compared against reference function `float std::log10(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hlog10, static_cast<float (*)(float)>(std::log10),
                            ULPValidatorBuilderFactory<float>(2));

MATH_UNARY_HP_KERNEL_DEF(h2log10);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2log10(x)` for all possible inputs. The results are
 * compared against reference function `float std::log10(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2log10, static_cast<float (*)(float)>(std::log10),
                            ULPValidatorBuilderFactory<float>(2));


MATH_UNARY_HP_KERNEL_DEF(hlog2);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hlog2(x)` for all possible inputs. The results are
 * compared against reference function `float std::log2(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hlog2, static_cast<float (*)(float)>(std::log2),
                            ULPValidatorBuilderFactory<float>(1));

MATH_UNARY_HP_KERNEL_DEF(h2log2);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2log2(x)` for all possible inputs. The results are
 * compared against reference function `float std::log2(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2log2, static_cast<float (*)(float)>(std::log2),
                            ULPValidatorBuilderFactory<float>(1));


MATH_UNARY_HP_KERNEL_DEF(hsqrt);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hsqrt(x)` for all possible inputs. The results are
 * compared against reference function `float std::sqrt(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hsqrt, static_cast<float (*)(float)>(std::sqrt),
                            ULPValidatorBuilderFactory<float>(1));

MATH_UNARY_HP_KERNEL_DEF(h2sqrt);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2sqrt(x)` for all possible inputs. The results are
 * compared against reference function `float std::sqrt(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2sqrt, static_cast<float (*)(float)>(std::sqrt),
                            ULPValidatorBuilderFactory<float>(1));


MATH_UNARY_HP_KERNEL_DEF(hceil);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hceil(x)` for all possible inputs. The results are
 * compared against reference function `float std::ceil(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hceil, static_cast<float (*)(float)>(std::ceil),
                            EqValidatorBuilderFactory<float>());

MATH_UNARY_HP_KERNEL_DEF(h2ceil);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2ceil(x)` for all possible inputs. The results are
 * compared against reference function `float std::ceil(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2ceil, static_cast<float (*)(float)>(std::ceil),
                            EqValidatorBuilderFactory<float>());


MATH_UNARY_HP_KERNEL_DEF(hfloor);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hfloor(x)` for all possible inputs. The results are
 * compared against reference function `float std::floor(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hfloor, static_cast<float (*)(float)>(std::floor),
                            EqValidatorBuilderFactory<float>());

MATH_UNARY_HP_KERNEL_DEF(h2floor);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2floor(x)` for all possible inputs. The results are
 * compared against reference function `float std::floor(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2floor, static_cast<float (*)(float)>(std::floor),
                            EqValidatorBuilderFactory<float>());


MATH_UNARY_HP_KERNEL_DEF(htrunc);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `htrunc(x)` for all possible inputs. The results are
 * compared against reference function `float std::trunc(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(htrunc, static_cast<float (*)(float)>(std::trunc),
                            EqValidatorBuilderFactory<float>());

MATH_UNARY_HP_KERNEL_DEF(h2trunc);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2trunc(x)` for all possible inputs. The results are
 * compared against reference function `float std::trunc(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2trunc, static_cast<float (*)(float)>(std::trunc),
                            EqValidatorBuilderFactory<float>());


static float hrcp_ref(float x) { return 1.0f / x; }

MATH_UNARY_HP_KERNEL_DEF(hrcp);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hrcp(x)` for all possible inputs.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hrcp, hrcp_ref, EqValidatorBuilderFactory<float>());

MATH_UNARY_HP_KERNEL_DEF(h2rcp);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2rcp(x)` for all possible inputs.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2rcp, hrcp_ref, EqValidatorBuilderFactory<float>());


static float hrsqrt_ref(float x) { return 1.0f / std::sqrt(x); }

MATH_UNARY_HP_KERNEL_DEF(hrsqrt);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hrsqrt(x)` for all possible inputs.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hrsqrt, hrsqrt_ref, EqValidatorBuilderFactory<float>());

MATH_UNARY_HP_KERNEL_DEF(h2rsqrt);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2rsqrt(x)` for all possible inputs.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2rsqrt, hrsqrt_ref, EqValidatorBuilderFactory<float>());


MATH_UNARY_HP_KERNEL_DEF(hrint);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `hrint(x)` for all possible inputs. The results are
 * compared against reference function `float std::rint(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(hrint, static_cast<float (*)(float)>(std::rint),
                            EqValidatorBuilderFactory<float>());

MATH_UNARY_HP_KERNEL_DEF(h2rint);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `h2rint(x)` for all possible inputs. The results are
 * compared against reference function `float std::rint(float)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/half_precision_math.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_HP_TEST_DEF_IMPL(h2rint, static_cast<float (*)(float)>(std::rint),
                            EqValidatorBuilderFactory<float>());

/**
 * End doxygen group MathTest.
 * @}
 */
