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
#include "special_common.hh"
#include "math_special_func_kernels_rtc.hh"

#include <boost/math/special_functions.hpp>


/**
 * @addtogroup SpecialMathFuncs SpecialMathFuncs
 * @{
 * @ingroup MathTest
 */

/********** Unary Functions **********/

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `erff(x)` for all possible inputs and `erf(x)` against a
 * table of difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::erf(T)`. The maximum ulp error is 2.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(erf, 2, 2)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for erff and erf.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_erf_erff_Negative_RTC") { NegativeTestRTCWrapper<4>(kErf); }

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `erfcf(x)` for all possible inputs and `erfc(x)` against a
 * table of difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::erfc(T)`. The maximum ulp error for single
 * precision is 4 and for double precision is 5.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(erfc, 4, 5)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for erfcf and erfc.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_erfc_erfcf_Negative_RTC") { NegativeTestRTCWrapper<4>(kErfc); }

MATH_UNARY_KERNEL_DEF(erfinv)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `erfinvf(x)` for all possible inputs. The results are
 * compared against reference function `double boost::math::erf_inv(double)`. The maximum ulp error
 * is 2.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_erfinvf_Accuracy_Positive") {
  auto erfinv_ref = [](double arg) -> double {
    if (arg == 0) return 0;
    if (arg == 1)
      return std::numeric_limits<double>::infinity();
    else if (arg == -1)
      return -std::numeric_limits<double>::infinity();
    else if (arg < -1 || arg > 1)
      return std::numeric_limits<double>::quiet_NaN();
    return boost::math::erf_inv(arg);
  };
  double (*ref)(double) = erfinv_ref;
  UnarySinglePrecisionTest(erfinv_kernel<float>, ref, ULPValidatorBuilderFactory<float>(2));
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `erfinv(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are
 * compared against reference function `long double boost::math::erf_inv(long double)`. The maximum
 * ulp error is 5.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_erfinv_Accuracy_Positive") {
  auto erfinv_ref = [](long double arg) -> long double {
    if (arg == 0) return 0;
    if (arg == 1)
      return std::numeric_limits<long double>::infinity();
    else if (arg == -1)
      return -std::numeric_limits<long double>::infinity();
    else if (arg < -1 || arg > 1)
      return std::numeric_limits<long double>::quiet_NaN();
    return boost::math::erf_inv(arg);
  };
  long double (*ref)(long double) = erfinv_ref;
  UnaryDoublePrecisionTest(erfinv_kernel<double>, ref, ULPValidatorBuilderFactory<double>(5));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for erfinvf and erfinv.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_erfinv_erfinvf_Negative_RTC") { NegativeTestRTCWrapper<4>(kErfinv); }

MATH_UNARY_KERNEL_DEF(erfcinv)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `erfcinvf(x)` for all possible inputs. The results are
 * compared against reference function `double boost::math::erfc_inv(double)`. The maximum ulp error
 * is 4.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_erfcinvf_Accuracy_Positive") {
  auto erfcinv_ref = [](double arg) -> double {
    if (arg == 0)
      return std::numeric_limits<double>::infinity();
    else if (arg == 2)
      return -std::numeric_limits<double>::infinity();
    else if (arg < 0 || arg > 2)
      return std::numeric_limits<double>::quiet_NaN();
    return boost::math::erfc_inv(arg);
  };
  double (*ref)(double) = erfcinv_ref;
  UnarySinglePrecisionTest(erfcinv_kernel<float>, ref, ULPValidatorBuilderFactory<float>(4));
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `erfcinv(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are
 * compared against reference function `long double boost::math::erfc_inv(long double)`. The maximum
 * ulp error is 6.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_erfcinv_Accuracy_Positive") {
  auto erfcinv_ref = [](long double arg) -> long double {
    if (arg == 0)
      return std::numeric_limits<long double>::infinity();
    else if (arg == 2)
      return -std::numeric_limits<long double>::infinity();
    else if (arg < 0 || arg > 2)
      return std::numeric_limits<long double>::quiet_NaN();
    return boost::math::erfc_inv(arg);
  };
  long double (*ref)(long double) = erfcinv_ref;
  UnaryDoublePrecisionTest(erfcinv_kernel<double>, ref, ULPValidatorBuilderFactory<double>(6));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for erfcinvf and erfcinv.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_erfcinv_erfcinvf_Negative_RTC") { NegativeTestRTCWrapper<4>(kErfcinv); }

MATH_UNARY_KERNEL_DEF(erfcx)

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `erfcxf(x)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_erfcxf_Sanity_Positive") {
  constexpr std::array<float, 11> input{-std::numeric_limits<float>::infinity(),
                                        -1000.f,
                                        -100.f,
                                        -5.f,
                                        -0.5f,
                                        0.,
                                        0.75f,
                                        15.f,
                                        200.f,
                                        500.f,
                                        std::numeric_limits<float>::infinity()};
  constexpr std::array<float, 11> reference{std::numeric_limits<float>::infinity(),
                                            std::numeric_limits<float>::infinity(),
                                            std::numeric_limits<float>::infinity(),
                                            1.44009806e11f,
                                            1.95236027f,
                                            1.0f,
                                            5.06937683e-1f,
                                            3.75296101e-2f,
                                            2.82091252e-3f,
                                            1.12837693e-3f,
                                            0.f};
  SpecialSimpleTest<float>(erfcx_kernel<float>, ULPValidatorBuilderFactory<float>(4), input.data(),
                           reference.data(), input.size());
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `erfcx(x)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_erfcx_Sanity_Positive") {
  constexpr std::array<double, 11> input{
      -std::numeric_limits<double>::infinity(), -1000., -100., -5., -0.5, 0., 0.75, 15., 200., 500.,
      std::numeric_limits<double>::infinity()};
  constexpr std::array<double, 11> reference{std::numeric_limits<double>::infinity(),
                                             std::numeric_limits<double>::infinity(),
                                             std::numeric_limits<double>::infinity(),
                                             1.4400979867466104e11,
                                             1.9523604891825568,
                                             1.0,
                                             5.0693765029314475e-1,
                                             3.7529606388505762e-2,
                                             2.8209126572120466e-3,
                                             1.1283769103507188e-3,
                                             0.};
  SpecialSimpleTest<double>(erfcx_kernel<double>, ULPValidatorBuilderFactory<double>(4),
                            input.data(), reference.data(), input.size());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for erfcxf and erfcx.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_erfcx_erfcxf_Negative_RTC") { NegativeTestRTCWrapper<4>(kErfcx); }

MATH_UNARY_KERNEL_DEF(normcdf)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `normcdff(x)` for all possible inputs. The maximum ulp error
 * is 5.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_normcdff_Accuracy_Positive") {
  auto normcdf_ref = [](double arg) -> double { return std::erfc(-arg / std::sqrt(2)) / 2; };
  double (*ref)(double) = normcdf_ref;
  UnarySinglePrecisionTest(normcdf_kernel<float>, ref, ULPValidatorBuilderFactory<float>(5));
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `normcdf(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The maximum ulp error is 5.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_normcdf_Accuracy_Positive") {
  auto normcdf_ref = [](long double arg) -> long double {
    return std::erfc(-arg / std::sqrt(2.L)) / 2;
  };
  long double (*ref)(long double) = normcdf_ref;
  UnaryDoublePrecisionTest(normcdf_kernel<double>, ref, ULPValidatorBuilderFactory<double>(5));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for normcdff and normcdf.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_normcdf_normcdff_Negative_RTC") { NegativeTestRTCWrapper<4>(kNormcdf); }

MATH_UNARY_KERNEL_DEF(normcdfinv)

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `normcdfinvf(x)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_normcdfinvf_Sanity_Positive") {
  constexpr std::array<float, 9> input{0.f, 0.1f, 0.25f, 0.4f, 0.5f, 0.6f, 0.75f, 0.9f, 1.f};
  constexpr std::array<float, 9> reference{-std::numeric_limits<float>::infinity(),
                                           -1.28155160f,
                                           -0.674489737f,
                                           -0.253347069f,
                                           0,
                                           0.253347158f,
                                           0.674489737f,
                                           1.28155148f,
                                           std::numeric_limits<float>::infinity()};
  SpecialSimpleTest<float>(normcdfinv_kernel<float>, ULPValidatorBuilderFactory<float>(5),
                           input.data(), reference.data(), input.size());
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `normcdfinv(x)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_normcdfinv_Sanity_Positive") {
  constexpr std::array<double, 9> input{0., 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.};
  constexpr std::array<double, 9> reference{-std::numeric_limits<float>::infinity(),
                                            -1.2815515655446004,
                                            -0.67448975019608159,
                                            -0.25334710313579972,
                                            0,
                                            0.25334710313579972,
                                            0.67448975019608159,
                                            1.2815515655446006,
                                            std::numeric_limits<float>::infinity()};
  SpecialSimpleTest<double>(normcdfinv_kernel<double>, ULPValidatorBuilderFactory<double>(5),
                            input.data(), reference.data(), input.size());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for normcdfinvf and normcdfinv.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_normcdfinv_normcdfinvf_Negative_RTC") {
  NegativeTestRTCWrapper<4>(kNormcdfinv);
}

MATH_UNARY_KERNEL_DEF(tgamma)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `tgammaf(x)` for all possible inputs below 171.7 and that
 * are not very small negative numbers, as they lead to overflow for IEEE compatible double. The
 * results are compared against reference function `double std::tgamma(double)`. The maximum ulp
 * error is 5.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_tgammaf_Accuracy_Limited_Positive") {
  double (*ref)(double) = std::tgamma;
  UnarySinglePrecisionRangeTest(tgamma_kernel<float>, ref, ULPValidatorBuilderFactory<float>(5),
                                std::numeric_limits<float>::lowest(), -0.001f);
  UnarySinglePrecisionRangeTest(tgamma_kernel<float>, ref, ULPValidatorBuilderFactory<float>(5), 0,
                                171.7);
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `tgamma(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are
 * compared against reference function `long double std::tgamma(long double)`. The maximum ulp error
 * is 10.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_tgamma_Accuracy_Limited_Positive") {
  long double (*ref)(long double) = std::tgamma;
  UnaryDoublePrecisionTest(tgamma_kernel<double>, ref, ULPValidatorBuilderFactory<double>(10));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for tgammaf and tgamma.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_tgamma_tgammaf_Negative_RTC") { NegativeTestRTCWrapper<4>(kTgamma); }

MATH_UNARY_KERNEL_DEF(lgamma)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `lgammaf(x)` for all possible inputs. The results are
 * compared against reference function `double std::lgamma(double)`. For `x` outside interval
 * -11.0001 … -2.2637, the maximum ulp error is 4, and larger otherwise.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_lgammaf_Accuracy_Limited_Positive") {
  double (*ref)(double) = std::lgamma;
  UnarySinglePrecisionRangeTest(lgamma_kernel<float>, ref, ULPValidatorBuilderFactory<float>(6),
                                std::numeric_limits<float>::lowest(), -11.0001f);
  UnarySinglePrecisionRangeTest(lgamma_kernel<float>, ref, ULPValidatorBuilderFactory<float>(6),
                                -2.2636f, std::numeric_limits<float>::max());
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `lgamma(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are compared against
 * reference function `long double std::lgamma(long double)`. For `x` outside interval -11.0001 …
 * -2.2637, the maximum ulp error is 4, and larger otherwise.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_lgamma_Accuracy_Limited_Positive") {
  long double (*ref)(long double) = std::lgamma;
  UnaryDoublePrecisionBruteForceTest(lgamma_kernel<double>, ref,
                                     ULPValidatorBuilderFactory<double>(4),
                                     std::numeric_limits<double>::lowest(), -11.0001);
  UnaryDoublePrecisionBruteForceTest(lgamma_kernel<double>, ref,
                                     ULPValidatorBuilderFactory<double>(4), -2.2636,
                                     std::numeric_limits<double>::max());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for lgammaf and lgamma.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_lgamma_lgammaf_Negative_RTC") { NegativeTestRTCWrapper<4>(kLgamma); }

MATH_UNARY_KERNEL_DEF(cyl_bessel_i0)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `cyl_bessel_i0f(x)` for all possible inputs in range [0,
 * 10000). The results are compared against reference function `double std::cyl_bessel_i(0,
 * double)`. The maximum ulp error is 6.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_cyl_bessel_i0f_Accuracy_Limited_Positive") {
  auto cyl_bessel_i0_ref = [](double arg) -> double { return std::cyl_bessel_i(0, arg); };
  double (*ref)(double) = cyl_bessel_i0_ref;
  UnarySinglePrecisionRangeTest(cyl_bessel_i0_kernel<float>, ref,
                                ULPValidatorBuilderFactory<float>(6), 0, 10000);
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `cyl_bessel_i0(x)` against a table of difficult values,
 * followed by a large number of randomly generated values from range [0, 10000). The results are
 * compared against reference function `long double std::cyl_bessel_i(0, long double)`. The maximum
 * ulp error is 6.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_cyl_bessel_i0_Accuracy_Limited_Positive") {
  auto cyl_bessel_i0_ref = [](long double arg) -> long double { return std::cyl_bessel_i(0, arg); };
  long double (*ref)(long double) = cyl_bessel_i0_ref;
  UnaryDoublePrecisionBruteForceTest(cyl_bessel_i0_kernel<double>, ref,
                                     ULPValidatorBuilderFactory<double>(6), 0, 10000);
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for cyl_bessel_i0f and cyl_bessel_i0.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_cyl_bessel_i0_cyl_bessel_i0f_Negative_RTC") {
  NegativeTestRTCWrapper<4>(kCylBesselI0);
}

MATH_UNARY_KERNEL_DEF(cyl_bessel_i1)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `cyl_bessel_i1f(x)` for all possible inputs in range [0,
 * 10000). The results are compared against reference function `double std::cyl_bessel_i(1,
 * double)`. The maximum ulp error is 6.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_cyl_bessel_i1f_Accuracy_Limited_Positive") {
  auto cyl_bessel_i1_ref = [](double arg) -> double { return std::cyl_bessel_i(1, arg); };
  double (*ref)(double) = cyl_bessel_i1_ref;
  UnarySinglePrecisionRangeTest(cyl_bessel_i1_kernel<float>, ref,
                                ULPValidatorBuilderFactory<float>(6), 0, 10000);
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `cyl_bessel_i1(x)` against a table of difficult values,
 * followed by a large number of randomly generated values from range [0, 10000). The results are
 * compared against reference function `long double std::cyl_bessel_i(1, long double)`. The maximum
 * ulp error is 6.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_cyl_bessel_i1_Accuracy_Limited_Positive") {
  auto cyl_bessel_i1_ref = [](long double arg) -> long double { return std::cyl_bessel_i(1, arg); };
  long double (*ref)(long double) = cyl_bessel_i1_ref;
  UnaryDoublePrecisionBruteForceTest(cyl_bessel_i1_kernel<double>, ref,
                                     ULPValidatorBuilderFactory<double>(6), 0, 10000);
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for cyl_bessel_i1f and cyl_bessel_i1.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_cyl_bessel_i1_cyl_bessel_i1f_Negative_RTC") {
  NegativeTestRTCWrapper<4>(kCylBesselI1);
}

/********** Bessel Functions **********/

MATH_UNARY_KERNEL_DEF(y0)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `y0f(x)` for all possible inputs. The results are
 * compared against reference function `double y0(double)`. For `x` outside [-8, 8], the maximum
 * absolute error is 2.2x10^-6, otherwise, the maximum ulp error is 9.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_y0f_Accuracy_Limited_Positive") {
#ifdef __unix__
  double (*ref)(double) = y0;
#elif _WIN64
  double (*ref)(double) = _y0;
#endif
  UnarySinglePrecisionRangeTest(y0_kernel<float>, ref, ULPValidatorBuilderFactory<float>(9), -8.f,
                                8.f);
  UnarySinglePrecisionRangeTest(y0_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022),
                                8.f, std::numeric_limits<float>::max());
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `y0(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are
 * compared against reference function `long double y0l(long double)`. The maximum absolute error is
 * 5x10^-12.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_y0_Accuracy_Limited_Positive") {
#ifdef __unix__
  long double (*ref)(long double) = y0l;
#elif _WIN64
  long double (*ref)(long double) = _y0l;
#endif
  UnaryDoublePrecisionBruteForceTest(y0_kernel<double>, ref,
                                     AbsValidatorBuilderFactory<float>(5.e-12), -8.,
                                     std::numeric_limits<double>::max());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for y0f and y0.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_y0_y0f_Negative_RTC") { NegativeTestRTCWrapper<4>(kY0); }

MATH_UNARY_KERNEL_DEF(y1)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `y1f(x)` for all possible inputs. The results are
 * compared against reference function `double y1(double)`. For `x` outside [-8, 8], the maximum
 * absolute error is 2.2x10^-6, otherwise, the maximum ulp error is 9.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_y1f_Accuracy_Limited_Positive") {
#ifdef __unix__
  double (*ref)(double) = y1;
#elif _WIN64
  double (*ref)(double) = _y1;
#endif
  UnarySinglePrecisionRangeTest(y1_kernel<float>, ref, ULPValidatorBuilderFactory<float>(9), -8.f,
                                8.f);
  UnarySinglePrecisionRangeTest(y1_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022),
                                8.f, std::numeric_limits<float>::max());
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `y1(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are
 * compared against reference function `long double y1l(long double)`. The maximum absolute error is
 * 5x10^-12.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_y1_Accuracy_Limited_Positive") {
#ifdef __unix__
  long double (*ref)(long double) = y1l;
#elif _WIN64
  long double (*ref)(long double) = _y1l;
#endif
  UnaryDoublePrecisionBruteForceTest(y1_kernel<double>, ref,
                                     AbsValidatorBuilderFactory<float>(5.e-12), -8.,
                                     std::numeric_limits<double>::max());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for y1f and y1.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_y1_y1f_Negative_RTC") { NegativeTestRTCWrapper<4>(kY1); }

MATH_BESSEL_N_KERNEL_DEF(yn)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `ynf(x)` for all possible inputs and n equal to 5, 25 or
 * 120. The results are compared against reference function `double yn(int, double)`. For `x` larger
 * than n, the maximum absolute error is 2.2x10^-6.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_ynf_Accuracy_Limited_Positive") {
#ifdef __unix__
  double (*ref)(int, double) = yn;
#elif _WIN64
  double (*ref)(int, double) = _yn;
#endif
  int n = GENERATE(5, 25, 120);
  BesselSinglePrecisionRangeTest(yn_kernel, ref, AbsValidatorBuilderFactory<float>(0.0000022), n, n,
                                 std::numeric_limits<float>::max());
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `yn(x)` against a table of difficult values,
 * followed by a large number of randomly generated values from range and n equal to 5, 25, or 120.
 * The results are compared against reference function `long double ynl(int, long double)`. For `x`
 * larger than 1.5n, the  maximum absolute error is 5x10^-12.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_yn_Accuracy_Limited_Positive") {
#ifdef __unix__
  long double (*ref)(int, long double) = ynl;
#elif _WIN64
  long double (*ref)(int, long double) = _ynl;
#endif
  int n = GENERATE(5, 25, 120);
  BesselDoublePrecisionBruteForceTest(yn_kernel<double>, ref,
                                      AbsValidatorBuilderFactory<double>(5.e-12), n, 1.5 * n,
                                      std::numeric_limits<double>::max());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for ynf and yn.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_yn_ynf_Negative_RTC") { NegativeTestRTCWrapper<8>(kYn); }

MATH_UNARY_KERNEL_DEF(j0)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `j0f(x)` for all possible inputs. The results are
 * compared against reference function `double j0(double)`. For `x` outside [-8, 8], the maximum
 * absolute error is 2.2x10^-6, otherwise, the maximum ulp error is 9.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_j0f_Accuracy_Limited_Positive") {
#ifdef __unix__
  double (*ref)(double) = j0;
#elif _WIN64
  double (*ref)(double) = _j0;
#endif
  UnarySinglePrecisionRangeTest(j0_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022),
                                std::numeric_limits<float>::lowest(), -8.f);
  UnarySinglePrecisionRangeTest(j0_kernel<float>, ref, ULPValidatorBuilderFactory<float>(9), -8.f,
                                8.f);
  UnarySinglePrecisionRangeTest(j0_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022),
                                8.f, std::numeric_limits<float>::max());
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `j0(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are
 * compared against reference function `long double j0l(long double)`. The maximum absolute error is
 * 5x10^-12.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_j0_Accuracy_Limited_Positive") {
#ifdef __unix__
  long double (*ref)(long double) = j0l;
#elif _WIN64
  long double (*ref)(long double) = _j0l;
#endif
  UnaryDoublePrecisionBruteForceTest(
      j0_kernel<double>, ref, AbsValidatorBuilderFactory<float>(5.e-12),
      std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for j0f and j0.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_j0_j0f_Negative_RTC") { NegativeTestRTCWrapper<4>(kJ0); }

MATH_UNARY_KERNEL_DEF(j1)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `j1f(x)` for all possible inputs. The results are
 * compared against reference function `double j1(double)`. For `x` outside [-8, 8], the maximum
 * absolute error is 2.2x10^-6, otherwise, the maximum ulp error is 9.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_j1f_Accuracy_Limited_Positive") {
#ifdef __unix__
  double (*ref)(double) = j1;
#elif _WIN64
  double (*ref)(double) = _j1;
#endif
  UnarySinglePrecisionRangeTest(j1_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022),
                                std::numeric_limits<float>::lowest(), -8.f);
  UnarySinglePrecisionRangeTest(j1_kernel<float>, ref, ULPValidatorBuilderFactory<float>(9), -8.f,
                                8.f);
  UnarySinglePrecisionRangeTest(j1_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022),
                                8.f, std::numeric_limits<float>::max());
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `j1(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are
 * compared against reference function `long double j1l(long double)`. The maximum absolute error is
 * 5x10^-12.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_j1_Accuracy_Limited_Positive") {
#ifdef __unix__
  long double (*ref)(long double) = j1l;
#elif _WIN64
  long double (*ref)(long double) = _j1l;
#endif
  UnaryDoublePrecisionBruteForceTest(
      j1_kernel<double>, ref, AbsValidatorBuilderFactory<double>(5.e-12),
      std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for j1f and j1.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_j1_j1f_Negative_RTC") { NegativeTestRTCWrapper<4>(kJ1); }

MATH_BESSEL_N_KERNEL_DEF(jn)

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `jnf(x)` for all possible inputs and n equal to 5, 25 or
 * 120. The results are compared against reference function `double jn(int, double)`. For `x` larger
 * than n, the maximum absolute error is 2.2x10^-6.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_jnf_Accuracy_Limited_Positive") {
#ifdef __unix__
  double (*ref)(int, double) = jn;
#elif _WIN64
  double (*ref)(int, double) = _jn;
#endif
  int n = GENERATE(5, 25, 120);
  BesselSinglePrecisionRangeTest(jn_kernel, ref, AbsValidatorBuilderFactory<float>(0.0000022), n, n,
                                 std::numeric_limits<float>::max());
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `jn(x)` against a table of difficult values,
 * followed by a large number of randomly generated values from range and n equal to 5, 25, or 120.
 * The results are compared against reference function `long double jnl(int, long double)`. The
 * maximum absolute error is 5x10^-12.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_jn_Accuracy_Limited_Positive") {
#ifdef __unix__
  long double (*ref)(int, long double) = jnl;
#elif _WIN64
  long double (*ref)(int, long double) = _jnl;
#endif
  int n = GENERATE(5, 25, 120);
  BesselDoublePrecisionBruteForceTest(
      jn_kernel<double>, ref, AbsValidatorBuilderFactory<double>(5.e-12), n,
      std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for jnf and jn.
 *
 * Test source
 * ------------------------
 *    - unit/math/special_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_jn_jnf_Negative_RTC") { NegativeTestRTCWrapper<8>(kJn); }

/**
 * End doxygen group MathTest.
 * @}
 */
