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
#include "math_log_negative_kernels_rtc.hh"

/**
 * @addtogroup LogMathFuncs LogMathFuncs
 * @{
 * @ingroup MathTest
 */

/********** Unary Functions **********/

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `logf(x)` for all possible inputs and `log(x)` against a
 * table of difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::log(T)`. The maximum ulp error is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(log, 1, 1)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for logf and log.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_log_logf_Negative_RTC") { NegativeTestRTCWrapper<4>(kLog); }

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `log2f(x)` for all possible inputs and `log2(x)` against a
 * table of difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::log2(T)`. The maximum ulp error is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(log2, 1, 1)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for log2f and log2.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_log2_log2f_Negative_RTC") { NegativeTestRTCWrapper<4>(kLog2); }

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `log10f(x)` for all possible inputs and `log10(x)` against a
 * table of difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::log10(T)`. The maximum ulp error for single
 * precision is 2 and for double precision is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(log10, 2, 1)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for log10f and log10.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_log10_log10f_Negative_RTC") { NegativeTestRTCWrapper<4>(kLog10); }

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `log1pf(x)` for all possible inputs and `log1p(x)` against a
 * table of difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::log1p(T)`. The maximum ulp error is 1.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(log1p, 1, 1)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for log1pf and log1p.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_log1p_log1pf_Negative_RTC") { NegativeTestRTCWrapper<4>(kLog1p); }

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `logb(x)` for all possible inputs and `logb(x)` against a
 * table of difficult values, followed by a large number of randomly generated values. The results
 * are compared against reference function `T std::logb(T)`. The maximum ulp error is 0.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(logb, 0, 0)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for logbf and logb.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_logb_logbf_Negative_RTC") { NegativeTestRTCWrapper<4>(kLogb); }


template <typename T>
__global__ void ilogb_kernel(int* const ys, const size_t num_xs, T* const xs) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (auto i = tid; i < num_xs; i += stride) {
    if constexpr (std::is_same_v<float, T>) {
      ys[i] = ilogbf(xs[i]);
    } else if constexpr (std::is_same_v<double, T>) {
      ys[i] = ilogb(xs[i]);
    }
  }
}

template <typename T> int ilogb_ref(T arg) {
  if (arg == 0) {
    return std::numeric_limits<int>::min();
  } else if (std::isnan(arg)) {
    return std::numeric_limits<int>::min();
  } else if (std::isinf(arg)) {
    return std::numeric_limits<int>::max();
  } else {
    return std::ilogb(arg);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `ilogbf(x)` for all possible inputs. The results are
 * compared against reference function `int std::ilogb(double)`. The maximum ulp error is 0.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_ilogbf_Accuracy_Positive") {
  UnarySinglePrecisionTest(ilogb_kernel<float>, ilogb_ref<double>,
                           EqValidatorBuilderFactory<int>());
}

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `ilogb(x)` against a table of difficult values,
 * followed by a large number of randomly generated values. The results are
 * compared against reference function `int std::ilogb(long double)`. The maximum ulp error is 0.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_ilogb_Accuracy_Positive") {
  UnaryDoublePrecisionTest(ilogb_kernel<double>, ilogb_ref<long double>,
                           EqValidatorBuilderFactory<int>());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for ilogbf and ilogb.
 *
 * Test source
 * ------------------------
 *    - unit/math/log_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_ilogb_ilogbf_Negative_RTC") { NegativeTestRTCWrapper<4>(kIlogb); }

/**
 * End doxygen group MathTest.
 * @}
 */
