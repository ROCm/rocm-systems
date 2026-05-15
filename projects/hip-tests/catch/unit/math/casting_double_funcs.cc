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

#include "casting_common.hh"
#include "casting_double_negative_kernels_rtc.hh"

/**
 * @addtogroup CastingDoubleType CastingDoubleType
 * @{
 * @ingroup MathTest
 */

#define CAST_DOUBLE2INT_TEST_DEF(kern_name, T, ref_func)                                           \
  CAST_KERNEL_DEF(kern_name, T, double)                                                            \
  CAST_F2I_REF_DEF(kern_name, T, double, ref_func)                                                 \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(double) = kern_name##_ref;                                                            \
    CastDoublePrecisionTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>());              \
  }

#define CAST_DOUBLE2INT_RZ_TEST_DEF(kern_name, T)                                                  \
  CAST_KERNEL_DEF(kern_name, T, double)                                                            \
  CAST_F2I_RZ_REF_DEF(kern_name, T, double)                                                        \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(double) = kern_name##_ref;                                                            \
    CastDoublePrecisionTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>());              \
  }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2int_rd` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function
 * `std::floor`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2INT_TEST_DEF(__double2int_rd, int, std::floor)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2int_rn` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function
 * `std::rint`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2INT_TEST_DEF(__double2int_rn, int, std::rint)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2int_ru` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function
 * `std::ceil`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2INT_TEST_DEF(__double2int_ru, int, std::ceil)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2int_rz` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function which
 * performs cast to int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2INT_RZ_TEST_DEF(__double2int_rz, int)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __double2int_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___double2int_Negative_RTC") { NegativeTestRTCWrapper<12>(kDouble2Int); }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2uint_rd` against a table of difficult values, followed by a
 * large number of randomly generated values. The results are compared against reference function
 * `std::floor`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2INT_TEST_DEF(__double2uint_rd, unsigned int, std::floor)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2uint_rn` against a table of difficult values, followed by a
 * large number of randomly generated values. The results are compared against reference function
 * `std::rint`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2INT_TEST_DEF(__double2uint_rn, unsigned int, std::rint)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2uint_ru` against a table of difficult values, followed by a
 * large number of randomly generated values. The results are compared against reference function
 * `std::ceil`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2INT_TEST_DEF(__double2uint_ru, unsigned int, std::ceil)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2uint_rz` against a table of difficult values, followed by a
 * large number of randomly generated values. The results are compared against reference function
 * which performs cast to unsigned int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2INT_RZ_TEST_DEF(__double2uint_rz, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __double2uint_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___double2uint_Negative_RTC") { NegativeTestRTCWrapper<12>(kDouble2Uint); }

#define CAST_DOUBLE2LL_TEST_DEF(kern_name, T, ref_func)                                            \
  CAST_KERNEL_DEF(kern_name, T, double)                                                            \
  CAST_F2I_REF_DEF(kern_name, T, double, ref_func)                                                 \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(double) = kern_name##_ref;                                                            \
    UnaryDoublePrecisionBruteForceTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>(),    \
                                       static_cast<double>(std::numeric_limits<T>::min()),         \
                                       static_cast<double>(std::numeric_limits<T>::max()));        \
  }

#define CAST_DOUBLE2LL_RZ_TEST_DEF(kern_name, T)                                                   \
  CAST_KERNEL_DEF(kern_name, T, double)                                                            \
  CAST_F2I_RZ_REF_DEF(kern_name, T, double)                                                        \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(double) = kern_name##_ref;                                                            \
    UnaryDoublePrecisionBruteForceTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>(),    \
                                       static_cast<double>(std::numeric_limits<T>::min()),         \
                                       static_cast<double>(std::numeric_limits<T>::max()));        \
  }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2ll_rd` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function
 * `std::floor`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2LL_TEST_DEF(__double2ll_rd, long long int, std::floor)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2ll_rn` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function
 * `std::rint`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2LL_TEST_DEF(__double2ll_rn, long long int, std::rint)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2ll_ru` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function
 * `std::ceil`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2LL_TEST_DEF(__double2ll_ru, long long int, std::ceil)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2ll_rz` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function which
 * performs cast to long long int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2LL_RZ_TEST_DEF(__double2ll_rz, long long int)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __double2ll_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___double2ll_Negative_RTC") { NegativeTestRTCWrapper<12>(kDouble2LL); }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2ull_rd` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function
 * `std::floor`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2LL_TEST_DEF(__double2ull_rd, unsigned long long int, std::floor)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2ull_rn` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function
 * `std::rint`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2LL_TEST_DEF(__double2ull_rn, unsigned long long int, std::rint)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2ull_ru` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function
 * `std::ceil`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2LL_TEST_DEF(__double2ull_ru, unsigned long long int, std::ceil)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2ull_rz` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function which
 * performs cast to unsigned long long int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2LL_RZ_TEST_DEF(__double2ull_rz, unsigned long long int)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __double2ull_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___double2ull_Negative_RTC") { NegativeTestRTCWrapper<12>(kDouble2ULL); }

#define CAST_DOUBLE2FLOAT_TEST_DEF(kern_name, round_dir)                                           \
  CAST_KERNEL_DEF(kern_name, float, double)                                                        \
  CAST_RND_REF_DEF(kern_name, float, double, round_dir)                                            \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    float (*ref)(double) = kern_name##_ref;                                                        \
    CastDoublePrecisionTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<float>());          \
  }

#define CAST_DOUBLE2FLOAT_RN_TEST_DEF(kern_name)                                                   \
  CAST_KERNEL_DEF(kern_name, float, double)                                                        \
  CAST_REF_DEF(kern_name, float, double)                                                           \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    float (*ref)(double) = kern_name##_ref;                                                        \
    CastDoublePrecisionTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<float>());          \
  }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2float_rd` against a table of difficult values, followed by a
 * large number of randomly generated values. The results are compared against reference function
 * which performs cast to float with rounding mode FE_DOWNWARD.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2FLOAT_TEST_DEF(__double2float_rd, FE_DOWNWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2float_rn` against a table of difficult values, followed by a
 * large number of randomly generated values. The results are compared against reference function
 * which performs cast to float.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2FLOAT_RN_TEST_DEF(__double2float_rn)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2float_ru` against a table of difficult values, followed by a
 * large number of randomly generated values. The results are compared against reference function
 * which performs cast to float with rounding mode FE_UPWARD.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2FLOAT_TEST_DEF(__double2float_ru, FE_UPWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2float_rz` against a table of difficult values, followed by a
 * large number of randomly generated values. The results are compared against reference function
 * which performs cast to float with rounding mode FE_TOWARDZERO.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_DOUBLE2FLOAT_TEST_DEF(__double2float_rz, FE_TOWARDZERO)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __double2float_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___double2float_Negative_RTC") { NegativeTestRTCWrapper<12>(kDouble2Float); }

CAST_KERNEL_DEF(__double2hiint, int, double)

int __double2hiint_ref(double arg) {
  int tmp[2];
  memcpy(tmp, &arg, sizeof(tmp));
  return tmp[1];
}

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2hiint` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function which
 * performs copy of higher part of double value to int variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___double2hiint_Positive") {
  int (*ref)(double) = __double2hiint_ref;
  CastDoublePrecisionTest(__double2hiint_kernel, ref, EqValidatorBuilderFactory<int>());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __double2hiint.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___double2hiint_Negative_RTC") { NegativeTestRTCWrapper<3>(kDouble2Hiint); }

CAST_KERNEL_DEF(__double2loint, int, double)

int __double2loint_ref(double arg) {
  int tmp[2];
  memcpy(tmp, &arg, sizeof(tmp));
  return tmp[0];
}

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double2loint` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function which
 * performs copy of lower part of double value to int variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___double2loint_Positive") {
  int (*ref)(double) = __double2loint_ref;
  CastDoublePrecisionTest(__double2loint_kernel, ref, EqValidatorBuilderFactory<int>());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __double2loint.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___double2loint_Negative_RTC") { NegativeTestRTCWrapper<3>(kDouble2Loint); }

CAST_KERNEL_DEF(__double_as_longlong, long long int, double)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__double_as_longlong` against a table of difficult values, followed by a
 * large number of randomly generated values. The results are compared against reference function
 * which performs copy of double value to long long int variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___double_as_longlong_Positive") {
  long long int (*ref)(double) = type2_as_type1_ref<long long int, double>;
  CastDoublePrecisionTest(__double_as_longlong_kernel, ref,
                          EqValidatorBuilderFactory<long long int>());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __double_as_longlong.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_double_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___double_as_longlong_Negative_RTC") {
  NegativeTestRTCWrapper<3>(kDoubleAsLonglong);
}

/**
 * End doxygen group MathTest.
 * @}
 */
