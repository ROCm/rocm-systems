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
#include "casting_float_negative_kernels_rtc.hh"

/**
 * @addtogroup CastingFloatType CastingFloatType
 * @{
 * @ingroup MathTest
 */

#define CAST_FLOAT2INT_TEST_DEF(kern_name, T, ref_func)                                            \
  CAST_KERNEL_DEF(kern_name, T, float)                                                             \
  CAST_F2I_REF_DEF(kern_name, T, float, ref_func)                                                  \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(float) = kern_name##_ref;                                                             \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>(),         \
                                  std::numeric_limits<float>::lowest(),                            \
                                  std::numeric_limits<float>::max());                              \
  }

#define CAST_FLOAT2INT_RZ_TEST_DEF(kern_name, T)                                                   \
  CAST_KERNEL_DEF(kern_name, T, float)                                                             \
  CAST_F2I_RZ_REF_DEF(kern_name, T, float)                                                         \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(float) = kern_name##_ref;                                                             \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>(),         \
                                  std::numeric_limits<float>::lowest(),                            \
                                  std::numeric_limits<float>::max());                              \
  }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2int_rd` for all possible inputs. The results are compared against
 * reference function `std::floor`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2INT_TEST_DEF(__float2int_rd, int, std::floor)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2int_rn` for all possible inputs. The results are compared against
 * reference function `std::rint`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2INT_TEST_DEF(__float2int_rn, int, std::rint)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2int_ru` for all possible inputs. The results are compared against
 * reference function `std::ceil`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2INT_TEST_DEF(__float2int_ru, int, std::ceil)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2int_rz` for all possible inputs. The results are compared against
 * reference function `std::trunc`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2INT_TEST_DEF(__float2int_rz, int, std::trunc)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __float2int_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float2int_Negative_RTC") { NegativeTestRTCWrapper<12>(kFloat2Int); }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2uint_rd` for all possible inputs. The results are compared
 * against reference function `std::floor`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2INT_TEST_DEF(__float2uint_rd, unsigned int, std::floor)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2uint_rn` for all possible inputs. The results are compared
 * against reference function `std::rint`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2INT_TEST_DEF(__float2uint_rn, unsigned int, std::rint)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2uint_ru` for all possible inputs. The results are compared
 * against reference function `std::ceil`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2INT_TEST_DEF(__float2uint_ru, unsigned int, std::ceil)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2uint_rz` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function which
 * performs cast to unsigned int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2INT_RZ_TEST_DEF(__float2uint_rz, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __float2uint_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float2uint_Negative_RTC") { NegativeTestRTCWrapper<12>(kFloat2Uint); }

#define CAST_FLOAT2LL_TEST_DEF(kern_name, T, ref_func)                                             \
  CAST_KERNEL_DEF(kern_name, T, float)                                                             \
  CAST_F2I_REF_DEF(kern_name, T, float, ref_func)                                                  \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(float) = kern_name##_ref;                                                             \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>(),         \
                                  static_cast<float>(std::numeric_limits<T>::min()),               \
                                  static_cast<float>(std::numeric_limits<T>::max()));              \
  }

#define CAST_FLOAT2LL_RZ_TEST_DEF(kern_name, T)                                                    \
  CAST_KERNEL_DEF(kern_name, T, float)                                                             \
  CAST_F2I_RZ_REF_DEF(kern_name, T, float)                                                         \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(float) = kern_name##_ref;                                                             \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>(),         \
                                  static_cast<float>(std::numeric_limits<T>::min()),               \
                                  static_cast<float>(std::numeric_limits<T>::max()));              \
  }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2ll_rd` for all possible inputs. The results are compared against
 * reference function `std::floor`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2LL_TEST_DEF(__float2ll_rd, long long int, std::floor)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2ll_rn` for all possible inputs between lowest and maximal long
 * long int value. The results are compared against reference function `std::rint`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2LL_TEST_DEF(__float2ll_rn, long long int, std::rint)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2ll_ru` for all possible inputs between lowest and maximal long
 * long int value. The results are compared against reference function `std::ceil`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2LL_TEST_DEF(__float2ll_ru, long long int, std::ceil)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2ll_rz` for all possible inputs between lowest and maximal long
 * long int value. The results are compared against reference function which performs cast to long
 * long int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2LL_RZ_TEST_DEF(__float2ll_rz, long long int)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __float2ll_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float2ll_Negative_RTC") { NegativeTestRTCWrapper<12>(kFloat2LL); }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2ull_rd` for all possible inputs between lowest and maximal
 * unsigned long long int value. The results are compared against reference function `std::floor`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2LL_TEST_DEF(__float2ull_rd, unsigned long long int, std::floor)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2ull_rn` for all possible inputs between lowest and maximal
 * unsigned long long int value. The results are compared against reference function `std::rint`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2LL_TEST_DEF(__float2ull_rn, unsigned long long int, std::rint)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2ull_ru` for all possible inputs between lowest and maximal
 * unsigned long long int value. The results are compared against reference function `std::ceil`.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2LL_TEST_DEF(__float2ull_ru, unsigned long long int, std::ceil)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2ll_rz` for all possible inputs between lowest and maximal
 * unsigned long long int value. The results are compared against reference function which performs
 * cast to unsigned long long int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2LL_RZ_TEST_DEF(__float2ull_rz, unsigned long long int)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __float2ull_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float2ull_Negative_RTC") { NegativeTestRTCWrapper<12>(kFloat2ULL); }

CAST_KERNEL_DEF(__float_as_int, int, float)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float_as_int` for all possible inputs. The results are compared against
 * reference function which performs copy of float value to int variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float_as_int_Positive") {
  int (*ref)(float) = type2_as_type1_ref<int, float>;
  UnarySinglePrecisionTest(__float_as_int_kernel, ref, EqValidatorBuilderFactory<int>());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __float_as_int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float_as_int_Negative_RTC") { NegativeTestRTCWrapper<3>(kFloatAsInt); }

CAST_KERNEL_DEF(__float_as_uint, unsigned int, float)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float_as_uint` for all possible inputs. The results are compared
 * against reference function which performs copy of float value to unsigned int variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float_as_uint_Positive") {
  unsigned int (*ref)(float) = type2_as_type1_ref<unsigned int, float>;
  UnarySinglePrecisionTest(__float_as_uint_kernel, ref, EqValidatorBuilderFactory<unsigned int>());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __float_as_uint.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float_as_uint_Negative_RTC") { NegativeTestRTCWrapper<3>(kFloatAsUint); }

/**
 * End doxygen group MathTest.
 * @}
 */
