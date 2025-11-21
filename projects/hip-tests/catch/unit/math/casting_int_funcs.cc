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
#include "casting_int_negative_kernels_rtc.hh"

/**
 * @addtogroup CastingIntTypes CastingIntTypes
 * @{
 * @ingroup MathTest
 */

#define CAST_INT2FLOAT_TEST_DEF(kern_name, T1, T2, round_dir)                                      \
  CAST_KERNEL_DEF(kern_name, T1, T2)                                                               \
  CAST_RND_REF_DEF(kern_name, T1, T2, round_dir)                                                   \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T1 (*ref)(T2) = kern_name##_ref;                                                               \
    CastIntRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T1>());                    \
  }

#define CAST_INT2FLOAT_RN_TEST_DEF(kern_name, T1, T2)                                              \
  CAST_KERNEL_DEF(kern_name, T1, T2)                                                               \
  CAST_REF_DEF(kern_name, T1, T2)                                                                  \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T1 (*ref)(T2) = kern_name##_ref;                                                               \
    CastIntRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T1>());                    \
  }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__int2float_rd` for all possible inputs. The results are compared against
 * reference function which performs cast to float with FE_DOWNWARD rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2FLOAT_TEST_DEF(__int2float_rd, float, int, FE_DOWNWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__int2float_rn` for all possible inputs. The results are compared against
 * reference function which performs cast to float.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2FLOAT_RN_TEST_DEF(__int2float_rn, float, int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__int2float_ru` for all possible inputs. The results are compared against
 * reference function which performs cast to float with FE_UPWARD rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2FLOAT_TEST_DEF(__int2float_ru, float, int, FE_UPWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__int2float_rz` for all possible inputs. The results are compared against
 * reference function which performs cast to float with FE_TOWARDZERO rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2FLOAT_TEST_DEF(__int2float_rz, float, int, FE_TOWARDZERO)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __int2float_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_int2float___Negative_RTC") { NegativeTestRTCWrapper<12>(kInt2Float); }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__uint2float_rd` for all possible inputs. The results are compared
 * against reference function which performs cast to float with FE_DOWNWARD rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2FLOAT_TEST_DEF(__uint2float_rd, float, unsigned int, FE_DOWNWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__uint2float_rn` for all possible inputs. The results are compared
 * against reference function which performs cast to float.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2FLOAT_RN_TEST_DEF(__uint2float_rn, float, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__uint2float_ru` for all possible inputs. The results are compared
 * against reference function which performs cast to float with FE_UPWARD rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2FLOAT_TEST_DEF(__uint2float_ru, float, unsigned int, FE_UPWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__uint2float_rz` for all possible inputs. The results are compared
 * against reference function which performs cast to float with FE_TOWARDZERO rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2FLOAT_TEST_DEF(__uint2float_rz, float, unsigned int, FE_TOWARDZERO)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __uint2float_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___uint2float_Negative_RTC") { NegativeTestRTCWrapper<12>(kUint2Float); }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__int2double_rn` for all possible inputs. The results are compared
 * against reference function which performs cast to double.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2FLOAT_RN_TEST_DEF(__int2double_rn, double, int)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __int2double_rn.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___int2double_Negative_RTC") { NegativeTestRTCWrapper<3>(kInt2Double); }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__uint2double_rn` for all possible inputs. The results are compared
 * against reference function which performs cast to double.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2FLOAT_RN_TEST_DEF(__uint2double_rn, double, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __uint2double_rn.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___uint2double_Negative_RTC") { NegativeTestRTCWrapper<3>(kUint2Double); }

#define CAST_LL2FLOAT_TEST_DEF(kern_name, T1, T2, round_dir)                                       \
  CAST_KERNEL_DEF(kern_name, T1, T2)                                                               \
  CAST_RND_REF_DEF(kern_name, T1, T2, round_dir)                                                   \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T1 (*ref)(T2) = kern_name##_ref;                                                               \
    CastIntBruteForceTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T1>());               \
  }

#define CAST_LL2FLOAT_RN_TEST_DEF(kern_name, T1, T2)                                               \
  CAST_KERNEL_DEF(kern_name, T1, T2)                                                               \
  CAST_REF_DEF(kern_name, T1, T2)                                                                  \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T1 (*ref)(T2) = kern_name##_ref;                                                               \
    CastIntBruteForceTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T1>());               \
  }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ll2float_rd` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to float with FE_DOWNWARD
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_TEST_DEF(__ll2float_rd, float, long long int, FE_DOWNWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ll2float_rn` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to float.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_RN_TEST_DEF(__ll2float_rn, float, long long int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ll2float_ru` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to float with FE_UPWARD
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_TEST_DEF(__ll2float_ru, float, long long int, FE_UPWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ll2float_rz` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to float with FE_TOWARDZERO
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_TEST_DEF(__ll2float_rz, float, long long int, FE_TOWARDZERO)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __ll2float_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___ll2float_Negative_RTC") { NegativeTestRTCWrapper<12>(kLL2Float); }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ull2float_rd` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to float with FE_DOWNWARD
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_TEST_DEF(__ull2float_rd, float, unsigned long long int, FE_DOWNWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ull2float_rn` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to float.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_RN_TEST_DEF(__ull2float_rn, float, unsigned long long int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ull2float_ru` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to float with FE_UPWARD
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_TEST_DEF(__ull2float_ru, float, unsigned long long int, FE_UPWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ull2float_rz` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to float with FE_TOWARDZERO
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_TEST_DEF(__ull2float_rz, float, unsigned long long int, FE_TOWARDZERO)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __ull2float_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___ull2float_Negative_RTC") { NegativeTestRTCWrapper<12>(kULL2Float); }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ll2double_rd` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to double with FE_DOWNWARD
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_TEST_DEF(__ll2double_rd, double, long long int, FE_DOWNWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ll2double_rn` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to double.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_RN_TEST_DEF(__ll2double_rn, double, long long int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ll2double_ru` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to double with FE_UPWARD
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_TEST_DEF(__ll2double_ru, double, long long int, FE_UPWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ll2double_rz` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to double with FE_TOWARDZERO
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_TEST_DEF(__ll2double_rz, double, long long int, FE_TOWARDZERO)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __ll2double_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___ll2double_Negative_RTC") { NegativeTestRTCWrapper<12>(kLL2Double); }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ull2double_rd` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to double with FE_DOWNWARD
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_TEST_DEF(__ull2double_rd, double, unsigned long long int, FE_DOWNWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ull2double_rn` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to double.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_RN_TEST_DEF(__ull2double_rn, double, unsigned long long int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ull2double_ru` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to double with FE_UPWARD
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_TEST_DEF(__ull2double_ru, double, unsigned long long int, FE_UPWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ull2double_rz` against a large number of randomly generated values. The
 * results are compared against reference function which performs cast to double with FE_TOWARDZERO
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2FLOAT_TEST_DEF(__ull2double_rz, double, unsigned long long int, FE_TOWARDZERO)

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __ull2double_[rd,rn,ru,rz].
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___ull2double_Negative_RTC") { NegativeTestRTCWrapper<12>(kULL2Double); }

CAST_KERNEL_DEF(__int_as_float, float, int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__int_as_float` for all possible inputs. The results are compared against
 * reference function which performs copy of int value to float variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___int_as_float_Positive") {
  float (*ref)(int) = type2_as_type1_ref<float, int>;
  CastIntRangeTest(__int_as_float_kernel, ref, EqValidatorBuilderFactory<float>());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __int_as_float.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___int_as_float_Negative_RTC") { NegativeTestRTCWrapper<3>(kIntAsFloat); }

CAST_KERNEL_DEF(__uint_as_float, float, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__uint_as_float` for all possible inputs. The results are compared
 * against reference function which performs copy of unsigned int value to float variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___uint_as_float_Positive") {
  float (*ref)(unsigned int) = type2_as_type1_ref<float, unsigned int>;
  CastIntRangeTest(__uint_as_float_kernel, ref, EqValidatorBuilderFactory<float>());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __uint_as_float.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___uint_as_float_Negative_RTC") { NegativeTestRTCWrapper<3>(kUintAsFloat); }

CAST_KERNEL_DEF(__longlong_as_double, double, long long int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__longlong_as_double` against a large number of randomly generated
 * values. The results are compared against reference function which performs copy of long long int
 * value to double variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___longlong_as_double_Positive") {
  double (*ref)(long long int) = type2_as_type1_ref<double, long long int>;
  CastIntBruteForceTest(__longlong_as_double_kernel, ref, EqValidatorBuilderFactory<double>());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __longlong_as_double.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___longlong_as_double_Negative_RTC") {
  NegativeTestRTCWrapper<3>(kLonglongAsDouble);
}

__global__ void __hiloint2double_kernel(double* const ys, const size_t num_xs, int* const x1s,
                                        int* const x2s) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (auto i = tid; i < num_xs; i += stride) {
    ys[i] = __hiloint2double(x1s[i], x2s[i]);
  }
}

double __hiloint2double_ref(int hi, int lo) {
  uint64_t tmp0 = (static_cast<uint64_t>(hi) << 32ull) | static_cast<uint32_t>(lo);
  double tmp1;
  memcpy(&tmp1, &tmp0, sizeof(tmp0));

  return tmp1;
}

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__hiloint2double` for all possible inputs for hi value. The results are
 * compared against reference function which performs copy of hi int value to higher part of double
 * variable and copy of lo int value to lower part of double variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___hiloint2double_Positive") {
  double (*ref)(int, int) = __hiloint2double_ref;
  CastBinaryIntRangeTest(__hiloint2double_kernel, ref, EqValidatorBuilderFactory<double>());
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for __hiloint2double.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___hiloint2double_Negative_RTC") { NegativeTestRTCWrapper<5>(kHilo2Double); }

/**
 * End doxygen group MathTest.
 * @}
 */
