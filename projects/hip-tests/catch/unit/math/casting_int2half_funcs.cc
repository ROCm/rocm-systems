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
#include "casting_common.hh"

/**
 * @addtogroup HalfPrecisionCastingIntTypes HalfPrecisionCastingIntTypes
 * @{
 * @ingroup MathTest
 */

#define CAST_INT2HALF_RN_TEST_DEF(kern_name, T)                                                    \
  CAST_KERNEL_DEF(kern_name, Float16, T)                                                           \
  CAST_REF_DEF(kern_name, Float16, T)                                                              \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive") {                                      \
    Float16 (*ref)(T) = kern_name##_ref;                                                           \
    CastIntRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<Float16>());               \
  }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__int2half_rn` for all possible inputs. The results are compared against
 * reference function which performs int cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__int2half_rn, int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__int2half_rz` for all possible inputs. The results are compared against
 * reference function which performs int cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__int2half_rz, int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__int2half_rd` for all possible inputs. The results are compared against
 * reference function which performs int cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__int2half_rd, int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__int2half_ru` for all possible inputs. The results are compared against
 * reference function which performs int cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__int2half_ru, int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__uint2half_rn` for all possible inputs. The results are compared against
 * reference function which performs unsigned int cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__uint2half_rn, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__uint2half_rz` for all possible inputs. The results are compared against
 * reference function which performs unsigned int cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__uint2half_rz, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__uint2half_rd` for all possible inputs. The results are compared against
 * reference function which performs unsigned int cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__uint2half_rd, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__uint2half_ru` for all possible inputs. The results are compared against
 * reference function which performs unsigned int cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__uint2half_ru, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__short2half_rn` for all possible inputs. The results are compared
 * against reference function which performs short cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__short2half_rn, short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__short2half_rz` for all possible inputs. The results are compared
 * against reference function which performs short cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__short2half_rz, short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__short2half_rd` for all possible inputs. The results are compared
 * against reference function which performs short cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__short2half_rd, short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__short2half_ru` for all possible inputs. The results are compared
 * against reference function which performs short cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__short2half_ru, short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ushort2half_rn` for all possible inputs. The results are compared
 * against reference function which performs unsigned short cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__ushort2half_rn, unsigned short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ushort2half_rz` for all possible inputs. The results are compared
 * against reference function which performs unsigned short cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__ushort2half_rz, unsigned short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ushort2half_rd` for all possible inputs. The results are compared
 * against reference function which performs unsigned short cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__ushort2half_rd, unsigned short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ushort2half_ru` for all possible inputs. The results are compared
 * against reference function which performs unsigned short cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_INT2HALF_RN_TEST_DEF(__ushort2half_ru, unsigned short)

#define CAST_LL2HALF_TEST_DEF(kern_name, T)                                                        \
  CAST_KERNEL_DEF(kern_name, Float16, T)                                                           \
  CAST_REF_DEF(kern_name, Float16, T)                                                              \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive") {                                      \
    Float16 (*ref)(T) = kern_name##_ref;                                                           \
    CastIntBruteForceTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<Float16>());          \
  }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ll2half_rn` against a large number of randomly generated values. The
 * results are compared against reference function which performs long long cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2HALF_TEST_DEF(__ll2half_rn, long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ll2half_rz` against a large number of randomly generated values. The
 * results are compared against reference function which performs long long cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2HALF_TEST_DEF(__ll2half_rz, long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ll2half_rd` against a large number of randomly generated values. The
 * results are compared against reference function which performs long long cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2HALF_TEST_DEF(__ll2half_rd, long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ll2half_ru` against a large number of randomly generated values. The
 * results are compared against reference function which performs long long cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2HALF_TEST_DEF(__ll2half_ru, long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ull2half_rn` against a large number of randomly generated values. The
 * results are compared against reference function which performs unsigned long long cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2HALF_TEST_DEF(__ull2half_rn, unsigned long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ull2half_rz` against a large number of randomly generated values. The
 * results are compared against reference function which performs unsigned long long cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2HALF_TEST_DEF(__ull2half_rz, unsigned long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ull2half_rd` against a large number of randomly generated values. The
 * results are compared against reference function which performs unsigned long long cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2HALF_TEST_DEF(__ull2half_rd, unsigned long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ull2half_ru` against a large number of randomly generated values. The
 * results are compared against reference function which performs unsigned long long cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_LL2HALF_TEST_DEF(__ull2half_ru, unsigned long long)

CAST_KERNEL_DEF(__short_as_half, Float16, short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__short_as_half` for all possible inputs. The results are compared
 * against reference function which performs copy of short value to __half variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___short_as_half_Accuracy_Positive") {
  Float16 (*ref)(short) = type2_as_type1_ref<Float16, short>;
  CastIntBruteForceTest(__short_as_half_kernel, ref, EqValidatorBuilderFactory<Float16>());
}

CAST_KERNEL_DEF(__ushort_as_half, Float16, unsigned short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__ushort_as_half` for all possible inputs. The results are compared
 * against reference function which performs copy of unsigned short value to __half variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_int2half_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___ushort_as_half_Accuracy_Positive") {
  Float16 (*ref)(unsigned short) = type2_as_type1_ref<Float16, unsigned short>;
  CastIntBruteForceTest(__ushort_as_half_kernel, ref, EqValidatorBuilderFactory<Float16>());
}

/**
 * End doxygen group MathTest.
 * @}
 */
