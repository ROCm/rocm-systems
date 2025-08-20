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

#define CAST_HALF2INT_RN_TEST_DEF(kern_name, T)                                                    \
  CAST_KERNEL_DEF(kern_name, T, Float16)                                                           \
  CAST_F2I_RZ_REF_DEF(kern_name, T, Float16)                                                       \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive") {                                      \
    T (*ref)(Float16) = kern_name##_ref;                                                           \
    CastUnaryHalfPrecisionTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>());           \
  }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2int_rn` for all possible inputs. The results are compared against
 * reference function which performs __half cast to int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2int_rn, int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2int_rz` for all possible inputs. The results are compared against
 * reference function which performs __half cast to int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2int_rz, int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2int_rd` for all possible inputs. The results are compared against
 * reference function which performs __half cast to int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2int_rd, int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2int_ru` for all possible inputs. The results are compared against
 * reference function which performs __half cast to int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2int_ru, int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2uint_rn` for all possible inputs. The results are compared against
 * reference function which performs __half cast to unsigned int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2uint_rn, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2uint_rz` for all possible inputs. The results are compared against
 * reference function which performs __half cast to unsigned int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2uint_rz, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2uint_rd` for all possible inputs. The results are compared against
 * reference function which performs __half cast to unsigned int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2uint_rd, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2uint_ru` for all possible inputs. The results are compared against
 * reference function which performs __half cast to unsigned int.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2uint_ru, unsigned int)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2short_rn` for all possible inputs. The results are compared
 * against reference function which performs __half cast to short.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2short_rn, short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2short_rz` for all possible inputs. The results are compared
 * against reference function which performs __half cast to short.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2short_rz, short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2short_rd` for all possible inputs. The results are compared
 * against reference function which performs __half cast to short.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2short_rd, short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2short_ru` for all possible inputs. The results are compared
 * against reference function which performs __half cast to short.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2short_ru, short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2ushort_rn` for all possible inputs. The results are compared
 * against reference function which performs __half cast to unsigned short.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2ushort_rn, unsigned short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2ushort_rz` for all possible inputs. The results are compared
 * against reference function which performs __half cast to unsigned short.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2ushort_rz, unsigned short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2ushort_rd` for all possible inputs. The results are compared
 * against reference function which performs __half cast to unsigned short.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2ushort_rd, unsigned short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2ushort_ru` for all possible inputs. The results are compared
 * against reference function which performs __half cast to unsigned short.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2ushort_ru, unsigned short)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2ll_rn` for all possible inputs. The results are compared against
 * reference function which performs __half cast to long long.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2ll_rn, long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2ll_rz` for all possible inputs. The results are compared against
 * reference function which performs __half cast to long long.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2ll_rz, long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2ll_rd` for all possible inputs. The results are compared against
 * reference function which performs __half cast to long long.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2ll_rd, long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2ll_ru` for all possible inputs. The results are compared against
 * reference function which performs __half cast to long long.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2ll_ru, long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2ull_rn` for all possible inputs. The results are compared against
 * reference function which performs __half cast to unsigned long long.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2ull_rn, unsigned long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2ull_rz` for all possible inputs. The results are compared against
 * reference function which performs __half cast to unsigned long long.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2ull_rz, unsigned long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2ull_rd` for all possible inputs. The results are compared against
 * reference function which performs __half cast to unsigned long long.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2ull_rd, unsigned long long)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2ull_ru` for all possible inputs. The results are compared against
 * reference function which performs __half cast to unsigned long long.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_HALF2INT_RN_TEST_DEF(__half2ull_ru, unsigned long long)

CAST_KERNEL_DEF(__half_as_short, short, Float16)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half_as_short` for all possible inputs. The results are compared
 * against reference function which performs copy of __half value to short variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___half_as_short_Accuracy_Positive") {
  short (*ref)(Float16) = type2_as_type1_ref<short, Float16>;
  CastUnaryHalfPrecisionTest(__half_as_short_kernel, ref, EqValidatorBuilderFactory<short>());
}

CAST_KERNEL_DEF(__half_as_ushort, unsigned short, Float16)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half_as_ushort` for all possible inputs. The results are compared
 * against reference function which performs copy of __half value to unsigned short variable.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2int_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___half_as_ushort_Accuracy_Positive") {
  unsigned short (*ref)(Float16) = type2_as_type1_ref<unsigned short, Float16>;
  CastUnaryHalfPrecisionTest(__half_as_ushort_kernel, ref,
                             EqValidatorBuilderFactory<unsigned short>());
}

/**
 * End doxygen group MathTest.
 * @}
 */
