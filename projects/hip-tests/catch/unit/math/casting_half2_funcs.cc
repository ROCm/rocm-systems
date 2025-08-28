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
#include "casting_half2_common.hh"

/**
 * @addtogroup HalfPrecisionCastingHalf2 HalfPrecisionCastingHalf2
 * @{
 * @ingroup MathTest
 */

/********** half -> half2 **********/

CAST_KERNEL_DEF(__half2half2, __half2, Float16)

static __half2 __half2half2_ref(Float16 x) { return __half2{x, x}; }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2half2` for all possible inputs. The results are compared against
 * reference function which returns __half2 value created from one __half value.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___half2half2_Accuracy_Positive") {
  UnaryHalfPrecisionTest(__half2half2_kernel, __half2half2_ref,
                         Half2ValidatorBuilderFactory(EqValidatorBuilderFactory<Float16>()));
}

CAST_BINARY_KERNEL_DEF(make_half2, __half2, Float16)

static __half2 make_half2_ref(Float16 x, Float16 y) { return __half2{x, y}; }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `make_half2` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function which
 * returns __half2 value created from two __half values.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_make_half2_Accuracy_Positive") {
  BinaryFloatingPointTest(make_half2_kernel, make_half2_ref,
                          Half2ValidatorBuilderFactory(EqValidatorBuilderFactory<Float16>()));
}

CAST_BINARY_KERNEL_DEF(__halves2half2, __half2, Float16)

static __half2 __halves2half2_ref(Float16 x, Float16 y) { return __half2{x, y}; }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__halves2half2` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function which
 * returns __half2 value created from two __half values.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___halves2half2_Accuracy_Positive") {
  BinaryFloatingPointTest(__halves2half2_kernel, __halves2half2_ref,
                          Half2ValidatorBuilderFactory(EqValidatorBuilderFactory<Float16>()));
}

/********** half2 -> half **********/


CAST_HALF2_KERNEL_DEF(__low2half, Float16)

static Float16 __low2half_ref(Float16 x) { return x; }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__low2half` for all possible inputs. The results are compared against
 * reference function which returns __half value created from lower __half2 element.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___low2half_Accuracy_Positive") {
  UnaryHalfPrecisionTest(__low2half_kernel, __low2half_ref, EqValidatorBuilderFactory<Float16>());
}

CAST_HALF2_KERNEL_DEF(__high2half, Float16)

static Float16 __high2half_ref(Float16 x) { return -x; }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__high2half` for all possible inputs. The results are compared against
 * reference function which returns __half value created from higher __half2 element.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___high2half_Accuracy_Positive") {
  UnaryHalfPrecisionTest(__high2half_kernel, __high2half_ref, EqValidatorBuilderFactory<Float16>());
}

/********** half2 -> half2 **********/

CAST_HALF2_KERNEL_DEF(__low2half2, __half2)

static __half2 __low2half2_ref(Float16 x) { return __half2{x, x}; }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__low2half2` for all possible inputs. The results are compared against
 * reference function which returns __half2 value created from two lower __half2 elements.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___low2half2_Accuracy_Positive") {
  UnaryHalfPrecisionTest(__low2half2_kernel, __low2half2_ref,
                         Half2ValidatorBuilderFactory(EqValidatorBuilderFactory<Float16>()));
}

CAST_HALF2_KERNEL_DEF(__high2half2, __half2)

static __half2 __high2half2_ref(Float16 x) { return __half2{-x, -x}; }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__high2half2` for all possible inputs. The results are compared against
 * reference function which returns __half2 value created from two higher __half2 elements.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___high2half2_Accuracy_Positive") {
  UnaryHalfPrecisionTest(__high2half2_kernel, __high2half2_ref,
                         Half2ValidatorBuilderFactory(EqValidatorBuilderFactory<Float16>()));
}

CAST_HALF2_KERNEL_DEF(__lowhigh2highlow, __half2)

static __half2 __lowhigh2highlow_ref(Float16 x) { return __half2{-x, x}; }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__lowhigh2highlow` for all possible inputs. The results are compared
 * against reference function which returns __half2 value created from higher and lower __half2
 * elements.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___lowhigh2highlow_Accuracy_Positive") {
  UnaryHalfPrecisionTest(__lowhigh2highlow_kernel, __lowhigh2highlow_ref,
                         Half2ValidatorBuilderFactory(EqValidatorBuilderFactory<Float16>()));
}

CAST_BINARY_HALF2_KERNEL_DEF(__lows2half2, __half2)

static __half2 __lows2half2_ref(Float16 x, Float16 y) { return __half2{x, y}; }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__lows2half2` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function which
 * returns __half2 value created from lower elements of two __half2 values.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___lows2half2_Accuracy_Positive") {
  BinaryFloatingPointTest(__lows2half2_kernel, __lows2half2_ref,
                          Half2ValidatorBuilderFactory(EqValidatorBuilderFactory<Float16>()));
}

CAST_BINARY_HALF2_KERNEL_DEF(__highs2half2, __half2)

static __half2 __highs2half2_ref(Float16 x, Float16 y) { return __half2{-x, -y}; }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__highs2half2` against a table of difficult values, followed by a large
 * number of randomly generated values. The results are compared against reference function which
 * returns __half2 value created from higher elements of two __half2 values.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___highs2half2_Accuracy_Positive") {
  BinaryFloatingPointTest(__highs2half2_kernel, __highs2half2_ref,
                          Half2ValidatorBuilderFactory(EqValidatorBuilderFactory<Float16>()));
}

/********** float -> half2 **********/

CAST_KERNEL_DEF(__float2half2_rn, __half2, float)

static __half2 __float2half2_rn_ref(float x) {
  return __half2{static_cast<Float16>(x), static_cast<Float16>(x)};
}

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2half2_rn` for all possible inputs. The results are compared
 * against reference function which returns __half2 value created from one casted float value.
 * elements.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float2half2_rn_Accuracy_Positive") {
  UnarySinglePrecisionTest(__float2half2_rn_kernel, __float2half2_rn_ref,
                           Half2ValidatorBuilderFactory(EqValidatorBuilderFactory<Float16>()));
}

CAST_BINARY_KERNEL_DEF(__floats2half2_rn, __half2, float)

static __half2 __floats2half2_rn_ref(float x, float y) {
  return __half2{static_cast<Float16>(x), static_cast<Float16>(y)};
}

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__floats2half2_rn` against a table of difficult values, followed by a
 * large number of randomly generated values. The results are compared against reference function
 * which returns __half2 value created from two casted float values.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___floats2half2_rn_Accuracy_Positive") {
  BinaryFloatingPointTest(__floats2half2_rn_kernel, __floats2half2_rn_ref,
                          Half2ValidatorBuilderFactory(EqValidatorBuilderFactory<Float16>()));
}

/********** float2 -> half2 **********/

__global__ void __float22half2_rn_kernel(__half2* const ys, const size_t num_xs, float* const xs) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (auto i = tid; i < num_xs; i += stride) {
    ys[i] = __float22half2_rn(make_float2(xs[i], -xs[i]));
  }
}

static __half2 __float22half2_rn_ref(float x) {
  return __half2{static_cast<Float16>(x), static_cast<Float16>(-x)};
}

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float22half2_rn` for all possible inputs. The results are compared
 * against reference function which returns __half2 value created from two casted float values.
 * elements.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float22half2_rn_Accuracy_Positive") {
  UnarySinglePrecisionTest(__float22half2_rn_kernel, __float22half2_rn_ref,
                           Half2ValidatorBuilderFactory(EqValidatorBuilderFactory<Float16>()));
}

/********** half2 -> float **********/

CAST_HALF2_KERNEL_DEF(__low2float, float)

static float __low2float_ref(Float16 x) { return static_cast<float>(x); }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__low2float` for all possible inputs. The results are compared
 * against reference function which returns float value created from lower __half2 element.
 * elements.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___low2float_Accuracy_Positive") {
  UnaryHalfPrecisionTest(__low2float_kernel, __low2float_ref, EqValidatorBuilderFactory<float>());
}

CAST_HALF2_KERNEL_DEF(__high2float, float)

static float __high2float_ref(Float16 x) { return static_cast<float>(-x); }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__high2float` for all possible inputs. The results are compared
 * against reference function which returns float value created from higher __half2 element.
 * elements.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___high2float_Accuracy_Positive") {
  UnaryHalfPrecisionTest(__high2float_kernel, __high2float_ref, EqValidatorBuilderFactory<float>());
}

/********** half2 -> float2 **********/

CAST_HALF2_KERNEL_DEF(__half22float2, float2)

static float2 __half22float2_ref(Float16 x) {
  return make_float2(static_cast<float>(x), static_cast<float>(-x));
}

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half22float2` for all possible inputs. The results are compared against
 * reference function which returns float2 value created from casted elements of one __half2 value.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half2_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___half22float2_Accuracy_Positive") {
  UnaryHalfPrecisionTest(__half22float2_kernel, __half22float2_ref,
                         Float2ValidatorBuilderFactory(EqValidatorBuilderFactory<float>()));
}

/**
 * End doxygen group MathTest.
 * @}
 */
