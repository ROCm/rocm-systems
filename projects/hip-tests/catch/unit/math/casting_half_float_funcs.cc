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
 * @addtogroup HalfPrecisionCastingFloat HalfPrecisionCastingFloat
 * @{
 * @ingroup MathTest
 */

#define CAST_FLOAT2HALF_TEST_DEF(kern_name, round_dir)                                             \
  CAST_KERNEL_DEF(kern_name, Float16, float)                                                       \
  CAST_RND_REF_DEF(kern_name, Float16, float, round_dir)                                           \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Limited_Positive") {                              \
    Float16 (*ref)(float) = kern_name##_ref;                                                       \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<Float16>(),   \
                                  std::numeric_limits<float>::min(), 0.f);                         \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<Float16>(),   \
                                  0.0001f, std::numeric_limits<float>::max());                     \
  }

#define CAST_FLOAT2HALF_RN_TEST_DEF(kern_name)                                                     \
  CAST_KERNEL_DEF(kern_name, Float16, float)                                                       \
  CAST_REF_DEF(kern_name, Float16, float)                                                          \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive") {                                      \
    Float16 (*ref)(float) = kern_name##_ref;                                                       \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<Float16>(),   \
                                  std::numeric_limits<float>::min(),                               \
                                  std::numeric_limits<float>::max());                              \
  }

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2half_rd` for all possible inputs apart from very small positive
 * values. Rounding behaviour is not correct for host functions for this range. The results are
 * compared against reference function which performs float cast to __half with FE_DOWNWARD rounding
 * mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2HALF_TEST_DEF(__float2half_rd, FE_DOWNWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2half_rn` for all possible inputs. The results are compared
 * against reference function which performs float cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2HALF_RN_TEST_DEF(__float2half_rn)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2half` for all possible inputs. The results are compared against
 * reference function which performs float cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2HALF_RN_TEST_DEF(__float2half)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2half_ru` for all possible inputs apart from very small positive
 * values. Rounding behaviour is not correct for host functions for this range. The results are
 * compared against reference function which performs float cast to __half with FE_UPWARD rounding
 * mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2HALF_TEST_DEF(__float2half_ru, FE_UPWARD)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2half_rz` for all possible inputs apart from very small positive
 * values. Rounding behaviour is not correct for host functions for this range. The results are
 * compared against reference function which performs float cast to __half with FE_TOWARDZERO
 * rounding mode.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2HALF_TEST_DEF(__float2half_rz, FE_TOWARDZERO)

/**
 * Test Description
 * ------------------------
 *    - Sanity test that checks `__float2half_rd` for very small positive values.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float2half_rd_SmallVals_Sanity_Positive") {
  const float input[] = {0.8859e-06f, 1.5454e-07f, 6.5955e-08f, 2.7955e-08f,
                         3.7956e-09f, 4.8995e-10f, 5.7997e-15f, 6.2117e-20f,
                         7.4999e-25f, 8.9999e-30f, 9.0001e-35f};
  const Float16 reference[] = {8.34465e-07, 1.19209e-07, 5.96046e-08, 0, 0, 0, 0, 0, 0, 0, 0};
  LinearAllocGuard<float> input_dev{LinearAllocs::hipMalloc, sizeof(float)};
  LinearAllocGuard<Float16> out(LinearAllocs::hipMallocManaged, sizeof(Float16));


  for (int i = 0; i < 11; ++i) {
    HIP_CHECK(hipMemcpy(input_dev.ptr(), input + i, sizeof(float), hipMemcpyHostToDevice));

    __float2half_rd_kernel<<<1, 1>>>(out.ptr(), 1, input_dev.ptr());
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(out.ptr()[0] == reference[i]);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test that checks `__float2half_ru` for very small positive values.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float2half_ru_SmallVals_Sanity_Positive") {
  const float input[] = {0.8859e-06f, 1.5454e-07f, 6.5955e-08f, 2.7955e-08f,
                         3.7956e-09f, 4.8995e-10f, 5.7997e-15f, 6.2117e-20f,
                         7.4999e-25f, 8.9999e-30f, 9.0001e-35f};
  const Float16 reference[] = {8.9407e-07,  1.78814e-07, 1.19209e-07, 5.96046e-08,
                               5.96046e-08, 5.96046e-08, 5.96046e-08, 5.96046e-08,
                               5.96046e-08, 5.96046e-08, 5.96046e-08};
  LinearAllocGuard<float> input_dev{LinearAllocs::hipMalloc, sizeof(float)};
  LinearAllocGuard<Float16> out(LinearAllocs::hipMallocManaged, sizeof(Float16));


  for (int i = 0; i < 11; ++i) {
    HIP_CHECK(hipMemcpy(input_dev.ptr(), input + i, sizeof(float), hipMemcpyHostToDevice));

    __float2half_ru_kernel<<<1, 1>>>(out.ptr(), 1, input_dev.ptr());
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(out.ptr()[0] == reference[i]);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test that checks `__float2half_rz` for very small positive values.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___float2half_rz_SmallVals_Sanity_Positive") {
  const float input[] = {0.8859e-06f, 1.5454e-07f, 6.5955e-08f, 2.7955e-08f,
                         3.7956e-09f, 4.8995e-10f, 5.7997e-15f, 6.2117e-20f,
                         7.4999e-25f, 8.9999e-30f, 9.0001e-35f};
  const Float16 reference[] = {8.34465e-07, 1.19209e-07, 5.96046e-08, 0, 0, 0, 0, 0, 0, 0, 0};
  LinearAllocGuard<float> input_dev{LinearAllocs::hipMalloc, sizeof(float)};
  LinearAllocGuard<Float16> out(LinearAllocs::hipMallocManaged, sizeof(Float16));


  for (int i = 0; i < 11; ++i) {
    HIP_CHECK(hipMemcpy(input_dev.ptr(), input + i, sizeof(float), hipMemcpyHostToDevice));

    __float2half_rz_kernel<<<1, 1>>>(out.ptr(), 1, input_dev.ptr());
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(out.ptr()[0] == reference[i]);
  }
}

CAST_KERNEL_DEF(__half2float, float, Float16)
CAST_REF_DEF(__half2float, float, Float16)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2float` for all possible inputs. The results are compared against
 * reference function which performs __half cast to float.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___half2float_Accuracy_Positive") {
  float (*ref)(Float16) = __half2float_ref;
  UnaryHalfPrecisionTest(__half2float_kernel, ref, EqValidatorBuilderFactory<float>());
}

/**
 * End doxygen group MathTest.
 * @}
 */
