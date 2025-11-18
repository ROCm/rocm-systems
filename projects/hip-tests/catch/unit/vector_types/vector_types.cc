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

#include "vector_operations_common.hh"
#if HT_AMD
#include "negative_negate_unsigned_rtc.hh"
#include "negative_bitwise_float_double_rtc.hh"
#include "negative_calculate_assign_with_value_rtc.hh"
#endif

/**
 * @addtogroup make_vector make_vector
 * @{
 * @ingroup VectorTypeTest
 */

/**
 * Test Description
 * ------------------------
 *    - Creates vectors for all supported types:
 *        -# make_char1, make_char2, make_char3, make_char4
 *        -# make_uchar1, make_uchar2, make_uchar3, make_uchar4
 *        -# make_short1, make_short2, make_short3, make_short4
 *        -# make_ushort1, make_ushort2, make_ushort3, make_ushort4
 *        -# make_int1, make_int2, make_int3, make_int4
 *        -# make_uint1, make_uint2, make_uint4, make_uint4
 *        -# make_long1, make_long2, make_long3, make_long4
 *        -# make_ulong1, make_ulong2, make_ulong3, make_ulong4
 *        -# make_longlong1, make_longlong2, make_longlong3, make_longlong4
 *        -# make_ulonglong1, make_ulonglong2, make_ulonglong3, make_ulonglong4
 *        -# make_float1, make_float2, make_float3, make_float4
 *        -# make_double1, make_double2, make_double3, make_double4
 *    - Checks that each vector type is created as expected
 *    - Calls make function from the host side
 * Test source
 * ------------------------
 *    - unit/vector_types/vector_types.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_make_vector_SanityCheck_Basic_Host", "", char1, uchar1, char2, uchar2,
                   char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2, short3, ushort3,
                   short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4, uint4, long1,
                   ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1, ulonglong1,
                   longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4, float1,
                   float2, float3, float4, double1, double2, double3, double4) {
  auto value = GetTestValue<decltype(TestType().x)>(0);
  TestType vector = MakeVectorTypeHost<TestType>(value);
  SanityCheck(vector, value);
}

/**
 * Test Description
 * ------------------------
 *    - Creates vectors for all supported types:
 *        -# make_char1, make_char2, make_char3, make_char4
 *        -# make_uchar1, make_uchar2, make_uchar3, make_uchar4
 *        -# make_short1, make_short2, make_short3, make_short4
 *        -# make_ushort1, make_ushort2, make_ushort3, make_ushort4
 *        -# make_int1, make_int2, make_int3, make_int4
 *        -# make_uint1, make_uint2, make_uint4, make_uint4
 *        -# make_long1, make_long2, make_long3, make_long4
 *        -# make_ulong1, make_ulong2, make_ulong3, make_ulong4
 *        -# make_longlong1, make_longlong2, make_longlong3, make_longlong4
 *        -# make_ulonglong1, make_ulonglong2, make_ulonglong3, make_ulonglong4
 *        -# make_float1, make_float2, make_float3, make_float4
 *        -# make_double1, make_double2, make_double3, make_double4
 *    - Checks that each vector type is created as expected
 *    - Calls make function from the device side
 * Test source
 * ------------------------
 *    - unit/vector_types/vector_types.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_make_vector_SanityCheck_Basic_Device", "", char1, uchar1, char2, uchar2,
                   char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2, short3, ushort3,
                   short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4, uint4, long1,
                   ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1, ulonglong1,
                   longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4, float1,
                   float2, float3, float4, double1, double2, double3, double4) {
  auto value = GetTestValue<decltype(TestType().x)>(0);
  TestType vector = MakeVectorTypeDevice<TestType>(value);
  SanityCheck(vector, value);
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 *    - Performs supported operations between all supported vector types
 *    - Checks that the operators are overloaded as expected by comparing results to the manually
 * calculated ones
 *    - Calls operations from the host side
 * Test source
 * ------------------------
 *    - unit/vector_types/vector_types.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_VectorAndVectorOperations_SanityCheck_Basic_Host", "", char1, uchar1,
                   char2, uchar2, char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2,
                   short3, ushort3, short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4,
                   uint4, long1, ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1,
                   ulonglong1, longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4,
                   float1, float2, float3, float4, double1, double2, double3, double4) {
  auto value1 = GetTestValue<decltype(TestType().x)>(0);
  auto value2 = GetTestValue<decltype(TestType().x)>(1);

  for (const auto operation : {VectorOperation::kIncrementPrefix,
                               VectorOperation::kIncrementPostfix,
                               VectorOperation::kDecrementPrefix,
                               VectorOperation::kDecrementPostfix,
                               VectorOperation::kAddAssign,
                               VectorOperation::kSubtractAssign,
                               VectorOperation::kMultiplyAssign,
                               VectorOperation::kDivideAssign,
                               VectorOperation::kNegate,
                               VectorOperation::kBitwiseNot,
                               VectorOperation::kModuloAssign,
                               VectorOperation::kBitwiseXorAssign,
                               VectorOperation::kBitwiseOrAssign,
                               VectorOperation::kBitwiseAndAssign,
                               VectorOperation::kRightShiftAssign,
                               VectorOperation::kLeftShiftAssign,
                               VectorOperation::kAdd,
                               VectorOperation::kSubtract,
                               VectorOperation::kMultiply,
                               VectorOperation::kDivide,
                               VectorOperation::kEqual,
                               VectorOperation::kNotEqual,
                               VectorOperation::kModulo,
                               VectorOperation::kBitwiseXor,
                               VectorOperation::kBitwiseOr,
                               VectorOperation::kBitwiseAnd,
                               VectorOperation::kRightShift,
                               VectorOperation::kLeftShift,
                               VectorOperation::kSubscript}) {
    DYNAMIC_SECTION("operation: " << to_string(operation)) {
      TestType vector = PerformVectorOperationHost<TestType>(operation, value1, value2);
      SanityCheck(operation, vector, value1, value2);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Performs supported operations between vector and underlying vector type (scalar)
 *    - Checks that the operators are overloaded as expected by comparing results to the manually
 * calculated ones
 *    - Calls operations from the host side
 * Test source
 * ------------------------
 *    - unit/vector_types/vector_types.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_VectorAndValueTypeOperations_SanityCheck_Basic_Host", "", char1, uchar1,
                   char2, uchar2, char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2,
                   short3, ushort3, short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4,
                   uint4, long1, ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1,
                   ulonglong1, longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4,
                   float1, float2, float3, float4, double1, double2, double3, double4) {
  auto value1 = GetTestValue<decltype(TestType().x)>(0);
  auto value2 = GetTestValue<decltype(TestType().x)>(1);

  for (const auto operation :
       {VectorOperation::kAddAssign, VectorOperation::kSubtractAssign,
        VectorOperation::kMultiplyAssign, VectorOperation::kDivideAssign, VectorOperation::kAdd,
        VectorOperation::kSubtract, VectorOperation::kMultiply, VectorOperation::kDivide,
        VectorOperation::kEqual, VectorOperation::kNotEqual, VectorOperation::kModulo,
        VectorOperation::kBitwiseXor, VectorOperation::kBitwiseOr, VectorOperation::kBitwiseAnd,
        VectorOperation::kRightShift, VectorOperation::kLeftShift, VectorOperation::kSubscript}) {
    DYNAMIC_SECTION("operation: " << to_string(operation)) {
      TestType vector = PerformVectorOperationHost<TestType, false>(operation, value1, value2);
      SanityCheck(operation, vector, value1, value2);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Performs supported operations between all supported vector types
 *    - Checks that the operators are overloaded as expected by comparing results to the manually
 * calculated ones
 *    - Calls operations from the device side
 * Test source
 * ------------------------
 *    - unit/vector_types/vector_types.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_VectorAndVectorOperations_SanityCheck_Basic_Device", "", char1, uchar1,
                   char2, uchar2, char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2,
                   short3, ushort3, short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4,
                   uint4, long1, ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1,
                   ulonglong1, longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4,
                   float1, float2, float3, float4, double1, double2, double3, double4) {
  auto value1 = GetTestValue<decltype(TestType().x)>(0);
  auto value2 = GetTestValue<decltype(TestType().x)>(1);

  for (const auto operation : {VectorOperation::kIncrementPrefix,
                               VectorOperation::kIncrementPostfix,
                               VectorOperation::kDecrementPrefix,
                               VectorOperation::kDecrementPostfix,
                               VectorOperation::kAddAssign,
                               VectorOperation::kSubtractAssign,
                               VectorOperation::kMultiplyAssign,
                               VectorOperation::kDivideAssign,
                               VectorOperation::kNegate,
                               VectorOperation::kBitwiseNot,
                               VectorOperation::kModuloAssign,
                               VectorOperation::kBitwiseXorAssign,
                               VectorOperation::kBitwiseOrAssign,
                               VectorOperation::kBitwiseAndAssign,
                               VectorOperation::kRightShiftAssign,
                               VectorOperation::kLeftShiftAssign,
                               VectorOperation::kAdd,
                               VectorOperation::kSubtract,
                               VectorOperation::kMultiply,
                               VectorOperation::kDivide,
                               VectorOperation::kEqual,
                               VectorOperation::kNotEqual,
                               VectorOperation::kModulo,
                               VectorOperation::kBitwiseXor,
                               VectorOperation::kBitwiseOr,
                               VectorOperation::kBitwiseAnd,
                               VectorOperation::kRightShift,
                               VectorOperation::kLeftShift,
                               VectorOperation::kSubscript}) {
    DYNAMIC_SECTION("operation: " << to_string(operation)) {
      TestType vector = PerformVectorOperationDevice<TestType>(operation, value1, value2);
      SanityCheck(operation, vector, value1, value2);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Performs supported operations between vector and underlying vector type (scalar)
 *    - Checks that the operators are overloaded as expected by comparing results to the manually
 * calculated ones
 *    - Calls operations from the device side
 * Test source
 * ------------------------
 *    - unit/vector_types/vector_types.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_VectorAndValueTypeOperations_SanityCheck_Basic_Device", "", char1, uchar1,
                   char2, uchar2, char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2,
                   short3, ushort3, short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4,
                   uint4, long1, ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1,
                   ulonglong1, longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4,
                   float1, float2, float3, float4, double1, double2, double3, double4) {
  auto value1 = GetTestValue<decltype(TestType().x)>(0);
  auto value2 = GetTestValue<decltype(TestType().x)>(1);

  for (const auto operation :
       {VectorOperation::kAddAssign, VectorOperation::kSubtractAssign,
        VectorOperation::kMultiplyAssign, VectorOperation::kDivideAssign, VectorOperation::kAdd,
        VectorOperation::kSubtract, VectorOperation::kMultiply, VectorOperation::kDivide,
        VectorOperation::kEqual, VectorOperation::kNotEqual, VectorOperation::kModulo,
        VectorOperation::kBitwiseXor, VectorOperation::kBitwiseOr, VectorOperation::kBitwiseAnd,
        VectorOperation::kRightShift, VectorOperation::kLeftShift, VectorOperation::kSubscript}) {
    DYNAMIC_SECTION("operation: " << to_string(operation)) {
      TestType vector = PerformVectorOperationDevice<TestType, false>(operation, value1, value2);
      SanityCheck(operation, vector, value1, value2);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Checks that vectors can be used with structured bindings
 *    - Tests from the host side
 * Test source
 * ------------------------
 *    - unit/vector_types/vector_types.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_VectorStructuredBindings_SanityCheck_Basic_host", "", float3, double3) {
  auto value = GetTestValue<decltype(TestType().x)>(0);

  TestType vec3 = {value, value, value};

  auto&& [ret1, ret2, ret3] = vec3;
  REQUIRE(ret1 == value);
}

__host__ __device__ constexpr bool func() {
  int3 vec3 = int3{0};
  int exp = int{0};
  return vec3.x == exp;
}

__global__ void generate_my_kernel() { static_assert(func()); }


/**
 * Test Description
 * ------------------------
 *    - Checks that vectors can work with constexpr
 *    - Tests from the host and device side
 * Test source
 * ------------------------
 *    - unit/vector_types/vector_types.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_VectorConstexpr_SanityCheck_Basic_host_device", "") {
  generate_my_kernel<<<1, 1>>>();
  static_assert(func());
}

struct padded_struct {
    int2 data1;
    float3 data2;
};

__host__ __device__ void check_alignment() {
  // char/uchar
  static_assert(std::alignment_of_v<char1> == 1);
  static_assert(std::alignment_of_v<char2> == 2);
  static_assert(std::alignment_of_v<char3> == 1);
  static_assert(std::alignment_of_v<char4> == 4);
  static_assert(std::alignment_of_v<uchar1> == 1);
  static_assert(std::alignment_of_v<uchar2> == 2);
  static_assert(std::alignment_of_v<uchar3> == 1);
  static_assert(std::alignment_of_v<uchar4> == 4);

  // short/ushort
  static_assert(std::alignment_of_v<short1> == 2);
  static_assert(std::alignment_of_v<short2> == 4);
  static_assert(std::alignment_of_v<short3> == 2);
  static_assert(std::alignment_of_v<short4> == 8);
  static_assert(std::alignment_of_v<ushort1> == 2);
  static_assert(std::alignment_of_v<ushort2> == 4);
  static_assert(std::alignment_of_v<ushort3> == 2);
  static_assert(std::alignment_of_v<ushort4> == 8);

  // int/uint
  static_assert(std::alignment_of_v<int1> == 4);
  static_assert(std::alignment_of_v<int2> == 8);
  static_assert(std::alignment_of_v<int3> == 4);
  static_assert(std::alignment_of_v<int4> == 16);
  static_assert(std::alignment_of_v<uint1> == 4);
  static_assert(std::alignment_of_v<uint2> == 8);
  static_assert(std::alignment_of_v<uint3> == 4);
  static_assert(std::alignment_of_v<uint4> == 16);

  // long/ulong
  constexpr size_t long_size = sizeof(long);  // needed to handle MSVC long defintion
  static_assert(std::alignment_of_v<long1> == long_size);
  static_assert(std::alignment_of_v<long2> == 2 * long_size);
  static_assert(std::alignment_of_v<long3> == long_size);
  static_assert(std::alignment_of_v<long4> == 4 * long_size);
  static_assert(std::alignment_of_v<ulong1> == long_size);
  static_assert(std::alignment_of_v<ulong2> == 2 * long_size);
  static_assert(std::alignment_of_v<ulong3> == long_size);
  static_assert(std::alignment_of_v<ulong4> == 4 * long_size);

  // longlong/ulonglong
  static_assert(std::alignment_of_v<longlong1> == 8);
  static_assert(std::alignment_of_v<longlong2> == 16);
  static_assert(std::alignment_of_v<longlong3> == 8);
  static_assert(std::alignment_of_v<longlong4> == 32);
  static_assert(std::alignment_of_v<ulonglong1> == 8);
  static_assert(std::alignment_of_v<ulonglong2> == 16);
  static_assert(std::alignment_of_v<ulonglong3> == 8);
  static_assert(std::alignment_of_v<ulonglong4> == 32);

  // float
  static_assert(std::alignment_of_v<float1> == 4);
  static_assert(std::alignment_of_v<float2> == 8);
  static_assert(std::alignment_of_v<float3> == 4);
  static_assert(std::alignment_of_v<float4> == 16);

  // double
  static_assert(std::alignment_of_v<double1> == 8);
  static_assert(std::alignment_of_v<double2> == 16);
  static_assert(std::alignment_of_v<double3> == 8);
  static_assert(std::alignment_of_v<double4> == 32);

  // padded struct
  static_assert(std::alignment_of_v<padded_struct> == 8);
}

__global__ void check_alignment_device() { check_alignment(); }

/**
 * Test Description
 * ------------------------
 *    - Compile-time test checking vector type alignement
 *    - Tests from the host and device side
 * Test source
 * ------------------------
 *    - unit/vector_types/vector_types.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Vector_alignment_check", "") {
  check_alignment_device<<<1, 1>>>();
  check_alignment();
}

__host__ __device__ void check_size() {
  // char/uchar
  constexpr size_t char_size = sizeof(char);
  static_assert(sizeof(char1) == 1 * char_size);
  static_assert(sizeof(char2) == 2 * char_size);
  static_assert(sizeof(char3) == 3 * char_size);
  static_assert(sizeof(char4) == 4 * char_size);
  static_assert(sizeof(uchar1) == 1 * char_size);
  static_assert(sizeof(uchar2) == 2 * char_size);
  static_assert(sizeof(uchar3) == 3 * char_size);
  static_assert(sizeof(uchar4) == 4 * char_size);

  // short/ushort
  constexpr size_t short_size = sizeof(short);
  static_assert(sizeof(short1) == 1 * short_size);
  static_assert(sizeof(short2) == 2 * short_size);
  static_assert(sizeof(short3) == 3 * short_size);
  static_assert(sizeof(short4) == 4 * short_size);
  static_assert(sizeof(ushort1) == 1 * short_size);
  static_assert(sizeof(ushort2) == 2 * short_size);
  static_assert(sizeof(ushort3) == 3 * short_size);
  static_assert(sizeof(ushort4) == 4 * short_size);

  // int/uint
  constexpr size_t int_size = sizeof(int);
  static_assert(sizeof(int1) == 1 * int_size);
  static_assert(sizeof(int2) == 2 * int_size);
  static_assert(sizeof(int3) == 3 * int_size);
  static_assert(sizeof(int4) == 4 * int_size);
  static_assert(sizeof(uint1) == 1 * int_size);
  static_assert(sizeof(uint2) == 2 * int_size);
  static_assert(sizeof(uint3) == 3 * int_size);
  static_assert(sizeof(uint4) == 4 * int_size);

  // long/ulong
  constexpr size_t long_size = sizeof(long);
  static_assert(sizeof(long1) == 1 * long_size);
  static_assert(sizeof(long2) == 2 * long_size);
  static_assert(sizeof(long3) == 3 * long_size);
  static_assert(sizeof(long4) == 4 * long_size);
  static_assert(sizeof(ulong1) == 1 * long_size);
  static_assert(sizeof(ulong2) == 2 * long_size);
  static_assert(sizeof(ulong3) == 3 * long_size);
  static_assert(sizeof(ulong4) == 4 * long_size);

  // longlong/ulonglong
  constexpr size_t longlong_size = sizeof(long long);
  static_assert(sizeof(longlong1) == 1 * longlong_size);
  static_assert(sizeof(longlong2) == 2 * longlong_size);
  static_assert(sizeof(longlong3) == 3 * longlong_size);
  static_assert(sizeof(longlong4) == 4 * longlong_size);
  static_assert(sizeof(ulonglong1) == 1 * longlong_size);
  static_assert(sizeof(ulonglong2) == 2 * longlong_size);
  static_assert(sizeof(ulonglong3) == 3 * longlong_size);
  static_assert(sizeof(ulonglong4) == 4 * longlong_size);

  // float
  constexpr size_t float_size = sizeof(float);
  static_assert(sizeof(float1) == 1 * float_size);
  static_assert(sizeof(float2) == 2 * float_size);
  static_assert(sizeof(float3) == 3 * float_size);
  static_assert(sizeof(float4) == 4 * float_size);

  // double
  constexpr size_t double_size = sizeof(double);
  static_assert(sizeof(double1) == 1 * double_size);
  static_assert(sizeof(double2) == 2 * double_size);
  static_assert(sizeof(double3) == 3 * double_size);
  static_assert(sizeof(double4) == 4 * double_size);

  // padded struct
  static_assert(sizeof(padded_struct) == 24);
}

__global__ void check_size_device() { check_size(); }

/**
 * Test Description
 * ------------------------
 *    - Compile-time test checking vector type size
 *    - Tests from the host and device side
 * Test source
 * ------------------------
 *    - unit/vector_types/vector_types.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Vector_size_check", "") {
  check_size_device<<<1, 1>>>();
  check_size();
}
#endif  // HT_AMD

/**
 * End doxygen group VectorTypeTest.
 * @}
 */
