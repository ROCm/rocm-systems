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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

constexpr auto kIntegerTestValueFirst = 42;
constexpr auto kIntegerTestValueSecond = 4;
constexpr auto kFloatingPointTestValueFirst = 42.125;
constexpr auto kFloatingPointTestValueSecond = 4.875;

template <typename T> T GetTestValue(int index) {
  if (index == 0) {
    return std::is_floating_point_v<T> ? static_cast<T>(kIntegerTestValueFirst)
                                       : static_cast<T>(kFloatingPointTestValueFirst);
  } else {
    return std::is_floating_point_v<T> ? static_cast<T>(kIntegerTestValueSecond)
                                       : static_cast<T>(kFloatingPointTestValueSecond);
  }
}

template <typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T().x)) == 1>::type SanityCheck(
    T vector, decltype(T().x) expected_value) {
  REQUIRE(vector.x == expected_value);
}

template <typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T().x)) == 2>::type SanityCheck(
    T vector, decltype(T().x) expected_value) {
  REQUIRE(vector.x == expected_value);
  REQUIRE(vector.y == expected_value);
}

template <typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T().x)) == 3>::type SanityCheck(
    T vector, decltype(T().x) expected_value) {
  REQUIRE(vector.x == expected_value);
  REQUIRE(vector.y == expected_value);
  REQUIRE(vector.z == expected_value);
}

template <typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T().x)) == 4>::type SanityCheck(
    T vector, decltype(T().x) expected_value) {
  REQUIRE(vector.x == expected_value);
  REQUIRE(vector.y == expected_value);
  REQUIRE(vector.z == expected_value);
  REQUIRE(vector.w == expected_value);
}

template <typename T>
__host__ __device__ void MakeVectorType(T* vector_ptr, decltype(T().x) value) {
  if constexpr (std::is_same_v<T, char1>) {
    *vector_ptr = make_char1(value);
  } else if constexpr (std::is_same_v<T, uchar1>) {
    *vector_ptr = make_uchar1(value);
  } else if constexpr (std::is_same_v<T, char2>) {
    *vector_ptr = make_char2(value, value);
  } else if constexpr (std::is_same_v<T, uchar2>) {
    *vector_ptr = make_uchar2(value, value);
  } else if constexpr (std::is_same_v<T, char3>) {
    *vector_ptr = make_char3(value, value, value);
  } else if constexpr (std::is_same_v<T, uchar3>) {
    *vector_ptr = make_uchar3(value, value, value);
  } else if constexpr (std::is_same_v<T, char4>) {
    *vector_ptr = make_char4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, uchar4>) {
    *vector_ptr = make_uchar4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, short1>) {
    *vector_ptr = make_short1(value);
  } else if constexpr (std::is_same_v<T, ushort1>) {
    *vector_ptr = make_ushort1(value);
  } else if constexpr (std::is_same_v<T, short2>) {
    *vector_ptr = make_short2(value, value);
  } else if constexpr (std::is_same_v<T, ushort2>) {
    *vector_ptr = make_ushort2(value, value);
  } else if constexpr (std::is_same_v<T, short3>) {
    *vector_ptr = make_short3(value, value, value);
  } else if constexpr (std::is_same_v<T, ushort3>) {
    *vector_ptr = make_ushort3(value, value, value);
  } else if constexpr (std::is_same_v<T, short4>) {
    *vector_ptr = make_short4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, ushort4>) {
    *vector_ptr = make_ushort4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, int1>) {
    *vector_ptr = make_int1(value);
  } else if constexpr (std::is_same_v<T, uint1>) {
    *vector_ptr = make_uint1(value);
  } else if constexpr (std::is_same_v<T, int2>) {
    *vector_ptr = make_int2(value, value);
  } else if constexpr (std::is_same_v<T, uint2>) {
    *vector_ptr = make_uint2(value, value);
  } else if constexpr (std::is_same_v<T, int3>) {
    *vector_ptr = make_int3(value, value, value);
  } else if constexpr (std::is_same_v<T, uint3>) {
    *vector_ptr = make_uint3(value, value, value);
  } else if constexpr (std::is_same_v<T, int4>) {
    *vector_ptr = make_int4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, uint4>) {
    *vector_ptr = make_uint4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, long1>) {
    *vector_ptr = make_long1(value);
  } else if constexpr (std::is_same_v<T, ulong1>) {
    *vector_ptr = make_ulong1(value);
  } else if constexpr (std::is_same_v<T, long2>) {
    *vector_ptr = make_long2(value, value);
  } else if constexpr (std::is_same_v<T, ulong2>) {
    *vector_ptr = make_ulong2(value, value);
  } else if constexpr (std::is_same_v<T, long3>) {
    *vector_ptr = make_long3(value, value, value);
  } else if constexpr (std::is_same_v<T, ulong3>) {
    *vector_ptr = make_ulong3(value, value, value);
  } else if constexpr (std::is_same_v<T, long4>) {
    *vector_ptr = make_long4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, ulong4>) {
    *vector_ptr = make_ulong4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, longlong1>) {
    *vector_ptr = make_longlong1(value);
  } else if constexpr (std::is_same_v<T, ulonglong1>) {
    *vector_ptr = make_ulonglong1(value);
  } else if constexpr (std::is_same_v<T, longlong2>) {
    *vector_ptr = make_longlong2(value, value);
  } else if constexpr (std::is_same_v<T, ulonglong2>) {
    *vector_ptr = make_ulonglong2(value, value);
  } else if constexpr (std::is_same_v<T, longlong3>) {
    *vector_ptr = make_longlong3(value, value, value);
  } else if constexpr (std::is_same_v<T, ulonglong3>) {
    *vector_ptr = make_ulonglong3(value, value, value);
  } else if constexpr (std::is_same_v<T, longlong4>) {
    *vector_ptr = make_longlong4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, ulonglong4>) {
    *vector_ptr = make_ulonglong4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, float1>) {
    *vector_ptr = make_float1(value);
  } else if constexpr (std::is_same_v<T, float2>) {
    *vector_ptr = make_float2(value, value);
  } else if constexpr (std::is_same_v<T, float3>) {
    *vector_ptr = make_float3(value, value, value);
  } else if constexpr (std::is_same_v<T, float4>) {
    *vector_ptr = make_float4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, double1>) {
    *vector_ptr = make_double1(value);
  } else if constexpr (std::is_same_v<T, double2>) {
    *vector_ptr = make_double2(value, value);
  } else if constexpr (std::is_same_v<T, double3>) {
    *vector_ptr = make_double3(value, value, value);
  } else if constexpr (std::is_same_v<T, double4>) {
    *vector_ptr = make_double4(value, value, value, value);
  }
}

template <typename T> __host__ __device__ T MakeVector(decltype(T().x) value) {
  T vector{};
  MakeVectorType(&vector, value);
  return vector;
}

template <typename T> T MakeVectorTypeHost(decltype(T().x) value) {
  T vector{};
  MakeVectorType(&vector, value);
  return vector;
}

template <typename T> __global__ void VectorTypeKernel(T* vector, decltype(T().x) value) {
  MakeVectorType(vector, value);
}

template <typename T> T MakeVectorTypeDevice(decltype(T().x) value) {
  T vector_h{};
  T* vector_d;
  HIP_CHECK(hipMalloc(&vector_d, sizeof(T)));
  HIP_CHECK(hipMemcpy(vector_d, &vector_h, sizeof(T), hipMemcpyHostToDevice));
  VectorTypeKernel<<<1, 1, 0, 0>>>(vector_d, value);
  HIP_CHECK(hipMemcpy(&vector_h, vector_d, sizeof(T), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(vector_d));
  return vector_h;
}
