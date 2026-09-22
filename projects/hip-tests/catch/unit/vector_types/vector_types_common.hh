/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <type_traits>

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

/**
 * @brief Template helper to map base types and dimensions to HIP vector types.
 *
 * This trait maps a base type T and dimension N to the corresponding HIP vector type.
 * For example: vector_type_helper<char, 2>::type is char2
 * Dimension 0 returns the base type itself.
 *
 * @tparam T Base type (char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double)
 * @tparam N Vector dimension (0 = base type, 1, 2, 3, or 4)
 */
template <typename T, size_t N>
struct vector_type_helper {
  // Primary template is undefined - only valid specializations are defined below
  // This ensures compile-time errors for invalid type/dimension combinations
  static_assert(!std::is_same_v<T, T>,
                "vector_type_helper: Invalid base type or dimension. "
                "Valid base types: char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double. "
                "Valid dimensions: 0 (base type), 1, 2, 3, 4");
};

// Specialization for dimension 0 (base type itself)
template <typename T>
struct vector_type_helper<T, 0> {
  using type = T;
};

// Specialization for char base type
template <>
struct vector_type_helper<char, 1> {
  using type = char1;
  static_assert(sizeof(type) == sizeof(char), "char1 size mismatch");
};

template <>
struct vector_type_helper<char, 2> {
  using type = char2;
  static_assert(sizeof(type) == 2 * sizeof(char), "char2 size mismatch");
};

template <>
struct vector_type_helper<char, 3> {
  using type = char3;
  static_assert(sizeof(type) == 3 * sizeof(char), "char3 size mismatch");
};

template <>
struct vector_type_helper<char, 4> {
  using type = char4;
  static_assert(sizeof(type) == 4 * sizeof(char), "char4 size mismatch");
};

// Specialization for unsigned char base type
template <>
struct vector_type_helper<unsigned char, 1> {
  using type = uchar1;
  static_assert(sizeof(type) == sizeof(unsigned char), "uchar1 size mismatch");
};

template <>
struct vector_type_helper<unsigned char, 2> {
  using type = uchar2;
  static_assert(sizeof(type) == 2 * sizeof(unsigned char), "uchar2 size mismatch");
};

template <>
struct vector_type_helper<unsigned char, 3> {
  using type = uchar3;
  static_assert(sizeof(type) == 3 * sizeof(unsigned char), "uchar3 size mismatch");
};

template <>
struct vector_type_helper<unsigned char, 4> {
  using type = uchar4;
  static_assert(sizeof(type) == 4 * sizeof(unsigned char), "uchar4 size mismatch");
};

// Specialization for short base type
template <>
struct vector_type_helper<short, 1> {
  using type = short1;
  static_assert(sizeof(type) == sizeof(short), "short1 size mismatch");
};

template <>
struct vector_type_helper<short, 2> {
  using type = short2;
  static_assert(sizeof(type) == 2 * sizeof(short), "short2 size mismatch");
};

template <>
struct vector_type_helper<short, 3> {
  using type = short3;
  static_assert(sizeof(type) == 3 * sizeof(short), "short3 size mismatch");
};

template <>
struct vector_type_helper<short, 4> {
  using type = short4;
  static_assert(sizeof(type) == 4 * sizeof(short), "short4 size mismatch");
};

// Specialization for unsigned short base type
template <>
struct vector_type_helper<unsigned short, 1> {
  using type = ushort1;
  static_assert(sizeof(type) == sizeof(unsigned short), "ushort1 size mismatch");
};

template <>
struct vector_type_helper<unsigned short, 2> {
  using type = ushort2;
  static_assert(sizeof(type) == 2 * sizeof(unsigned short), "ushort2 size mismatch");
};

template <>
struct vector_type_helper<unsigned short, 3> {
  using type = ushort3;
  static_assert(sizeof(type) == 3 * sizeof(unsigned short), "ushort3 size mismatch");
};

template <>
struct vector_type_helper<unsigned short, 4> {
  using type = ushort4;
  static_assert(sizeof(type) == 4 * sizeof(unsigned short), "ushort4 size mismatch");
};

// Specialization for int base type
template <>
struct vector_type_helper<int, 1> {
  using type = int1;
  static_assert(sizeof(type) == sizeof(int), "int1 size mismatch");
};

template <>
struct vector_type_helper<int, 2> {
  using type = int2;
  static_assert(sizeof(type) == 2 * sizeof(int), "int2 size mismatch");
};

template <>
struct vector_type_helper<int, 3> {
  using type = int3;
  static_assert(sizeof(type) == 3 * sizeof(int), "int3 size mismatch");
};

template <>
struct vector_type_helper<int, 4> {
  using type = int4;
  static_assert(sizeof(type) == 4 * sizeof(int), "int4 size mismatch");
};

// Specialization for unsigned int base type
template <>
struct vector_type_helper<unsigned int, 1> {
  using type = uint1;
  static_assert(sizeof(type) == sizeof(unsigned int), "uint1 size mismatch");
};

template <>
struct vector_type_helper<unsigned int, 2> {
  using type = uint2;
  static_assert(sizeof(type) == 2 * sizeof(unsigned int), "uint2 size mismatch");
};

template <>
struct vector_type_helper<unsigned int, 3> {
  using type = uint3;
  static_assert(sizeof(type) == 3 * sizeof(unsigned int), "uint3 size mismatch");
};

template <>
struct vector_type_helper<unsigned int, 4> {
  using type = uint4;
  static_assert(sizeof(type) == 4 * sizeof(unsigned int), "uint4 size mismatch");
};

// Specialization for long base type
template <>
struct vector_type_helper<long, 1> {
  using type = long1;
  static_assert(sizeof(type) == sizeof(long), "long1 size mismatch");
};

template <>
struct vector_type_helper<long, 2> {
  using type = long2;
  static_assert(sizeof(type) == 2 * sizeof(long), "long2 size mismatch");
};

template <>
struct vector_type_helper<long, 3> {
  using type = long3;
  static_assert(sizeof(type) == 3 * sizeof(long), "long3 size mismatch");
};

template <>
struct vector_type_helper<long, 4> {
  using type = long4;
  static_assert(sizeof(type) == 4 * sizeof(long), "long4 size mismatch");
};

// Specialization for unsigned long base type
template <>
struct vector_type_helper<unsigned long, 1> {
  using type = ulong1;
  static_assert(sizeof(type) == sizeof(unsigned long), "ulong1 size mismatch");
};

template <>
struct vector_type_helper<unsigned long, 2> {
  using type = ulong2;
  static_assert(sizeof(type) == 2 * sizeof(unsigned long), "ulong2 size mismatch");
};

template <>
struct vector_type_helper<unsigned long, 3> {
  using type = ulong3;
  static_assert(sizeof(type) == 3 * sizeof(unsigned long), "ulong3 size mismatch");
};

template <>
struct vector_type_helper<unsigned long, 4> {
  using type = ulong4;
  static_assert(sizeof(type) == 4 * sizeof(unsigned long), "ulong4 size mismatch");
};

// Specialization for long long base type
template <>
struct vector_type_helper<long long, 1> {
  using type = longlong1;
  static_assert(sizeof(type) == sizeof(long long), "longlong1 size mismatch");
};

template <>
struct vector_type_helper<long long, 2> {
  using type = longlong2;
  static_assert(sizeof(type) == 2 * sizeof(long long), "longlong2 size mismatch");
};

template <>
struct vector_type_helper<long long, 3> {
  using type = longlong3;
  static_assert(sizeof(type) == 3 * sizeof(long long), "longlong3 size mismatch");
};

template <>
struct vector_type_helper<long long, 4> {
  using type = longlong4;
  static_assert(sizeof(type) == 4 * sizeof(long long), "longlong4 size mismatch");
};

// Specialization for unsigned long long base type
template <>
struct vector_type_helper<unsigned long long, 1> {
  using type = ulonglong1;
  static_assert(sizeof(type) == sizeof(unsigned long long), "ulonglong1 size mismatch");
};

template <>
struct vector_type_helper<unsigned long long, 2> {
  using type = ulonglong2;
  static_assert(sizeof(type) == 2 * sizeof(unsigned long long), "ulonglong2 size mismatch");
};

template <>
struct vector_type_helper<unsigned long long, 3> {
  using type = ulonglong3;
  static_assert(sizeof(type) == 3 * sizeof(unsigned long long), "ulonglong3 size mismatch");
};

template <>
struct vector_type_helper<unsigned long long, 4> {
  using type = ulonglong4;
  static_assert(sizeof(type) == 4 * sizeof(unsigned long long), "ulonglong4 size mismatch");
};

// Specialization for float base type
template <>
struct vector_type_helper<float, 1> {
  using type = float1;
  static_assert(sizeof(type) == sizeof(float), "float1 size mismatch");
};

template <>
struct vector_type_helper<float, 2> {
  using type = float2;
  static_assert(sizeof(type) == 2 * sizeof(float), "float2 size mismatch");
};

template <>
struct vector_type_helper<float, 3> {
  using type = float3;
  static_assert(sizeof(type) == 3 * sizeof(float), "float3 size mismatch");
};

template <>
struct vector_type_helper<float, 4> {
  using type = float4;
  static_assert(sizeof(type) == 4 * sizeof(float), "float4 size mismatch");
};

// Specialization for double base type
template <>
struct vector_type_helper<double, 1> {
  using type = double1;
  static_assert(sizeof(type) == sizeof(double), "double1 size mismatch");
};

template <>
struct vector_type_helper<double, 2> {
  using type = double2;
  static_assert(sizeof(type) == 2 * sizeof(double), "double2 size mismatch");
};

template <>
struct vector_type_helper<double, 3> {
  using type = double3;
  static_assert(sizeof(type) == 3 * sizeof(double), "double3 size mismatch");
};

template <>
struct vector_type_helper<double, 4> {
  using type = double4;
  static_assert(sizeof(type) == 4 * sizeof(double), "double4 size mismatch");
};

// Convenience alias template for easier usage
template <typename T, size_t N>
using vector_type_helper_t = typename vector_type_helper<T, N>::type;
