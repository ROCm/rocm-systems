/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip/hip_runtime.h>
#include <type_traits>

/**
 * @brief Template helper to map base types and dimensions to HIP vector types.
 *
 * This trait maps a base type T and dimension N to the corresponding HIP vector type.
 * For example: vector_type_helper<char, 2>::type is char2
 * Dimension 0 returns the base type itself.
 *
 * @tparam T Base type (char, uchar, short, ushort, int, uint, float)
 * @tparam N Vector dimension (0 = base type, 1, 2, or 4)
 */
template <typename T, size_t N>
struct vector_type_helper {
  // Primary template is undefined - only valid specializations are defined below
  // This ensures compile-time errors for invalid type/dimension combinations
  static_assert(!std::is_same_v<T, T>,
                "vector_type_helper: Invalid base type or dimension. "
                "Valid base types: char, uchar, short, ushort, int, uint, float. "
                "Valid dimensions: 0 (base type), 1, 2, 4");
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
struct vector_type_helper<unsigned int, 4> {
  using type = uint4;
  static_assert(sizeof(type) == 4 * sizeof(unsigned int), "uint4 size mismatch");
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
struct vector_type_helper<float, 4> {
  using type = float4;
  static_assert(sizeof(type) == 4 * sizeof(float), "float4 size mismatch");
};

// Convenience alias template for easier usage
template <typename T, size_t N>
using vector_type_helper_t = typename vector_type_helper<T, N>::type;
