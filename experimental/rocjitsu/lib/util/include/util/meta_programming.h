// Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_META_PROGRAMMING_H_
#define UTIL_META_PROGRAMMING_H_

#include <concepts>
#include <type_traits>

namespace util {
namespace metaprogramming {

/// @brief Concept used to determine if a type is an array.
template <typename T>
concept IsArray = std::is_array_v<T>;

/// @brief Concept used to determine if a type is integral.
template <typename T>
concept IsIntegral = std::is_integral_v<T>;

/// @brief Concept used to determine if a type is unsigned.
template <typename T>
concept IsUnsigned = std::is_unsigned_v<T>;

/// @brief Concept used to determine if a type is unsigned and integral.
template <typename T>
concept IsUnsignedInt = IsIntegral<T> && IsUnsigned<T>;

/// @brief Concept used to determine if a type is arithmetic (integral or floating-point).
template <typename T>
concept IsArithmetic = std::is_arithmetic_v<T>;

/// @brief Concept used to determine if a type is void.
template <typename T>
concept IsVoid = std::is_void_v<T>;

/// @brief Concept used to determine if a type is not void.
template <typename T>
concept IsNonVoid = !std::is_void_v<T>;

template <template <typename> typename TagT, typename... Options> struct GetOption {
  using Type = void;
};

template <template <typename> typename TagT, typename T, typename... Options>
struct GetOption<TagT, T, Options...> : GetOption<TagT, Options...> {};

template <template <typename> typename TagT, typename T, typename... Options>
struct GetOption<TagT, TagT<T>, Options...> : GetOption<TagT, Options...> {
  using Type = T;
};

} // namespace metaprogramming
} // namespace util

#endif // UTIL_META_PROGRAMMING_H_
