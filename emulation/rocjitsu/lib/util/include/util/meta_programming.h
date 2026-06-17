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

/// @brief Helper variable template that is always false but depends on a type parameter.
///
/// @details Used in `if constexpr` / `static_assert` chains to produce a
/// deferred (non-eagerly evaluated) compile error for unhandled template
/// specializations.  A bare `static_assert(false, ...)` inside a constexpr
/// branch is ill-formed even when the branch is never taken; replacing it with
/// `static_assert(always_false_v<T>, ...)` is well-formed and only fires when
/// the branch is actually instantiated.
///
/// @tparam T  Any type — the assertion fires when the enclosing template is
///            instantiated with a specialization that has no handler.
///
/// Example:
/// @code
/// template <AccMode M> void resolve_acc() {
///   if constexpr (M == AccMode::Unified) { ... }
///   else if constexpr (M == AccMode::Separate) { ... }
///   else { static_assert(util::always_false_v<AccMode>, "unhandled AccMode"); }
/// }
/// @endcode
template <typename T> inline constexpr bool always_false_v = false;

} // namespace util

#endif // UTIL_META_PROGRAMMING_H_
