// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rocprofsys::trace_cache
{

namespace type_traits
{

template <typename T>
struct is_span : std::false_type
{};

template <typename T>
struct is_span<std::span<T>> : std::true_type
{};

template <typename T>
inline constexpr bool is_span_v = is_span<T>::value;

template <typename T>
struct is_vector : std::false_type
{};

template <typename T>
struct is_vector<std::vector<T>> : std::true_type
{};

template <typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

template <typename T>
struct is_array : std::false_type
{};

template <typename T, size_t N>
struct is_array<std::array<T, N>> : std::true_type
{};

template <typename T>
inline constexpr bool is_array_v = is_array<T>::value;

template <typename T>
concept string_like = std::is_same_v<std::decay_t<T>, std::string_view> ||
                      std::is_same_v<std::decay_t<T>, std::string>;

template <typename T>
struct is_optional : std::false_type
{};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type
{};

template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

template <typename T>
inline constexpr bool is_supported_type_v =
    is_span_v<T> || std::is_integral_v<T> || std::is_floating_point_v<T> ||
    string_like<T> || is_vector_v<T> || is_optional_v<T> || is_array_v<T> ||
    std::is_enum_v<T>;

template <typename T>
concept supported_cache_type = is_supported_type_v<std::decay_t<T>>;

template <typename T, bool = std::is_enum_v<T>>
struct is_enum_class : std::false_type
{};

template <typename T>
struct is_enum_class<T, true>
: std::bool_constant<!std::is_convertible_v<T, std::underlying_type_t<T>>>
{};

template <typename T>
inline constexpr bool is_enum_class_v = is_enum_class<T>::value;

/**
 * Opt-in for a struct whose bytes may be copied as they are: trivially copyable, no
 * padding, no pointers. Specialize it next to the record that uses the struct.
 */
template <typename T>
struct explicitly_raw : std::false_type
{};

/** Types that are cached by copying their bytes: numbers, scoped enums, opted-in structs.
 */
template <typename T>
struct is_raw_cacheable
: std::bool_constant<std::is_arithmetic_v<T> || is_enum_class_v<T> ||
                     explicitly_raw<T>::value>
{};

template <typename T, std::size_t N>
struct is_raw_cacheable<std::array<T, N>> : is_raw_cacheable<T>
{};

template <typename T, typename TypeIdentifierEnum>
concept has_type_identifier = is_enum_class_v<TypeIdentifierEnum> && requires {
    { T::type_identifier } -> std::convertible_to<TypeIdentifierEnum>;
};

template <typename T, typename TypeIdentifierEnum, typename CacheableType>
concept sample_processor = requires(T t, TypeIdentifierEnum e, const CacheableType& c) {
    { t.execute_sample_processing(e, c) } -> std::same_as<void>;
};

}  // namespace type_traits
}  // namespace rocprofsys::trace_cache
