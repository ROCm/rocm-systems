// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "common/span.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{

namespace type_traits
{

template <typename T>
struct always_false : std::false_type
{};

}  // namespace type_traits

template <typename T>
void
serialize(std::uint8_t*, const T&)
{
    static_assert(type_traits::always_false<T>::value, "serialize<T> not specialized");
}

template <typename T>
T
deserialize(std::uint8_t*&)
{
    static_assert(type_traits::always_false<T>::value, "deserialize<T> not specialized");
    return T{};
}

template <typename T>
size_t
get_size(const T&)
{
    static_assert(type_traits::always_false<T>::value, "get_size(T) not specialized");
    return 0;
}

namespace type_traits
{

template <typename T>
struct tuple_to_variant;

template <typename... Types>
struct tuple_to_variant<std::tuple<Types...>>
{
    using type = std::variant<Types...>;
};

template <class...>
using void_t = void;

template <typename T>
struct is_span : std::false_type
{};

template <typename T>
struct is_span<span<T>> : std::true_type
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
static constexpr bool is_string_view_v =
    std::is_same_v<std::decay_t<T>, std::string_view>;

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
    is_string_view_v<T> || is_vector_v<T> || is_optional_v<T> || is_array_v<T>;

template <typename T>
struct is_enum_class
: std::bool_constant<std::is_enum_v<T> &&
                     !std::is_convertible_v<T, std::underlying_type_t<T>>>
{};

template <typename T>
inline constexpr bool is_enum_class_v = is_enum_class<T>::value;

template <typename T, typename TypeIdentifierEnum, typename = void>
struct has_type_identifier : std::false_type
{};

template <class T, typename TypeIdentifierEnum>
struct has_type_identifier<T, TypeIdentifierEnum, void_t<decltype(T::type_identifier)>>
: std::bool_constant<
      is_enum_class_v<TypeIdentifierEnum> &&
      std::is_convertible_v<decltype(T::type_identifier), TypeIdentifierEnum>>
{};

template <typename T, typename = void>
struct has_serialize : std::false_type
{};

template <typename T>
struct has_serialize<T, std::void_t<decltype(serialize(std::declval<std::uint8_t*>(),
                                                       std::declval<const T&>()))>>
: std::true_type
{};

template <typename T, typename = void>
struct has_deserialize : std::false_type
{};

template <typename T>
struct has_deserialize<
    T, void_t<std::is_same<decltype(deserialize<T>(std::declval<std::uint8_t*&>())), T>>>
: std::true_type
{};

template <typename T, typename = void>
struct has_get_size : std::false_type
{};

template <typename T>
struct has_get_size<T, void_t<decltype(get_size(std::declval<const T&>()))>>
: std::true_type
{};

template <typename T, typename TypeIdentifierEnum>
__attribute__((always_inline)) inline constexpr void
check_type()
{
    static_assert(has_serialize<T>::value, "Type doesn't have `serialize` function.");
    static_assert(has_deserialize<T>::value, "Type doesn't have `deserialize` function.");
    static_assert(has_get_size<T>::value, "Type doesn't have `get_size` function.");
    static_assert(has_type_identifier<T, TypeIdentifierEnum>::value,
                  "Type doesn't have `type_identifier` member with correct type.");
}

template <typename T, typename TypeIdentifierEnum, typename CacheableType,
          typename = void>
struct has_execute_processing : std::false_type
{};

template <typename T, typename TypeIdentifierEnum, typename CacheableType>
struct has_execute_processing<
    T, TypeIdentifierEnum, CacheableType,
    void_t<decltype(std::declval<T>().execute_sample_processing(
        std::declval<TypeIdentifierEnum>(), std::declval<const CacheableType&>()))>>
: std::bool_constant<std::is_void_v<decltype(std::declval<T>().execute_sample_processing(
      std::declval<TypeIdentifierEnum>(), std::declval<const CacheableType&>()))>>
{};

}  // namespace type_traits
}  // namespace trace_cache
}  // namespace rocprofsys
