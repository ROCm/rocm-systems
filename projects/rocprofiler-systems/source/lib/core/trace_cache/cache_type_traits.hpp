// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "common/span.hpp"
#include <array>
#include <concepts>
#include <cstddef>
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
concept supported_cache_type = is_supported_type_v<std::decay_t<T>>;

template <typename T>
struct is_enum_class
: std::bool_constant<std::is_enum_v<T> &&
                     !std::is_convertible_v<T, std::underlying_type_t<T>>>
{};

template <typename T>
inline constexpr bool is_enum_class_v = is_enum_class<T>::value;

template <typename T>
concept serializable = requires(std::uint8_t* dst, const T& v) { serialize(dst, v); };

template <typename T>
concept deserializable = requires(std::uint8_t*& src) {
    { deserialize<T>(src) } -> std::same_as<T>;
};

template <typename T>
concept has_get_size = requires(const T& v) {
    { get_size(v) } -> std::convertible_to<std::size_t>;
};

template <typename T, typename TypeIdentifierEnum>
concept has_type_identifier = is_enum_class_v<TypeIdentifierEnum> && requires {
    { T::type_identifier } -> std::convertible_to<TypeIdentifierEnum>;
};

template <typename T, typename TypeIdentifierEnum>
concept cacheable = serializable<T> && deserializable<T> && has_get_size<T> &&
                    has_type_identifier<T, TypeIdentifierEnum>;

template <typename T, typename TypeIdentifierEnum, typename CacheableType>
concept sample_processor = requires(T t, TypeIdentifierEnum e, const CacheableType& c) {
    { t.execute_sample_processing(e, c) } -> std::same_as<void>;
};

}  // namespace type_traits
}  // namespace trace_cache
}  // namespace rocprofsys
