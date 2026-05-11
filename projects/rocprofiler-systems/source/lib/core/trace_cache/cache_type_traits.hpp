// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "common/span.hpp"
#include "core/trace_cache/archive.hpp"

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

// Free-function wrappers that dispatch through the member-archive path. They
// exist so legacy call sites that already write
// `serialize(buf, value)` / `deserialize<T>(cursor)` / `get_size(value)`
// continue to compile and work bytewise-identical. The implementation is the
// archive in archive.hpp; per-type bodies live as
// `template <class Archive> serialize(Archive&)` members on the type.
template <typename T>
ROCPROFSYS_INLINE void
serialize(std::uint8_t* buffer, const T& item)
{
    serialize_at(buffer, item);
}

template <typename T>
ROCPROFSYS_INLINE T
deserialize(std::uint8_t*& buffer)
{
    return deserialize_from<T>(buffer);
}

template <typename T>
ROCPROFSYS_INLINE std::size_t
                  get_size(const T& item)
{
    return serialized_size(item);
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
struct is_enum_class
: std::bool_constant<std::is_enum_v<T> &&
                     !std::is_convertible_v<T, std::underlying_type_t<T>>>
{};

template <typename T>
inline constexpr bool is_enum_class_v = is_enum_class<T>::value;

template <typename T>
concept archive_serializable = requires(T v, output_archive& oa) { v.serialize(oa); };

template <typename T, typename TypeIdentifierEnum>
concept has_type_identifier = is_enum_class_v<TypeIdentifierEnum> && requires {
    { T::type_identifier } -> std::convertible_to<TypeIdentifierEnum>;
};

template <typename T, typename TypeIdentifierEnum>
concept cacheable = archive_serializable<T> && has_type_identifier<T, TypeIdentifierEnum>;

template <typename T, typename TypeIdentifierEnum, typename CacheableType>
concept sample_processor = requires(T t, TypeIdentifierEnum e, const CacheableType& c) {
    { t.execute_sample_processing(e, c) } -> std::same_as<void>;
};

}  // namespace type_traits
}  // namespace trace_cache
}  // namespace rocprofsys
