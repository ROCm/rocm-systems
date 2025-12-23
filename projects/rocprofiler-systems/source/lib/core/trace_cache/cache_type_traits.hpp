// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include "common/span.hpp"
#include <cstdint>
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
serialize(uint8_t*, const T&)
{
    static_assert(type_traits::always_false<T>::value, "serialize<T> not specialized");
}

template <typename T>
T
deserialize(uint8_t*&)
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
static constexpr bool is_string_view_v =
    std::is_same_v<std::decay_t<T>, std::string_view>;

template <typename T>
inline constexpr bool is_supported_type_v =
    is_span_v<T> || std::is_integral_v<T> || std::is_floating_point_v<T> ||
    is_string_view_v<T> || is_vector_v<T>;

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
struct has_serialize<T, std::void_t<decltype(serialize(std::declval<uint8_t*>(),
                                                       std::declval<const T&>()))>>
: std::true_type
{};

template <typename T, typename = void>
struct has_deserialize : std::false_type
{};

template <typename T>
struct has_deserialize<
    T, void_t<std::is_same<decltype(deserialize<T>(std::declval<uint8_t*&>())), T>>>
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
