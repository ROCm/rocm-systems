// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace profiler_hub::common::traits
{

namespace impl
{
template <typename T>
struct is_string_literal_impl : std::false_type
{};

template <>
struct is_string_literal_impl<const char*> : std::true_type
{};

template <typename T>
inline constexpr bool is_string_literal_impl_v = is_string_literal_impl<T>::value;

}  // namespace impl

template <typename T>
constexpr bool
is_string_literal()
{
    using tp_t = std::decay_t<T>;
    return impl::is_string_literal_impl_v<tp_t>;
}

template <typename T>
struct is_optional : std::false_type
{};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type
{};

template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

template <typename T>
struct is_std_unordered_map : std::false_type
{};

template <typename K, typename V, typename Hash, typename KeyEqual, typename Alloc>
struct is_std_unordered_map<std::unordered_map<K, V, Hash, KeyEqual, Alloc>>
: std::true_type
{};

template <typename T>
inline constexpr bool is_unordered_map_v = is_std_unordered_map<T>::value;

template <typename T>
static constexpr bool is_int64_bindable_v =
    std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ||
    std::is_same_v<T, size_t>;

template <typename T>
static constexpr bool is_int32_bindable_v =
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>;

template <typename T>
static constexpr bool is_text_bindable_v = std::is_same_v<T, const char*>;

template <typename T>
static constexpr bool is_string_bindable_v = std::is_same_v<T, std::string>;

template <typename T>
static constexpr bool is_double_bindable_v = std::is_floating_point_v<T>;

template <typename T>
static constexpr bool is_supported_bind_type_v =
    is_int64_bindable_v<T> || is_int32_bindable_v<T> || is_text_bindable_v<T> ||
    is_double_bindable_v<T>;

}  // namespace profiler_hub::common::traits
