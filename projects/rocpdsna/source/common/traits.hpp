
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace rocpdsna::common::traits
{

namespace
{
template <typename T>
struct is_string_literal_impl : std::false_type
{};

template <>
struct is_string_literal_impl<const char*> : std::true_type
{};

template <typename T>
inline constexpr bool is_string_literal_impl_v = is_string_literal_impl<T>::value;

}  // namespace

template <typename T>
constexpr bool
is_string_literal()
{
    using tp_t = std::decay_t<T>;
    return is_string_literal_impl_v<tp_t>;
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

}  // namespace rocpdsna::common::traits
