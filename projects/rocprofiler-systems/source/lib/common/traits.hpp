// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include <optional>
#include <string>
#include <type_traits>

namespace rocprofsys
{
inline namespace common
{
namespace traits
{

namespace
{
template <typename T>
struct is_string_literal_impl : std::false_type
{};

template <>
struct is_string_literal_impl<std::string_view> : std::true_type
{};

template <>
struct is_string_literal_impl<const char*> : std::true_type
{};

template <>
struct is_string_literal_impl<char*> : std::true_type
{};

template <>
struct is_string_literal_impl<std::string> : std::true_type
{};

template <typename T>
inline constexpr bool is_string_literal_impl_v = is_string_literal_impl<T>::value;

}  // namespace

template <typename T>
constexpr bool
is_string_literal()
{
    using Tp = std::decay_t<T>;
    return is_string_literal_impl_v<Tp>;
}

template <typename T>
struct is_optional : std::false_type
{};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type
{};

template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

}  // namespace traits
}  // namespace common
}  // namespace rocprofsys
