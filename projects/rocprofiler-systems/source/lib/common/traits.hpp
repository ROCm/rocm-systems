// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include <concepts>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

namespace rocprofsys
{
inline namespace common
{
namespace traits
{
template <typename T>
concept string_literal =
    std::same_as<std::decay_t<T>, std::string> ||
    std::same_as<std::decay_t<T>, std::string_view> ||
    std::same_as<std::decay_t<T>, const char*> || std::same_as<std::decay_t<T>, char*>;

template <typename T>
constexpr bool
is_string_literal()
{
    return string_literal<T>;
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
