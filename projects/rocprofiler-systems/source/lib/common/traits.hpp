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
