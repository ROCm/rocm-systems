// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <ios>
#include <sstream>
#include <string>
#include <type_traits>

#if !defined(OMNITRACE_FOLD_EXPRESSION)
#    define OMNITRACE_FOLD_EXPRESSION(...) ((__VA_ARGS__), ...)
#endif

namespace omnitrace
{
inline namespace common
{
namespace
{
template <typename Tp>
struct is_string_impl : std::false_type
{};

template <>
struct is_string_impl<std::string> : std::true_type
{};

template <>
struct is_string_impl<std::string_view> : std::true_type
{};

template <>
struct is_string_impl<const char*> : std::true_type
{};

template <>
struct is_string_impl<char*> : std::true_type
{};

template <typename Tp>
struct is_string : is_string_impl<std::remove_cv_t<std::decay_t<Tp>>>
{};

template <typename ArgT>
auto
as_string(ArgT&& _v, std::enable_if_t<is_string<ArgT>::value, int> = 0)
{
    return std::string{ "\"" } + _v + std::string{ "\"" };
}

template <typename ArgT>
auto
as_string(ArgT&& _v, std::enable_if_t<!is_string<ArgT>::value, long> = 0)
{
    return _v;
}

template <typename DelimT, typename... Args>
auto
join(DelimT&& _delim, Args&&... _args)
{
    std::stringstream _ss{};
    _ss << std::boolalpha;
    OMNITRACE_FOLD_EXPRESSION(_ss << _delim << as_string(_args));
    auto _ret = _ss.str();
    if constexpr(std::is_same<DelimT, char>::value)
    {
        return (_ret.length() > 1) ? _ret.substr(1) : std::string{};
    }
    else
    {
        auto&& _len = std::string{ _delim }.length();
        return (_ret.length() > _len) ? _ret.substr(_len) : std::string{};
    }
}
}  // namespace
}  // namespace common
}  // namespace omnitrace
