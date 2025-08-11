// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <array>
#include <initializer_list>
#include <ios>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

#if !defined(ROCPROFSYS_FOLD_EXPRESSION)
#    define ROCPROFSYS_FOLD_EXPRESSION(...) ((__VA_ARGS__), ...)
#endif

namespace rocprofsys
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
    if constexpr(std::is_pointer<std::decay_t<ArgT>>::value)
    {
        return (_v == nullptr) ? std::string{ "\"\"" }
                               : (std::string{ "\"" } + _v + std::string{ "\"" });
    }
    else
    {
        return std::string{ "\"" } + _v + std::string{ "\"" };
    }
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
    using delim_type = std::remove_cv_t<std::remove_reference_t<DelimT>>;

    std::stringstream _ss{};
    _ss << std::boolalpha;

    if constexpr(std::is_same<delim_type, char>::value)
    {
        const char _delim_c[2] = { _delim, '\0' };
        ROCPROFSYS_FOLD_EXPRESSION(_ss << _delim_c << _args);
        auto _ret = _ss.str();
        return (_ret.length() > 1) ? _ret.substr(1) : std::string{};
    }
    else
    {
        ROCPROFSYS_FOLD_EXPRESSION(_ss << _delim << _args);
        auto   _ret = _ss.str();
        auto&& _len = std::string{ _delim }.length();
        return (_ret.length() > _len) ? _ret.substr(_len) : std::string{};
    }
}

struct QuoteStrings
{};

template <typename DelimT, typename... Args>
auto
join(QuoteStrings&&, DelimT&& _delim, Args&&... _args)
{
    using delim_type = std::remove_cv_t<std::remove_reference_t<DelimT>>;

    std::stringstream _ss{};
    _ss << std::boolalpha;

    if constexpr(std::is_same<delim_type, char>::value)
    {
        const char _delim_c[2] = { _delim, '\0' };
        ROCPROFSYS_FOLD_EXPRESSION(_ss << _delim_c << as_string(_args));
        auto _ret = _ss.str();
        return (_ret.length() > 1) ? _ret.substr(1) : std::string{};
    }
    else
    {
        ROCPROFSYS_FOLD_EXPRESSION(_ss << _delim << as_string(_args));
        auto   _ret = _ss.str();
        auto&& _len = std::string{ _delim }.length();
        return (_ret.length() > _len) ? _ret.substr(_len) : std::string{};
    }
}

template <typename... Args>
auto
join(std::array<std::string_view, 3>&& _delim, Args&&... _args)
{
    return join("", std::get<0>(_delim),
                join(std::get<1>(_delim), std::forward<Args>(_args)...),
                std::get<2>(_delim));
}

template <typename... Args>
auto
join(QuoteStrings&&, std::array<std::string_view, 3>&& _delim, Args&&... _args)
{
    return join(QuoteStrings{}, "", std::get<0>(_delim),
                join(std::get<1>(_delim), std::forward<Args>(_args)...),
                std::get<2>(_delim));
}

template <typename DelimB, typename DelimT, typename DelimE, typename... Args>
auto
join(std::tuple<DelimB, DelimT, DelimE>&& _delim, Args&&... _args)
{
    return join("", std::get<0>(_delim),
                join(std::get<1>(_delim), std::forward<Args>(_args)...),
                std::get<2>(_delim));
}

template <typename DelimB, typename DelimT, typename DelimE, typename... Args>
auto
join(QuoteStrings&&, std::tuple<DelimB, DelimT, DelimE>&& _delim, Args&&... _args)
{
    return join(QuoteStrings{}, "", std::get<0>(_delim),
                join(std::get<1>(_delim), std::forward<Args>(_args)...),
                std::get<2>(_delim));
}
}  // namespace
}  // namespace common
}  // namespace rocprofsys
