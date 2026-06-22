// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <initializer_list>
#include <ios>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

#include "traits.hpp"

#if !defined(ROCPROFSYS_FOLD_EXPRESSION)
#    define ROCPROFSYS_FOLD_EXPRESSION(...) ((__VA_ARGS__), ...)
#endif

namespace rocprofsys
{
inline namespace common
{
namespace
{
template <typename ArgT>
    requires traits::string_literal<ArgT>
auto
as_string(const ArgT& _v)
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
    requires(!traits::string_literal<ArgT>)
auto
as_string(const ArgT& _v)
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
