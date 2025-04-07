// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/cxx/details/mpl.hpp>

#include <array>
#include <cstring>
#include <initializer_list>
#include <ios>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace rocprofiler
{
namespace sdk
{
namespace join
{
template <typename... ArgsT>
inline void
consume_args(ArgsT&&...)
{}

enum
{
    NoQuoteStrings = 0x0,
    QuoteStrings   = 0x1
};

template <size_t Idx>
struct triplet_config
{
    static constexpr auto index() { return Idx; }
    std::string_view      delimiter = {};
    std::string_view      prefix    = {};
    std::string_view      suffix    = {};
};

using generic_config = triplet_config<0>;
using array_config   = triplet_config<1>;
using pair_config    = triplet_config<2>;

struct config : generic_config
{
    using format_flags_t = std::ios_base::fmtflags;
    using base_type      = generic_config;

    config()                  = default;
    ~config()                 = default;
    config(const config&)     = default;
    config(config&&) noexcept = default;

    config& operator=(const config&) = default;
    config& operator=(config&&) noexcept = default;

    // converting constructor
    config(std::string_view _delim)
    : base_type{_delim}
    {}

    // converting constructor
    config(const char* const _delim)
    : base_type{_delim}
    {}

    config(generic_config _cfg)
    : base_type{_cfg}
    {}

    config(array_config _cfg)
    : array{_cfg}
    {}

    config(pair_config _cfg)
    : pair{_cfg}
    {}

    config(generic_config _generic, array_config _array)
    : base_type{_generic}
    , array{_array}
    {}

    config(generic_config _generic, pair_config _pair)
    : base_type{_generic}
    , pair{_pair}
    {}

    config(array_config _array, pair_config _pair)
    : array{_array}
    , pair{_pair}
    {}

    format_flags_t flags = std::ios_base::boolalpha;
    array_config   array = {", ", "[", "]"};
    pair_config    pair  = {", ", "{", "}"};
};

namespace impl
{
template <int TraitT, typename ArgT>
inline decltype(auto)
join_arg(config _cfg, ArgT&& _v)
{
    using arg_type = mpl::basic_identity_t<ArgT>;

    constexpr bool _is_string_type  = mpl::is_string_type<arg_type>::value;
    constexpr bool _is_iterable     = mpl::is_iterable<arg_type>(0);
    constexpr bool _has_traits_type = mpl::has_traits<arg_type>(0);
    constexpr bool _has_key_type    = mpl::has_key_type<arg_type>(0);
    constexpr bool _has_value_type  = mpl::has_value_type<arg_type>(0);
    constexpr bool _has_mapped_type = mpl::has_mapped_type<arg_type>(0);

    if constexpr(_is_string_type)
    {
        if constexpr(TraitT == QuoteStrings)
        {
            return std::string{"\""} + std::string{std::forward<ArgT>(_v)} + std::string{"\""};
        }
        else
        {
            return std::forward<ArgT>(_v);
        }
    }
    else if constexpr(_is_iterable && !_has_traits_type &&
                      (_has_value_type || (_has_key_type && _has_mapped_type)))
    {
        if constexpr(_has_key_type && _has_mapped_type)
        {
            std::stringstream _ss{};
            _ss.setf(_cfg.flags);
            for(auto&& itr : std::forward<ArgT>(_v))
                _ss << _cfg.array.delimiter << _cfg.pair.prefix << join_arg<TraitT>(_cfg, itr.first)
                    << _cfg.pair.delimiter << join_arg<TraitT>(_cfg, itr.second)
                    << _cfg.pair.suffix;
            auto   _ret = _ss.str();
            auto&& _len = _cfg.array.delimiter.length();
            return (_ret.length() > _len) ? (std::string{_cfg.array.prefix} + _ret.substr(_len) +
                                             std::string{_cfg.array.suffix})
                                          : std::string{};
        }
        else if constexpr(_has_value_type)
        {
            std::stringstream _ss{};
            _ss.setf(_cfg.flags);
            for(auto&& itr : std::forward<ArgT>(_v))
                _ss << _cfg.array.delimiter << join_arg<TraitT>(_cfg, itr);
            auto   _ret = _ss.str();
            auto&& _len = _cfg.array.delimiter.length();
            return (_ret.length() > _len) ? (std::string{_cfg.array.prefix} + _ret.substr(_len) +
                                             std::string{_cfg.array.suffix})
                                          : std::string{};
        }
    }
    else if constexpr(mpl::supports_ostream<ArgT>(0))
    {
        return std::forward<ArgT>(_v);
    }
    else
    {
        static_assert(_is_iterable, "Type is not iterable");
        static_assert(!_has_traits_type, "Type has a traits type");
        if constexpr(!_has_value_type)
        {
            static_assert(_has_key_type && _has_mapped_type,
                          "Type must have a key_type and mapped_type if there is no value_type");
        }
        else
        {
            static_assert(_has_value_type,
                          "Type must have a value_type if there is no key_type and mapped_type");
        }
        static_assert(std::is_empty<ArgT>::value,
                      "Error! argument type cannot be written to output stream");
    }
    // suppress any unused but set variable warnings
    consume_args(_is_string_type,
                 _is_iterable,
                 _has_traits_type,
                 _has_key_type,
                 _has_value_type,
                 _has_mapped_type);
}
}  // namespace impl

template <int TraitT = NoQuoteStrings, typename... Args>
auto
join(config _cfg, Args&&... _args)
{
    static_assert(std::is_trivially_copyable<config>::value,
                  "Error! config is not trivially copyable");

    std::stringstream _ss{};
    _ss.setf(_cfg.flags);
    ((_ss << ((mpl::is_empty(_args)) ? std::string_view{} : std::string_view{_cfg.delimiter})
          << impl::join_arg<TraitT>(_cfg, _args)),
     ...);
    auto   _ret = _ss.str();
    auto&& _len = _cfg.delimiter.length();
    auto   _cmp =
        strncmp(std::string_view{_ret}.data(), std::string_view{_cfg.delimiter}.data(), _len) == 0;
    return (_ret.length() > _len) ? (std::string{_cfg.prefix} +
                                     ((_cmp) ? _ret.substr(_len) : _ret) + std::string{_cfg.suffix})
                                  : std::string{};
}

template <int TraitT = NoQuoteStrings, typename... Args>
auto
join(std::array<std::string_view, 3>&& _delims, Args&&... _args)
{
    auto _cfg      = config{};
    _cfg.delimiter = _delims.at(0);
    _cfg.prefix    = _delims.at(1);
    _cfg.suffix    = _delims.at(2);
    return join(_cfg, std::forward<Args>(_args)...);
}

template <int TraitT = NoQuoteStrings,
          typename DelimT,
          typename... Args,
          std::enable_if_t<!mpl::is_basic_same<config, DelimT>::value, int> = 0>
auto
join(DelimT&& _delim, Args&&... _args)
{
    using delim_type = mpl::basic_identity_t<DelimT>;

    if constexpr(std::is_constructible<config, delim_type>::value)
    {
        auto _cfg = config{std::forward<DelimT>(_delim)};
        return join<TraitT>(_cfg, std::forward<Args>(_args)...);
    }
    else if constexpr(std::is_same<delim_type, char>::value)
    {
        auto       _cfg        = config{};
        const char _delim_c[2] = {_delim, '\0'};
        _cfg.delimiter         = _delim_c;
        return join<TraitT>(_cfg, std::forward<Args>(_args)...);
    }
    else
    {
        auto _cfg      = config{};
        _cfg.delimiter = std::string_view{_delim};
        return join<TraitT>(_cfg, std::forward<Args>(_args)...);
    }
}

template <typename ArgT>
auto
quoted(ArgT&& _arg)
{
    auto _cfg   = config{};
    _cfg.prefix = "\"";
    _cfg.suffix = "\"";
    return join(_cfg, std::forward<ArgT>(_arg));
}
}  // namespace join
}  // namespace sdk
}  // namespace rocprofiler
