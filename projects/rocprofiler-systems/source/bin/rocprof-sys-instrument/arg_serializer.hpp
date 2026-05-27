// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Serialization helpers for the trace-args payload consumed by
// rocprofsys_push_trace_with_args

#include "core/demangler.hpp"

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

// These aliases mirror the ones declared in fwd.hpp
using string_t       = std::string;
using string_view_t  = std::string_view;
using stringstream_t = std::stringstream;

template <typename NameT>
inline string_view_t
get_serialized_arg_name(NameT&& name)
{
    using raw_type   = std::remove_reference_t<NameT>;
    using value_type = std::decay_t<NameT>;
    static_assert(std::is_same_v<value_type, string_t> ||
                      std::is_same_v<value_type, string_view_t> ||
                      std::is_same_v<value_type, const char*> ||
                      std::is_same_v<value_type, char*>,
                  "serialized argument names must be string_t, string_view_t, "
                  "or a C string");

    if constexpr(std::is_pointer_v<raw_type>)
    {
        return name ? string_view_t{ name } : string_view_t{};
    }
    else
    {
        return string_view_t{ name };
    }
}

template <typename Tp>
inline string_t
get_serialized_arg_type()
{
    using value_type = std::decay_t<Tp>;
    if constexpr(std::is_same_v<value_type, string_t> ||
                 std::is_same_v<value_type, string_view_t> ||
                 std::is_same_v<value_type, const char*> ||
                 std::is_same_v<value_type, char*>)
    {
        return "string";
    }
    else
    {
        return rocprofsys::utility::demangle<value_type>();
    }
}

template <typename Tp>
inline string_t
get_serialized_arg_value(Tp&& value)
{
    using raw_type   = std::remove_reference_t<Tp>;
    using value_type = std::decay_t<Tp>;

    if constexpr(std::is_pointer_v<raw_type> &&
                 (std::is_same_v<value_type, const char*> ||
                  std::is_same_v<value_type, char*>) )
    {
        return value ? string_t{ value } : string_t{};
    }
    else if constexpr(std::is_same_v<value_type, string_t> ||
                      std::is_same_v<value_type, string_view_t>)
    {
        return string_t{ std::forward<Tp>(value) };
    }
    else
    {
        stringstream_t ss;
        ss << std::forward<Tp>(value);
        return ss.str();
    }
}

// Appends serialized arguments directly to the output string in the format:
//   <arg_number>;;<arg_type>;;<arg_name>;;<arg_value>;;
template <typename NameT, typename ValueT, typename... TailT>
inline size_t
append_serialized_args(string_t& out, size_t idx, NameT&& name, ValueT&& value,
                       bool enabled, TailT&&... tail)
{
    constexpr const char* kDelim = ";;";

    if(enabled)
    {
        auto arg_name = get_serialized_arg_name(std::forward<NameT>(name));
        if(!arg_name.empty())
        {
            out.append(std::to_string(idx));
            out.append(kDelim);
            out.append(get_serialized_arg_type<ValueT>());
            out.append(kDelim);
            out.append(arg_name);
            out.append(kDelim);
            out.append(get_serialized_arg_value(std::forward<ValueT>(value)));
            out.append(kDelim);
            ++idx;
        }
    }

    if constexpr(sizeof...(TailT) > 0)
    {
        return append_serialized_args(out, idx, std::forward<TailT>(tail)...);
    }

    return idx;
}

// Builds a serialized string of arguments for rocprofsys_push_trace_with_args.
// Output format is: <arg_number>;;<arg_type>;;<arg_name>;;<arg_value>;;
template <typename... Args>
inline string_t
rocprofsys_get_serialized_args(Args&&... args)
{
    static_assert(sizeof...(Args) % 3 == 0,
                  "serialized args must be passed as name/value/enabled triples");

    string_t out;
    if constexpr(sizeof...(Args) > 0)
    {
        out.reserve((sizeof...(Args) / 3) * 64);
        append_serialized_args(out, 0, std::forward<Args>(args)...);
    }
    return out;
}
