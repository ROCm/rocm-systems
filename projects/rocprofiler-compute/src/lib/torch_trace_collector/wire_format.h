// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <string_view>

namespace torch_trace_collector::detail
{

inline constexpr const char* kEncodedPercent = "%25";
inline constexpr const char* kEncodedSlash   = "%2F";
inline constexpr const char* kUnavailable    = "n/a";

inline std::string encode_marker_name(std::string_view name)
{
    std::string out;
    out.reserve(name.size());
    for(char c : name)
    {
        if(c == '%')
        {
            out += kEncodedPercent;
        }
        else if(c == '/')
        {
            out += kEncodedSlash;
        }
        else
        {
            out += c;
        }
    }
    return out;
}

inline std::string build_range_name(std::string_view name,
                                    std::string_view context,
                                    std::string_view seq,
                                    std::string_view tid,
                                    std::string_view ftid,
                                    std::string_view scope,
                                    std::string_view args,
                                    std::string_view backend)
{
    std::string out = encode_marker_name(name);
    out += ':';
    out += context;
    out += "|seq=";
    out += seq;
    out += "|tid=";
    out += tid;
    out += "|ftid=";
    out += ftid;
    out += "|scope=";
    out += scope;
    out += "|args=";
    out += args;
    if(!backend.empty())
    {
        out += '|';
        out += backend;
    }
    return out;
}

}  // namespace torch_trace_collector::detail
