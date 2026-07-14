// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "stack_entry.h"

#include <string>
#include <vector>

namespace roctx_recordfn::detail
{

// Percent-encoding of the two characters that would otherwise collide with the
// marker-path grammar. The inverse decode lives with the Python readers
// (utils/inject_roctx/core.py decode_marker_name, utils/utils_analysis.py).
inline constexpr const char* kEncodedPercent = "%25";
inline constexpr const char* kEncodedSlash   = "%2F";

// Appends name to out with '%' and '/' percent-encoded so an embedded '/' is
// not read as the frame separator in build_marker_string.
inline void encode_marker_segment(const std::string& name, std::string& out)
{
    for (char c : name)
    {
        if (c == '%')
            out += kEncodedPercent;
        else if (c == '/')
            out += kEncodedSlash;
        else
            out += c;
    }
}

// Appends the frames to out as a '/'-separated list, using select_field to
// render each frame's chosen field.
template<typename SelectField>
void append_joined_frames(const std::vector<StackEntry>& stack, std::string& out, SelectField select_field)
{
    bool first = true;
    for (const auto& entry : stack)
    {
        if (!first)
            out += '/';
        select_field(entry, out);
        first = false;
    }
}

// Renders the stack as "marker1/.../markerN:context1/.../contextN". Marker names
// are percent-encoded so an embedded '/' is not read as the frame separator.
inline std::string build_marker_string(const std::vector<StackEntry>& stack)
{
    std::size_t marker_len = 0;
    std::size_t ctx_len    = 0;
    for (const auto& entry : stack)
    {
        marker_len += entry.marker.size() + 1;
        // Each '%' or '/' expands from one char to three when encoded.
        for (char c : entry.marker)
            if (c == '%' || c == '/')
                marker_len += 2;
        ctx_len += entry.context.size() + 1;
    }
    std::string out;
    out.reserve(marker_len + ctx_len + 1);

    append_joined_frames(stack,
                         out,
                         [](const StackEntry& entry, std::string& dst)
                         { encode_marker_segment(entry.marker, dst); });
    out += ':';
    append_joined_frames(stack,
                         out,
                         [](const StackEntry& entry, std::string& dst) { dst += entry.context; });
    return out;
}

}  // namespace roctx_recordfn::detail
