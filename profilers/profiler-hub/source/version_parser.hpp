// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/version.hpp"

#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>

namespace profiler_hub
{

/** Parses an unsigned integer that occupies the whole text. */
inline std::optional<std::uint32_t>
parse_u32(std::string_view text)
{
    if(text.empty()) return std::nullopt;

    std::uint32_t value  = 0;
    const char*   begin  = text.data();
    const char*   end    = begin + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value);
    if(ec != std::errc{} || ptr != end) return std::nullopt;
    return value;
}

/**
 * Parses major[.minor[.patch]] text into a version.
 * "3" is 3.0.0. "3.0" is 3.0.0. "3.0.1" fills all three fields.
 * Each component must be a whole-text unsigned integer. A trailing dot,
 * an empty component, or a fourth component is rejected.
 */
inline std::optional<version_t>
parse_version(std::string_view text)
{
    version_t            version{ 0, 0, 0 };
    std::uint32_t* const fields[] = { &version.major, &version.minor, &version.patch };
    std::size_t          count    = 0;
    std::size_t          start    = 0;

    while(start <= text.size() && count < 3)
    {
        const std::size_t      dot  = text.find('.', start);
        const std::string_view part = text.substr(
            start, dot == std::string_view::npos ? text.size() - start : dot - start);
        const auto parsed = parse_u32(part);
        if(!parsed.has_value()) return std::nullopt;
        *fields[count] = parsed.value();
        ++count;
        if(dot == std::string_view::npos)
        {
            start = text.size();
            break;
        }
        start = dot + 1;
        if(start == text.size()) return std::nullopt;
    }
    if(count == 0 || start < text.size()) return std::nullopt;
    return version;
}

}  // namespace profiler_hub
