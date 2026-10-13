// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/version.hpp"

#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>

namespace profiler_hub::data_storage
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
 * Parses an older rocpd `schema_version` value.
 * "3" is major 3 with minor and patch 0. "3.0.1" fills all three fields.
 * "3.0" is major 3, minor 0, patch 0.
 */
inline std::optional<profiler_hub::version_t>
parse_schema_version_text(std::string_view text)
{
    profiler_hub::version_t version{ 0, 0, 0 };
    std::uint32_t* const    fields[] = { &version.major, &version.minor, &version.patch };
    std::size_t             count    = 0;
    std::size_t             start    = 0;

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

}  // namespace profiler_hub::data_storage
