// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "track_read_options.hpp"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <system_error>

namespace profiler_hub
{

size_t
parse_size(const char* text, size_t fallback) noexcept
{
    if(text == nullptr || text[0] == '\0') return fallback;

    const char* const last = text + std::strlen(text);
    size_t            value{};
    const auto [end, error] = std::from_chars(text, last, value);
    if(error != std::errc{} || end != last) return fallback;
    return value;
}

track_read_options
track_read_options::from_env(const env_lookup_t& lookup)
{
    track_read_options options;
    options.parallel_read_min_events =
        parse_size(lookup("PH_READ_MIN_EVENTS"), options.parallel_read_min_events);
    options.parallel_read_parts = std::max<size_t>(
        1, parse_size(lookup("PH_READ_PARTS"), options.parallel_read_parts));
    return options;
}

}  // namespace profiler_hub
