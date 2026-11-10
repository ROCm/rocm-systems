// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kineto_format.hpp"

#include <algorithm>
#include <charconv>

namespace profiler_hub::pytorch
{
namespace
{
bool
has_key(std::string_view text, std::string_view quoted_key)
{
    return text.find(quoted_key) != std::string_view::npos;
}

void
skip_space(std::string_view& text)
{
    text.remove_prefix(std::min(text.find_first_not_of(" \t\r\n"), text.size()));
}
}  // namespace

bool
looks_like_kineto(std::string_view text)
{
    if(text.starts_with("\xEF\xBB\xBF")) text.remove_prefix(3);
    const auto first = text.find_first_not_of(" \t\r\n");
    if(first == std::string_view::npos || text[first] != '{') return false;

    // Some traces have no schemaVersion, and some put distributedInfo past the probe
    // window, so the "cpu_op" category also identifies PyTorch profiler output.
    return has_key(text, "\"traceEvents\"") &&
           (has_key(text, "\"schemaVersion\"") || has_key(text, "\"deviceProperties\"") ||
            has_key(text, "\"distributedInfo\"") ||
            has_key(text, "\"baseTimeNanoseconds\"") || has_key(text, "\"cpu_op\""));
}

std::uint32_t
schema_version_in_prefix(std::string_view text)
{
    constexpr std::string_view key = "\"schemaVersion\"";
    for(auto at = text.find(key); at != std::string_view::npos;
        at      = text.find(key, at + 1))
    {
        auto rest = text.substr(at + key.size());
        skip_space(rest);
        if(!rest.starts_with(':')) continue;
        rest.remove_prefix(1);
        skip_space(rest);

        std::uint32_t value  = 0;
        const char*   last   = rest.data() + rest.size();
        const auto [end, ec] = std::from_chars(rest.data(), last, value);
        // A digit run that reaches the end of the prefix may be cut short.
        if(ec != std::errc{} || end == last || *end == '.' || *end == 'e' || *end == 'E')
        {
            return 0;
        }
        return value;
    }
    return 0;
}
}  // namespace profiler_hub::pytorch
