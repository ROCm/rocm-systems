// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kineto_format.hpp"

namespace profiler_hub::pytorch
{
namespace
{
bool
has_key(std::string_view text, std::string_view quoted_key)
{
    return text.find(quoted_key) != std::string_view::npos;
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
}  // namespace profiler_hub::pytorch
