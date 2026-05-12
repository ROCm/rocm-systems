// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

namespace rocprofsys::common
{

inline std::string
format_size_human(std::optional<std::uintmax_t> size_bytes)
{
    if(!size_bytes) return "?";

    constexpr std::uintmax_t KIB = 1024ULL;
    constexpr std::uintmax_t MIB = KIB * 1024ULL;
    constexpr std::uintmax_t GIB = MIB * 1024ULL;

    const auto value = *size_bytes;

    double      scaled = 0.0;
    const char* unit   = nullptr;
    if(value < MIB)
    {
        scaled = static_cast<double>(value) / static_cast<double>(KIB);
        unit   = "KB";
    }
    else if(value < GIB)
    {
        scaled = static_cast<double>(value) / static_cast<double>(MIB);
        unit   = "MB";
    }
    else
    {
        scaled = static_cast<double>(value) / static_cast<double>(GIB);
        unit   = "GB";
    }

    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f %s", scaled, unit);
    return std::string{ buffer };
}

}  // namespace rocprofsys::common
