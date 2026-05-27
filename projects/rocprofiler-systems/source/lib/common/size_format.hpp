// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <spdlog/fmt/fmt.h>

#include <cstdint>
#include <optional>
#include <string>

namespace rocprofsys
{
inline namespace common
{
[[nodiscard]] inline std::string
format_size_human(std::uintmax_t bytes)
{
    constexpr std::uintmax_t KIB = 1024ULL;
    constexpr std::uintmax_t MIB = KIB * 1024ULL;
    constexpr std::uintmax_t GIB = MIB * 1024ULL;

    if(bytes < KIB) return fmt::format("{} B", bytes);
    if(bytes < MIB) return fmt::format("{:.2f} KiB", static_cast<double>(bytes) / KIB);
    if(bytes < GIB) return fmt::format("{:.2f} MiB", static_cast<double>(bytes) / MIB);
    return fmt::format("{:.2f} GiB", static_cast<double>(bytes) / GIB);
}

[[nodiscard]] inline std::string
format_size_human(std::optional<std::uintmax_t> size_bytes)
{
    if(!size_bytes) return "?";
    return format_size_human(*size_bytes);
}

}  // namespace common
}  // namespace rocprofsys
