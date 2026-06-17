// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/units.hpp"

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
    if(bytes < units::kilobyte) return fmt::format("{} B", bytes);
    if(bytes < units::megabyte)
        return fmt::format("{:.2f} KB", static_cast<double>(bytes) / units::kilobyte);
    if(bytes < units::gigabyte)
        return fmt::format("{:.2f} MB", static_cast<double>(bytes) / units::megabyte);
    return fmt::format("{:.2f} GB", static_cast<double>(bytes) / units::gigabyte);
}

[[nodiscard]] inline std::string
format_size_human(std::optional<std::uintmax_t> size_bytes)
{
    if(!size_bytes) return "?";
    return format_size_human(*size_bytes);
}

}  // namespace common
}  // namespace rocprofsys
