// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <spdlog/fmt/fmt.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unistd.h>

namespace rocprofsys::inline common::units
{
inline constexpr std::int64_t nsec = 1;
inline constexpr std::int64_t usec = 1000 * nsec;
inline constexpr std::int64_t msec = 1000 * usec;
inline constexpr std::int64_t sec  = 1000 * msec;

inline constexpr std::int64_t byte     = 1;
inline constexpr std::int64_t kilobyte = 1000 * byte;
inline constexpr std::int64_t megabyte = 1000 * kilobyte;
inline constexpr std::int64_t gigabyte = 1000 * megabyte;

inline constexpr std::int64_t nanowatt = 1;
inline constexpr std::int64_t watt     = 1000 * 1000 * 1000 * nanowatt;

inline std::int64_t
get_page_size()
{
    static const std::int64_t page_size = ::sysconf(_SC_PAGESIZE);
    return page_size;
}

}  // namespace rocprofsys::inline common::units

namespace rocprofsys::inline common
{
inline constexpr std::string_view UNKNOWN_VALUE_PLACEHOLDER = "?";

[[nodiscard]] inline std::string
datasize_to_string(std::uint64_t bytes)
{
    if(bytes < units::kilobyte) return fmt::format("{} B", bytes);
    if(bytes < units::megabyte)
        return fmt::format("{:.2f} KB", static_cast<double>(bytes) / units::kilobyte);
    if(bytes < units::gigabyte)
        return fmt::format("{:.2f} MB", static_cast<double>(bytes) / units::megabyte);
    return fmt::format("{:.2f} GB", static_cast<double>(bytes) / units::gigabyte);
}

}  // namespace rocprofsys::inline common
