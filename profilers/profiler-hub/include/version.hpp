// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace profiler_hub
{

struct version_t
{
    uint32_t major = 0;
    uint32_t minor = 0;
    uint32_t patch = 0;

    // {0, 0, 0} is the "use latest available" sentinel.
    [[nodiscard]] bool is_latest() const noexcept
    {
        return major == 0 && minor == 0 && patch == 0;
    }

    [[nodiscard]] std::string to_string() const
    {
        return std::to_string(major) + "." + std::to_string(minor) + "." +
               std::to_string(patch);
    }

    friend bool operator==(const version_t& lhs, const version_t& rhs) noexcept
    {
        return lhs.major == rhs.major && lhs.minor == rhs.minor && lhs.patch == rhs.patch;
    }

    friend bool operator<(const version_t& lhs, const version_t& rhs) noexcept
    {
        if(lhs.major != rhs.major) return lhs.major < rhs.major;
        if(lhs.minor != rhs.minor) return lhs.minor < rhs.minor;
        return lhs.patch < rhs.patch;
    }
};
}  // namespace profiler_hub
