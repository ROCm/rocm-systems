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

    /**
     * @brief Format the version as "MAJOR.MINOR.PATCH".
     * @return The version string, for example "3.0.1"
     */
    [[nodiscard]] std::string to_string() const
    {
        return std::to_string(major) + "." + std::to_string(minor) + "." +
               std::to_string(patch);
    }

    /**
     * @brief Compare two versions field by field.
     * @param lhs Left-hand version
     * @param rhs Right-hand version
     * @return true when major, minor, and patch are all equal
     */
    friend bool operator==(const version_t& lhs, const version_t& rhs) noexcept
    {
        return lhs.major == rhs.major && lhs.minor == rhs.minor && lhs.patch == rhs.patch;
    }

    /**
     * @brief Ordering for versions, compared major first, then minor, then patch.
     * @param lhs Left-hand version
     * @param rhs Right-hand version
     * @return true when `lhs` is strictly older than `rhs`
     */
    friend bool operator<(const version_t& lhs, const version_t& rhs) noexcept
    {
        if(lhs.major != rhs.major) return lhs.major < rhs.major;
        if(lhs.minor != rhs.minor) return lhs.minor < rhs.minor;
        return lhs.patch < rhs.patch;
    }
};
}  // namespace profiler_hub
