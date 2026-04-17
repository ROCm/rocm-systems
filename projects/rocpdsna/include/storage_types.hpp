// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

namespace rocpdsna
{

struct version_t
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;

    constexpr uint32_t get_major() const noexcept { return major; }
    constexpr uint32_t get_minor() const noexcept { return minor; }
    constexpr uint32_t get_patch() const noexcept { return patch; }

    constexpr bool operator==(const version_t& other) const noexcept
    {
        return major == other.major && minor == other.minor && patch == other.patch;
    }

    constexpr bool operator<(const version_t& other) const noexcept
    {
        if(major != other.major) return major < other.major;
        if(minor != other.minor) return minor < other.minor;
        return patch < other.patch;
    }

    constexpr bool operator!=(const version_t& other) const noexcept
    {
        return !(*this == other);
    }

    constexpr bool operator>(const version_t& other) const noexcept
    {
        return other < *this;
    }

    constexpr bool operator<=(const version_t& other) const noexcept
    {
        return !(other < *this);
    }

    constexpr bool operator>=(const version_t& other) const noexcept
    {
        return !(*this < other);
    }
};
}  // namespace rocpdsna
