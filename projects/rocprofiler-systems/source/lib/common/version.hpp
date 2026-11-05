// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <compare>  // NOLINT(misc-include-cleaner)
#include <cstdint>

namespace rocprofsys::inline common
{

struct version
{
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t patch = 0;

    [[nodiscard]] constexpr auto formatted() const
    {
        constexpr auto major_multiplier = 10000U;
        constexpr auto minor_multiplier = 100U;
        return (major * major_multiplier) + (minor * minor_multiplier) + patch;
    }

    [[nodiscard]] static constexpr version from_formatted(std::uint32_t formatted)
    {
        constexpr auto major_multiplier          = 10000u;
        constexpr auto minor_multiplier          = 100u;
        constexpr auto version_component_modulus = 100u;  // keep 2 digits
        return version{ .major = formatted / major_multiplier,
                        .minor =
                            (formatted / minor_multiplier) % version_component_modulus,
                        .patch = formatted % version_component_modulus };
    }

    constexpr auto operator<=>(const version&) const = default;
};

}  // namespace rocprofsys::inline common
