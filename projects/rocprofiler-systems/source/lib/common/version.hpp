// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <fmt/format.h>

#include <compare>  // NOLINT(misc-include-cleaner)
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

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

/**
 * @brief Compare the version a library was built against with the version loaded.
 *
 * @param library_name Name used in the message.
 * @param built Version the headers were configured against.
 * @param loaded Version reported by the library at runtime.
 * @return `std::nullopt` when the versions are equal, otherwise a message describing
 * the mismatch.
 */
[[nodiscard]] inline std::optional<std::string>
describe_version_mismatch(std::string_view library_name, const version& built,
                          const version& loaded)
{
    if(built == loaded)
    {
        return std::nullopt;
    }

    return fmt::format("{} version mismatch: built against v{}.{}.{} but v{}.{}.{} is "
                       "loaded",
                       library_name, built.major, built.minor, built.patch, loaded.major,
                       loaded.minor, loaded.patch);
}

}  // namespace rocprofsys::inline common
