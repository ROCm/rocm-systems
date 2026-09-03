// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <ratio>
#include <string_view>

namespace rocprofsys::inline common::units
{

namespace detail
{

inline constexpr std::intmax_t k_seconds_per_minute = 60;
inline constexpr std::intmax_t k_seconds_per_hour   = 3600;

}  // namespace detail

/**
 * Converts a duration expressed in (possibly non-finite or out-of-range) seconds
 * into std::chrono::nanoseconds, saturating to the representable range instead of
 * invoking undefined behaviour on a subsequent narrowing cast.
 *
 * Intended for period-from-frequency computations (e.g. `1.0 / freq_hz`), where a
 * misconfigured zero or negative frequency would otherwise yield +-inf or NaN.
 * Non-positive or non-finite input saturates to zero; input at or beyond the
 * representable range saturates to std::chrono::nanoseconds::max().
 */
[[nodiscard]] inline std::chrono::nanoseconds
seconds_to_duration(double seconds) noexcept
{
    using rep = std::chrono::nanoseconds::rep;
    // `!(seconds > 0.0)`, not `seconds <= 0.0`: also catches NaN, for which every
    // relational comparison (including <=) is false.
    if(!(seconds > 0.0))
    {
        return std::chrono::nanoseconds::zero();
    }
    constexpr double k_max_sec =
        static_cast<double>(std::numeric_limits<rep>::max()) / std::nano::den;
    return seconds >= k_max_sec ? std::chrono::nanoseconds::max()
                                : std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::duration<double>{ seconds });
}

/**
 * Printable suffix for a std::chrono::duration @p Period (e.g. "ms" for
 * std::milli).
 *
 * Left undefined for unknown periods so that printing an unregistered unit is a
 * compile error; specialise it to add a new unit's suffix.
 */
template <typename Period>
struct duration_suffix;

template <>
struct duration_suffix<std::nano>
{
    static constexpr std::string_view k_value = "ns";
};
template <>
struct duration_suffix<std::micro>
{
    static constexpr std::string_view k_value = "us";
};
template <>
struct duration_suffix<std::milli>
{
    static constexpr std::string_view k_value = "ms";
};
template <>
struct duration_suffix<std::ratio<1>>
{
    static constexpr std::string_view k_value = "s";
};
template <>
struct duration_suffix<std::ratio<detail::k_seconds_per_minute>>
{
    static constexpr std::string_view k_value = "min";
};
template <>
struct duration_suffix<std::ratio<detail::k_seconds_per_hour>>
{
    static constexpr std::string_view k_value = "h";
};

}  // namespace rocprofsys::inline common::units
