// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>

namespace rocprofsys::utility::string
{
/// @brief Convert a string to lowercase (ASCII) in place.
/// @param value The string which will be modified.
inline void
to_lower_in_place(std::string& value)
{
    std::ranges::transform(value, value.begin(), [](unsigned char chr) {
        return static_cast<char>(std::tolower(chr));
    });
}

/// @brief Convert a string to lowercase (ASCII).
/// @param value The string to convert.
/// @return A copy of @p value with every character lowercased.
[[nodiscard]] inline std::string
to_lower(std::string_view value)
{
    std::string str_copy{ value };
    to_lower_in_place(str_copy);
    return str_copy;
}

/// @brief Convert a string to uppercase (ASCII) in place.
/// @param value The string which will be modified.
inline void
to_upper_in_place(std::string& value)
{
    std::ranges::transform(value, value.begin(), [](unsigned char chr) {
        return static_cast<char>(std::toupper(chr));
    });
}

/// @brief Convert a string to uppercase (ASCII).
/// @param value The string to convert.
/// @return A copy of @p value with every character uppercased.
[[nodiscard]] inline std::string
to_upper(std::string_view value)
{
    std::string str_copy{ value };
    to_upper_in_place(str_copy);
    return str_copy;
}

/// @brief Strip leading whitespace (" \t\n\r\f\v"). Zero-copy: returns a view
///        into @p value, safe for hot parsing paths.
/// @param value The string to trim.
/// @return A view of @p value with leading whitespace removed; empty if
///         @p value is empty or all whitespace.
[[nodiscard]] inline std::string_view
ltrim(std::string_view value) noexcept
{
    constexpr std::string_view k_whitespace = " \t\n\r\f\v";
    const auto                 pos          = value.find_first_not_of(k_whitespace);
    return pos == std::string_view::npos ? std::string_view{} : value.substr(pos);
}

/// @brief Strip trailing whitespace (" \t\n\r\f\v"). Zero-copy: returns a view
///        into @p value, safe for hot parsing paths.
/// @param value The string to trim.
/// @return A view of @p value with trailing whitespace removed; empty if
///         @p value is empty or all whitespace.
[[nodiscard]] inline std::string_view
rtrim(std::string_view value) noexcept
{
    constexpr std::string_view k_whitespace = " \t\n\r\f\v";
    const auto                 pos          = value.find_last_not_of(k_whitespace);
    return pos == std::string_view::npos ? std::string_view{} : value.substr(0, pos + 1);
}

/// @brief Strip leading and trailing whitespace (" \t\n\r\f\v").
/// @param value The string to trim.
/// @return A copy of @p value with leading/trailing whitespace removed; empty
///         if @p value is empty or all whitespace.
[[nodiscard]] inline std::string_view
trim(std::string_view value)
{
    return rtrim(ltrim(value));
}

[[nodiscard]] constexpr bool
equals_ignore_case(std::string_view lhs, std::string_view rhs) noexcept
{
    return std::ranges::equal(lhs, rhs, [](char left, char right) {
        return std::tolower(left) == std::tolower(right);
    });
}

/// @brief Parse a string into a boolean.
///
/// Leading and trailing whitespace is trimmed before interpretation. All-digit
/// strings are truthy when non-zero (an overflowing digit string is also truthy);
/// other values are matched case-insensitively against the false tokens
/// off/false/no/n/f/0 (anything else is truthy). An empty or all-whitespace string
/// yields @p fallback.
/// @param value    The string to interpret.
/// @param fallback Returned when @p value is empty or all whitespace.
/// @return The parsed boolean.
[[nodiscard]] inline bool
to_bool(std::string_view value, bool fallback = false)
{
    const auto trimmed = trim(value);
    if(trimmed.empty())
    {
        return fallback;  // empty or all whitespace
    }

    if(trimmed.find_first_not_of("0123456789") == std::string::npos)
    {
        std::uint64_t numeric{};
        const auto*   last   = trimmed.data() + trimmed.size();
        const auto [ptr, ec] = std::from_chars(trimmed.data(), last, numeric);
        if(ec == std::errc::result_out_of_range)
        {
            return true;
        }
        if(ec == std::errc{} && ptr == last)
        {
            return numeric != 0;
        }
        return true;
    }

    std::string lower{ to_lower(trimmed) };

    constexpr auto k_false_values = std::array{
        std::string_view{ "off" }, std::string_view{ "false" }, std::string_view{ "no" },
        std::string_view{ "n" },   std::string_view{ "f" },
    };
    return !std::ranges::any_of(k_false_values,
                                [&lower](std::string_view val) { return lower == val; });
}

/// @brief Normalize a POSIX clock identifier name, e.g. "CLOCK_MONOTONIC" -> "monotonic".
/// @param value The clock identifier name.
/// @return The lowercased name with the "clock_" prefix stripped;
///         "process_cputime_id" is further mapped to "cputime".
[[nodiscard]] inline std::string
clock_name(std::string_view value)
{
    constexpr std::string_view k_clock_prefix = "clock_";

    std::string name{ to_lower(value) };
    if(name.starts_with(k_clock_prefix))
    {
        name = name.substr(k_clock_prefix.length());
    }
    if(name == "process_cputime_id")
    {
        name = "cputime";
    }
    return name;
}

/// @brief Strip the "rocprofsys_" prefix from an environment/setting name.
/// @param value The environment variable or setting name.
/// @return The lowercased name with the "rocprofsys_" prefix removed, if present.
[[nodiscard]] inline std::string
strip_rocprofsys_prefix(std::string_view value)
{
    constexpr std::string_view k_rocprofsys_prefix = "rocprofsys_";

    std::string name{ to_lower(value) };
    if(name.starts_with(k_rocprofsys_prefix))
    {
        return name.substr(k_rocprofsys_prefix.length());
    }
    return name;
}

}  // namespace rocprofsys::utility::string
