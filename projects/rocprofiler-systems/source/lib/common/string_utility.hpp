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

/// @brief Convert a string to lowercase (ASCII).
/// @param value The string to convert.
/// @return A copy of @p value with every character lowercased.
[[nodiscard]] inline std::string
to_lower(std::string_view value)
{
    std::string str_copy{ value };

    std::ranges::transform(str_copy, str_copy.begin(), [](unsigned char chr) {
        return static_cast<char>(std::tolower(chr));
    });

    return str_copy;
}

/// @brief Convert a string to uppercase (ASCII).
/// @param value The string to convert.
/// @return A copy of @p value with every character uppercased.
[[nodiscard]] inline std::string
to_upper(std::string_view value)
{
    std::string str_copy{ value };

    std::ranges::transform(str_copy, str_copy.begin(), [](unsigned char chr) {
        return static_cast<char>(std::toupper(chr));
    });

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
[[nodiscard]] inline std::string
trim(std::string_view value)
{
    return std::string{ rtrim(ltrim(value)) };
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
}  // namespace rocprofsys::utility::string
