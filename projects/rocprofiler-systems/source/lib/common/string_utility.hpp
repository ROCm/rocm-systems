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

namespace rocprofsys::utility::string
{

inline std::string
to_lower(std::string_view value)
{
    std::string str_copy{ value };

    std::ranges::transform(str_copy, str_copy.begin(), [](unsigned char chr) {
        return static_cast<char>(std::tolower(chr));
    });

    return str_copy;
}

inline std::string
to_upper(std::string_view value)
{
    std::string str_copy{ value };

    std::ranges::transform(str_copy, str_copy.begin(), [](unsigned char chr) {
        return static_cast<char>(std::toupper(chr));
    });

    return str_copy;
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
    // trim leading/trailing whitespace before interpreting
    constexpr std::string_view k_whitespace = " \t\n\r\f\v";
    const auto                 first_pos    = value.find_first_not_of(k_whitespace);
    if(first_pos == std::string_view::npos)
    {
        return fallback;  // empty or all whitespace
    }
    const auto last_pos = value.find_last_not_of(k_whitespace);
    value               = value.substr(first_pos, last_pos - first_pos + 1);

    if(value.find_first_not_of("0123456789") == std::string_view::npos)
    {
        std::uint64_t numeric{};
        const auto*   last   = value.data() + value.size();
        const auto [ptr, ec] = std::from_chars(value.data(), last, numeric);
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

    std::string lower{ to_lower(value) };

    constexpr auto k_false_values = std::array{
        std::string_view{ "off" }, std::string_view{ "false" }, std::string_view{ "no" },
        std::string_view{ "n" },   std::string_view{ "f" },
    };
    return !std::ranges::any_of(k_false_values,
                                [&lower](std::string_view val) { return lower == val; });
}
}  // namespace rocprofsys::utility::string
