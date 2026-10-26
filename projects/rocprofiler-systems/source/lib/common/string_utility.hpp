// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
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

/// @brief Check whether a string is non-empty and consists only of ASCII digits.
[[nodiscard]] constexpr bool
is_numeric(std::string_view value) noexcept
{
    return !value.empty() &&
           std::ranges::all_of(value, [](char chr) { return chr >= '0' && chr <= '9'; });
}

/// @brief Number of decimal digits needed to print @p value (0 -> 1).
[[nodiscard]] constexpr size_t
decimal_width(size_t value) noexcept
{
    constexpr std::size_t k_ten = 10;
    size_t                width = 1;
    for(; value >= k_ten; value /= k_ten)
    {
        ++width;
    }
    return width;
}

/// @brief Non-owning key/value views of a "KEY=VALUE" entry.
struct key_value
{
    std::string_view key;
    std::string_view value;
};

/// @brief Split a "KEY=VALUE" string at the first '='.
/// @return The key/value views, or nullopt if @p entry has no '='.
[[nodiscard]] constexpr std::optional<key_value>
split_key_value(std::string_view entry) noexcept
{
    auto const eq_pos = entry.find('=');
    if(eq_pos == std::string_view::npos)
    {
        return std::nullopt;
    }
    return key_value{ .key = entry.substr(0, eq_pos), .value = entry.substr(eq_pos + 1) };
}

/// @brief Check whether a "KEY=VALUE" entry belongs to @p key.
[[nodiscard]] constexpr bool
has_env_key(std::string_view entry, std::string_view key) noexcept
{
    return entry.starts_with(key) && entry.size() > key.size() &&
           entry[key.size()] == '=';
}

/// @brief Strictly parse a whole string as an integral Tp; nullopt on empty, partial or
///        overflow.
template <std::integral Tp>
[[nodiscard]] inline std::optional<Tp>
safe_from_chars(std::string_view s)
{
    if(s.empty())
    {
        return std::nullopt;
    }
    Tp         value  = 0;
    const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    // Reject partial parses (e.g. "12.5" -> 12) by checking entire string was consumed
    if(result.ec != std::errc{} || result.ptr != s.data() + s.size())
    {
        return std::nullopt;
    }
    return value;
}

/// @brief Strictly parse a whole string as an int; nullopt on empty, partial or overflow.
[[nodiscard]] inline std::optional<int>
safe_stoi(std::string_view s)
{
    return safe_from_chars<int>(s);
}

/// @brief Strictly parse a whole string as a long; nullopt on empty, partial or overflow.
[[nodiscard]] inline std::optional<long>
safe_stol(std::string_view s)
{
    return safe_from_chars<long>(s);
}

/// @brief Strictly parse a whole string as a double; nullopt on empty or partial parse.
[[nodiscard]] inline std::optional<double>
safe_stod(std::string_view number_str)
{
    if(number_str.empty())
    {
        return std::nullopt;
    }
    double value = 0.0;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    // Prefer from_chars: locale-independent, zero-allocation
    const auto result =
        std::from_chars(number_str.data(), number_str.data() + number_str.size(), value);
    if(result.ec != std::errc{} || result.ptr != number_str.data() + number_str.size())
    {
        return std::nullopt;
    }
#else
    // Fallback for older compilers (GCC <11, Clang <16).
    // Note: std::stod is locale-sensitive -assumes C/POSIX locale.
    try
    {
        std::size_t pos = 0;
        value           = std::stod(std::string{ s }, &pos);
        if(pos != s.size()) return std::nullopt;
    } catch(const std::exception&)
    {
        return std::nullopt;
    }
#endif
    return value;
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

/// @brief Strip ANSI escape (SGR/color) sequences from a string.
/// @param value The string to strip, potentially containing "\033[...m" sequences.
/// @return A copy of @p value with every "\033...m" escape sequence removed.
[[nodiscard]] inline std::string
strip_ansi(std::string_view value)
{
    std::string result;
    result.reserve(value.size());
    bool in_escape = false;
    for(const char chr : value)
    {
        if(in_escape)
        {
            in_escape = (chr != 'm');
            continue;
        }
        if(chr == '\033')
        {
            in_escape = true;
            continue;
        }
        result += chr;
    }
    return result;
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

inline constexpr size_t k_default_hex_width = 16;

/// @brief Format an integral value as a zero-padded, "0x"-prefixed hex string.
/// @param value The integral value to format.
/// @param width The minimum number of hex digits to zero-pad to.
/// @return @p value formatted as e.g. "0x0000000000000010" (width 16, value 16).
template <typename Tp>
[[nodiscard]] inline std::string
hex_padded(Tp value, size_t width = k_default_hex_width)
{
    return fmt::format("0x{:0{}x}", value, width);
}

}  // namespace rocprofsys::utility::string
