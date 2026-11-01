// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/value_text.hpp"

#include "common/string_utility.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>

#if !defined(__cpp_lib_to_chars) || __cpp_lib_to_chars < 201611L
#    include <cctype>
#    include <cstddef>
#    include <stdexcept>
#    include <string>
#endif

namespace rocprofsys::parser
{
std::optional<std::int64_t>
parse_integer(std::string_view value)
{
    std::int64_t result{};
    const auto*  last    = value.data() + value.size();
    const auto [ptr, ec] = std::from_chars(value.data(), last, result);
    if(ec != std::errc{} || ptr != last)
    {
        return std::nullopt;
    }
    return result;
}

std::optional<double>
parse_number(std::string_view value)
{
    double result{};
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    const auto* last     = value.data() + value.size();
    const auto [ptr, ec] = std::from_chars(value.data(), last, result);
    if(ec != std::errc{} || ptr != last)
    {
        return std::nullopt;
    }
#else
    // std::stod skips leading whitespace, which std::from_chars rejects.
    if(value.empty() || std::isspace(static_cast<unsigned char>(value.front())) != 0)
    {
        return std::nullopt;
    }
    try
    {
        std::size_t consumed = 0;
        result               = std::stod(std::string{ value }, &consumed);
        if(consumed != value.size())
        {
            return std::nullopt;
        }
    } catch(const std::logic_error&)
    {
        return std::nullopt;
    }
#endif
    if(!std::isfinite(result))
    {
        return std::nullopt;
    }
    return result;
}

std::optional<bool>
parse_boolean(std::string_view value)
{
    if(value.empty())
    {
        return std::nullopt;
    }

    constexpr auto k_tokens = std::array{
        std::pair{ std::string_view{ "true" }, true },
        std::pair{ std::string_view{ "false" }, false },
        std::pair{ std::string_view{ "yes" }, true },
        std::pair{ std::string_view{ "no" }, false },
        std::pair{ std::string_view{ "on" }, true },
        std::pair{ std::string_view{ "off" }, false },
        std::pair{ std::string_view{ "y" }, true },
        std::pair{ std::string_view{ "n" }, false },
        std::pair{ std::string_view{ "t" }, true },
        std::pair{ std::string_view{ "f" }, false },
        std::pair{ std::string_view{ "1" }, true },
        std::pair{ std::string_view{ "0" }, false },
    };

    for(const auto& [token, result] : k_tokens)
    {
        if(utility::string::equals_ignore_case(value, token))
        {
            return result;
        }
    }

    auto number = value;
    if(!number.empty() && number.front() == '+')
    {
        number.remove_prefix(1);
    }
    const auto parsed = parse_integer(number);
    if(!parsed)
    {
        return std::nullopt;
    }
    return *parsed != 0;
}

}  // namespace rocprofsys::parser
