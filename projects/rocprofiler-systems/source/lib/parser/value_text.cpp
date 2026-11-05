// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/value_text.hpp"

#include "common/string_utility.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
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
    if(number.front() == '+')
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

namespace
{
constexpr double k_int64_upper = 0x1p63;

[[nodiscard]] std::optional<std::int64_t>
exact_integer_bound(double bound)
{
    const auto lowest = static_cast<double>(std::numeric_limits<std::int64_t>::min());
    if(bound < lowest || bound >= k_int64_upper)
    {
        return std::nullopt;
    }
    const auto as_integer = static_cast<std::int64_t>(bound);
    if(static_cast<double>(as_integer) != bound)
    {
        return std::nullopt;
    }
    return as_integer;
}

[[nodiscard]] bool
integer_in_range(std::int64_t value, const schema_param& param)
{
    const auto holds = [](std::int64_t parsed, const std::optional<double>& bound,
                          bool minimum) {
        if(!bound)
        {
            return true;
        }
        const auto exact = exact_integer_bound(*bound);
        if(!exact)
        {
            return false;
        }
        return minimum ? parsed >= *exact : parsed <= *exact;
    };
    return holds(value, param.min, true) && holds(value, param.max, false);
}

[[nodiscard]] bool
number_in_range(double value, const schema_param& param)
{
    return (!param.min || value >= *param.min) && (!param.max || value <= *param.max);
}

[[nodiscard]] bool
choice_listed(const schema_param& param, std::string_view value)
{
    for(const auto& item : param.choices)
    {
        if(item.value == value)
        {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool
choice_accepted(const schema_param& param, std::string_view value)
{
    return choice_listed(param, value) || (param.allow_custom && !value.empty());
}
}  // namespace

bool
accepts_text(const schema_param& param, std::string_view value)
{
    switch(param.type)
    {
        case param_type::boolean: return parse_boolean(value).has_value();
        case param_type::integer:
        {
            const auto parsed = parse_integer(value);
            return parsed && integer_in_range(*parsed, param);
        }
        case param_type::number:
        {
            const auto parsed = parse_number(value);
            return parsed && number_in_range(*parsed, param);
        }
        case param_type::string: return true;
        case param_type::path: return !value.empty();
        case param_type::enumeration: return choice_accepted(param, value);
        case param_type::enumeration_list:
        {
            const auto parts = split_list(value, list_separator(param));
            if(!parts)
            {
                return false;
            }
            for(const auto& element : *parts)
            {
                if(element.empty() || !choice_accepted(param, element))
                {
                    return false;
                }
            }
            return true;
        }
    }
    return false;
}

}  // namespace rocprofsys::parser
