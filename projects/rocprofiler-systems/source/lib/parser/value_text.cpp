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
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if !defined(__cpp_lib_to_chars) || __cpp_lib_to_chars < 201611L
#    include <cctype>
#    include <cstddef>
#    include <cstdio>
#    include <stdexcept>
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

[[nodiscard]] std::string
format_integer(std::int64_t value)
{
    std::array<char, 32> buffer{};
    const auto           result =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    return std::string{ buffer.data(), result.ptr };
}

[[nodiscard]] std::string
format_number(double value)
{
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                      std::chars_format::general);
    if(result.ec == std::errc{})
    {
        return std::string{ buffer.data(), result.ptr };
    }
#else
    std::array<char, 64> buffer{};
    const auto written = std::snprintf(buffer.data(), buffer.size(), "%g", value);
    if(written > 0)
    {
        return std::string{ buffer.data() };
    }
#endif
    return {};
}

[[nodiscard]] std::string
format_list(const std::vector<std::string>& parts)
{
    std::string joined;
    for(std::size_t index = 0; index < parts.size(); ++index)
    {
        if(index != 0)
        {
            joined.push_back(',');
        }
        joined.append(parts[index]);
    }
    return joined;
}
}  // namespace

std::optional<config_value_t>
parse_value(const schema_param& param, std::string_view text)
{
    switch(param.type)
    {
        case param_type::boolean:
        {
            const auto parsed = parse_boolean(text);
            if(!parsed)
            {
                return std::nullopt;
            }
            return config_value_t{ *parsed };
        }
        case param_type::integer:
        {
            const auto parsed = parse_integer(text);
            if(!parsed || !integer_in_range(*parsed, param))
            {
                return std::nullopt;
            }
            return config_value_t{ *parsed };
        }
        case param_type::number:
        {
            const auto parsed = parse_number(text);
            if(!parsed || !number_in_range(*parsed, param))
            {
                return std::nullopt;
            }
            return config_value_t{ *parsed };
        }
        case param_type::string: return config_value_t{ std::string{ text } };
        case param_type::path:
        {
            if(text.empty())
            {
                return std::nullopt;
            }
            return config_value_t{ std::string{ text } };
        }
        case param_type::enumeration:
        {
            if(!choice_accepted(param, text))
            {
                return std::nullopt;
            }
            return config_value_t{ std::string{ text } };
        }
        case param_type::enumeration_list:
        {
            const auto parts = split_list(text, list_separator(param));
            if(!parts)
            {
                return std::nullopt;
            }
            for(const auto& element : *parts)
            {
                if(element.empty() || !choice_accepted(param, element))
                {
                    return std::nullopt;
                }
            }
            return config_value_t{ *parts };
        }
    }
    return std::nullopt;
}

std::string
format_value(const config_value_t& value)
{
    if(const auto* flag = std::get_if<bool>(&value))
    {
        return *flag ? "true" : "false";
    }
    if(const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return format_integer(*integer);
    }
    if(const auto* number = std::get_if<double>(&value))
    {
        return format_number(*number);
    }
    if(const auto* text = std::get_if<std::string>(&value))
    {
        return *text;
    }
    return format_list(std::get<std::vector<std::string>>(value));
}

bool
accepts_text(const schema_param& param, std::string_view value)
{
    return parse_value(param, value).has_value();
}

}  // namespace rocprofsys::parser
