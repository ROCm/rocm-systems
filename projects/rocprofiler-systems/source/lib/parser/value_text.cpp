// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/value_text.hpp"
#include "parser/schema.hpp"

#include "common/string_utility.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace rocprofsys::parser
{
std::optional<std::int64_t>
parse_integer(std::string_view value)
{
    const auto parsed = utility::string::safe_from_chars<std::int64_t>(value);
    if(parsed.is_error())
    {
        return std::nullopt;
    }
    return parsed.value();
}

std::optional<double>
parse_number(std::string_view value)
{
    const auto parsed = utility::string::safe_from_chars<double>(value);
    if(parsed.is_error() || !std::isfinite(parsed.value()))
    {
        return std::nullopt;
    }
    return parsed.value();
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

std::optional<std::int64_t>
exact_integer_bound(double bound)
{
    // The std::int64_t maximum rounds up to 2^63 as a double, which is out of range.
    constexpr auto k_int64_upper =
        static_cast<double>(std::numeric_limits<std::int64_t>::max());
    constexpr auto k_int64_lower =
        static_cast<double>(std::numeric_limits<std::int64_t>::min());
    if(std::isnan(bound) || bound < k_int64_lower || bound >= k_int64_upper)
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

namespace
{
// Room for any std::int64_t, and for any double in general format.
constexpr std::size_t k_integer_text_size = 32;
constexpr std::size_t k_number_text_size  = 64;

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
    return std::ranges::find(param.choices, value) != param.choices.end();
}

[[nodiscard]] bool
choice_accepted(const schema_param& param, std::string_view value)
{
    return choice_listed(param, value) || (param.allow_custom && !value.empty());
}

[[nodiscard]] std::string
format_integer(std::int64_t value)
{
    std::array<char, k_integer_text_size> buffer{};
    const auto                            result =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    return std::string{ buffer.data(), result.ptr };
}

[[nodiscard]] std::string
format_number(double value)
{
    std::array<char, k_number_text_size> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                      std::chars_format::general);
    if(result.ec != std::errc{})
    {
        return {};
    }
    return std::string{ buffer.data(), result.ptr };
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

// One overload per config_value_t alternative: each accepts the parameter types
// that are stored as that alternative.
[[nodiscard]] bool
accepts_held(const schema_param& param, bool /*value*/)
{
    return param.type == param_type::boolean;
}

[[nodiscard]] bool
accepts_held(const schema_param& param, std::int64_t value)
{
    return param.type == param_type::integer && integer_in_range(value, param);
}

[[nodiscard]] bool
accepts_held(const schema_param& param, double value)
{
    return param.type == param_type::number && number_in_range(value, param);
}

[[nodiscard]] bool
accepts_held(const schema_param& param, const std::string& value)
{
    switch(param.type)
    {
        case param_type::string: return true;
        case param_type::path: return !value.empty();
        case param_type::enumeration: return choice_accepted(param, value);
        case param_type::boolean:
        case param_type::integer:
        case param_type::number:
        case param_type::enumeration_list: return false;
    }
    return false;
}

[[nodiscard]] bool
accepts_held(const schema_param& param, const std::vector<std::string>& parts)
{
    return param.type == param_type::enumeration_list &&
           std::ranges::all_of(parts, [&](const std::string& element) {
               return !element.empty() && choice_accepted(param, element);
           });
}
}  // namespace

bool
accepts_value(const schema_param& param, const config_value_t& value)
{
    return std::visit([&param](const auto& held) { return accepts_held(param, held); },
                      value);
}

std::optional<config_value_t>
parse_value(const schema_param& param, std::string_view text)
{
    const auto                    trimmed = utility::string::trim(text);
    std::optional<config_value_t> candidate;
    switch(param.type)
    {
        case param_type::boolean:
            if(const auto parsed = parse_boolean(trimmed))
            {
                candidate = config_value_t{ *parsed };
            }
            break;
        case param_type::integer:
            if(const auto parsed = parse_integer(trimmed))
            {
                candidate = config_value_t{ *parsed };
            }
            break;
        case param_type::number:
            if(const auto parsed = parse_number(trimmed))
            {
                candidate = config_value_t{ *parsed };
            }
            break;
        case param_type::string:
        case param_type::path:
        case param_type::enumeration:
            candidate = config_value_t{ std::string{ trimmed } };
            break;
        case param_type::enumeration_list:
            if(auto parts = split_list(trimmed, list_separator(param)))
            {
                candidate = config_value_t{ std::move(*parts) };
            }
            break;
    }
    if(!candidate || !accepts_value(param, *candidate))
    {
        return std::nullopt;
    }
    return candidate;
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
