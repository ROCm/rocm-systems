// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema_validator.hpp"
#include "parser/schema.hpp"
#include "parser/value_text.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
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

bool
is_within_bounds(double value, const schema_param& param)
{
    return (!param.min || value >= *param.min) && (!param.max || value <= *param.max);
}

bool
integer_bound_holds(std::int64_t value, const std::optional<double>& bound, bool minimum)
{
    if(!bound)
    {
        return true;
    }
    const auto exact = exact_integer_bound(*bound);
    if(!exact)
    {
        return false;
    }
    return minimum ? value >= *exact : value <= *exact;
}

bool
is_valid_integer(const schema_param& param, std::string_view value)
{
    const auto parsed = parse_integer(value);
    return parsed && integer_bound_holds(*parsed, param.min, true) &&
           integer_bound_holds(*parsed, param.max, false);
}

bool
is_valid_number(const schema_param& param, std::string_view value)
{
    const auto parsed = parse_number(value);
    return parsed && is_within_bounds(*parsed, param);
}

bool
is_valid_choice(const schema_param& param, std::string_view value)
{
    const bool is_listed = std::ranges::any_of(
        param.choices, [value](const choice& item) { return item.value == value; });
    return is_listed || (param.allow_custom && !value.empty());
}

bool
is_valid_list(const schema_param& param, std::string_view value)
{
    const auto parts = split_list(value, list_separator(param));
    if(!parts)
    {
        return false;
    }
    return std::ranges::all_of(*parts, [&param](const std::string& element) {
        return !element.empty() && is_valid_choice(param, element);
    });
}

class configuration_checker
{
public:
    configuration_checker(const schema& config_schema, const value_map_t& values)
    : m_schema(config_schema)
    , m_values(values)
    {}

    std::vector<violation> run()
    {
        check_provided_values();
        for(const auto& param : m_schema.params())
        {
            check_exclusions(param);
            check_required(param);
        }
        return std::move(m_violations);
    }

private:
    [[nodiscard]] bool is_enabled(std::string_view param_id) const
    {
        const auto* param = m_schema.find_param(param_id);
        if(param == nullptr)
        {
            return false;
        }
        const auto provided = m_values.find(param_id);
        if(provided == m_values.end() || provided->second.empty())
        {
            return false;
        }
        return param->type != param_type::boolean ||
               parse_boolean(provided->second).value_or(false);
    }

    void report(std::string message, std::vector<std::string> params)
    {
        m_violations.push_back(
            violation{ .message = std::move(message), .params = std::move(params) });
    }

    void check_provided_values()
    {
        for(const auto& [param_id, value] : m_values)
        {
            const auto* param = m_schema.find_param(param_id);
            if(param == nullptr)
            {
                report(fmt::format("unknown parameter '{}'", param_id), { param_id });
            }
            else if(!is_valid_value(*param, value))
            {
                report(
                    fmt::format("invalid value '{}' for parameter '{}'", value, param_id),
                    { param_id });
            }
        }
    }

    void check_exclusions(const schema_param& param)
    {
        if(!is_enabled(param.id))
        {
            return;
        }
        for(const auto& other : param.conflicts)
        {
            auto key = std::minmax(param.id, other);
            if(is_enabled(other) &&
               m_reported_pairs.emplace(key.first, key.second).second)
            {
                report(fmt::format("parameters '{}' and '{}' are mutually exclusive",
                                   param.id, other),
                       { param.id, other });
            }
        }
    }

    void check_required(const schema_param& param)
    {
        if(!is_enabled(param.id))
        {
            return;
        }
        for(const auto& requirement : param.required)
        {
            if(!is_enabled(requirement))
            {
                report(fmt::format("parameter '{}' requires '{}'", param.id, requirement),
                       { param.id, requirement });
            }
        }
    }

    const schema&                                 m_schema;
    const value_map_t&                            m_values;
    std::vector<violation>                        m_violations;
    std::set<std::pair<std::string, std::string>> m_reported_pairs;
};

void
report_catalog(std::vector<violation>& errors, std::string message,
               std::vector<std::string> params)
{
    errors.push_back(
        violation{ .message = std::move(message), .params = std::move(params) });
}

void
require_known(const schema& config_schema, std::string_view owner,
              std::string_view target, std::vector<violation>& errors)
{
    if(config_schema.find_param(target) != nullptr)
    {
        return;
    }
    report_catalog(
        errors,
        fmt::format("parameter '{}' refers to unknown parameter '{}'", owner, target),
        { std::string{ owner }, std::string{ target } });
}

void
require_known_ids(const schema& config_schema, std::string_view owner,
                  const std::vector<std::string>& targets, std::vector<violation>& errors)
{
    for(const auto& target : targets)
    {
        require_known(config_schema, owner, target, errors);
    }
}

void
check_references(const schema& config_schema, const schema_param& param,
                 std::vector<violation>& errors)
{
    require_known_ids(config_schema, param.id, param.conflicts, errors);
    require_known_ids(config_schema, param.id, param.required, errors);
    for(const auto& [target, ignored] : param.assigns)
    {
        static_cast<void>(ignored);
        require_known(config_schema, param.id, target, errors);
    }
}

void
check_argument_counts(const schema_param& param, std::vector<violation>& errors)
{
    if(param.arg_min_count && param.arg_max_count &&
       *param.arg_min_count > *param.arg_max_count)
    {
        report_catalog(errors,
                       fmt::format("parameter '{}' accepts fewer arguments than its "
                                   "minimum",
                                   param.id),
                       { param.id });
    }
}

void
check_bounds(const schema_param& param, std::vector<violation>& errors)
{
    if(param.min && param.max && *param.min > *param.max)
    {
        report_catalog(
            errors,
            fmt::format("parameter '{}' has a minimum greater than its maximum",
                        param.id),
            { param.id });
    }
    if(param.type != param_type::integer)
    {
        return;
    }
    if(param.min && !exact_integer_bound(*param.min))
    {
        report_catalog(errors,
                       fmt::format("parameter '{}' has a non-integer minimum", param.id),
                       { param.id });
    }
    if(param.max && !exact_integer_bound(*param.max))
    {
        report_catalog(errors,
                       fmt::format("parameter '{}' has a non-integer maximum", param.id),
                       { param.id });
    }
}

void
check_param_shape(const schema& config_schema, const schema_param& param,
                  std::vector<violation>& errors)
{
    if(param.default_value && !is_valid_value(param, *param.default_value))
    {
        report_catalog(errors,
                       fmt::format("parameter '{}' has an invalid default '{}'", param.id,
                                   *param.default_value),
                       { param.id });
    }
    check_bounds(param, errors);
    check_argument_counts(param, errors);
    check_references(config_schema, param, errors);
}
}  // namespace

bool
is_valid_value(const schema_param& param, std::string_view value)
{
    switch(param.type)
    {
        case param_type::boolean: return parse_boolean(value).has_value();
        case param_type::integer: return is_valid_integer(param, value);
        case param_type::number: return is_valid_number(param, value);
        case param_type::string: return true;
        case param_type::path: return !value.empty();
        case param_type::enumeration: return is_valid_choice(param, value);
        case param_type::enumeration_list: return is_valid_list(param, value);
    }
    return false;
}

std::vector<violation>
validate_values(const schema& config_schema, const value_map_t& values)
{
    return configuration_checker{ config_schema, values }.run();
}

std::vector<violation>
validate_schema(const schema& config_schema)
{
    std::vector<violation> errors;
    for(const auto& param : config_schema.params())
    {
        check_param_shape(config_schema, param, errors);
    }
    return errors;
}

}  // namespace rocprofsys::parser
