// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema_validator.hpp"
#include "parser/schema.hpp"
#include "parser/value_text.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
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
matches_pattern(const schema_param&, std::string_view)
{
    return true;
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

bool
values_equal(const schema_param& param, std::string_view lhs, std::string_view rhs)
{
    if(param.type == param_type::boolean)
    {
        const auto lhs_value = parse_boolean(lhs);
        const auto rhs_value = parse_boolean(rhs);
        return lhs_value && rhs_value && *lhs_value == *rhs_value;
    }
    if(param.type == param_type::integer)
    {
        const auto lhs_value = parse_integer(lhs);
        const auto rhs_value = parse_integer(rhs);
        return lhs_value && rhs_value && *lhs_value == *rhs_value;
    }
    if(param.type == param_type::number)
    {
        const auto lhs_value = parse_number(lhs);
        const auto rhs_value = parse_number(rhs);
        return lhs_value && rhs_value && *lhs_value == *rhs_value;
    }
    return lhs == rhs;
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
            check_required(param);
            check_exclusions(param);
            check_prerequisites(param);
            check_dependency(param);
        }
        for(const auto& item : m_schema.constraints())
        {
            check_constraint(item);
        }
        return std::move(m_violations);
    }

private:
    [[nodiscard]] std::optional<std::string_view> effective_value(
        const schema_param& param) const
    {
        if(auto itr = m_values.find(param.id); itr != m_values.end())
        {
            return itr->second;
        }
        if(param.default_value)
        {
            return *param.default_value;
        }
        return std::nullopt;
    }

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

    [[nodiscard]] bool holds(const param_condition& term) const
    {
        const auto* param = m_schema.find_param(term.param);
        if(param == nullptr)
        {
            return false;
        }
        const auto value = effective_value(*param);
        return value && values_equal(*param, *value, term.equals);
    }

    [[nodiscard]] bool holds(const condition& when) const
    {
        const auto term_holds = [this](const param_condition& term) {
            return holds(term);
        };
        if(when.terms.empty())
        {
            return false;
        }
        return (when.mode == condition_mode::all)
                   ? std::ranges::all_of(when.terms, term_holds)
                   : std::ranges::any_of(when.terms, term_holds);
    }

    [[nodiscard]] bool forced_values_hold(const constraint& item) const
    {
        return std::ranges::all_of(item.forced_values, [this](const auto& entry) {
            return holds(param_condition{ .param = entry.first, .equals = entry.second });
        });
    }

    void report(severity level, std::string message, std::vector<std::string> params)
    {
        m_violations.push_back(violation{
            .level = level, .message = std::move(message), .params = std::move(params) });
    }

    void check_provided_values()
    {
        for(const auto& [param_id, value] : m_values)
        {
            const auto* param = m_schema.find_param(param_id);
            if(param == nullptr)
            {
                report(severity::error, fmt::format("unknown parameter '{}'", param_id),
                       { param_id });
            }
            else if(!is_valid_value(*param, value))
            {
                report(severity::error, invalid_value_message(value, param_id),
                       { param_id });
            }
        }
    }

    void check_required(const schema_param& param)
    {
        if(param.required && !effective_value(param))
        {
            report(severity::error, fmt::format("parameter '{}' is required", param.id),
                   { param.id });
        }
    }

    void check_exclusions(const schema_param& param)
    {
        if(!is_enabled(param.id))
        {
            return;
        }
        for(const auto& other : param.mutually_exclusive_with)
        {
            auto key = std::minmax(param.id, other);
            if(is_enabled(other) &&
               m_reported_pairs.emplace(key.first, key.second).second)
            {
                report(severity::error,
                       fmt::format("parameters '{}' and '{}' are mutually exclusive",
                                   param.id, other),
                       { param.id, other });
            }
        }
    }

    void check_prerequisites(const schema_param& param)
    {
        if(!is_enabled(param.id))
        {
            return;
        }
        for(const auto& prerequisite : param.prerequisites)
        {
            if(!is_enabled(prerequisite))
            {
                report(
                    severity::error,
                    fmt::format("parameter '{}' requires '{}'", param.id, prerequisite),
                    { param.id, prerequisite });
            }
        }
    }

    void check_dependency(const schema_param& param)
    {
        if(param.depends_on && m_values.contains(param.id) && !holds(*param.depends_on))
        {
            report(severity::warning,
                   fmt::format("parameter '{}' has no effect unless '{}' is '{}'",
                               param.id, param.depends_on->param,
                               param.depends_on->equals),
                   { param.id, param.depends_on->param });
        }
    }

    void check_constraint(const constraint& item)
    {
        if(item.kind == constraint_kind::mutually_exclusive)
        {
            std::vector<std::string> enabled;
            std::ranges::copy_if(
                item.params, std::back_inserter(enabled),
                [this](const std::string& param_id) { return is_enabled(param_id); });
            if(enabled.size() > 1)
            {
                auto message = item.message.empty()
                                   ? fmt::format("parameters {} are mutually exclusive",
                                                 fmt::join(enabled, ", "))
                                   : item.message;
                report(item.level, std::move(message), std::move(enabled));
            }
            return;
        }

        const bool is_satisfied =
            item.kind == constraint_kind::forces && forced_values_hold(item);
        if(holds(item.when) && !is_satisfied)
        {
            std::vector<std::string> params;
            params.reserve(item.when.terms.size());
            for(const auto& term : item.when.terms)
            {
                params.push_back(term.param);
            }
            report(item.level, item.message, std::move(params));
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
    errors.push_back(violation{ .level   = severity::error,
                                .message = std::move(message),
                                .params  = std::move(params) });
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
    if(param.depends_on)
    {
        require_known(config_schema, param.id, param.depends_on->param, errors);
    }
    require_known_ids(config_schema, param.id, param.mutually_exclusive_with, errors);
    require_known_ids(config_schema, param.id, param.prerequisites, errors);
    require_known_ids(config_schema, param.id, param.copy_to, errors);
    require_known_ids(config_schema, param.id, param.fills, errors);
    for(const auto& [target, ignored] : param.assigns)
    {
        static_cast<void>(ignored);
        require_known(config_schema, param.id, target, errors);
    }
    for(const auto& mapping : param.maps_to)
    {
        require_known(config_schema, param.id, mapping.target, errors);
    }
    for(const auto& item : param.choices)
    {
        for(const auto& [target, ignored] : item.sets)
        {
            static_cast<void>(ignored);
            require_known(config_schema, param.id, target, errors);
        }
    }
    if(param.arg)
    {
        require_known_ids(config_schema, param.id, param.arg->binds, errors);
    }
}

void
check_argument_counts(const schema_param& param, std::vector<violation>& errors)
{
    if(!param.arg)
    {
        return;
    }
    const auto& argument = *param.arg;
    if(argument.min_count && argument.max_count &&
       *argument.min_count > *argument.max_count)
    {
        report_catalog(errors,
                       fmt::format("parameter '{}' accepts fewer arguments than its "
                                   "minimum",
                                   param.id),
                       { param.id });
    }
    if(argument.max_count && argument.binds.size() > *argument.max_count)
    {
        report_catalog(
            errors,
            fmt::format("parameter '{}' binds more arguments than it accepts", param.id),
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
    if(!param.category.empty() && config_schema.find_category(param.category) == nullptr)
    {
        report_catalog(errors,
                       fmt::format("parameter '{}' uses unknown category '{}'", param.id,
                                   param.category),
                       { param.id, param.category });
    }
    if(param.env && param.env->separator.empty())
    {
        report_catalog(
            errors, fmt::format("parameter '{}' has an empty list separator", param.id),
            { param.id });
    }
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

void
check_constraint_targets(const schema& config_schema, const constraint& item,
                         std::vector<violation>& errors)
{
    require_known_ids(config_schema, "constraint", item.params, errors);
    for(const auto& term : item.when.terms)
    {
        require_known(config_schema, "constraint", term.param, errors);
    }
    for(const auto& [target, ignored] : item.forced_values)
    {
        static_cast<void>(ignored);
        require_known(config_schema, "constraint", target, errors);
    }
}
}  // namespace

std::string
invalid_value_message(std::string_view value, std::string_view param_id)
{
    return fmt::format("invalid value '{}' for parameter '{}'", value, param_id);
}

bool
is_valid_value(const schema_param& param, std::string_view value)
{
    switch(param.type)
    {
        case param_type::boolean: return parse_boolean(value).has_value();
        case param_type::integer: return is_valid_integer(param, value);
        case param_type::number: return is_valid_number(param, value);
        case param_type::string: return matches_pattern(param, value);
        case param_type::path: return !value.empty() && matches_pattern(param, value);
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
    for(const auto& item : config_schema.constraints())
    {
        check_constraint_targets(config_schema, item, errors);
    }
    return errors;
}

}  // namespace rocprofsys::parser
