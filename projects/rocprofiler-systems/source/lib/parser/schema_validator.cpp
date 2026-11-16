// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema_validator.hpp"
#include "parser/schema.hpp"
#include "parser/value_text.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
class configuration_checker
{
public:
    configuration_checker(const schema& config_schema, const typed_values_t& values)
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
    [[nodiscard]] bool enabled(std::string_view param_id) const
    {
        const auto provided = m_values.find(param_id);
        return m_schema.find_param(param_id) != nullptr && provided != m_values.end() &&
               is_enabled(provided->second);
    }

    void report(std::string message, std::vector<std::string> params)
    {
        m_violations.push_back(
            violation{ .message = std::move(message), .params = std::move(params) });
    }

    void check_provided_values()
    {
        for(const auto& [param_id, ignored] : m_values)
        {
            static_cast<void>(ignored);
            if(m_schema.find_param(param_id) == nullptr)
            {
                report(fmt::format("unknown parameter '{}'", param_id), { param_id });
            }
        }
    }

    void check_exclusions(const schema_param& param)
    {
        if(!enabled(param.id))
        {
            return;
        }
        for(const auto& other : param.conflicts)
        {
            auto key = std::minmax(param.id, other);
            if(enabled(other) && m_reported_pairs.emplace(key.first, key.second).second)
            {
                report(fmt::format("parameters '{}' and '{}' are mutually exclusive",
                                   param.id, other),
                       { param.id, other });
            }
        }
    }

    void check_required(const schema_param& param)
    {
        if(!enabled(param.id))
        {
            return;
        }
        for(const auto& requirement : param.required)
        {
            if(!enabled(requirement))
            {
                report(fmt::format("parameter '{}' requires '{}'", param.id, requirement),
                       { param.id, requirement });
            }
        }
    }

    const schema&                                 m_schema;
    const typed_values_t&                         m_values;
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
    for(const auto& [target_id, text] : param.assigns)
    {
        require_known(config_schema, param.id, target_id, errors);
        const auto* target = config_schema.find_param(target_id);
        if(target != nullptr && !is_valid_value(*target, text))
        {
            report_catalog(
                errors,
                fmt::format("parameter '{}' assigns invalid value '{}' to '{}'", param.id,
                            text, target_id),
                { param.id, target_id });
        }
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
    return accepts_text(param, value);
}

std::string
invalid_value_message(std::string_view value, std::string_view param_id)
{
    return fmt::format("invalid value '{}' for parameter '{}'", value, param_id);
}

std::vector<violation>
validate_values(const schema& config_schema, const typed_values_t& values)
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
