// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/help.hpp"
#include "parser/schema.hpp"

#include <algorithm>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
bool
include_param(const schema_param& param, param_level max_level)
{
    return param.level == param_level::basic || max_level == param_level::advanced;
}

bool
already_listed(const std::vector<std::string>& names, std::string_view name)
{
    return std::find(names.begin(), names.end(), name) != names.end();
}

std::vector<std::string>
category_order(const schema& config_schema)
{
    std::vector<std::string> order;
    for(const auto& param : config_schema.params())
    {
        if(!param.category.empty() && !already_listed(order, param.category))
        {
            order.push_back(param.category);
        }
    }
    return order;
}

void
append_usage(std::ostream& out, const schema_param& param)
{
    out << "  ";
    if(param.arg)
    {
        out << *param.arg;
        for(const auto& alias : param.arg_aliases)
        {
            out << ", " << alias;
        }
        if(!param.flag)
        {
            out << " <value>";
        }
    }
    else
    {
        out << param.id;
    }
    out << '\n';
}

/// The row's own choices, then the names its registered backend lists, the names
/// backend_interface::accepts() checks a value against.
std::vector<std::string>
choice_names(const schema_param& param, const backend_registry* backends)
{
    auto        names = param.choices;
    const auto* found =
        backends == nullptr || !param.backend ? nullptr : backends->find(*param.backend);
    if(found != nullptr)
    {
        const auto listed = found->choices(param);
        names.insert(names.end(), listed.begin(), listed.end());
    }
    return names;
}

void
append_choices(std::ostream& out, const schema_param& param,
               const backend_registry* backends)
{
    const auto names = choice_names(param, backends);
    if(names.empty())
    {
        return;
    }
    out << "      choices:";
    for(const auto& name : names)
    {
        out << ' ' << name;
    }
    out << '\n';
}

void
append_details(std::ostream& out, const schema_param& param,
               const backend_registry* backends)
{
    if(!param.description.empty())
    {
        out << "      " << param.description << '\n';
    }
    append_choices(out, param, backends);
    if(param.env)
    {
        out << "      env: " << *param.env << '\n';
    }
    if(param.default_value)
    {
        out << "      default: " << *param.default_value << '\n';
    }
    out << '\n';
}

void
append_param(std::ostream& out, const schema_param& param, param_level max_level,
             const backend_registry* backends)
{
    if(!include_param(param, max_level))
    {
        return;
    }
    append_usage(out, param);
    append_details(out, param, backends);
}
}  // namespace

std::string
format_help(const schema& config_schema, param_level max_level, std::string_view topic,
            const backend_registry* backends)
{
    std::ostringstream out;
    auto               matched = topic.empty();
    for(const auto& category : category_order(config_schema))
    {
        if(!topic.empty() && topic != category)
        {
            continue;
        }
        matched = true;
        out << category << '\n';
        for(const auto* param : config_schema.params_in_category(category))
        {
            append_param(out, *param, max_level, backends);
        }
    }
    if(topic.empty())
    {
        for(const auto& param : config_schema.params())
        {
            if(!param.category.empty())
            {
                continue;
            }
            matched = true;
            append_param(out, param, max_level, backends);
        }
    }
    if(!matched)
    {
        out << "No parameters match topic '" << topic << "'.\n";
    }
    return out.str();
}

}  // namespace rocprofsys::parser
