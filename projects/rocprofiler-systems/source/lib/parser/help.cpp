// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/help.hpp"
#include "parser/schema.hpp"

#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

namespace rocprofsys::parser
{
namespace
{
std::string_view
type_name(param_type type)
{
    switch(type)
    {
        case param_type::boolean: return "bool";
        case param_type::integer: return "integer";
        case param_type::number: return "double";
        case param_type::string: return "string";
        case param_type::path: return "path";
        case param_type::enumeration: return "enum";
        case param_type::enumeration_list: return "enum_list";
    }
    return "value";
}

bool
include_param(const schema_param& param, help_style style)
{
    return style == help_style::full || param.level == param_level::basic;
}

bool
category_matches(const category& item, std::string_view topic)
{
    return topic.empty() || item.id == topic || item.name == topic;
}

void
append_usage(std::ostream& out, const schema_param& param)
{
    out << "  ";
    if(param.arg)
    {
        out << param.arg->name;
        if(param.arg->form == arg_form::value)
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

void
append_choices(std::ostream& out, const schema_param& param)
{
    if(param.choices.empty())
    {
        return;
    }
    out << "      choices:";
    for(const auto& item : param.choices)
    {
        out << ' ' << item.value;
    }
    out << '\n';
}

void
append_details(std::ostream& out, const schema_param& param, help_style style)
{
    if(!param.description.empty())
    {
        out << "      " << param.description << '\n';
    }
    if(style == help_style::full)
    {
        out << "      type: " << type_name(param.type) << '\n';
        append_choices(out, param);
    }
    if(param.env)
    {
        out << "      env: " << param.env->name << '\n';
    }
    if(param.default_value)
    {
        out << "      default: " << *param.default_value << '\n';
    }
    out << '\n';
}

void
append_category(std::ostream& out, const schema& config_schema, const category& item,
                help_style style)
{
    auto wrote_heading = false;
    for(const auto* param : config_schema.params_in_category(item.id))
    {
        if(param == nullptr || !include_param(*param, style))
        {
            continue;
        }
        if(!wrote_heading)
        {
            out << item.name << '\n';
            wrote_heading = true;
        }
        append_usage(out, *param);
        append_details(out, *param, style);
    }
}
}  // namespace

std::string
format_help(const schema& config_schema, help_style style, std::string_view topic)
{
    std::ostringstream out;
    out << "ROCm Systems Profiler options\n\n";
    auto matched = topic.empty();
    for(const auto& item : config_schema.categories())
    {
        if(!category_matches(item, topic))
        {
            continue;
        }
        matched = true;
        append_category(out, config_schema, item, style);
    }
    if(!matched)
    {
        out << "No parameters match topic '" << topic << "'.\n";
    }
    return out.str();
}

}  // namespace rocprofsys::parser
