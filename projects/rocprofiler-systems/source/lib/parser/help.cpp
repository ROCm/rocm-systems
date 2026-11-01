// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/help.hpp"
#include "parser/schema.hpp"
#include "parser/value_provider.hpp"

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

struct help_view
{
    const schema&                  config_schema;
    help_style                     style     = help_style::compact;
    const value_provider_registry* providers = nullptr;
};

std::vector<std::string>
choice_names(const schema_param& param, const value_provider_registry* providers)
{
    std::vector<std::string> names;
    names.reserve(param.choices.size());
    for(const auto& item : param.choices)
    {
        names.push_back(item.value);
    }
    const auto provider_name = param.value_provider.value_or(std::string{});
    if(!names.empty() || providers == nullptr || provider_name.empty())
    {
        return names;
    }
    const auto* provider = providers->find(provider_name);
    if(provider == nullptr)
    {
        return names;
    }
    return provider->choices(param);
}

void
append_choices(std::ostream& out, const schema_param& param,
               const value_provider_registry* providers)
{
    const auto names = choice_names(param, providers);
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
append_details(std::ostream& out, const schema_param& param, const help_view& view)
{
    if(!param.description.empty())
    {
        out << "      " << param.description << '\n';
    }
    if(view.style == help_style::full)
    {
        out << "      type: " << type_name(param.type) << '\n';
        append_choices(out, param, view.providers);
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
append_category(std::ostream& out, const help_view& view, const category& item)
{
    auto wrote_heading = false;
    for(const auto* param : view.config_schema.params_in_category(item.id))
    {
        if(param == nullptr || !include_param(*param, view.style))
        {
            continue;
        }
        if(!wrote_heading)
        {
            out << item.name << '\n';
            wrote_heading = true;
        }
        append_usage(out, *param);
        append_details(out, *param, view);
    }
}

bool
topic_selects(std::string_view topic, std::string_view category_id)
{
    return topic.empty() || topic == category_id;
}

void
append_heading(std::ostream& out, const help_view& view, std::string_view heading)
{
    if(!heading.empty())
    {
        out << heading << '\n';
    }
    for(const auto& param : view.config_schema.params())
    {
        if(param.category != heading || !include_param(param, view.style))
        {
            continue;
        }
        append_usage(out, param);
        append_details(out, param, view);
    }
}

void
append_unregistered(std::ostream& out, const help_view& view, std::string_view topic)
{
    std::vector<std::string> headings;
    headings.reserve(view.config_schema.params().size());
    for(const auto& param : view.config_schema.params())
    {
        if(view.config_schema.find_category(param.category) != nullptr)
        {
            continue;
        }
        if(!topic_selects(topic, param.category))
        {
            continue;
        }
        if(std::ranges::find(headings, param.category) == headings.end())
        {
            headings.push_back(param.category);
        }
    }
    for(const auto& heading : headings)
    {
        append_heading(out, view, heading);
    }
}
}  // namespace

std::string
format_help(const schema& config_schema, help_style style, std::string_view topic,
            const value_provider_registry* providers)
{
    std::ostringstream out;
    out << "ROCm Systems Profiler options\n\n";
    const auto view    = help_view{ .config_schema = config_schema,
                                    .style         = style,
                                    .providers     = providers };
    auto       matched = topic.empty();
    for(const auto& item : config_schema.categories())
    {
        if(!category_matches(item, topic))
        {
            continue;
        }
        matched = true;
        append_category(out, view, item);
    }
    const auto before = out.tellp();
    append_unregistered(out, view, topic);
    if(out.tellp() != before)
    {
        matched = true;
    }
    if(!matched)
    {
        out << "No parameters match topic '" << topic << "'.\n";
    }
    return out.str();
}

}  // namespace rocprofsys::parser
