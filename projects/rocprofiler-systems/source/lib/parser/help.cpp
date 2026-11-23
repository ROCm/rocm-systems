// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/help.hpp"
#include "parser/backend.hpp"
#include "parser/schema.hpp"

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
constexpr std::size_t k_column_gap = 2;

/// The rows shown under one heading. Rows with no category have an empty heading.
struct help_group
{
    std::string                      heading;
    std::vector<const schema_param*> params;
};

bool
include_param(const schema_param& param, param_level max_level)
{
    return param.level == param_level::basic || max_level == param_level::advanced;
}

bool
already_listed(const std::vector<std::string>& names, std::string_view name)
{
    return std::ranges::find(names, name) != names.end();
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

/// The rows shown at @p max_level, under each category in the order it first appears,
/// then the rows with no category. A heading with no row shown is left out.
std::vector<help_group>
help_groups(const schema& config_schema, param_level max_level)
{
    auto headings = category_order(config_schema);
    headings.emplace_back();
    std::vector<help_group> groups;
    for(auto& heading : headings)
    {
        auto shown = config_schema.params_in_category(heading);
        std::erase_if(shown, [max_level](const schema_param* param) {
            return !include_param(*param, max_level);
        });
        if(!shown.empty())
        {
            groups.push_back(
                { .heading = std::move(heading), .params = std::move(shown) });
        }
    }
    return groups;
}

/// The groups format_help() shows: all of them, the category @p topic names, or the
/// row @p topic spells, whatever its level.
std::vector<help_group>
topic_groups(const schema& config_schema, param_level max_level, std::string_view topic)
{
    auto groups = help_groups(config_schema, max_level);
    if(topic.empty())
    {
        return groups;
    }
    std::erase_if(groups,
                  [topic](const help_group& group) { return group.heading != topic; });
    const auto* param = config_schema.find_param_by_arg(topic);
    if(groups.empty() && param != nullptr)
    {
        groups.push_back({ .heading = param->category, .params = { param } });
    }
    return groups;
}

/// arg and then each of arg_aliases, as in "--output-format, -f". A row with no
/// command-line spelling is named by its id.
std::string
spellings(const schema_param& param)
{
    if(!param.arg)
    {
        return param.id;
    }
    auto text = *param.arg;
    for(const auto& alias : param.arg_aliases)
    {
        text.append(", ").append(alias);
    }
    return text;
}

/// Where a summary line's description starts: past the widest spelling.
std::size_t
description_column(const std::vector<help_group>& groups)
{
    // A plain loop: amdclang++ 22 (ROCm 7.2) crashes on max_element over a joined view.
    std::size_t widest = 0;
    for(const auto& group : groups)
    {
        for(const auto* param : group.params)
        {
            widest = std::max(widest, spellings(*param).size());
        }
    }
    return widest == 0 ? 0 : widest + k_column_gap;
}

void
append_heading(std::ostream& out, const help_group& group)
{
    if(!group.heading.empty())
    {
        out << group.heading << '\n';
    }
}

void
append_usage(std::ostream& out, const schema_param& param)
{
    out << "  " << spellings(param);
    if(param.arg && param.type != param_type::boolean)
    {
        out << " <value>";
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
append_summary_line(std::ostream& out, const schema_param& param, std::size_t column)
{
    const auto text = spellings(param);
    out << "  " << text;
    if(!param.description.empty())
    {
        out << std::string(column - text.size(), ' ') << param.description;
    }
    out << '\n';
}
}  // namespace

std::string
format_help(const schema& config_schema, param_level max_level, std::string_view topic,
            const backend_registry* backends)
{
    std::ostringstream out;
    const auto         groups = topic_groups(config_schema, max_level, topic);
    for(const auto& group : groups)
    {
        append_heading(out, group);
        for(const auto* param : group.params)
        {
            append_usage(out, *param);
            append_details(out, *param, backends);
        }
    }
    if(groups.empty() && !topic.empty())
    {
        out << "No parameters match topic '" << topic << "'.\n";
    }
    return out.str();
}

std::string
format_summary(const schema& config_schema, param_level max_level)
{
    std::ostringstream out;
    const auto         groups = help_groups(config_schema, max_level);
    const auto         column = description_column(groups);
    for(const auto& group : groups)
    {
        append_heading(out, group);
        for(const auto* param : group.params)
        {
            append_summary_line(out, *param, column);
        }
        out << '\n';
    }
    return out.str();
}

}  // namespace rocprofsys::parser
