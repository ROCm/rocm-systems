// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema.hpp"

#include "common/string_utility.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
std::string_view
list_separator(const schema_param& param) noexcept
{
    return param.env ? std::string_view{ param.env->separator } : std::string_view{ "," };
}

std::optional<std::vector<std::string>>
split_list(std::string_view value, std::string_view separator)
{
    if(separator.empty())
    {
        return std::nullopt;
    }

    std::vector<std::string> parts;
    if(value.empty())
    {
        return parts;
    }

    std::size_t start = 0;
    while(start <= value.size())
    {
        const auto position = value.find(separator, start);
        const auto end = (position == std::string_view::npos) ? value.size() : position;
        parts.emplace_back(utility::string::trim(value.substr(start, end - start)));
        if(position == std::string_view::npos)
        {
            break;
        }
        start = position + separator.size();
    }
    return parts;
}

void
schema::set_version(std::string version)
{
    m_version = std::move(version);
}

bool
schema::add_category(category value)
{
    if(m_category_index.contains(value.id))
    {
        return false;
    }

    m_category_index.emplace(value.id, m_categories.size());
    m_categories.push_back(std::move(value));
    return true;
}

bool
schema::add_param(schema_param param)
{
    const bool env_taken = param.env && m_env_index.contains(param.env->name);
    const bool arg_taken = param.arg && m_arg_index.contains(param.arg->name);
    if(m_param_index.contains(param.id) || env_taken || arg_taken)
    {
        return false;
    }

    const auto position = m_params.size();
    m_param_index.emplace(param.id, position);
    if(param.env)
    {
        m_env_index.emplace(param.env->name, position);
    }
    if(param.arg)
    {
        m_arg_index.emplace(param.arg->name, position);
    }
    m_params.push_back(std::move(param));
    return true;
}

void
schema::add_constraint(constraint value)
{
    m_constraints.push_back(std::move(value));
}

const category*
schema::find_category(std::string_view category_id) const
{
    auto itr = m_category_index.find(category_id);
    return (itr != m_category_index.end()) ? &m_categories.at(itr->second) : nullptr;
}

const schema_param*
schema::find_param(std::string_view param_id) const
{
    return find_indexed(m_param_index, param_id);
}

const schema_param*
schema::find_param_by_env(std::string_view name) const
{
    return find_indexed(m_env_index, name);
}

const schema_param*
schema::find_param_by_arg(std::string_view name) const
{
    return find_indexed(m_arg_index, name);
}

std::vector<const schema_param*>
schema::params_in_category(std::string_view category_id) const
{
    std::vector<const schema_param*> result;
    for(const auto& param : m_params)
    {
        if(param.category == category_id)
        {
            result.push_back(&param);
        }
    }
    return result;
}

const schema_param*
schema::find_indexed(const index_map_t& index, std::string_view key) const
{
    auto itr = index.find(key);
    return (itr != index.end()) ? &m_params.at(itr->second) : nullptr;
}

}  // namespace rocprofsys::parser
