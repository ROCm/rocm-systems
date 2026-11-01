// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"

#include "common/string_utility.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
bool
known_param(const schema& catalog, std::string_view param_id)
{
    return !param_id.empty() && catalog.find_param(param_id) != nullptr;
}

bool
accepted_text(const schema& catalog, std::string_view param_id, std::string_view text)
{
    const auto* param = catalog.find_param(param_id);
    return param != nullptr && is_valid_value(*param, text);
}

bool
constraint_refs_are_usable(const schema& catalog, const constraint& item)
{
    const auto params_known =
        std::ranges::all_of(item.params, [&catalog](const std::string& param_id) {
            return known_param(catalog, param_id);
        });
    const auto terms_fit =
        std::ranges::all_of(item.when.terms, [&catalog](const param_condition& term) {
            return accepted_text(catalog, term.param, term.equals);
        });
    const auto forced_fit =
        std::ranges::all_of(item.forced_values, [&catalog](const auto& entry) {
            return accepted_text(catalog, entry.first, entry.second);
        });
    return params_known && terms_fit && forced_fit;
}
}  // namespace

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

namespace
{
[[nodiscard]] bool
alias_matches(const choice& item, std::string_view token)
{
    return std::ranges::any_of(
        item.aliases, [token](const std::string& alias) { return alias == token; });
}

std::string
canonical_token(const schema_param& param, std::string_view token)
{
    for(const auto& item : param.choices)
    {
        if(item.value == token)
        {
            return item.value;
        }
    }
    for(const auto& item : param.choices)
    {
        if(alias_matches(item, token))
        {
            return item.value;
        }
    }
    return std::string{ token };
}
}  // namespace

std::string
canonical_choice(const schema_param& param, std::string_view value)
{
    if(param.type == param_type::enumeration)
    {
        return canonical_token(param, value);
    }
    if(param.type != param_type::enumeration_list)
    {
        return std::string{ value };
    }

    const auto parts = split_list(value, list_separator(param));
    if(!parts)
    {
        return std::string{ value };
    }

    std::string joined;
    for(std::size_t index = 0; index < parts->size(); ++index)
    {
        if(index != 0)
        {
            joined.append(list_separator(param));
        }
        joined.append(canonical_token(param, (*parts)[index]));
    }
    return joined;
}

void
schema::set_version(std::string version)
{
    m_version = std::move(version);
}

bool
schema::add_category(category value)
{
    if(value.id.empty() || m_category_index.contains(value.id))
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
    const bool empty_env = param.env && param.env->name.empty();
    const bool empty_arg = param.arg && param.arg->name.empty();
    const bool env_taken = param.env && m_env_index.contains(param.env->name);
    const bool arg_taken = param.arg && m_arg_index.contains(param.arg->name);
    if(param.id.empty() || empty_env || empty_arg || m_param_index.contains(param.id) ||
       env_taken || arg_taken)
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

bool
schema::add_constraint(constraint value)
{
    if(!constraint_refs_are_usable(*this, value))
    {
        return false;
    }
    m_constraints.push_back(std::move(value));
    return true;
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
