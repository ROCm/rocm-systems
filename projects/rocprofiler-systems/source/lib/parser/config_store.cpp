// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/config_store.hpp"
#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"
#include "parser/value_text.hpp"

#include "common/string_utility.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
const schema_param&
require_param(const schema& config_schema, std::string_view param_id)
{
    const auto* param = config_schema.find_param(param_id);
    if(param == nullptr)
    {
        throw std::runtime_error(fmt::format("unknown parameter '{}'", param_id));
    }
    return *param;
}

void
notify(std::string_view param_id, std::string_view value,
       const std::vector<config_store::change_callback_t>& callbacks)
{
    for(const auto& callback : callbacks)
    {
        callback(param_id, value);
    }
}

std::vector<std::string>
split_list(std::string_view text, std::string_view separator)
{
    std::vector<std::string> items;
    if(text.empty())
    {
        return items;
    }
    if(separator.empty())
    {
        items.emplace_back(text);
        return items;
    }

    while(true)
    {
        const auto position = text.find(separator);
        const auto element  = utility::string::trim(text.substr(0, position));
        items.emplace_back(element);
        if(position == std::string_view::npos)
        {
            return items;
        }
        text.remove_prefix(position + separator.size());
    }
}

template <typename ValueT>
struct dependent_false : std::false_type
{};

template <typename ValueT>
std::optional<ValueT>
convert(param_type type, std::string_view text, std::string_view separator)
{
    if constexpr(std::is_same_v<ValueT, bool>)
    {
        return type == param_type::boolean ? parse_boolean(text) : std::nullopt;
    }
    else if constexpr(std::is_same_v<ValueT, std::int64_t>)
    {
        return type == param_type::integer ? parse_integer(text) : std::nullopt;
    }
    else if constexpr(std::is_same_v<ValueT, double>)
    {
        return type == param_type::number ? parse_number(text) : std::nullopt;
    }
    else if constexpr(std::is_same_v<ValueT, std::string>)
    {
        const bool is_text = type == param_type::string || type == param_type::path ||
                             type == param_type::enumeration;
        return is_text ? std::optional<std::string>{ text } : std::nullopt;
    }
    else if constexpr(std::is_same_v<ValueT, std::vector<std::string>>)
    {
        return type == param_type::enumeration_list
                   ? std::optional{ split_list(text, separator) }
                   : std::nullopt;
    }
    else
    {
        static_assert(dependent_false<ValueT>::value, "unsupported config value type");
    }
}
}  // namespace

config_store::config_store(schema config_schema)
: m_schema{ std::make_shared<const schema>(std::move(config_schema)) }
{}

config_store&
config_store::instance()
{
    static config_store s_store{ schema{} };
    return s_store;
}

void
config_store::reset(schema config_schema)
{
    auto replacement = std::make_shared<const schema>(std::move(config_schema));
    const std::unique_lock<std::shared_mutex> lock{ m_mutex };
    m_schema = std::move(replacement);
    m_values.clear();
    m_callbacks.clear();
}

std::shared_ptr<const schema>
config_store::locked_schema() const
{
    const std::shared_lock<std::shared_mutex> lock{ m_mutex };
    return m_schema;
}

void
config_store::set(std::string_view param_id, std::string_view value, config_source source)
{
    const auto  config_schema = locked_schema();
    const auto& param         = require_param(*config_schema, param_id);
    if(!is_valid_value(param, value))
    {
        throw std::runtime_error(
            fmt::format("invalid value '{}' for parameter '{}'", value, param_id));
    }

    std::vector<change_callback_t> callbacks;
    {
        const std::unique_lock<std::shared_mutex> lock{ m_mutex };
        static_cast<void>(require_param(*m_schema, param_id));

        const auto key           = std::string{ param_id };
        auto [entry, inserted]   = m_values.try_emplace(key);
        const bool value_changed = inserted || entry->second.value != value;
        entry->second.value      = std::string{ value };
        entry->second.source     = source;
        const auto registered    = m_callbacks.find(key);
        if(value_changed && registered != m_callbacks.end())
        {
            callbacks = registered->second;
        }
    }

    notify(param_id, value, callbacks);
}

std::optional<config_store::stored_text>
config_store::lookup(std::string_view param_id) const
{
    const std::shared_lock<std::shared_mutex> lock{ m_mutex };
    const auto                                entry = m_values.find(param_id);
    if(entry == m_values.end())
    {
        return std::nullopt;
    }
    const auto* param = m_schema->find_param(param_id);
    if(param == nullptr)
    {
        return std::nullopt;
    }
    return stored_text{ .type      = param->type,
                        .text      = entry->second.value,
                        .separator = std::string{ list_separator(*param) } };
}

std::optional<std::string>
config_store::get(std::string_view param_id) const
{
    const auto stored = lookup(param_id);
    return stored ? std::optional{ stored->text } : std::nullopt;
}

bool
config_store::contains(std::string_view param_id) const
{
    const std::shared_lock<std::shared_mutex> lock{ m_mutex };
    return m_values.contains(param_id);
}

std::optional<config_source>
config_store::source_of(std::string_view param_id) const
{
    const std::shared_lock<std::shared_mutex> lock{ m_mutex };
    const auto                                entry = m_values.find(param_id);
    return entry == m_values.end() ? std::nullopt : std::optional{ entry->second.source };
}

template <typename ValueT>
std::optional<ValueT>
config_store::get_as(std::string_view param_id) const
{
    const auto stored = lookup(param_id);
    return stored ? convert<ValueT>(stored->type, stored->text, stored->separator)
                  : std::nullopt;
}

template std::optional<bool>         config_store::get_as<bool>(std::string_view) const;
template std::optional<std::int64_t> config_store::get_as<std::int64_t>(
    std::string_view) const;
template std::optional<double>      config_store::get_as<double>(std::string_view) const;
template std::optional<std::string> config_store::get_as<std::string>(
    std::string_view) const;
template std::optional<std::vector<std::string>>
    config_store::get_as<std::vector<std::string>>(std::string_view) const;

std::vector<std::string>
config_store::ids_in_category(std::string_view category_id) const
{
    const std::shared_lock<std::shared_mutex> lock{ m_mutex };
    std::vector<std::string>                  ids;
    ids.reserve(m_schema->params().size());
    for(const auto& param : m_schema->params())
    {
        if(param.category == category_id && m_values.contains(param.id))
        {
            ids.push_back(param.id);
        }
    }
    return ids;
}

value_map_t
config_store::values() const
{
    const std::shared_lock<std::shared_mutex> lock{ m_mutex };
    value_map_t                               result;
    for(const auto& [param_id, stored] : m_values)
    {
        result.emplace(param_id, stored.value);
    }
    return result;
}

void
config_store::on_change(std::string_view param_id, change_callback_t callback)
{
    const std::unique_lock<std::shared_mutex> lock{ m_mutex };
    static_cast<void>(require_param(*m_schema, param_id));
    m_callbacks[std::string{ param_id }].push_back(std::move(callback));
}

void
config_store::clear()
{
    const std::unique_lock<std::shared_mutex> lock{ m_mutex };
    m_values.clear();
}

std::shared_ptr<const schema>
config_store::config_schema() const
{
    return locked_schema();
}

}  // namespace rocprofsys::parser
