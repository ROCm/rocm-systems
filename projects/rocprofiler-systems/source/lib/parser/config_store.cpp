// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/config_store.hpp"
#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"
#include "parser/value_text.hpp"

#include <fmt/format.h>

#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
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
}  // namespace

config_store::config_store(schema config_schema)
: m_schema{ std::make_shared<const schema>(std::move(config_schema)) }
{}

void
config_store::set(std::string_view param_id, std::string_view value, config_source source)
{
    const auto& param  = require_param(*m_schema, param_id);
    const auto  parsed = parse_value(param, value);
    if(!parsed)
    {
        throw std::runtime_error(invalid_value_message(value, param_id));
    }
    store_parsed(param_id, *parsed, source);
}

void
config_store::set_value(std::string_view param_id, config_value_t value,
                        config_source source)
{
    const auto& param = require_param(*m_schema, param_id);
    if(!accepts_value(param, value))
    {
        throw std::runtime_error(invalid_value_message(format_value(value), param_id));
    }
    store_parsed(param_id, std::move(value), source);
}

void
config_store::store_parsed(std::string_view param_id, config_value_t value,
                           config_source source)
{
    const auto                     text = format_value(value);
    std::vector<change_callback_t> callbacks;
    {
        const std::unique_lock<std::shared_mutex> lock{ m_mutex };

        const auto key           = std::string{ param_id };
        auto [entry, inserted]   = m_values.try_emplace(key);
        const bool value_changed = inserted || entry->second.value != value;
        entry->second.value      = std::move(value);
        entry->second.source     = source;
        const auto registered    = m_callbacks.find(key);
        if(value_changed && registered != m_callbacks.end())
        {
            callbacks = registered->second;
        }
    }

    notify(param_id, text, callbacks);
}

std::optional<config_store::stored_value>
config_store::lookup(std::string_view param_id) const
{
    const std::shared_lock<std::shared_mutex> lock{ m_mutex };
    const auto                                entry = m_values.find(param_id);
    if(entry == m_values.end())
    {
        return std::nullopt;
    }
    return entry->second;
}

std::optional<std::string>
config_store::get(std::string_view param_id) const
{
    const auto stored = lookup(param_id);
    return stored ? std::optional{ format_value(stored->value) } : std::nullopt;
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
        result.emplace(param_id, format_value(stored.value));
    }
    return result;
}

std::vector<config_slot>
config_store::slots() const
{
    const std::shared_lock<std::shared_mutex> lock{ m_mutex };
    std::vector<config_slot>                  result;
    result.reserve(m_values.size());
    for(const auto& [param_id, stored] : m_values)
    {
        result.push_back(config_slot{
            .id = param_id, .value = stored.value, .source = stored.source });
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
    return m_schema;
}

}  // namespace rocprofsys::parser
