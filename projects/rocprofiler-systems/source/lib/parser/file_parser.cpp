// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/file_parser.hpp"
#include "parser/config_store.hpp"
#include "parser/parse_result.hpp"
#include "parser/precedence.hpp"
#include "parser/schema.hpp"
#include "parser/value_text.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
/// Text for a JSON scalar that is checked as text. A number keeps its JSON spelling.
std::optional<std::string>
scalar_text(const nlohmann::json& value)
{
    if(value.is_string())
    {
        return value.get<std::string>();
    }
    if(value.is_boolean())
    {
        return std::string{ value.get<bool>() ? "true" : "false" };
    }
    if(value.is_number())
    {
        return value.dump();
    }
    return std::nullopt;
}

/// A JSON scalar that already has the parameter's type. Anything else, including an
/// integer that does not fit int64, falls back to text and is rejected there.
std::optional<config_value_t>
typed_scalar(const schema_param& param, const nlohmann::json& value)
{
    if(param.type == param_type::boolean && value.is_boolean())
    {
        return config_value_t{ value.get<bool>() };
    }
    if(param.type == param_type::integer && value.is_number_unsigned())
    {
        const auto unsigned_value = value.get<std::uint64_t>();
        if(unsigned_value >
           static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        {
            return std::nullopt;
        }
        return config_value_t{ static_cast<std::int64_t>(unsigned_value) };
    }
    if(param.type == param_type::integer && value.is_number_integer())
    {
        return config_value_t{ value.get<std::int64_t>() };
    }
    if(param.type == param_type::number && value.is_number())
    {
        return config_value_t{ value.get<double>() };
    }
    return std::nullopt;
}

std::optional<std::string>
joined_array(const nlohmann::json& value, std::string_view separator)
{
    std::vector<std::string> items;
    items.reserve(value.size());
    for(const auto& item : value)
    {
        const auto text = scalar_text(item);
        if(!text)
        {
            return std::nullopt;
        }
        items.push_back(*text);
    }
    return fmt::format("{}", fmt::join(items, separator));
}

struct decoded_value
{
    std::optional<config_value_t> typed;
    std::optional<std::string>    text;
    bool                          array_rejected = false;
};

decoded_value
decode_value(const schema_param& param, const nlohmann::json& value)
{
    if(value.is_array())
    {
        if(param.type != param_type::enumeration_list)
        {
            return { .array_rejected = true };
        }
        return { .text = joined_array(value, list_separator(param)) };
    }
    if(auto typed = typed_scalar(param, value))
    {
        return { .typed = std::move(typed) };
    }
    return { .text = scalar_text(value) };
}

[[nodiscard]] bool
ignored_entry(std::string_view key, const nlohmann::json& value)
{
    return (!key.empty() && key.front() == '_') || value.is_null();
}

/// The JSON form of a stored value. A list becomes a JSON array. Parentheses are
/// used because nlohmann::json{x} with one element would build a one-element array.
[[nodiscard]] nlohmann::json
json_value(const config_value_t& value)
{
    if(const auto* flag = std::get_if<bool>(&value))
    {
        return nlohmann::json(*flag);
    }
    if(const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return nlohmann::json(*integer);
    }
    if(const auto* number = std::get_if<double>(&value))
    {
        return nlohmann::json(*number);
    }
    if(const auto* text = std::get_if<std::string>(&value))
    {
        return nlohmann::json(*text);
    }
    return nlohmann::json(std::get<std::vector<std::string>>(value));
}

void
append_parameter(nlohmann::json& object, const schema_param& param,
                 const resolved_config& resolved)
{
    const auto found = resolved.find(param.id);
    if(found == resolved.end() || found->second.source == config_source::default_value)
    {
        return;
    }
    object[param.id] = json_value(found->second.value);
}

/// Categories in the order each word first appears on a parameter row.
std::vector<std::string>
category_order(const schema& config_schema)
{
    std::vector<std::string> order;
    for(const auto& param : config_schema.params())
    {
        if(!param.category.empty() &&
           std::find(order.begin(), order.end(), param.category) == order.end())
        {
            order.push_back(param.category);
        }
    }
    return order;
}

/// A category is known when at least one parameter row uses it.
[[nodiscard]] bool
known_category(const schema& config_schema, std::string_view category_id)
{
    return !category_id.empty() && !config_schema.params_in_category(category_id).empty();
}

void
append_category(nlohmann::json& document, const schema& config_schema,
                const std::string& category_id, const resolved_config& resolved)
{
    auto object = nlohmann::json::object();
    for(const auto* param : config_schema.params_in_category(category_id))
    {
        append_parameter(object, *param, resolved);
    }
    if(!object.empty())
    {
        document[category_id] = std::move(object);
    }
}

class document_reader
{
public:
    document_reader(const schema& config_schema, parse_result& result)
    : m_schema(config_schema)
    , m_result(result)
    {}

    void read_document(const nlohmann::json& document)
    {
        for(const auto& [key, value] : document.items())
        {
            if(ignored_entry(key, value))
            {
                continue;
            }
            read_category(key, value);
        }
    }

private:
    void read_category(const std::string& category_id, const nlohmann::json& value)
    {
        if(!known_category(m_schema, category_id))
        {
            m_result.errors.push_back(fmt::format("unknown category '{}'", category_id));
            return;
        }
        if(!value.is_object())
        {
            m_result.errors.push_back(
                fmt::format("category '{}' must be a JSON object", category_id));
            return;
        }
        for(const auto& [key, entry] : value.items())
        {
            if(ignored_entry(key, entry))
            {
                continue;
            }
            read_parameter(category_id, key, entry);
        }
    }

    void read_parameter(std::string_view category_id, const std::string& key,
                        const nlohmann::json& value)
    {
        const auto* param = m_schema.find_param(key);
        if(param == nullptr)
        {
            m_result.errors.push_back(fmt::format("unknown parameter '{}'", key));
            return;
        }
        if(param->category != category_id)
        {
            m_result.errors.push_back(fmt::format(
                "parameter '{}' is not in category '{}'", param->id, category_id));
            return;
        }
        store_decoded(*param, value);
    }

    void store_decoded(const schema_param& param, const nlohmann::json& value)
    {
        const auto decoded = decode_value(param, value);
        if(decoded.array_rejected)
        {
            m_result.errors.push_back(
                fmt::format("parameter '{}' does not accept an array", param.id));
            return;
        }
        if(decoded.typed)
        {
            m_result.store_typed(param, *decoded.typed);
            return;
        }
        if(!decoded.text)
        {
            m_result.errors.push_back(
                fmt::format("unsupported value for parameter '{}'", param.id));
            return;
        }
        m_result.store_if_valid(param, *decoded.text);
    }

    const schema& m_schema;
    parse_result& m_result;
};
}  // namespace

nlohmann::json
catalog_document(const schema& config_schema, const resolved_config& resolved)
{
    auto document = nlohmann::json::object();
    for(const auto& category_id : category_order(config_schema))
    {
        append_category(document, config_schema, category_id, resolved);
    }
    return document;
}

parse_result
file_parser::parse(const schema& config_schema, const nlohmann::json& document) const
{
    parse_result result{};
    if(!document.is_object())
    {
        result.errors.emplace_back("configuration must be a JSON object");
        return result;
    }

    document_reader reader{ config_schema, result };
    reader.read_document(document);
    return result;
}

parse_result
file_parser::parse_file(const schema& config_schema, const std::string& filepath) const
{
    std::error_code directory_error;
    if(std::filesystem::is_directory(filepath, directory_error))
    {
        throw std::runtime_error(
            fmt::format("configuration path '{}' is a directory", filepath));
    }

    std::ifstream input{ filepath };
    if(!input.is_open())
    {
        parse_result result{};
        result.errors.push_back(
            fmt::format("failed to open configuration file '{}'", filepath));
        return result;
    }

    try
    {
        return parse(config_schema, nlohmann::json::parse(input));
    } catch(const nlohmann::json::parse_error& error)
    {
        parse_result result{};
        result.errors.push_back(fmt::format("failed to parse configuration file '{}': {}",
                                            filepath, error.what()));
        return result;
    }
}

}  // namespace rocprofsys::parser
