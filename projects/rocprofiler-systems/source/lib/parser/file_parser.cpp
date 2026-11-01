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

#include <cstdint>
#include <filesystem>
#include <fstream>
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
constexpr std::int64_t k_unused_integer = 0;
constexpr double       k_unused_number  = 0.0;

std::optional<std::string>
scalar_text(const nlohmann::json& value)
{
    if(value.is_string())
    {
        return value.get<std::string>();
    }
    if(value.is_boolean() || value.is_number())
    {
        return value.dump();
    }
    return std::nullopt;
}

struct decoded_value
{
    bool                       array_rejected = false;
    std::optional<std::string> text;
};

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
        items.push_back(text.value_or(std::string{}));
    }
    return fmt::format("{}", fmt::join(items, separator));
}

decoded_value
decode_value(const schema_param& param, const nlohmann::json& value)
{
    if(!value.is_array())
    {
        return { .text = scalar_text(value) };
    }
    if(param.type != param_type::enumeration_list)
    {
        return { .array_rejected = true };
    }
    return { .text = joined_array(value, list_separator(param)) };
}

[[nodiscard]] bool
ignored_entry(std::string_view key, const nlohmann::json& value)
{
    return (!key.empty() && key.front() == '_') || value.is_null();
}

[[nodiscard]] nlohmann::json
json_leaf(const schema_param& param, std::string_view text)
{
    if(param.type == param_type::boolean)
    {
        return parse_boolean(text).value_or(false);
    }
    if(param.type == param_type::integer)
    {
        return parse_integer(text).value_or(k_unused_integer);
    }
    if(param.type == param_type::number)
    {
        return parse_number(text).value_or(k_unused_number);
    }
    if(param.type == param_type::enumeration_list)
    {
        auto       items = nlohmann::json::array();
        const auto parts = split_list(text, list_separator(param));
        for(const auto& part : parts.value_or(std::vector<std::string>{}))
        {
            items.push_back(part);
        }
        return items;
    }
    return std::string{ text };
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
    object[param.id] = json_leaf(param, found->second.value);
}

void
append_category(nlohmann::json& document, const schema& config_schema,
                const category& group, const resolved_config& resolved)
{
    auto object = nlohmann::json::object();
    for(const auto* param : config_schema.params_in_category(group.id))
    {
        append_parameter(object, *param, resolved);
    }
    if(!object.empty())
    {
        document[group.id] = std::move(object);
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
        if(m_schema.find_category(category_id) == nullptr)
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
        if(!decoded.text)
        {
            m_result.errors.push_back(
                fmt::format("unsupported value for parameter '{}'", param.id));
            return;
        }
        m_result.store_if_valid(param, decoded.text.value_or(std::string{}));
    }

    const schema& m_schema;
    parse_result& m_result;
};
}  // namespace

nlohmann::json
catalog_document(const schema& config_schema, const resolved_config& resolved)
{
    auto document = nlohmann::json::object();
    for(const auto& group : config_schema.categories())
    {
        append_category(document, config_schema, group, resolved);
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
