// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/file_parser.hpp"
#include "parser/parse_result.hpp"
#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <fstream>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
const schema_param*
find_key(const schema& config_schema, std::string_view key)
{
    if(const auto* param = config_schema.find_param(key))
    {
        return param;
    }
    return config_schema.find_param_by_env(key);
}

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
        items.push_back(*text);
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

class document_reader
{
public:
    document_reader(const schema& config_schema, parse_result& result)
    : m_schema(config_schema)
    , m_result(result)
    {}

    void read(const std::string& key, const nlohmann::json& value)
    {
        const auto* param = find_key(m_schema, key);
        if(param == nullptr)
        {
            m_result.errors.push_back(fmt::format("unknown parameter '{}'", key));
            return;
        }
        if(m_seen.contains(param->id))
        {
            reject_duplicate(*param);
            return;
        }

        const auto decoded = decode_value(*param, value);
        if(decoded.array_rejected)
        {
            m_result.errors.push_back(
                fmt::format("parameter '{}' does not accept an array", param->id));
            return;
        }
        if(!decoded.text.has_value())
        {
            m_result.errors.push_back(
                fmt::format("unsupported value for parameter '{}'", param->id));
            return;
        }

        const auto text   = decoded.text.value_or(std::string{});
        const auto stored = canonical_choice(*param, text);
        if(!is_valid_value(*param, stored))
        {
            m_result.errors.push_back(
                fmt::format("invalid value '{}' for parameter '{}'", text, param->id));
            return;
        }
        m_seen.insert(param->id);
        m_result.values.emplace(param->id, stored);
    }

private:
    void reject_duplicate(const schema_param& param)
    {
        m_result.values.erase(param.id);
        m_result.errors.push_back(
            fmt::format("parameter '{}' is given by more than one key", param.id));
    }

    const schema&                      m_schema;
    parse_result&                      m_result;
    std::set<std::string, std::less<>> m_seen;
};
}  // namespace

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
    for(const auto& [key, value] : document.items())
    {
        if((!key.empty() && key.front() == '_') || value.is_null())
        {
            continue;
        }
        reader.read(key, value);
    }
    return result;
}

parse_result
file_parser::parse_file(const schema& config_schema, const std::string& filepath) const
{
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
