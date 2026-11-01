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
#include <optional>
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

std::optional<std::string>
value_text(const nlohmann::json& value, std::string_view separator)
{
    if(!value.is_array())
    {
        return scalar_text(value);
    }

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

void
read_entry(const schema& config_schema, const std::string& key,
           const nlohmann::json& value, parse_result& result)
{
    const auto* param = find_key(config_schema, key);
    if(param == nullptr)
    {
        result.errors.push_back(fmt::format("unknown parameter '{}'", key));
        return;
    }

    const auto text = value_text(value, list_separator(*param));
    if(!text)
    {
        result.errors.push_back(
            fmt::format("unsupported value for parameter '{}'", param->id));
        return;
    }
    const auto stored = canonical_choice(*param, *text);
    if(!is_valid_value(*param, stored))
    {
        result.errors.push_back(
            fmt::format("invalid value '{}' for parameter '{}'", *text, param->id));
        return;
    }
    result.values.insert_or_assign(param->id, stored);
}
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

    for(const auto& [key, value] : document.items())
    {
        if((!key.empty() && key.front() == '_') || value.is_null())
        {
            continue;
        }
        read_entry(config_schema, key, value, result);
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
