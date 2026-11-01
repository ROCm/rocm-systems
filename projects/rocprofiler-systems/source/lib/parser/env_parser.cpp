// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/env_parser.hpp"
#include "parser/parse_result.hpp"
#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"

#include "common/string_utility.hpp"

#include <fmt/format.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace rocprofsys::parser
{
namespace
{
std::optional<std::string>
process_environment(std::string_view name)
{
    const auto* value = std::getenv(std::string{ name }.c_str());
    return value == nullptr ? std::nullopt : std::optional<std::string>{ value };
}
}  // namespace

env_parser::env_parser(lookup_t lookup)
: m_lookup(std::move(lookup))
{}

env_parser
env_parser::from_process()
{
    return env_parser{ process_environment };
}

parse_result
env_parser::parse(const schema& config_schema) const
{
    parse_result result{};
    for(const auto& param : config_schema.params())
    {
        if(!param.env)
        {
            continue;
        }
        const auto value = m_lookup(param.env->name);
        if(!value)
        {
            continue;
        }
        const auto text = std::string{ utility::string::trim(*value) };
        if(param.type == param_type::enumeration_list && text.empty())
        {
            continue;
        }
        const auto stored = canonical_choice(param, text);
        if(!is_valid_value(param, stored))
        {
            result.errors.push_back(
                fmt::format("invalid value '{}' for {}", text, param.env->name));
            continue;
        }
        result.values.emplace(param.id, stored);
    }
    return result;
}

}  // namespace rocprofsys::parser
