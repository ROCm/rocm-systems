// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/env_parser.hpp"
#include "parser/parse_result.hpp"
#include "parser/schema.hpp"

#include "common/string_utility.hpp"

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
        if(text.empty())
        {
            continue;
        }
        result.store_if_valid(param, text);
    }
    return result;
}

}  // namespace rocprofsys::parser
