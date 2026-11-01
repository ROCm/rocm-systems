// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/preset_parser.hpp"
#include "parser/parse_result.hpp"
#include "parser/schema.hpp"

#include <fmt/format.h>

namespace rocprofsys::parser
{
parse_result
preset_parser::parse(const schema& config_schema, const settings_t& settings) const
{
    parse_result result{};
    for(const auto& [variable, value] : settings)
    {
        const auto* param = config_schema.find_param_by_env(variable);
        if(param == nullptr)
        {
            result.errors.push_back(
                fmt::format("unknown environment variable '{}'", variable));
            continue;
        }
        result.store_if_valid(*param, value);
    }
    return result;
}

}  // namespace rocprofsys::parser
