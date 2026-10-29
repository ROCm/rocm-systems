// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/default_parser.hpp"
#include "parser/parse_result.hpp"
#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"

#include <fmt/format.h>

namespace rocprofsys::parser
{
parse_result
default_parser::parse(const schema& config_schema) const
{
    parse_result result{};
    for(const auto& param : config_schema.params())
    {
        if(!param.default_value)
        {
            continue;
        }
        if(!is_valid_value(param, *param.default_value))
        {
            result.errors.push_back(
                fmt::format("invalid default value '{}' for parameter '{}'",
                            *param.default_value, param.id));
            continue;
        }
        result.values.emplace(param.id, *param.default_value);
    }
    return result;
}

}  // namespace rocprofsys::parser
