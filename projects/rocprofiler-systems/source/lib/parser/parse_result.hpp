// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::parser
{
/// Values read from one source, plus any input that could not be accepted.
struct parse_result
{
    value_map_t              values;
    std::vector<std::string> errors;
    /// The profiled command. Only the tokens after a standalone "--".
    std::vector<std::string> command;
    /// Remainder, including the first unknown token, when parsing stops early.
    std::vector<std::string> unparsed;

    /// Store @p value under @p param, or record invalid_value_message and store nothing.
    void store_if_valid(const schema_param& param, std::string_view value)
    {
        const auto stored = canonical_choice(param, value);
        if(!is_valid_value(param, stored))
        {
            errors.push_back(invalid_value_message(value, param.id));
            return;
        }
        values.insert_or_assign(param.id, stored);
    }
};

}  // namespace rocprofsys::parser
