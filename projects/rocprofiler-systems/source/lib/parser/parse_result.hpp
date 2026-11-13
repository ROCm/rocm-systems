// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"
#include "parser/value_text.hpp"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
/// Values read from one source, plus any input that could not be accepted.
struct parse_result
{
    typed_values_t           values;
    std::vector<std::string> errors;
    /// The profiled command. Only the tokens after a standalone "--".
    std::vector<std::string> command;
    /// Remainder, including the first unknown token, when parsing stops early.
    std::vector<std::string> unparsed;

    /// Canonical text of a stored value. @p param_id must be present in values.
    [[nodiscard]] std::string text(std::string_view param_id) const
    {
        return format_value(values.at(std::string{ param_id }));
    }

    /// Parse @p value once and store the typed result, or record
    /// invalid_value_message and store nothing.
    void store_if_valid(const schema_param& param, std::string_view value)
    {
        const auto parsed = parse_value(param, value);
        if(!parsed)
        {
            errors.push_back(invalid_value_message(value, param.id));
            return;
        }
        values.insert_or_assign(param.id, *parsed);
    }

    /// Store a value that is already typed, such as a JSON boolean, or record
    /// invalid_value_message and store nothing.
    void store_typed(const schema_param& param, config_value_t value)
    {
        if(!accepts_value(param, value))
        {
            errors.push_back(invalid_value_message(format_value(value), param.id));
            return;
        }
        values.insert_or_assign(param.id, std::move(value));
    }
};

}  // namespace rocprofsys::parser
