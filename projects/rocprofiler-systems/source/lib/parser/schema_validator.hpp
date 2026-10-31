// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::parser
{
/**
 * Check whether @p value is an acceptable textual value for @p param.
 *
 * The whole string must parse as the parameter's type. Numbers must be finite and
 * within min/max, strings must match the pattern (if any), and enumerations must
 * name a choice unless custom values are allowed. An enumeration_list is split on
 * list_separator(); each trimmed element must be a valid choice, and an empty string
 * is the empty list. Boolean tokens (true/false, yes/no, on/off, 1/0) are matched
 * case-insensitively.
 */
[[nodiscard]] bool
is_valid_value(const schema_param& param, std::string_view value);

struct violation
{
    severity                 level = severity::error;
    std::string              message;
    std::vector<std::string> params;
};

/**
 * Validate a set of explicitly provided parameter values against @p config_schema.
 *
 * Reports unknown parameters, invalid values, and missing required parameters, then
 * evaluates inline and schema-level constraints. Constraints use each parameter's
 * effective value: the provided value, otherwise its default. A parameter is enabled
 * when its effective value is non-empty and, for booleans, true. depends_on is only
 * reported for explicitly provided values, and a forces constraint only when one of
 * the values it sets does not hold.
 */
[[nodiscard]] std::vector<violation>
validate_values(const schema& config_schema, const value_map_t& values);

}  // namespace rocprofsys::parser
