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
 * within min/max, and enumerations must name a choice unless custom values are
 * allowed. An enumeration_list is split on list_separator(); each trimmed element
 * must be a valid choice, and an empty string is the empty list. Boolean tokens
 * (true/false, yes/no, on/off, y/n, t/f, 1/0) are matched case-insensitively. Any
 * other integer is true when it is not zero. An empty boolean is not a value.
 * pattern is stored and is not applied.
 */
[[nodiscard]] bool
is_valid_value(const schema_param& param, std::string_view value);

/// Shared wording for a value the catalog refuses.
[[nodiscard]] std::string
invalid_value_message(std::string_view value, std::string_view param_id);

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
 * evaluates inline and schema-level constraints. A parameter is enabled only when a
 * value was provided for it: non-empty, and true for a boolean. A condition may fall
 * back to a default when that parameter was not provided. depends_on is reported only
 * for a provided parameter, and a forces constraint only when one of the values it
 * names does not hold. forces does not write those values.
 */
[[nodiscard]] std::vector<violation>
validate_values(const schema& config_schema, const value_map_t& values);

/**
 * Check the catalog itself, before any user input.
 *
 * Reports invalid defaults, unknown parameter ids in rules and effect fields, a
 * non-empty category that was not registered, an empty list separator, inverted or
 * non-integer bounds, and argument counts that disagree with binds. pattern is not
 * evaluated.
 */
[[nodiscard]] std::vector<violation>
validate_schema(const schema& config_schema);

}  // namespace rocprofsys::parser
