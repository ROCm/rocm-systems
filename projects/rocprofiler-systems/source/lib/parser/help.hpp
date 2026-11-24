// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/backend.hpp"
#include "parser/schema.hpp"

#include <string>
#include <string_view>

namespace rocprofsys::parser
{
/**
 * Render help for @p config_schema.
 *
 * Parameters are grouped by schema_param::category, in the order each word first
 * appears. The heading is that word, and the text starts with the first one; the
 * caller writes its own title. A category with no row shown has no heading, and rows
 * with no category come last, under no heading. @p topic limits the text to one
 * category or, when no category has that name, to the parameter it spells: its arg
 * or one of arg_aliases, such as "--output-format", whatever that parameter's level.
 * @p max_level basic omits advanced parameters. advanced includes them. A row's usage
 * line lists arg and then each of arg_aliases, as in "--output-format, -f <value>".
 * Every shown row prints its description, environment name, default, and choices:
 * its own, then the names its registered backend lists. A null @p backends skips
 * that lookup.
 */
[[nodiscard]] std::string
format_help(const schema& config_schema, param_level max_level = param_level::basic,
            std::string_view topic = {}, const backend_registry* backends = nullptr);

/**
 * One line per parameter, grouped as format_help() groups them: arg and each of
 * arg_aliases, then the description, which starts in the same column on every line.
 * @p max_level is as in format_help().
 */
[[nodiscard]] std::string
format_summary(const schema& config_schema, param_level max_level = param_level::basic);

}  // namespace rocprofsys::parser
