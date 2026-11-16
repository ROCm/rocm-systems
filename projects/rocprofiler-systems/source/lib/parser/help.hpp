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
 * caller writes its own title. @p topic limits the text to one category. An empty
 * category has no heading. @p max_level basic omits advanced parameters. advanced
 * includes them. A row's usage line lists arg and then each of arg_aliases, as in
 * "--output-format, -f <value>". Every shown row prints its description,
 * environment name, default, and choices: its own, then the names its registered
 * backend lists. A null @p backends skips that lookup.
 */
[[nodiscard]] std::string
format_help(const schema& config_schema, param_level max_level = param_level::basic,
            std::string_view topic = {}, const backend_registry* backends = nullptr);

}  // namespace rocprofsys::parser
