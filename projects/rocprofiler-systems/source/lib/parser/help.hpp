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
 * appears. The heading is that word. @p topic limits the text to one category.
 * An empty category has no heading. @p max_level basic omits advanced parameters.
 * advanced includes them. Every shown row prints its description, environment
 * name, default, and choices. When a row has no choices of its own, a registered
 * backend supplies them. A null @p backends skips that lookup.
 */
[[nodiscard]] std::string
format_help(const schema& config_schema, param_level max_level = param_level::basic,
            std::string_view topic = {}, const backend_registry* backends = nullptr);

}  // namespace rocprofsys::parser
