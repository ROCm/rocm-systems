// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/backend.hpp"
#include "parser/schema.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace rocprofsys::parser
{
enum class help_style : std::uint8_t
{
    /// Basic parameters only.
    compact = 0,
    /// Every parameter, including its type and choices.
    full,
};

/**
 * Render help for @p config_schema.
 *
 * Parameters are grouped by schema_param::category, in the order each word first
 * appears. The heading is that word. @p topic limits the text to one category.
 * An empty category has no heading. Compact style omits advanced and expert
 * parameters. When a row has no choices of its own, a registered backend supplies
 * them. A null @p backends skips that lookup.
 */
[[nodiscard]] std::string
format_help(const schema& config_schema, help_style style, std::string_view topic,
            const backend_registry* backends = nullptr);

}  // namespace rocprofsys::parser
