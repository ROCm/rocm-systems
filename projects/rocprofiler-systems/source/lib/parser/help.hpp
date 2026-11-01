// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"
#include "parser/value_provider.hpp"

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
 * @p topic limits the text to one category id or category name. An empty topic
 * includes every category. Compact style omits advanced and expert parameters.
 */
[[nodiscard]] std::string
format_help(const schema& config_schema, help_style style, std::string_view topic,
            const value_provider_registry* providers = nullptr);

}  // namespace rocprofsys::parser
