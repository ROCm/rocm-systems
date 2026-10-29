// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/config_store.hpp"
#include "parser/schema.hpp"

#include <string>
#include <vector>

namespace rocprofsys::parser
{
/// Values read from one source, plus any input that could not be accepted.
struct parse_result
{
    value_map_t              values;
    std::vector<std::string> errors;
    /// Tokens after a "--" separator. Not parsed as options.
    std::vector<std::string> command;
    /// Remainder, including the first unknown token, when parsing stops early.
    std::vector<std::string> unparsed;
};

/// Store parsed values under @p source. Ignores errors, the command, and unparsed tokens.
void
apply_parsed(config_store& store, const parse_result& parsed, config_source source);

}  // namespace rocprofsys::parser
