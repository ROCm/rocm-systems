// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

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
    /// The command. Tokens after "--", or from the first token that is not an option.
    std::vector<std::string> command;
    /// Remainder, including the first unknown token, when parsing stops early.
    std::vector<std::string> unparsed;
};

}  // namespace rocprofsys::parser
