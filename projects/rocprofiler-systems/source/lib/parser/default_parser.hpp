// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/parse_result.hpp"
#include "parser/schema.hpp"

namespace rocprofsys::parser
{
/// Reads each parameter's default_value. Parameters without one are skipped.
/// Catalog defects belong to validate_schema(); this parser does not report them.
class default_parser
{
public:
    [[nodiscard]] parse_result parse(const schema& config_schema) const;
};

}  // namespace rocprofsys::parser
