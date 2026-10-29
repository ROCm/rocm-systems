// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/parse_result.hpp"
#include "parser/schema.hpp"

namespace rocprofsys::parser
{
/// Reads each parameter's default_value. Parameters without one are skipped.
class default_parser
{
public:
    [[nodiscard]] parse_result parse(const schema& config_schema) const;
};

}  // namespace rocprofsys::parser
