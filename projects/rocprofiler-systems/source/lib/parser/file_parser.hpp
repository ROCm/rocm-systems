// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/parse_result.hpp"
#include "parser/schema.hpp"

#include <nlohmann/json_fwd.hpp>

#include <string>

namespace rocprofsys::parser
{
/**
 * Reads a JSON object of parameter values.
 *
 * A key is a parameter id or that parameter's environment variable name.
 * Keys that start with "_" (for example "_comment") and JSON null are ignored.
 * Scalars keep their JSON spelling, and arrays are joined with the parameter's
 * list separator. Values the schema rejects are reported and not stored.
 */
class file_parser
{
public:
    [[nodiscard]] parse_result parse(const schema&         config_schema,
                                     const nlohmann::json& document) const;

    [[nodiscard]] parse_result parse_file(const schema&      config_schema,
                                          const std::string& filepath) const;
};

}  // namespace rocprofsys::parser
