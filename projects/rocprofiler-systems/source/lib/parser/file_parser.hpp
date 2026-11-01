// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/parse_result.hpp"
#include "parser/precedence.hpp"
#include "parser/schema.hpp"

#include <nlohmann/json_fwd.hpp>

#include <string>

namespace rocprofsys::parser
{
/**
 * Reads a JSON object of parameter values grouped by category id.
 *
 * Each category value is an object whose keys are parameter ids. Environment
 * variable names are not keys. Keys that start with "_" and JSON null are
 * ignored. An unknown category, a category that is not an object, and a
 * parameter in the wrong category are errors. Scalars keep their JSON
 * spelling. A JSON array is accepted only for an enumeration_list and is
 * joined with that parameter's list separator. Choice aliases are stored as
 * the choice value. Values the schema rejects are reported and not stored.
 * catalog_document() writes this shape. It omits default_value sources and
 * parameters whose category is empty.
 */
class file_parser
{
public:
    [[nodiscard]] parse_result parse(const schema&         config_schema,
                                     const nlohmann::json& document) const;

    [[nodiscard]] parse_result parse_file(const schema&      config_schema,
                                          const std::string& filepath) const;
};

/// Category document for @p resolved. Defaults and uncategorized parameters are omitted.
[[nodiscard]] nlohmann::json
catalog_document(const schema& config_schema, const resolved_config& resolved);

}  // namespace rocprofsys::parser
