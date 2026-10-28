// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"

#include <nlohmann/json_fwd.hpp>

#include <string>

namespace rocprofsys::parser
{
/**
 * Build a schema from a parameter document.
 *
 * The document requires "schema_version" and a "params" array, and may contain
 * "categories" and "constraints" arrays. Unrecognized fields (e.g. "_comment",
 * "tool", "presets") are ignored.
 *
 * @throws std::runtime_error if a field is missing or has the wrong type, a
 *         parameter is internally inconsistent (e.g. a default outside its choices),
 *         ids/environment variables/argument names are reused, or a reference names an
 *         unknown parameter or category.
 */
[[nodiscard]] schema
load_schema(const nlohmann::json& document);

/**
 * Read and build a schema from a JSON file.
 * @throws std::runtime_error if the file cannot be opened or parsed, or for any of
 *         the reasons listed for load_schema().
 */
[[nodiscard]] schema
load_schema_file(const std::string& filepath);

}  // namespace rocprofsys::parser
