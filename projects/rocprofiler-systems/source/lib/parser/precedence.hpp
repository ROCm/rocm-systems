// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/config_store.hpp"
#include "parser/schema.hpp"

#include <map>
#include <string>
#include <vector>

namespace rocprofsys::parser
{
/// One source's accepted values. config_source order is lowest to highest precedence.
struct sourced_values
{
    config_source source = config_source::default_value;
    value_map_t   values;
};

struct resolved_value
{
    std::string   value;
    config_source source = config_source::default_value;
};

using resolved_config = std::map<std::string, resolved_value, std::less<>>;

/**
 * Merge sources so a higher config_source replaces a lower one for the same parameter.
 *
 * Order is default_value, preset, file, environment, then command_line. Sources of
 * equal rank keep their input order, and the later one wins.
 */
[[nodiscard]] resolved_config
resolve_precedence(std::vector<sourced_values> sources);

/// Store each resolved value under the source that supplied it.
void
apply_resolved(config_store& store, const resolved_config& resolved);

}  // namespace rocprofsys::parser
