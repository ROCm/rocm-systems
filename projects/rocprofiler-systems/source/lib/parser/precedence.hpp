// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/config_store.hpp"
#include "parser/value_text.hpp"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace rocprofsys::parser
{
/// One source's accepted values. config_source order is lowest to highest precedence.
struct sourced_values
{
    config_source  source = config_source::default_value;
    typed_values_t values;
};

/// The value that won the merge for one parameter, and the source that supplied it.
struct resolved_value
{
    config_value_t value;
    config_source  source = config_source::default_value;
};

/// Parameter id to the value that won the merge. Iterated in id order.
using resolved_config = std::map<std::string, resolved_value, std::less<>>;

/**
 * Merge sources so a higher config_source replaces a lower one for the same parameter.
 *
 * Order is default_value, file, environment, preset, then command_line. A preset
 * overrides the environment and the configuration file, matching rocprof-sys today.
 * Sources of equal rank keep their input order, and the later one wins.
 */
[[nodiscard]] resolved_config
resolve_precedence(std::vector<sourced_values> sources);

/**
 * Store each resolved value under the source that supplied it.
 *
 * Every value is checked before the first write. A rejection leaves the store
 * unchanged.
 *
 * @throws std::runtime_error if the store has no schema, or if any value is
 * rejected; the message lists every rejection, separated by "; ".
 * Any exception a change callback throws also propagates.
 */
void
apply_resolved(config_store& store, const resolved_config& resolved);

}  // namespace rocprofsys::parser
