// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_config_resolver.h
/// @brief Resolves a plugin's user config against its declared schema.
///
/// The resolver merges the user-supplied config object with the plugin's
/// `config_schema`, applying defaults, validating types, enforcing required
/// arguments, and passing through unknown keys. It is kept separate from the
/// loader so the contract can be unit-tested without dlopen()ing a plugin.

#pragma once

#include "flatbuffers/flexbuffers.h"

#include <string>

namespace rocjitsu::plugin_detail {

/// @brief Parse arbitrary JSON into a FlexBuffer. Returns false on parse error.
bool flexbuffer_from_json(const std::string &json, flexbuffers::Builder &out);

/// @brief Check whether @p v matches a schema type name
/// (`string`, `number`, `boolean`/`bool`). Unknown type names never match.
bool type_matches(const std::string &type, const flexbuffers::Reference &v);

/// @brief Merge @p user_cfg with @p schema_json, applying defaults and
/// validating types/required args. Produces a JSON object string in @p out.
///
/// A null/empty/`{}` schema passes the user config object through unchanged.
/// An unparseable or non-object schema is ignored (config passed through).
///
/// @returns false (and reports to the plugin log) only on a hard validation
///          failure: a provided arg with the wrong type, or a missing required
///          arg.
bool resolve_config(const std::string &plugin_name, const char *schema_json,
                    const flexbuffers::Reference &user_cfg, std::string &out);

/// @brief Test-friendly wrapper: resolve a user-config JSON string against a
/// schema JSON string. An empty or unparseable @p user_cfg_json is treated as
/// an empty config object.
///
/// @returns false on the same hard validation failures as resolve_config(), or
///          on an unparseable @p user_cfg_json.
bool resolve_config_json(const std::string &plugin_name, const char *schema_json,
                         const std::string &user_cfg_json, std::string &out);

} // namespace rocjitsu::plugin_detail
