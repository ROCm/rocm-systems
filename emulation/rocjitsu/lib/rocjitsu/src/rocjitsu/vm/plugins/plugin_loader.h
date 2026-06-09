// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_loader.h
/// @brief Host-side discovery and loading of execution plugins from shared
/// objects.
///
/// Plugins are enabled and configured through a `plugins` object in the
/// rocjitsu config file:
///
/// @code{.json}
///   "plugins": {
///     "race":    {},
///     "logging": { "verbose": true }
///   }
/// @endcode
///
/// Each key names a plugin; the loader opens `librocjitsu_plugin_<key>.so`
/// via the standard dynamic-linker search path, validates its ABI, resolves
/// the supplied configuration against the plugin's schema (filling in
/// defaults), instantiates the plugin, and adds it to the supplied group.

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <string>

namespace rocjitsu {

/// @brief Loads the plugins named in a config file's `plugins` section.
class PluginLoader {
public:
  /// Parse @p config_json (the full config-file contents), find the top-level
  /// `plugins` object, and load each listed plugin into @p group.
  ///
  /// Loaded shared objects are kept open for the lifetime of the process.
  /// Failures (missing library, ABI mismatch, bad config) are reported to
  /// stderr and skip that plugin without aborting the others.
  ///
  /// @returns The number of plugins successfully added to @p group.
  static int load_from_config(const std::string &config_json, ExecutionPluginGroup &group);
};

} // namespace rocjitsu
