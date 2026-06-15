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
///
/// Output sinks and the profiling group are also configured from the same
/// config file (see configure_plugin_group()):
///
/// @code{.json}
///   "profiled": true,
///   "sinks": { "types": ["stderr", "file"], "dir": "/tmp/out" }
/// @endcode

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <memory>
#include <string>

namespace rocjitsu {

/// @brief Loads the plugins named in a config file's `plugins` section.
class PluginLoader {
public:
  /// Parse @p config_json (the full config-file contents), find the top-level
  /// `plugins` object, and load each listed plugin into @p group.
  ///
  /// Loaded shared objects are kept open for the lifetime of the process.
  /// Failures (missing library, ABI mismatch, bad config) are reported to the
  /// plugin log and skip that plugin without aborting the others.
  ///
  /// @returns The number of plugins successfully added to @p group.
  static int load_from_config(const std::string &config_json, ExecutionPluginGroup &group);

  /// Build a fully configured plugin group from @p config_json: selects the
  /// plain or profiled group (`"profiled"` flag), wires output sinks
  /// (`"sinks"` object), and loads the plugins (`"plugins"` object). Shared by
  /// the local (interposer) and daemon launch paths so a given config behaves
  /// identically regardless of how the VM is brought up.
  ///
  /// @returns A non-null group (empty if the config declares no plugins).
  static std::shared_ptr<ExecutionPluginGroup>
  configure_plugin_group(const std::string &config_json);
};

} // namespace rocjitsu
