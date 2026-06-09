// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_abi.h
/// @brief Stable C ABI for runtime-loadable execution plugins.
///
/// Execution plugins are shipped as shared objects named
/// `librocjitsu_plugin_<name>.so` and discovered through the standard
/// dynamic-linker search path (RUNPATH, LD_LIBRARY_PATH, ld.so.cache).
///
/// Each plugin shared object must export exactly two symbols:
///
///   1. `rocjitsu_plugin_metadata` — returns a pointer to a static
///      ::rocjitsu::PluginMetadata describing the plugin (ABI version,
///      name, contact, version, and a JSON config schema).
///
///   2. `rocjitsu_plugin_create` — returns a `std::unique_ptr<ExecutionPlugin>`
///      constructed from a JSON configuration string. The host resolves the
///      plugin's config object (applying schema defaults) and passes it as a
///      JSON string; the plugin parses it however it likes.
///
/// Use the ROCJITSU_DEFINE_PLUGIN() helper to emit both exports.

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"

#include <memory>

namespace rocjitsu {

/// @brief ABI version. Bump on any incompatible change to the plugin
/// interface (this header, ExecutionPlugin's hook signatures, etc.).
inline constexpr int kPluginAbiVersion = 1;

/// @brief Static description of a plugin, returned by the metadata export.
///
/// All string members point to storage with static lifetime owned by the
/// plugin shared object; they remain valid for as long as the library is
/// loaded.
struct PluginMetadata {
  /// Must equal ::rocjitsu::kPluginAbiVersion. The loader rejects mismatches.
  int abi;
  /// Plugin name. Must match the `<name>` in `librocjitsu_plugin_<name>.so`
  /// and the key used to configure the plugin in the config file.
  const char *name;
  /// Maintainer contact (email, team, etc.). May be empty.
  const char *contact;
  /// Human-readable plugin version string. May be empty.
  const char *version;
  /// JSON object describing accepted config arguments. Each entry maps an
  /// argument name to `{ "type": "string|number|boolean",
  /// "description": "...", "default": <value> }`. An argument without a
  /// `default` is required. May be "{}" or null when the plugin takes no
  /// configuration.
  const char *config_schema;
};

/// @brief Signature of the exported metadata accessor.
using PluginMetadataFn = const PluginMetadata *(*)();

/// @brief Signature of the exported plugin factory.
/// @param config_json The plugin's resolved configuration object as a JSON
///        string (schema defaults already merged in by the host). Never null;
///        "{}" when there is no configuration.
using PluginCreateFn = std::unique_ptr<ExecutionPlugin> (*)(const char *config_json);

/// @brief Exported symbol name of the metadata accessor.
inline constexpr const char *kPluginMetadataSymbol = "rocjitsu_plugin_metadata";
/// @brief Exported symbol name of the plugin factory.
inline constexpr const char *kPluginCreateSymbol = "rocjitsu_plugin_create";

} // namespace rocjitsu

/// @brief Ensure ABI exports stay visible even under -fvisibility=hidden.
#if defined(_WIN32)
#define ROCJITSU_PLUGIN_EXPORT __declspec(dllexport)
#else
#define ROCJITSU_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

/// @brief Emit the two required plugin ABI exports.
///
/// @param PluginClass Concrete ::rocjitsu::ExecutionPlugin subclass. It must
///        be constructible from a single `const char *config_json` argument.
/// @param NAME        Plugin name string literal (matches the `.so` suffix).
/// @param CONTACT     Maintainer contact string literal.
/// @param VERSION     Version string literal.
/// @param CONFIG_SCHEMA JSON schema string literal ("{}" if none).
///
/// Example:
/// @code
///   ROCJITSU_DEFINE_PLUGIN(MyPlugin, "myplugin", "team@amd.com", "1.0", "{}")
/// @endcode
#define ROCJITSU_DEFINE_PLUGIN(PluginClass, NAME, CONTACT, VERSION, CONFIG_SCHEMA)                  \
  extern "C" ROCJITSU_PLUGIN_EXPORT const ::rocjitsu::PluginMetadata *rocjitsu_plugin_metadata() { \
    static const ::rocjitsu::PluginMetadata kMetadata{                                              \
        ::rocjitsu::kPluginAbiVersion, NAME, CONTACT, VERSION, CONFIG_SCHEMA};                      \
    return &kMetadata;                                                                              \
  }                                                                                                 \
  extern "C" ROCJITSU_PLUGIN_EXPORT std::unique_ptr<::rocjitsu::ExecutionPlugin>                    \
  rocjitsu_plugin_create(const char *config_json) {                                                 \
    return std::make_unique<PluginClass>(config_json);                                              \
  }
