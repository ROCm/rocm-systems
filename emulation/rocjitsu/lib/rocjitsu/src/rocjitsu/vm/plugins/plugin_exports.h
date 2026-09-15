// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_exports.h
/// @brief Loader boundary for repository-owned execution plugins.
///
/// Plugins are built in-tree and shipped with rocJitsu. This boundary does not
/// provide compatibility or versioning for independently built plugins; the
/// host and plugins must always be rebuilt and distributed together.
///
/// Execution plugins are shipped as shared objects named
/// `librocjitsu_plugin_<name>.so` and discovered through the standard
/// dynamic-linker search path (RUNPATH, LD_LIBRARY_PATH, ld.so.cache).
///
/// Each plugin shared object must export the `extern "C"` loader functions
/// described below. They form a C-shaped boundary: no C++ library types
/// (`std::unique_ptr`, `std::string`, ...) cross the boundary, and the plugin
/// owns allocation and destruction of its instance via its own allocator.
///
///   - `rocjitsu_plugin_metadata` — returns a pointer to a static
///     ::rocjitsu::PluginMetadata describing the plugin (name and a JSON config
///     schema).
///
///   - `rocjitsu_plugin_create` — constructs a plugin instance from a JSON
///     configuration string and returns it as an opaque
///     ::rocjitsu::PluginHandle. The host resolves the plugin's config object
///     (applying schema defaults) and passes it as a JSON string; the plugin
///     parses it however it likes.
///
///   - `rocjitsu_plugin_destroy` — destroys an instance previously returned by
///     `rocjitsu_plugin_create`, using the plugin's own allocator.
///
/// The opaque handle is an ::rocjitsu::ExecutionPlugin subclass instance, and
/// the host calls its virtual hooks directly. Use ROCJITSU_DEFINE_PLUGIN() to
/// emit the required loader exports for an in-tree plugin.

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"

#include "util/log.h"

#include <exception>

namespace rocjitsu {

/// @brief Opaque handle to a plugin instance returned by the create export.
///
/// In practice it points to an ::rocjitsu::ExecutionPlugin subclass owned by
/// the plugin library. The host treats it as opaque except to recover the
/// ExecutionPlugin base (see PluginLoader); only the plugin's own
/// `rocjitsu_plugin_destroy` may free it.
using PluginHandle = void *;

/// @brief Static description of a plugin, returned by the metadata export.
///
/// All string members point to storage with static lifetime owned by the
/// plugin shared object; they remain valid for as long as the library is
/// loaded.
struct PluginMetadata {
  /// Plugin name. Must match the `<name>` in `librocjitsu_plugin_<name>.so`
  /// and the key used to configure the plugin in the config file.
  const char *name;
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
/// @returns An opaque handle to the new instance, or nullptr on failure.
using PluginCreateFn = PluginHandle (*)(const char *config_json);

/// @brief Signature of the exported plugin destructor.
/// @param handle A handle previously returned by the create export. nullptr is
///        ignored.
using PluginDestroyFn = void (*)(PluginHandle handle);

/// @brief Exported symbol name of the metadata accessor.
inline constexpr const char *kPluginMetadataSymbol = "rocjitsu_plugin_metadata";
/// @brief Exported symbol name of the plugin factory.
inline constexpr const char *kPluginCreateSymbol = "rocjitsu_plugin_create";
/// @brief Exported symbol name of the plugin destructor.
inline constexpr const char *kPluginDestroySymbol = "rocjitsu_plugin_destroy";

} // namespace rocjitsu

/// @brief Ensure the plugin's exported symbols stay visible even under
/// -fvisibility=hidden.
#define ROCJITSU_PLUGIN_EXPORT RJ_API_EXPORT

/// @brief Emit the required plugin loader exports.
///
/// @param PluginClass Concrete ::rocjitsu::ExecutionPlugin subclass. It must
///        be constructible from a single `const char *config_json` argument.
/// @param NAME        Plugin name string literal (matches the `.so` suffix).
/// @param CONFIG_SCHEMA JSON schema string literal ("{}" if none).
///
/// Example:
/// @code
///   ROCJITSU_DEFINE_PLUGIN(MyPlugin, "myplugin", "{}")
/// @endcode
#define ROCJITSU_DEFINE_PLUGIN(PluginClass, NAME, CONFIG_SCHEMA)                                   \
  extern "C" ROCJITSU_PLUGIN_EXPORT const ::rocjitsu::PluginMetadata *rocjitsu_plugin_metadata() { \
    static const ::rocjitsu::PluginMetadata kMetadata{NAME, CONFIG_SCHEMA};                        \
    return &kMetadata;                                                                             \
  }                                                                                                \
  extern "C" ROCJITSU_PLUGIN_EXPORT ::rocjitsu::PluginHandle rocjitsu_plugin_create(               \
      const char *config_json) {                                                                   \
    try {                                                                                          \
      return static_cast<::rocjitsu::ExecutionPlugin *>(new PluginClass(config_json));             \
    } catch (const std::exception &e) {                                                            \
      ::util::Logger::warn("plugin '", NAME, "': create failed: ", e.what());                      \
      return nullptr;                                                                              \
    } catch (...) {                                                                                \
      ::util::Logger::warn("plugin '", NAME, "': create failed with unknown exception");           \
      return nullptr;                                                                              \
    }                                                                                              \
  }                                                                                                \
  extern "C" ROCJITSU_PLUGIN_EXPORT void rocjitsu_plugin_destroy(                                  \
      ::rocjitsu::PluginHandle handle) {                                                           \
    delete static_cast<::rocjitsu::ExecutionPlugin *>(handle);                                     \
  }
