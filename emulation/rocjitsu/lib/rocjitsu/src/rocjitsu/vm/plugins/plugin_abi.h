// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_abi.h
/// @brief In-tree toolchain contract for runtime-loadable execution plugins.
///
/// This is a toolchain contract, not a stable ABI: it only supports plugins
/// built in-tree, from the same source tree and toolchain as the host (see
/// "Toolchain contract" below). It does not offer the compatibility
/// guarantees a true stable ABI would (e.g. across compilers, standard
/// library versions, or releases).
///
/// Execution plugins are shipped as shared objects named
/// `librocjitsu_plugin_<name>.so` and discovered through the standard
/// dynamic-linker search path (RUNPATH, LD_LIBRARY_PATH, ld.so.cache).
///
/// Each plugin shared object must export exactly three `extern "C"` symbols.
/// They form a C-shaped boundary: no C++ library types (`std::unique_ptr`,
/// `std::string`, ...) cross the boundary, and the plugin owns allocation and
/// destruction of its instance via its own allocator.
///
///   1. `rocjitsu_plugin_metadata` — returns a pointer to a static
///      ::rocjitsu::PluginMetadata describing the plugin (contract version,
///      name, contact, version, and a JSON config schema).
///
///   2. `rocjitsu_plugin_create` — constructs a plugin instance from a JSON
///      configuration string and returns it as an opaque
///      ::rocjitsu::PluginHandle. The host resolves the plugin's config object
///      (applying schema defaults) and passes it as a JSON string; the plugin
///      parses it however it likes.
///
///   3. `rocjitsu_plugin_destroy` — destroys an instance previously returned by
///      `rocjitsu_plugin_create`, using the plugin's own allocator.
///
/// ## Toolchain contract
///
/// The opaque handle is an ::rocjitsu::ExecutionPlugin subclass instance; the
/// host calls its virtual hooks directly. Plugins must therefore be built
/// in-tree with the same toolchain and standard library as the host, against a
/// matching ::rocjitsu::ExecutionPlugin layout. The contract version below
/// only guards against mismatched in-tree builds (e.g. a stale plugin built
/// against an older header); it is not a stable ABI and provides no
/// cross-toolchain or cross-release compatibility guarantee.
///
/// Use the ROCJITSU_DEFINE_PLUGIN() helper to emit all three exports.

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"

#include "util/log.h"

#include <exception>

namespace rocjitsu {

/// @brief Toolchain-contract version. Bump on any incompatible change to the
/// plugin interface (this header, ExecutionPlugin's hook signatures, etc.).
/// This is not a stable ABI version: it only detects mismatched in-tree
/// builds, not compatibility across toolchains or releases (see the
/// "Toolchain contract" note above).
inline constexpr int kPluginAbiVersion = 2;

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
  /// Toolchain-contract version; must equal ::rocjitsu::kPluginAbiVersion.
  /// The loader rejects mismatches. Not a stable ABI version — see the
  /// "Toolchain contract" note above.
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

/// @brief Emit the three required plugin exports.
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
#define ROCJITSU_DEFINE_PLUGIN(PluginClass, NAME, CONTACT, VERSION, CONFIG_SCHEMA)                 \
  extern "C" ROCJITSU_PLUGIN_EXPORT const ::rocjitsu::PluginMetadata *rocjitsu_plugin_metadata() { \
    static const ::rocjitsu::PluginMetadata kMetadata{::rocjitsu::kPluginAbiVersion, NAME,         \
                                                      CONTACT, VERSION, CONFIG_SCHEMA};            \
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
