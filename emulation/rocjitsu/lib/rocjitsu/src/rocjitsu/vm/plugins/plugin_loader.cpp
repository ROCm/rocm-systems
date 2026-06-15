// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/plugin_loader.h"

#include "rocjitsu/vm/plugins/plugin_abi.h"
#include "rocjitsu/vm/plugins/plugin_config_resolver.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "rocjitsu/vm/plugins/profiled_execution_plugin_group.h"

#include "util/dynamic_loader.h"
#include "util/log.h"

#include "flatbuffers/flexbuffers.h"

#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

using plugin_detail::flexbuffer_from_json;
using plugin_detail::resolve_config;

/// Library handles are kept for the lifetime of the process: plugin objects
/// (and their vtables/code) live inside these libraries.
std::vector<util::LibraryHandle> &open_handles() {
  static std::vector<util::LibraryHandle> handles;
  return handles;
}

/// Plugin names are interpolated into `librocjitsu_plugin_<name>.so`. Restrict
/// them to a safe character set so a config key can never turn into a path
/// (e.g. `../evil`), which the dynamic linker would treat as a pathname and
/// load directly, bypassing the normal library-name lookup.
bool is_valid_plugin_name(const std::string &name) {
  if (name.empty())
    return false;
  for (char c : name) {
    bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
              c == '_' || c == '-';
    if (!ok)
      return false;
  }
  return true;
}

bool load_one(const std::string &name, const flexbuffers::Reference &user_cfg,
              ExecutionPluginGroup &group) {
  if (!is_valid_plugin_name(name)) {
    util::Logger::warn("plugin '", name,
                       "': invalid name (allowed: letters, digits, '_', '-'), skipping");
    return false;
  }

  std::string soname = "librocjitsu_plugin_" + name + ".so";
  util::LibraryHandle handle = util::open_library(soname.c_str());
  if (!handle) {
    util::Logger::warn("plugin '", name, "': cannot load ", soname, ": ",
                       util::last_library_error());
    return false;
  }

  auto meta_fn = util::lookup_symbol<PluginMetadataFn>(handle, kPluginMetadataSymbol);
  auto create_fn = util::lookup_symbol<PluginCreateFn>(handle, kPluginCreateSymbol);
  auto destroy_fn = util::lookup_symbol<PluginDestroyFn>(handle, kPluginDestroySymbol);
  if (!meta_fn || !create_fn || !destroy_fn) {
    util::Logger::warn("plugin '", name, "': ", soname, " is missing required ABI exports");
    util::close_library(handle);
    return false;
  }

  const PluginMetadata *meta = meta_fn();
  if (!meta || meta->abi != kPluginAbiVersion) {
    util::Logger::warn("plugin '", name, "': ABI version mismatch (got ", meta ? meta->abi : -1,
                       ", expected ", kPluginAbiVersion, ")");
    util::close_library(handle);
    return false;
  }
  if (meta->name && name != meta->name)
    util::Logger::warn("plugin '", name, "': metadata name '", meta->name,
                       "' differs from library name");

  std::string resolved;
  if (!resolve_config(name, meta->config_schema, user_cfg, resolved)) {
    util::close_library(handle);
    return false;
  }

  PluginHandle raw = create_fn(resolved.c_str());
  auto *plugin = static_cast<ExecutionPlugin *>(raw);
  if (!plugin) {
    util::Logger::warn("plugin '", name, "': create returned null");
    util::close_library(handle);
    return false;
  }

  // Own the instance through the plugin's own destroy export so allocation and
  // deallocation stay on the same side of the ABI boundary.
  OwnedPlugin owned(plugin, PluginDeleter{destroy_fn});
  if (!group.add(std::move(owned))) {
    util::Logger::warn("plugin '", name, "': already loaded, skipping duplicate");
    util::close_library(handle);
    return false;
  }

  open_handles().push_back(handle);
  util::Logger::plugins("plugin '", name, "' loaded",
                        (meta->version && *meta->version) ? " v" : "",
                        (meta->version && *meta->version) ? meta->version : "");
  return true;
}

/// Configure output sinks on @p group from the optional top-level `sinks`
/// object. Defaults to a single stderr sink when absent.
///
/// @code{.json}
///   "sinks": { "types": ["stderr", "file"], "dir": "/tmp/out" }
/// @endcode
void configure_sinks(const flexbuffers::Reference &root, ExecutionPluginGroup &group) {
  flexbuffers::Reference sinks = root.IsMap() ? root.AsMap()["sinks"] : flexbuffers::Reference();

  // Default: stderr only.
  if (!sinks.IsMap()) {
    group.add_sink(&StderrSink::instance());
    return;
  }

  auto sinks_map = sinks.AsMap();
  auto types = sinks_map["types"];
  std::string dir = sinks_map["dir"].IsString() ? sinks_map["dir"].AsString().c_str() : "";

  if (!types.IsVector()) {
    group.add_sink(&StderrSink::instance());
    return;
  }

  auto vec = types.AsVector();
  for (size_t i = 0; i < vec.size(); ++i) {
    std::string token = vec[i].IsString() ? vec[i].AsString().c_str() : "";
    if (token == "stderr")
      group.add_sink(&StderrSink::instance());
    else if (token == "stdout")
      group.add_sink(&StdoutSink::instance());
    else if (token == "file" && !dir.empty())
      group.set_sink_dir(dir);
  }
}

} // namespace

int PluginLoader::load_from_config(const std::string &config_json, ExecutionPluginGroup &group) {
  flexbuffers::Builder root_fbb;
  if (!flexbuffer_from_json(config_json, root_fbb))
    return 0;

  auto root = flexbuffers::GetRoot(root_fbb.GetBuffer());
  if (!root.IsMap())
    return 0;

  auto plugins = root.AsMap()["plugins"];
  if (!plugins.IsMap())
    return 0;

  auto pmap = plugins.AsMap();
  auto keys = pmap.Keys();
  auto vals = pmap.Values();
  int added = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    std::string name = keys[i].AsKey();
    if (load_one(name, vals[i], group))
      ++added;
  }
  return added;
}

std::shared_ptr<ExecutionPluginGroup>
PluginLoader::configure_plugin_group(const std::string &config_json) {
  flexbuffers::Builder root_fbb;
  bool parsed = flexbuffer_from_json(config_json, root_fbb);
  auto root = parsed ? flexbuffers::GetRoot(root_fbb.GetBuffer()) : flexbuffers::Reference();

  bool profiled =
      root.IsMap() && root.AsMap()["profiled"].IsBool() && root.AsMap()["profiled"].AsBool();

  std::shared_ptr<ExecutionPluginGroup> group =
      profiled ? std::make_shared<ProfiledExecutionPluginGroup>()
               : std::make_shared<ExecutionPluginGroup>();

  configure_sinks(root, *group);
  load_from_config(config_json, *group);
  return group;
}

} // namespace rocjitsu
