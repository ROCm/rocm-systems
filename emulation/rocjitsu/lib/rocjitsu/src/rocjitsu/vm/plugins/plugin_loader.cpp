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

#include <array>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace rocjitsu {
namespace {

using plugin_detail::flexbuffer_from_json;
using plugin_detail::is_valid_plugin_name;
using plugin_detail::resolve_config;

/// Library handles are kept for the lifetime of the process: plugin objects
/// (and their vtables/code) live inside these libraries.
std::vector<util::LibraryHandle> &open_handles() {
  static std::vector<util::LibraryHandle> handles;
  return handles;
}

std::string resolve_plugin_path(const std::string &soname, const std::string &plugin_dir) {
  if (!plugin_dir.empty())
    return plugin_dir + "/" + soname;

#if defined(__linux__)
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void *>(&resolve_plugin_path), &info) != 0 && info.dli_fname) {
    std::error_code error;
    const auto module_dir = std::filesystem::canonical(info.dli_fname, error).parent_path();
    if (!error) {
      // Shared-library hosts keep plugins beside librocjitsu. The statically
      // linked CLI uses <build>/tools/rocjitsu/rocjitsu in a build tree and
      // <prefix>/bin/rocjitsu when installed, so probe those layouts too.
      const std::array candidates{
          module_dir / soname,
          module_dir / ".." / "lib" / soname,
          module_dir / ".." / "lib64" / soname,
          module_dir / ".." / ".." / soname,
      };
      for (const auto &candidate : candidates) {
        if (std::filesystem::is_regular_file(candidate, error) && !error)
          return candidate.lexically_normal().string();
        error.clear();
      }
    }
  }
#endif

  return soname;
}

bool load_one(const std::string &name, const flexbuffers::Reference &user_cfg,
              ExecutionPluginGroup &group, const std::string &plugin_dir) {
  if (!is_valid_plugin_name(name)) {
    util::Logger::warn("plugin '", name,
                       "': invalid name (allowed: letters, digits, '_', '-'), skipping");
    return false;
  }

  // `name` is validated above, so it cannot contain a path separator; joining a
  // trusted `plugin_dir` in front of it stays inside that directory.
  std::string soname = "librocjitsu_plugin_" + name + ".so";
  std::string libpath = resolve_plugin_path(soname, plugin_dir);
  util::LibraryHandle handle = util::open_library(libpath.c_str());
  if (!handle) {
    util::Logger::warn("plugin '", name, "': cannot load ", libpath, ": ",
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
  // deallocation stay on the same side of the ABI boundary. Keep `handle` open
  // for the whole lifetime of `owned`: on the rejection path the instance is
  // destroyed via its PluginDeleter (the plugin's destroy_fn), which lives in
  // this library, so the library must still be loaded when that happens.
  bool added = false;
  {
    OwnedPlugin owned(plugin, PluginDeleter{destroy_fn});
    added = group.add(std::move(owned));
  }
  if (!added) {
    util::Logger::warn("plugin '", name, "': already loaded, skipping duplicate");
    util::close_library(handle);
    return false;
  }

  open_handles().push_back(handle);
  util::Logger::plugins("plugin '", name, "' loaded", (meta->version && *meta->version) ? " v" : "",
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
  bool configured = false;
  for (size_t i = 0; i < vec.size(); ++i) {
    std::string token = vec[i].IsString() ? vec[i].AsString().c_str() : "";
    if (token == "stderr") {
      group.add_sink(&StderrSink::instance());
      configured = true;
    } else if (token == "stdout") {
      group.add_sink(&StdoutSink::instance());
      configured = true;
    } else if (token == "file") {
      if (!dir.empty()) {
        group.set_sink_dir(dir);
        configured = true;
      } else
        util::Logger::warn("sink type 'file' requested but no 'dir' set");
    }
  }
  if (!configured)
    group.add_sink(&StderrSink::instance());
}

} // namespace

int PluginLoader::load_from_config(const std::string &config_json, ExecutionPluginGroup &group,
                                   const std::string &plugin_dir) {
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
    if (load_one(name, vals[i], group, plugin_dir))
      ++added;
  }
  return added;
}

std::shared_ptr<ExecutionPluginGroup>
PluginLoader::configure_plugin_group(const std::string &config_json, const std::string &plugin_dir,
                                     const simdojo::SimulationEngine::Config &engine_config) {
  flexbuffers::Builder root_fbb;
  bool parsed = flexbuffer_from_json(config_json, root_fbb);
  auto root = parsed ? flexbuffers::GetRoot(root_fbb.GetBuffer()) : flexbuffers::Reference();

  bool profiled =
      root.IsMap() && root.AsMap()["profiled"].IsBool() && root.AsMap()["profiled"].AsBool();
  if (profiled && engine_config.num_threads > 1) {
    util::Logger::warn("profiled plugin execution requires num_threads=1 (got ",
                       engine_config.num_threads, ")");
    throw std::invalid_argument("profiled plugin execution requires num_threads=1");
  }

  std::shared_ptr<ExecutionPluginGroup> group =
      profiled ? std::make_shared<ProfiledExecutionPluginGroup>()
               : std::make_shared<ExecutionPluginGroup>();

  configure_sinks(root, *group);
  load_from_config(config_json, *group, plugin_dir);
  return group;
}

} // namespace rocjitsu
