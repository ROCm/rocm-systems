// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/plugin_loader.h"

#include "rocjitsu/vm/plugins/plugin_config_resolver.h"
#include "rocjitsu/vm/plugins/plugin_exports.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"

#include "util/dynamic_loader.h"
#include "util/log.h"

#include "flatbuffers/flexbuffers.h"
#include "flatbuffers/idl.h"

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
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

bool map_has_key(const flexbuffers::Reference &root, std::string_view key) {
  if (!root.IsMap())
    return false;
  auto keys = root.AsMap().Keys();
  for (size_t i = 0; i < keys.size(); ++i)
    if (std::string_view(keys[i].AsKey()) == key)
      return true;
  return false;
}

util::FailureOr<bool> parse_require_all_plugins(const flexbuffers::Reference &root,
                                                util::DiagnosticEmitter emit_error) {
  if (!map_has_key(root, "require_all_plugins"))
    return false;

  auto value = root.AsMap()["require_all_plugins"];
  if (!value.IsBool())
    return emit_error.emit() << "top-level 'require_all_plugins' must be a boolean";
  return value.AsBool();
}

util::FailureOr<size_t> configured_plugin_count(const flexbuffers::Reference &root,
                                                bool require_all_plugins,
                                                util::DiagnosticEmitter emit_error) {
  if (!root.IsMap())
    return 0;
  if (!map_has_key(root, "plugins"))
    return 0;

  auto plugins = root.AsMap()["plugins"];
  if (!plugins.IsMap()) {
    if (require_all_plugins)
      return emit_error.emit()
             << "top-level 'plugins' must be an object when 'require_all_plugins' is true";
    return 0;
  }
  return plugins.AsMap().size();
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
    util::Logger::warn("plugin '", name, "': ", soname, " is missing required exports");
    util::close_library(handle);
    return false;
  }

  const PluginMetadata *meta = meta_fn();
  if (!meta) {
    util::Logger::warn("plugin '", name, "': metadata export returned null");
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
  // deallocation stay on the same side of the dynamic-library boundary. Keep `handle` open
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
  util::Logger::plugins("plugin '", name, "' loaded");
  return true;
}

/// Parse owned output sinks from the optional top-level `sinks` object.
/// Defaults to a single stderr sink when absent.
///
/// @code{.json}
///   "sinks": { "types": ["stderr", "file"], "dir": "/tmp/out" }
/// @endcode
PluginSinkConfig parse_sink_config(const flexbuffers::Reference &root) {
  PluginSinkConfig config;
  flexbuffers::Reference sinks = root.IsMap() ? root.AsMap()["sinks"] : flexbuffers::Reference();

  // Default: stderr only.
  if (!sinks.IsMap()) {
    config.emplace<StderrSink>();
    return config;
  }

  auto sinks_map = sinks.AsMap();
  auto types = sinks_map["types"];
  std::string dir = sinks_map["dir"].IsString() ? sinks_map["dir"].AsString().c_str() : "";

  if (!types.IsVector()) {
    config.emplace<StderrSink>();
    return config;
  }

  auto vec = types.AsVector();
  bool configured = false;
  for (size_t i = 0; i < vec.size(); ++i) {
    std::string token = vec[i].IsString() ? vec[i].AsString().c_str() : "";
    if (token == "stderr") {
      config.emplace<StderrSink>();
      configured = true;
    } else if (token == "stdout") {
      config.emplace<StdoutSink>();
      configured = true;
    } else if (token == "file") {
      if (!dir.empty()) {
        config.set_file_directory(dir);
        configured = true;
      } else
        util::Logger::warn("sink type 'file' requested but no 'dir' set");
    }
  }
  if (!configured)
    config.emplace<StderrSink>();
  return config;
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

util::FailureOr<std::shared_ptr<ExecutionPluginGroup>>
PluginLoader::configure_plugin_group(const std::string &config_json, const std::string &plugin_dir,
                                     util::DiagnosticEmitter emit_error) {
  flexbuffers::Builder root_fbb;
  flatbuffers::Parser parser;
  bool parsed =
      !config_json.empty() && parser.ParseFlexBuffer(config_json.c_str(), nullptr, &root_fbb);
  // ParseFlexBuffer accepts a single value without checking the following token.
  // Match the VM parser's whitespace and comment rules after that value.
  if (parsed) {
    size_t offset = static_cast<size_t>(parser.BytesConsumed());
    while ((offset = config_json.find_first_not_of(" \t\r\n", offset)) != std::string::npos) {
      if (config_json.compare(offset, 2, "//") == 0) {
        offset = config_json.find_first_of("\r\n", offset + 2);
        if (offset == std::string::npos)
          break;
      } else if (config_json.compare(offset, 2, "/*") == 0) {
        const size_t end = config_json.find("*/", offset + 2);
        if (end == std::string::npos)
          return emit_error.emit() << "invalid plugin configuration JSON: unterminated comment at "
                                      "offset "
                                   << offset;
        offset = end + 2;
      } else {
        return emit_error.emit()
               << "invalid plugin configuration JSON: unexpected content at offset " << offset;
      }
    }
  }
  // Empty input intentionally configures an empty group.
  if (!parsed && !config_json.empty())
    return emit_error.emit() << "invalid plugin configuration JSON: " << parser.error_;
  auto root = parsed ? flexbuffers::GetRoot(root_fbb.GetBuffer()) : flexbuffers::Reference();

  if (parsed && !root.IsMap())
    return emit_error.emit() << "plugin configuration must be an object";

  if (map_has_key(root, "profiled"))
    util::Logger::warn("hook profiling was removed; ignoring top-level 'profiled' config");

  const util::FailureOr<bool> require_all = parse_require_all_plugins(root, emit_error);
  if (require_all.failed())
    return util::Result::failure();
  const bool require_all_plugins = require_all.value();
  const util::FailureOr<size_t> requested =
      configured_plugin_count(root, require_all_plugins, emit_error);
  if (requested.failed())
    return util::Result::failure();
  const size_t requested_plugins = requested.value();
  auto group = std::make_shared<ExecutionPluginGroup>(parse_sink_config(root));
  group->defer_file_sinks();
  const int loaded_plugins = load_from_config(config_json, *group, plugin_dir);
  if (require_all_plugins && static_cast<size_t>(loaded_plugins) != requested_plugins) {
    return emit_error.emit() << "required plugin loading failed: loaded " << loaded_plugins
                             << " of " << requested_plugins << " configured plugins";
  }
  return group;
}

} // namespace rocjitsu
