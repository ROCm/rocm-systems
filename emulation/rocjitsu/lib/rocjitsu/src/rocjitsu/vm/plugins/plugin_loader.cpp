// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/plugin_loader.h"

#include "rocjitsu/vm/plugins/plugin_abi.h"

#include "util/dynamic_loader.h"

#include "flatbuffers/flexbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#include <cstdio>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

/// dlopen handles are kept for the lifetime of the process: plugin objects
/// (and their vtables/code) live inside these libraries.
std::vector<void *> &open_handles() {
  static std::vector<void *> handles;
  return handles;
}

/// Parse arbitrary JSON into a FlexBuffer. Returns false on parse error.
bool flexbuffer_from_json(const std::string &json, flexbuffers::Builder &out) {
  flatbuffers::Parser parser;
  return parser.ParseFlexBuffer(json.c_str(), nullptr, &out);
}

bool type_matches(const std::string &type, const flexbuffers::Reference &v) {
  if (type == "string")
    return v.IsString();
  if (type == "number")
    return v.IsInt() || v.IsUInt() || v.IsFloat();
  if (type == "boolean" || type == "bool")
    return v.IsBool();
  return false; // Unknown schema type.
}

void append_key(std::string &out, const std::string &key) {
  flatbuffers::EscapeString(key.c_str(), key.size(), &out, /*allow_non_utf8=*/true,
                            /*natural_utf8=*/false);
  out += ": ";
}

/// Append a JSON-encoded value. strings_quoted + keys_quoted produce a value
/// that is itself valid JSON.
void append_value(std::string &out, const flexbuffers::Reference &v) {
  v.ToString(/*strings_quoted=*/true, /*keys_quoted=*/true, out);
}

std::string trimmed(const std::string &s) {
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos)
    return "";
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

/// Merge the user-provided config object with the plugin's schema, applying
/// defaults and validating types/required args. Produces a JSON object string
/// in @p out. Returns false (and reports) on validation failure.
bool resolve_config(const std::string &plugin_name, const char *schema_json,
                    const flexbuffers::Reference &user_cfg, std::string &out) {
  std::string schema_str = trimmed(schema_json ? schema_json : "");

  // No schema: pass the user config object through unchanged.
  if (schema_str.empty() || schema_str == "{}" || schema_str == "null") {
    if (user_cfg.IsMap())
      user_cfg.ToString(true, true, out);
    else
      out = "{}";
    return true;
  }

  flexbuffers::Builder schema_fbb;
  if (!flexbuffer_from_json(schema_str, schema_fbb)) {
    std::fprintf(stderr, "[rocjitsu] plugin '%s': invalid config_schema, ignoring\n",
                 plugin_name.c_str());
    if (user_cfg.IsMap())
      user_cfg.ToString(true, true, out);
    else
      out = "{}";
    return true;
  }

  auto schema_root = flexbuffers::GetRoot(schema_fbb.GetBuffer());
  if (!schema_root.IsMap()) {
    if (user_cfg.IsMap())
      user_cfg.ToString(true, true, out);
    else
      out = "{}";
    return true;
  }

  auto schema_map = schema_root.AsMap();
  auto schema_keys = schema_map.Keys();
  auto schema_vals = schema_map.Values();
  flexbuffers::Map user_map = user_cfg.IsMap() ? user_cfg.AsMap() : flexbuffers::Map::EmptyMap();

  out = "{";
  bool first = true;
  auto emit = [&](const std::string &key, const flexbuffers::Reference &val) {
    if (!first)
      out += ", ";
    first = false;
    append_key(out, key);
    append_value(out, val);
  };

  for (size_t i = 0; i < schema_keys.size(); ++i) {
    std::string arg = schema_keys[i].AsKey();
    auto spec = schema_vals[i];
    std::string type = spec.IsMap() ? std::string(spec.AsMap()["type"].AsString().c_str()) : "";

    auto provided = user_map[arg.c_str()];
    if (!provided.IsNull()) {
      if (!type.empty() && !type_matches(type, provided)) {
        std::fprintf(stderr,
                     "[rocjitsu] plugin '%s': config arg '%s' has wrong type (expected %s)\n",
                     plugin_name.c_str(), arg.c_str(), type.c_str());
        return false;
      }
      emit(arg, provided);
      continue;
    }

    auto def = spec.IsMap() ? spec.AsMap()["default"] : flexbuffers::Reference();
    if (!def.IsNull()) {
      emit(arg, def);
    } else {
      std::fprintf(stderr, "[rocjitsu] plugin '%s': missing required config arg '%s'\n",
                   plugin_name.c_str(), arg.c_str());
      return false;
    }
  }

  // Pass through (and warn about) any user keys not described by the schema.
  auto user_keys = user_map.Keys();
  auto user_vals = user_map.Values();
  for (size_t i = 0; i < user_keys.size(); ++i) {
    std::string key = user_keys[i].AsKey();
    if (schema_map[key.c_str()].IsNull()) {
      std::fprintf(stderr, "[rocjitsu] plugin '%s': config arg '%s' not in schema, passed through\n",
                   plugin_name.c_str(), key.c_str());
      emit(key, user_vals[i]);
    }
  }

  out += "}";
  return true;
}

bool load_one(const std::string &name, const flexbuffers::Reference &user_cfg,
              ExecutionPluginGroup &group) {
  std::string soname = "librocjitsu_plugin_" + name + ".so";
  void *handle = dlopen(soname.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    std::fprintf(stderr, "[rocjitsu] plugin '%s': cannot load %s: %s\n", name.c_str(),
                 soname.c_str(), dlerror());
    return false;
  }

  auto meta_fn = util::lookup_symbol<PluginMetadataFn>(handle, kPluginMetadataSymbol);
  auto create_fn = util::lookup_symbol<PluginCreateFn>(handle, kPluginCreateSymbol);
  if (!meta_fn || !create_fn) {
    std::fprintf(stderr, "[rocjitsu] plugin '%s': %s is missing required ABI exports\n",
                 name.c_str(), soname.c_str());
    dlclose(handle);
    return false;
  }

  const PluginMetadata *meta = meta_fn();
  if (!meta || meta->abi != kPluginAbiVersion) {
    std::fprintf(stderr, "[rocjitsu] plugin '%s': ABI version mismatch (got %d, expected %d)\n",
                 name.c_str(), meta ? meta->abi : -1, kPluginAbiVersion);
    dlclose(handle);
    return false;
  }
  if (meta->name && name != meta->name) {
    std::fprintf(stderr, "[rocjitsu] plugin '%s': metadata name '%s' differs from library name\n",
                 name.c_str(), meta->name);
  }

  std::string resolved;
  if (!resolve_config(name, meta->config_schema, user_cfg, resolved)) {
    dlclose(handle);
    return false;
  }

  std::unique_ptr<ExecutionPlugin> plugin = create_fn(resolved.c_str());
  if (!plugin) {
    std::fprintf(stderr, "[rocjitsu] plugin '%s': create returned null\n", name.c_str());
    dlclose(handle);
    return false;
  }

  if (!group.add(std::move(plugin))) {
    std::fprintf(stderr, "[rocjitsu] plugin '%s': already loaded, skipping duplicate\n",
                 name.c_str());
    dlclose(handle);
    return false;
  }

  open_handles().push_back(handle);
  std::fprintf(stderr, "[rocjitsu] plugin '%s' loaded%s%s\n", name.c_str(),
               (meta->version && *meta->version) ? " v" : "",
               (meta->version && *meta->version) ? meta->version : "");
  return true;
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

} // namespace rocjitsu
