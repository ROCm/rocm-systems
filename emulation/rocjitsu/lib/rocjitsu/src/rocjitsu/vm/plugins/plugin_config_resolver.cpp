// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/plugin_config_resolver.h"

#include "util/log.h"

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

namespace rocjitsu::plugin_detail {
namespace {

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

} // namespace

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
    util::Logger::warn("plugin '", plugin_name, "': invalid config_schema, ignoring");
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
    if (!spec.IsMap()) {
      util::Logger::warn("plugin '", plugin_name, "': config schema entry '", arg,
                         "' must be an object");
      return false;
    }
    const auto type_ref = spec.AsMap()["type"];
    if (!type_ref.IsString()) {
      util::Logger::warn("plugin '", plugin_name, "': config schema entry '", arg,
                         "' must have a string 'type'");
      return false;
    }
    std::string type = type_ref.AsString().c_str();

    auto provided = user_map[arg.c_str()];
    if (!provided.IsNull()) {
      if (!type_matches(type, provided)) {
        util::Logger::warn("plugin '", plugin_name, "': config arg '", arg,
                           "' has wrong type (expected ", type, ")");
        return false;
      }
      emit(arg, provided);
      continue;
    }

    auto def = spec.AsMap()["default"];
    if (!def.IsNull()) {
      if (!type_matches(type, def)) {
        util::Logger::warn("plugin '", plugin_name, "': config arg '", arg,
                           "' has a default with wrong type (expected ", type, ")");
        return false;
      }
      emit(arg, def);
    } else {
      util::Logger::warn("plugin '", plugin_name, "': missing required config arg '", arg, "'");
      return false;
    }
  }

  // Pass through (and warn about) any user keys not described by the schema.
  auto user_keys = user_map.Keys();
  auto user_vals = user_map.Values();
  for (size_t i = 0; i < user_keys.size(); ++i) {
    std::string key = user_keys[i].AsKey();
    if (schema_map[key.c_str()].IsNull()) {
      util::Logger::warn("plugin '", plugin_name, "': config arg '", key,
                         "' not in schema, passed through");
      emit(key, user_vals[i]);
    }
  }

  out += "}";
  return true;
}

bool resolve_config_json(const std::string &plugin_name, const char *schema_json,
                         const std::string &user_cfg_json, std::string &out) {
  std::string trimmed_user = trimmed(user_cfg_json);
  if (trimmed_user.empty()) {
    flexbuffers::Builder empty;
    empty.Map([&] {});
    empty.Finish();
    return resolve_config(plugin_name, schema_json, flexbuffers::GetRoot(empty.GetBuffer()), out);
  }

  flexbuffers::Builder user_fbb;
  if (!flexbuffer_from_json(trimmed_user, user_fbb)) {
    util::Logger::warn("plugin '", plugin_name, "': invalid config JSON");
    return false;
  }
  return resolve_config(plugin_name, schema_json, flexbuffers::GetRoot(user_fbb.GetBuffer()), out);
}

} // namespace rocjitsu::plugin_detail
