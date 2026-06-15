// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Unit tests for the plugin config resolver. The resolver merges a plugin's
// declared `config_schema` with the user-supplied config: applying defaults,
// validating types, enforcing required args, and passing unknown keys through.
// These tests exercise the contract directly, without dlopen()ing a plugin.

#include "rocjitsu/vm/plugins/plugin_config_resolver.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <string>

namespace {

using rocjitsu::plugin_detail::flexbuffer_from_json;
using rocjitsu::plugin_detail::resolve_config_json;
using rocjitsu::plugin_detail::type_matches;

// Resolve helper: returns the resolved JSON via out-param and the bool result.
bool resolve(const char *schema, const std::string &user, std::string &out) {
  return resolve_config_json("test_plugin", schema, user, out);
}

TEST(PluginConfigResolver, NoSchemaPassesUserConfigThrough) {
  std::string out;
  EXPECT_TRUE(resolve(nullptr, R"({"a": 1, "b": "x"})", out));
  EXPECT_NE(out.find("\"a\": 1"), std::string::npos);
  EXPECT_NE(out.find("\"b\": \"x\""), std::string::npos);
}

TEST(PluginConfigResolver, EmptySchemaWithEmptyUserConfigYieldsEmptyObject) {
  std::string out;
  EXPECT_TRUE(resolve("{}", "", out));
  // An empty object may be rendered with insignificant whitespace.
  out.erase(std::remove_if(out.begin(), out.end(), [](unsigned char c) { return std::isspace(c); }),
            out.end());
  EXPECT_EQ(out, "{}");
}

TEST(PluginConfigResolver, AppliesDefaultsForMissingArgs) {
  const char *schema = R"({
    "level": { "type": "number", "default": 3 },
    "label": { "type": "string", "default": "info" }
  })";
  std::string out;
  EXPECT_TRUE(resolve(schema, "", out));
  EXPECT_NE(out.find("\"level\": 3"), std::string::npos);
  EXPECT_NE(out.find("\"label\": \"info\""), std::string::npos);
}

TEST(PluginConfigResolver, ProvidedValueOverridesDefault) {
  const char *schema = R"({ "level": { "type": "number", "default": 3 } })";
  std::string out;
  EXPECT_TRUE(resolve(schema, R"({"level": 7})", out));
  EXPECT_NE(out.find("\"level\": 7"), std::string::npos);
  EXPECT_EQ(out.find("\"level\": 3"), std::string::npos);
}

TEST(PluginConfigResolver, MissingRequiredArgFails) {
  // An arg with a type but no default is required.
  const char *schema = R"({ "path": { "type": "string" } })";
  std::string out;
  EXPECT_FALSE(resolve(schema, "", out));
}

TEST(PluginConfigResolver, RequiredArgProvidedSucceeds) {
  const char *schema = R"({ "path": { "type": "string" } })";
  std::string out;
  EXPECT_TRUE(resolve(schema, R"({"path": "/tmp/log"})", out));
  EXPECT_NE(out.find("\"path\": \"/tmp/log\""), std::string::npos);
}

TEST(PluginConfigResolver, WrongTypeFails) {
  const char *schema = R"({ "level": { "type": "number", "default": 3 } })";
  std::string out;
  EXPECT_FALSE(resolve(schema, R"({"level": "not-a-number"})", out));
}

TEST(PluginConfigResolver, UnknownKeyPassedThrough) {
  const char *schema = R"({ "level": { "type": "number", "default": 3 } })";
  std::string out;
  EXPECT_TRUE(resolve(schema, R"({"extra": "kept"})", out));
  EXPECT_NE(out.find("\"level\": 3"), std::string::npos);
  EXPECT_NE(out.find("\"extra\": \"kept\""), std::string::npos);
}

TEST(PluginConfigResolver, InvalidSchemaIsIgnored) {
  std::string out;
  EXPECT_TRUE(resolve("{ this is not json", R"({"a": 1})", out));
  EXPECT_NE(out.find("\"a\": 1"), std::string::npos);
}

TEST(PluginConfigResolver, InvalidUserJsonFails) {
  const char *schema = R"({ "level": { "type": "number", "default": 3 } })";
  std::string out;
  EXPECT_FALSE(resolve(schema, "{ not json", out));
}

TEST(PluginConfigResolver, TypeMatchesValidatesAgainstValueKinds) {
  flexbuffers::Builder fbb;
  ASSERT_TRUE(flexbuffer_from_json(R"({"s": "hello", "n": 42, "f": 1.5, "b": true})", fbb));
  auto root = flexbuffers::GetRoot(fbb.GetBuffer()).AsMap();

  EXPECT_TRUE(type_matches("string", root["s"]));
  EXPECT_FALSE(type_matches("number", root["s"]));

  EXPECT_TRUE(type_matches("number", root["n"]));
  EXPECT_TRUE(type_matches("number", root["f"]));
  EXPECT_FALSE(type_matches("string", root["n"]));

  EXPECT_TRUE(type_matches("boolean", root["b"]));
  EXPECT_TRUE(type_matches("bool", root["b"]));
  EXPECT_FALSE(type_matches("number", root["b"]));

  // Unknown schema type names never match.
  EXPECT_FALSE(type_matches("widget", root["s"]));
}

} // namespace
