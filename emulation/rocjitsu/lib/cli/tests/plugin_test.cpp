// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "plugin_host.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>

namespace {

const char *plugin_path() {
  if (const char *path = std::getenv("RJ_CLI_PLUGIN_PATH"))
    return path;
  return RJ_CLI_PLUGIN_PATH;
}

} // namespace

TEST(CliPlugin, RetainedFlagPinsLibraryBeyondTreeLifetime) {
  util::StringDiagnostic diagnostic;
  std::shared_ptr<rocjitsu::cli::Flag> retained;
  std::weak_ptr<void> library;
  {
    rocjitsu::cli::Command root{"tool"};
    auto loaded = load_plugin(plugin_path(), root, diagnostic.emitter());
    ASSERT_TRUE(loaded.succeeded()) << diagnostic.message();
    library = loaded.value();
    loaded.value().reset();
    const std::array<std::string_view, 5> args{"plugin", "inspect", "-f", "json", "input.bin"};
    auto parsed = rocjitsu::cli::parse(root, args, diagnostic.emitter());
    ASSERT_TRUE(parsed.succeeded()) << diagnostic.message();
    EXPECT_EQ(parsed.value().positional, std::vector<std::string>{"input.bin"});
    retained = parsed.value().command->find_flag("--format");
    EXPECT_EQ(retained->format_value(), "json");
  }
  ASSERT_FALSE(library.expired());
  ASSERT_TRUE(retained->parse_and_set("text", diagnostic.emitter()).succeeded());
  EXPECT_EQ(retained->format_value(), "text");
  EXPECT_TRUE(retained->parse_and_set("bad", diagnostic.emitter()).failed());
  EXPECT_EQ(diagnostic.message(), "--format: expected text or json");
  retained.reset();
  EXPECT_TRUE(library.expired());
  void *unexpected = dlopen(plugin_path(), RTLD_NOW | RTLD_NOLOAD);
  EXPECT_EQ(unexpected, nullptr) << "fixture must really unload to exercise the lifetime guard";
  if (unexpected)
    dlclose(unexpected);
}

TEST(CliPlugin, MissingLibraryAndRegistrationCollisionFail) {
  util::StringDiagnostic diagnostic;
  rocjitsu::cli::Command root{"tool"};
  EXPECT_TRUE(load_plugin("/no/such/rocjitsu-cli-plugin.so", root, diagnostic.emitter()).failed());
  auto first = load_plugin(plugin_path(), root, diagnostic.emitter());
  ASSERT_TRUE(first.succeeded());
  EXPECT_TRUE(load_plugin(plugin_path(), root, diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("duplicate command 'plugin'"), std::string::npos);
}
