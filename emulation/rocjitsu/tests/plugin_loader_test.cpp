// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/plugin_loader.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

#ifndef PLUGIN_LOADER_FIXTURE_DIR
#error "PLUGIN_LOADER_FIXTURE_DIR must be defined"
#endif

namespace {

class PluginLoaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(setenv("ROCJITSU_PLUGIN_TEST_TRACE", trace_.path().c_str(), 1), 0);
  }

  void TearDown() override { unsetenv("ROCJITSU_PLUGIN_TEST_TRACE"); }

  std::string trace() const {
    std::ifstream input(trace_.path());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }

  int load(const char *name, rocjitsu::ExecutionPluginGroup &group) const {
    const std::string config = std::string{"{\"plugins\":{\""} + name + "\":{}}}";
    return rocjitsu::PluginLoader::load_from_config(config, group, PLUGIN_LOADER_FIXTURE_DIR);
  }

  rocjitsu::test::ScopedTempFile trace_{"rocjitsu-plugin-loader-"};
};

TEST_F(PluginLoaderTest, LoadsPluginAndDispatchesLifecycle) {
  rocjitsu::ExecutionPluginGroup group(rocjitsu::PluginSinkConfig{});
  EXPECT_EQ(load("good", group), 1);
  EXPECT_EQ(group.num_plugins(), 1u);
  EXPECT_NE(trace().find("good:create\n"), std::string::npos);
  group.onInit();
  EXPECT_NE(trace().find("good:init\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, WarnsAndLoadsWhenRemovedProfiledConfigIsPresent) {
  testing::internal::CaptureStderr();
  auto group = rocjitsu::PluginLoader::configure_plugin_group(
      R"({"profiled":false,"plugins":{"good":{}}})", PLUGIN_LOADER_FIXTURE_DIR);
  const std::string error = testing::internal::GetCapturedStderr();

  ASSERT_TRUE(group.succeeded());
  ASSERT_EQ(group.value()->num_plugins(), 1u);
  EXPECT_NE(trace().find("good:create\n"), std::string::npos);
  EXPECT_NE(error.find("hook profiling was removed; ignoring top-level 'profiled' config"),
            std::string::npos);
}

TEST_F(PluginLoaderTest, RejectsMissingRequiredExport) {
  rocjitsu::ExecutionPluginGroup group(rocjitsu::PluginSinkConfig{});
  EXPECT_EQ(load("missing", group), 0);
  EXPECT_TRUE(group.empty());
  EXPECT_EQ(trace().find("missing:create\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, PluginFailuresRemainBestEffortByDefault) {
  auto group = rocjitsu::PluginLoader::configure_plugin_group(R"({"plugins":{"missing":{}}})",
                                                              PLUGIN_LOADER_FIXTURE_DIR);

  ASSERT_TRUE(group.succeeded());
  EXPECT_TRUE(group.value()->empty());
}

TEST_F(PluginLoaderTest, RequiredMissingPluginReturnsFailure) {
  util::StringDiagnostic diagnostic;
  const auto group = rocjitsu::PluginLoader::configure_plugin_group(
      R"({"require_all_plugins":true,"plugins":{"missing":{}}})", PLUGIN_LOADER_FIXTURE_DIR,
      diagnostic.emitter());
  EXPECT_TRUE(group.failed());
  EXPECT_EQ(diagnostic.message(),
            "required plugin loading failed: loaded 0 of 1 configured plugins");
}

TEST_F(PluginLoaderTest, FailedRequiredGroupDestroysCreatedPlugins) {
  const auto group = rocjitsu::PluginLoader::configure_plugin_group(
      R"({"require_all_plugins":true,"plugins":{"good":{},"missing":{}}})",
      PLUGIN_LOADER_FIXTURE_DIR);
  EXPECT_TRUE(group.failed());
  EXPECT_NE(trace().find("good:create\n"), std::string::npos);
  EXPECT_NE(trace().find("good:destroy\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, RequiredPluginLoads) {
  auto group = rocjitsu::PluginLoader::configure_plugin_group(
      R"({"require_all_plugins":true,"plugins":{"good":{}}})", PLUGIN_LOADER_FIXTURE_DIR);

  ASSERT_TRUE(group.succeeded());
  ASSERT_EQ(group.value()->num_plugins(), 1u);
  EXPECT_NE(trace().find("good:create\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, RequiringAllPluginsAllowsNoPlugins) {
  auto group = rocjitsu::PluginLoader::configure_plugin_group(R"({"require_all_plugins":true})",
                                                              PLUGIN_LOADER_FIXTURE_DIR);

  ASSERT_TRUE(group.succeeded());
  EXPECT_TRUE(group.value()->empty());
}

TEST_F(PluginLoaderTest, RequiringAllPluginsRejectsNonObjectPlugins) {
  for (const char *plugins : {"[]", "null", R"("good")"}) {
    SCOPED_TRACE(plugins);
    const std::string config =
        std::string{"{\"require_all_plugins\":true,\"plugins\":"} + plugins + "}";

    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(rocjitsu::PluginLoader::configure_plugin_group(config, PLUGIN_LOADER_FIXTURE_DIR,
                                                               diagnostic.emitter())
                    .failed());
    EXPECT_EQ(diagnostic.message(),
              "top-level 'plugins' must be an object when 'require_all_plugins' is true");
  }
}

TEST_F(PluginLoaderTest, BestEffortLoadingIgnoresNonObjectPlugins) {
  for (const char *plugins : {"[]", "null", R"("good")"}) {
    SCOPED_TRACE(plugins);
    const std::string config = std::string{"{\"plugins\":"} + plugins + "}";

    auto group = rocjitsu::PluginLoader::configure_plugin_group(config, PLUGIN_LOADER_FIXTURE_DIR);
    ASSERT_TRUE(group.succeeded());
    EXPECT_TRUE(group.value()->empty());
  }
}

TEST_F(PluginLoaderTest, RejectsNonBooleanRequireAllPlugins) {
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(rocjitsu::PluginLoader::configure_plugin_group(
                  R"({"require_all_plugins":"true","plugins":{"good":{}}})",
                  PLUGIN_LOADER_FIXTURE_DIR, diagnostic.emitter())
                  .failed());
  EXPECT_EQ(diagnostic.message(), "top-level 'require_all_plugins' must be a boolean");
  EXPECT_EQ(trace().find("good:create\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, DestroysRejectedDuplicateBeforeUnload) {
  rocjitsu::ExecutionPluginGroup group(rocjitsu::PluginSinkConfig{});
  ASSERT_EQ(load("good", group), 1);
  ASSERT_EQ(load("duplicate", group), 0);

  const std::string events = trace();
  const size_t created = events.find("duplicate:create\n");
  const size_t destroyed = events.find("duplicate:destroy\n");
  const size_t unloaded = events.find("duplicate:unload\n");
  ASSERT_NE(created, std::string::npos) << events;
  ASSERT_NE(destroyed, std::string::npos) << events;
  ASSERT_NE(unloaded, std::string::npos) << events;
  EXPECT_LT(created, destroyed) << events;
  EXPECT_LT(destroyed, unloaded) << events;
  EXPECT_EQ(group.num_plugins(), 1u);
}

TEST_F(PluginLoaderTest, PluginFileSinkFailureFallsBackToStderr) {
  testing::internal::CaptureStderr();
  auto group = rocjitsu::PluginLoader::configure_plugin_group(
      R"({"plugins":{"good":{}},"sinks":{"types":["file"],"dir":"/dev/null"}})",
      PLUGIN_LOADER_FIXTURE_DIR);
  ASSERT_TRUE(group.succeeded());
  ASSERT_EQ(group.value()->num_plugins(), 1u);
  group.value()->onInit();
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_NE(error.find("cannot open plugin sink '/dev/null/boundary.log'"), std::string::npos);
  EXPECT_NE(error.find("boundary:init"), std::string::npos);
}

TEST_F(PluginLoaderTest, FileSinkFailureDoesNotDuplicateConfiguredStderr) {
  testing::internal::CaptureStderr();
  auto group = rocjitsu::PluginLoader::configure_plugin_group(
      R"({"plugins":{"good":{}},"sinks":{"types":["stderr","file"],"dir":"/dev/null"}})",
      PLUGIN_LOADER_FIXTURE_DIR);
  ASSERT_TRUE(group.succeeded());
  group.value()->onInit();
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_NE(error.find("cannot open plugin sink '/dev/null/boundary.log'"), std::string::npos);
  const size_t output = error.find("boundary:init");
  ASSERT_NE(output, std::string::npos);
  EXPECT_EQ(error.find("boundary:init", output + 1), std::string::npos);
}

TEST_F(PluginLoaderTest, FileSinkWithoutDirectoryUsesDefaultStderrSink) {
  testing::internal::CaptureStderr();
  auto group = rocjitsu::PluginLoader::configure_plugin_group(
      R"({"plugins":{"good":{}},"sinks":{"types":["file"]}})", PLUGIN_LOADER_FIXTURE_DIR);
  ASSERT_TRUE(group.succeeded());
  group.value()->onInit();
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_NE(error.find("sink type 'file' requested but no 'dir' set"), std::string::npos);
  EXPECT_NE(error.find("boundary:init"), std::string::npos);
}

} // namespace
