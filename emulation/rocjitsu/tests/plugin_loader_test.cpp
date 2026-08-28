// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/plugin_loader.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
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

TEST_F(PluginLoaderTest, LoadsMatchingAbi) {
  rocjitsu::ExecutionPluginGroup group;
  EXPECT_EQ(load("good", group), 1);
  EXPECT_EQ(group.num_plugins(), 1u);
  EXPECT_NE(trace().find("good:create\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, RejectsAbiMismatchBeforeCreate) {
  rocjitsu::ExecutionPluginGroup group;
  EXPECT_EQ(load("badabi", group), 0);
  EXPECT_TRUE(group.empty());
  EXPECT_EQ(trace().find("badabi:create\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, RejectsMissingRequiredExport) {
  rocjitsu::ExecutionPluginGroup group;
  EXPECT_EQ(load("missing", group), 0);
  EXPECT_TRUE(group.empty());
  EXPECT_EQ(trace().find("missing:create\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, DestroysRejectedDuplicateBeforeUnload) {
  rocjitsu::ExecutionPluginGroup group;
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

TEST_F(PluginLoaderTest, RejectsProfiledGroupWithMultipleThreads) {
  const simdojo::SimulationEngine::Config engine_config{.num_threads = 2};
  EXPECT_THROW(
      rocjitsu::PluginLoader::configure_plugin_group(R"({"profiled":true})", "", engine_config),
      std::invalid_argument);
}

TEST_F(PluginLoaderTest, ProfileFileSinkFailureFallsBackToStderr) {
  testing::internal::CaptureStderr();
  auto group = rocjitsu::PluginLoader::configure_plugin_group(
      R"({"profiled":true,"sinks":{"types":["file"],"dir":"/dev/null"}})");
  group->onInit();
  group->onShutdown();
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_NE(error.find("cannot open plugin sink '/dev/null/profile.log'"), std::string::npos);
  EXPECT_NE(error.find("total emulation time"), std::string::npos);
}

TEST_F(PluginLoaderTest, PluginFileSinkFailureFallsBackToStderr) {
  testing::internal::CaptureStderr();
  auto group = rocjitsu::PluginLoader::configure_plugin_group(
      R"({"plugins":{"good":{}},"sinks":{"types":["file"],"dir":"/dev/null"}})",
      PLUGIN_LOADER_FIXTURE_DIR);
  ASSERT_EQ(group->num_plugins(), 1u);
  group->onInit();
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_NE(error.find("cannot open plugin sink '/dev/null/boundary.log'"), std::string::npos);
  EXPECT_NE(error.find("boundary:init"), std::string::npos);
}

TEST_F(PluginLoaderTest, FileSinkFailureDoesNotDuplicateConfiguredStderr) {
  testing::internal::CaptureStderr();
  auto group = rocjitsu::PluginLoader::configure_plugin_group(
      R"({"profiled":true,"sinks":{"types":["stderr","file"],"dir":"/dev/null"}})");
  group->onInit();
  group->onShutdown();
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_NE(error.find("cannot open plugin sink '/dev/null/profile.log'"), std::string::npos);
  const size_t output = error.find("total emulation time");
  ASSERT_NE(output, std::string::npos);
  EXPECT_EQ(error.find("total emulation time", output + 1), std::string::npos);
}

TEST_F(PluginLoaderTest, FileSinkWithoutDirectoryUsesDefaultStderrSink) {
  testing::internal::CaptureStderr();
  auto group = rocjitsu::PluginLoader::configure_plugin_group(
      R"({"profiled":true,"sinks":{"types":["file"]}})");
  group->onInit();
  group->onShutdown();
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_NE(error.find("sink type 'file' requested but no 'dir' set"), std::string::npos);
  EXPECT_NE(error.find("total emulation time"), std::string::npos);
}

} // namespace
