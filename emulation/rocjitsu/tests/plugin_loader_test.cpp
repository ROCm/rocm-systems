// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/plugin_loader.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unistd.h>

#ifndef PLUGIN_LOADER_FIXTURE_DIR
#error "PLUGIN_LOADER_FIXTURE_DIR must be defined"
#endif

namespace {

class PluginLoaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    trace_ = std::filesystem::temp_directory_path() /
             ("rocjitsu_plugin_loader_" + std::to_string(getpid()) + ".trace");
    std::filesystem::remove(trace_);
    ASSERT_EQ(setenv("ROCJITSU_PLUGIN_TEST_TRACE", trace_.c_str(), 1), 0);
  }

  void TearDown() override {
    unsetenv("ROCJITSU_PLUGIN_TEST_TRACE");
    std::filesystem::remove(trace_);
  }

  std::string trace() const {
    std::ifstream input(trace_);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }

  int load(const char *name, rocjitsu::ExecutionPluginGroup &group) const {
    const std::string config = std::string{"{\"plugins\":{\""} + name + "\":{}}}";
    return rocjitsu::PluginLoader::load_from_config(config, group, PLUGIN_LOADER_FIXTURE_DIR);
  }

  std::filesystem::path trace_;
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
  EXPECT_THROW(rocjitsu::PluginLoader::configure_plugin_group(R"({"profiled":true})", "", 2),
               std::invalid_argument);
}

} // namespace
