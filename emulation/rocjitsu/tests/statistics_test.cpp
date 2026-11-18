// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/stats/statistics.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace {

class StatisticsTest : public testing::Test {
protected:
  void SetUp() override {
    directory_ = std::filesystem::temp_directory_path() /
                 ("rocjitsu-statistics-test-" +
                  std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  }
  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }
  std::filesystem::path directory_;
};

TEST_F(StatisticsTest, DisabledConfigurationHasNoFilesystemEffects) {
  rocjitsu::StatisticsConfig config;
  config.directory = directory_.string();
  auto session = rocjitsu::StatisticsSession::create(config, "node", "functional");
  ASSERT_TRUE(session.succeeded());
  EXPECT_EQ(session.value(), nullptr);
  EXPECT_FALSE(std::filesystem::exists(directory_));
}

TEST_F(StatisticsTest, DispatchDetailLimitDoesNotLimitComponentAggregates) {
  rocjitsu::StatisticsConfig config{
      .enabled = true, .directory = directory_.string(), .max_dispatch_scopes = 1};
  auto result = rocjitsu::StatisticsSession::create(config, "node", "functional");
  ASSERT_TRUE(result.succeeded());
  auto session = std::move(result).value();
  ASSERT_NE(session->scope("dispatch0", 0, "dispatch"), nullptr);
  EXPECT_EQ(session->scope("dispatch1", 0, "dispatch"), nullptr);
  ASSERT_NE(session->scope("gpu0", 0), nullptr);
  ASSERT_NE(session->scope("gpu1", 1), nullptr);
  session->mark_run_complete();
  session->finish();
  std::ifstream file(session->output_path());
  const std::string output{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  EXPECT_NE(output.find("\"dropped_dispatch_scopes\":\"1\""), std::string::npos);
  EXPECT_NE(output.find("\"record\":\"session_end\",\"complete\":false"), std::string::npos);
}

TEST_F(StatisticsTest, ScopeIdentityIncludesGpuAndAdapterPreservesRunMetadata) {
  rocjitsu::StatisticsConfig config{
      .enabled = true, .scopes = {"component"}, .directory = directory_.string()};
  auto result = rocjitsu::StatisticsSession::create(config, "node\n\"", "functional");
  ASSERT_TRUE(result.succeeded());
  auto session = std::move(result).value();
  auto first = session->scope("l2", 0);
  auto second = session->scope("l2", 1);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first, second);
  EXPECT_EQ(session->scope("l2", 0), first);
  EXPECT_EQ(session->scope("dispatch0", 0, "dispatch"), nullptr);
  ASSERT_NE(first->counter("cache.hits", "requests", "Demand hits"), nullptr);
  session->mark_run_complete();
  session->mark_incomplete();
  session->finish();
  std::ifstream file(session->output_path());
  const std::string output{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  EXPECT_NE(output.find("\"schema\":\"rocjitsu.statistics.v1\""), std::string::npos);
  EXPECT_NE(output.find("node\\u000a\\\""), std::string::npos);
  EXPECT_NE(output.find("\"execution_model\":\"functional\""), std::string::npos);
  EXPECT_NE(output.find("\"gpu_id\":0"), std::string::npos);
  EXPECT_NE(output.find("\"gpu_id\":1"), std::string::npos);
  EXPECT_NE(output.find("\"domain\":\"model_events\""), std::string::npos);
  EXPECT_NE(output.find("\"record\":\"session_end\",\"complete\":false"), std::string::npos);
}

} // namespace
