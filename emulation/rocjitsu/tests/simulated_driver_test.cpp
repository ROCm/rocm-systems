// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simulated_driver_test.cpp
/// @brief Tests for SimulatedDriver creation, open/close, and topology generation.

#include "rocjitsu/kmd/linux/simulated_driver.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

namespace {

const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/amdgpu_cdna4.json";
const std::string SCHEMA_PATH = std::string(SCHEMA_DIR) + "/simulation_config.fbs";

class SimulatedDriverTest : public ::testing::Test {
protected:
  void SetUp() override {
    setenv("RJ_CONFIG", CONFIG_PATH.c_str(), 1);
    setenv("RJ_SCHEMA", SCHEMA_PATH.c_str(), 1);
  }
};

TEST_F(SimulatedDriverTest, CreateDefault) {
  auto driver = rocjitsu::SimulatedDriver::create_default();
  ASSERT_NE(driver, nullptr);
}

TEST_F(SimulatedDriverTest, OpenAndClose) {
  auto driver = rocjitsu::SimulatedDriver::create_default();
  ASSERT_NE(driver, nullptr);

  int fd = driver->open();
  EXPECT_GE(fd, 0);

  int ret = driver->close();
  EXPECT_EQ(ret, 0);
}

TEST_F(SimulatedDriverTest, TopologyDirectoryExists) {
  auto driver = rocjitsu::SimulatedDriver::create_default();
  ASSERT_NE(driver, nullptr);

  const auto &path = driver->topology().path();
  EXPECT_FALSE(path.empty());
  EXPECT_TRUE(std::filesystem::exists(path));
  EXPECT_TRUE(std::filesystem::exists(path + "/generation_id"));
  EXPECT_TRUE(std::filesystem::exists(path + "/nodes/0/properties"));
  EXPECT_TRUE(std::filesystem::exists(path + "/nodes/1/properties"));
}

} // namespace
