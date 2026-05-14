// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "test_utils.h"

#include <gtest/gtest.h>

#include <filesystem>

// Test that we can access test assets directory
TEST(TestUtilsTest, TestAssetsAccess) {
  // Verify test assets directory exists and is accessible
  EXPECT_NO_THROW({
    auto assets_dir = test_utils::get_test_assets_dir();
    EXPECT_TRUE(std::filesystem::exists(assets_dir));
    EXPECT_TRUE(std::filesystem::is_directory(assets_dir));
  });

  // Verify we can get paths to generated kpack files
  EXPECT_NO_THROW({
    auto noop_kpack = test_utils::get_test_asset("test_noop.kpack");
    EXPECT_TRUE(std::filesystem::is_regular_file(noop_kpack));
  });

  EXPECT_NO_THROW({
    auto zstd_kpack = test_utils::get_test_asset("test_zstd.kpack");
    EXPECT_TRUE(std::filesystem::is_regular_file(zstd_kpack));
  });
}
