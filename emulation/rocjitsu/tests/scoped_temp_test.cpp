// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>

namespace {

TEST(ScopedTempFileTest, MoveTransferCleansReplacedAndFinalFiles) {
  std::filesystem::path transferred_path;
  std::filesystem::path replaced_path;

  {
    rocjitsu::test::ScopedTempFile source("rocjitsu-temp-file-source-");
    transferred_path = source.path();
    source.write("transferred contents");

    rocjitsu::test::ScopedTempFile moved(std::move(source));
    EXPECT_TRUE(source.path().empty());
    EXPECT_EQ(std::filesystem::path(moved.path()), transferred_path);

    rocjitsu::test::ScopedTempFile destination("rocjitsu-temp-file-destination-");
    replaced_path = destination.path();
    destination.write("replaced contents");

    destination = std::move(moved);
    EXPECT_TRUE(moved.path().empty());
    EXPECT_EQ(std::filesystem::path(destination.path()), transferred_path);
    EXPECT_FALSE(std::filesystem::exists(replaced_path));

    std::ifstream input(destination.path(), std::ios::binary);
    ASSERT_TRUE(input);
    const std::string contents((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
    EXPECT_EQ(contents, "transferred contents");
  }

  EXPECT_FALSE(std::filesystem::exists(transferred_path));
  EXPECT_FALSE(std::filesystem::exists(replaced_path));
}

TEST(ScopedTempDirectoryTest, MoveTransferRecursivelyCleansReplacedAndFinalDirectories) {
  std::filesystem::path transferred_path;
  std::filesystem::path replaced_path;

  {
    rocjitsu::test::ScopedTempDirectory source("rocjitsu-temp-dir-source-");
    transferred_path = source.path();
    const auto transferred_child = transferred_path / "transferred.txt";
    {
      std::ofstream output(transferred_child);
      ASSERT_TRUE(output);
      output << "transferred contents";
    }
    ASSERT_TRUE(std::filesystem::exists(transferred_child));

    rocjitsu::test::ScopedTempDirectory moved(std::move(source));
    EXPECT_TRUE(source.path().empty());
    EXPECT_EQ(std::filesystem::path(moved.path()), transferred_path);

    rocjitsu::test::ScopedTempDirectory destination("rocjitsu-temp-dir-destination-");
    replaced_path = destination.path();
    const auto replaced_child = replaced_path / "replaced.txt";
    {
      std::ofstream output(replaced_child);
      ASSERT_TRUE(output);
      output << "replaced contents";
    }
    ASSERT_TRUE(std::filesystem::exists(replaced_child));

    destination = std::move(moved);
    EXPECT_TRUE(moved.path().empty());
    EXPECT_EQ(std::filesystem::path(destination.path()), transferred_path);
    EXPECT_FALSE(std::filesystem::exists(replaced_path));
    EXPECT_TRUE(std::filesystem::exists(transferred_child));
  }

  EXPECT_FALSE(std::filesystem::exists(transferred_path));
  EXPECT_FALSE(std::filesystem::exists(replaced_path));
}

} // namespace
