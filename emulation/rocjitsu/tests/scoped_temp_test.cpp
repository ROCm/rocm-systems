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

TEST(ScopedTempTest, UnavailableTemporaryDirectoryLeavesResourcesUninitialized) {
  rocjitsu::test::ScopedTempDirectory parent;
  ASSERT_TRUE(parent.initialize("rocjitsu-temp-failure-"));
  rocjitsu::test::ScopedTempFile file;
  rocjitsu::test::ScopedTempDirectory directory;
  {
    rocjitsu::test::ScopedEnvironmentVariable temporary_directory("TMPDIR",
                                                                  parent.path() + "/absent");
    EXPECT_FALSE(file.initialize("file-"));
    EXPECT_FALSE(directory.initialize("directory-"));
    EXPECT_TRUE(file.path().empty());
    EXPECT_TRUE(directory.path().empty());
    EXPECT_FALSE(file.write("must not be written"));
  }
  ASSERT_TRUE(file.initialize("rocjitsu-temp-retry-"));
  ASSERT_TRUE(directory.initialize("rocjitsu-temp-retry-"));
}

TEST(ScopedTempTest, CreationFailureDoesNotRetainTheUncreatedPattern) {
  rocjitsu::test::ScopedTempDirectory parent;
  ASSERT_TRUE(parent.initialize("rocjitsu-temp-create-failure-"));
  rocjitsu::test::ScopedTempFile file;
  rocjitsu::test::ScopedTempDirectory directory;
  EXPECT_FALSE(file.initialize(parent.path() + "/absent/file-"));
  EXPECT_FALSE(directory.initialize(parent.path() + "/absent/directory-"));
  EXPECT_TRUE(file.path().empty());
  EXPECT_TRUE(directory.path().empty());
}

TEST(ScopedTempFileTest, ReportsWriteOpenAndFlushFailures) {
  rocjitsu::test::ScopedTempFile file;
  ASSERT_TRUE(file.initialize("rocjitsu-temp-write-failure-"));
  ASSERT_TRUE(std::filesystem::remove(file.path()));
  ASSERT_TRUE(std::filesystem::create_directory(file.path()));
  EXPECT_FALSE(file.write("cannot open a directory"));
  ASSERT_TRUE(std::filesystem::remove(file.path()));
  std::filesystem::create_symlink("/dev/full", file.path());
  EXPECT_FALSE(file.write("cannot flush to a full device"));
}

TEST(ScopedTempFileTest, MoveTransferCleansReplacedAndFinalFiles) {
  std::filesystem::path transferred_path;
  std::filesystem::path replaced_path;

  {
    rocjitsu::test::ScopedTempFile source;
    ASSERT_TRUE(source.initialize("rocjitsu-temp-file-source-"));
    transferred_path = source.path();
    ASSERT_TRUE(source.write("transferred contents"));

    rocjitsu::test::ScopedTempFile moved(std::move(source));
    EXPECT_TRUE(source.path().empty());
    EXPECT_EQ(std::filesystem::path(moved.path()), transferred_path);

    rocjitsu::test::ScopedTempFile destination;
    ASSERT_TRUE(destination.initialize("rocjitsu-temp-file-destination-"));
    replaced_path = destination.path();
    ASSERT_TRUE(destination.write("replaced contents"));

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
    rocjitsu::test::ScopedTempDirectory source;
    ASSERT_TRUE(source.initialize("rocjitsu-temp-dir-source-"));
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

    rocjitsu::test::ScopedTempDirectory destination;
    ASSERT_TRUE(destination.initialize("rocjitsu-temp-dir-destination-"));
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
