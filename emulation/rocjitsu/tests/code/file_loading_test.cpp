// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/rj_code.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace rocjitsu {
namespace {

TEST(FileLoadingTest, RejectsDirectoryWithoutThrowing) {
  const test::ScopedTempDirectory directory("rocjitsu_file_loading_");
  EXPECT_TRUE(detail::read_file_bytes(directory.path()).failed());
  EXPECT_FALSE(Executable(directory.path()).is_valid());
  EXPECT_FALSE(AmdGpuCodeObject(directory.path()).is_valid());

  rj_code_executable_t *executable = nullptr;
  EXPECT_EQ(rj_code_executable_create(directory.path().c_str(), &executable),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(executable, nullptr);
  rj_code_executable_destroy(executable);
}

TEST(FileLoadingTest, RejectsMissingFileWithoutThrowing) {
  const test::ScopedTempDirectory directory("rocjitsu_file_loading_");
  const std::string path = directory.path() + "/missing";
  EXPECT_TRUE(detail::read_file_bytes(path).failed());
  EXPECT_FALSE(Executable(path).is_valid());
  EXPECT_FALSE(AmdGpuCodeObject(path).is_valid());

  rj_code_executable_t *executable = nullptr;
  EXPECT_EQ(rj_code_executable_create(path.c_str(), &executable),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(executable, nullptr);
  rj_code_executable_destroy(executable);
}

TEST(FileLoadingTest, ReadsEmptyFileButRejectsItAsCode) {
  const test::ScopedTempFile file("rocjitsu_file_loading_");
  util::FailureOr<std::vector<char>> image = detail::read_file_bytes(file.path());
  ASSERT_TRUE(image.succeeded());
  EXPECT_TRUE(image.value().empty());
  EXPECT_FALSE(Executable(file.path()).is_valid());
  EXPECT_FALSE(AmdGpuCodeObject(file.path()).is_valid());

  rj_code_executable_t *executable = nullptr;
  EXPECT_EQ(rj_code_executable_create(file.path().c_str(), &executable),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(executable, nullptr);
  rj_code_executable_destroy(executable);
}

TEST(FileLoadingTest, RejectsUnreadableRegularFileWithoutThrowing) {
  const test::ScopedTempFile file("rocjitsu_file_loading_");
  file.write("not empty");
  std::error_code error;
  std::filesystem::permissions(file.path(), std::filesystem::perms::none, error);
  ASSERT_FALSE(error) << error.message();
  ASSERT_TRUE(std::filesystem::is_regular_file(file.path(), error));
  ASSERT_FALSE(error) << error.message();

  {
    std::ifstream probe(file.path(), std::ios::binary);
    if (probe.is_open())
      GTEST_SKIP() << "The process can still read a file with mode 000";
  }

  EXPECT_TRUE(detail::read_file_bytes(file.path()).failed());
  EXPECT_FALSE(Executable(file.path()).is_valid());
  EXPECT_FALSE(AmdGpuCodeObject(file.path()).is_valid());

  rj_code_executable_t *executable = nullptr;
  EXPECT_EQ(rj_code_executable_create(file.path().c_str(), &executable),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(executable, nullptr);
  rj_code_executable_destroy(executable);
}

} // namespace
} // namespace rocjitsu
