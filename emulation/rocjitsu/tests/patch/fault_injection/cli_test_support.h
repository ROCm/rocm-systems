// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <system_error>

namespace rocjitsu::test {

inline testing::AssertionResult mutation_cli_path(std::string &path) {
  std::error_code error;
  const std::filesystem::path executable = std::filesystem::read_symlink("/proc/self/exe", error);
  if (error)
    return testing::AssertionFailure()
           << "Cannot locate running test executable: " << error.message();
  // Relocated tests must use their package even while the build tree exists.
  const std::filesystem::path cli =
      executable == RJ_CODE_MUTATE_TEST_PATH
          ? std::filesystem::path(RJ_CODE_MUTATE_PATH)
          : (executable.parent_path() / RJ_INSTALLED_CODE_MUTATE_PATH).lexically_normal();
  if (!std::filesystem::is_regular_file(cli, error))
    return testing::AssertionFailure() << "Mutation CLI is not a regular file: " << cli
                                       << (error ? ": " + error.message() : "");
  path = cli.string();
  return testing::AssertionSuccess();
}

} // namespace rocjitsu::test
