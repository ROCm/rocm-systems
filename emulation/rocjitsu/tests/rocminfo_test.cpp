// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rocminfo_test.cpp
/// @brief Verifies that rocminfo runs successfully against the simulated GPU
///        via the rocjitsu CLI launcher and reports expected topology.

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

struct ProcessResult {
  std::string output;
  int exit_code;
};

ProcessResult run_command(const std::string &cmd) {
  ProcessResult result;
  std::array<char, 4096> buf;
  std::string full_cmd = cmd + " 2>&1";

  FILE *pipe = popen(full_cmd.c_str(), "r");
  if (!pipe) {
    result.exit_code = -1;
    return result;
  }

  while (fgets(buf.data(), buf.size(), pipe) != nullptr)
    result.output += buf.data();

  int status = pclose(pipe);
  result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  return result;
}

const ProcessResult &rocminfo_output() {
  static const ProcessResult result = run_command(std::string(ROCJITSU_BIN) + " --config " +
                                                  RJ_CONFIG_PATH + " -- " + ROCMINFO_PATH);
  return result;
}

TEST(RocminfoTest, ExitsSuccessfully) {
  EXPECT_EQ(rocminfo_output().exit_code, 0)
      << "rocminfo failed with exit code " << rocminfo_output().exit_code << "\nOutput:\n"
      << rocminfo_output().output;
}

TEST(RocminfoTest, InterposerActive) {
  EXPECT_NE(rocminfo_output().output.find("gfx950"), std::string::npos)
      << "Interposer does not appear to be active (no gfx950 agent).\nOutput:\n"
      << rocminfo_output().output;
}

TEST(RocminfoTest, HsaSystemAttributes) {
  ASSERT_EQ(rocminfo_output().exit_code, 0) << rocminfo_output().output;
  EXPECT_NE(rocminfo_output().output.find("HSA System Attributes"), std::string::npos)
      << "rocminfo did not report HSA System Attributes.\nOutput:\n"
      << rocminfo_output().output;
}

TEST(RocminfoTest, DetectsGpuAgent) {
  ASSERT_EQ(rocminfo_output().exit_code, 0) << rocminfo_output().output;
  EXPECT_NE(rocminfo_output().output.find("Device Type:             GPU"), std::string::npos)
      << "rocminfo did not detect a GPU agent.\nOutput:\n"
      << rocminfo_output().output;
}

TEST(RocminfoTest, ReportsGfx950) {
  ASSERT_EQ(rocminfo_output().exit_code, 0) << rocminfo_output().output;
  EXPECT_NE(rocminfo_output().output.find("gfx950"), std::string::npos)
      << "rocminfo did not report gfx950 target.\nOutput:\n"
      << rocminfo_output().output;
}

TEST(RocminfoTest, ReportsWavefrontSize) {
  ASSERT_EQ(rocminfo_output().exit_code, 0) << rocminfo_output().output;
  EXPECT_NE(rocminfo_output().output.find("Wavefront Size:"), std::string::npos)
      << "rocminfo did not report Wavefront Size.\nOutput:\n"
      << rocminfo_output().output;
}

} // namespace
