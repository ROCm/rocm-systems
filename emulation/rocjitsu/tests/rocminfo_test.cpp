// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rocminfo_test.cpp
/// @brief Verifies that rocminfo runs successfully against the simulated GPU
///        via the LD_PRELOAD KMD interposer and reports expected topology.
///
/// Requires LD_PRELOAD=librocjitsu_kmd.so, RJ_CONFIG/RJ_SCHEMA env vars,
/// and the rocminfo binary installed (typically at /opt/rocm/bin/rocminfo).

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

/// Run a command and capture its combined stdout+stderr and exit code.
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

/// Path to rocminfo, injected via CMake compile definition ROCMINFO_PATH.
const char *rocminfo_path() { return ROCMINFO_PATH; }

/// Lazily-initialised singleton so rocminfo is only invoked once.
const ProcessResult &rocminfo_output() {
  static const ProcessResult result = run_command(rocminfo_path());
  return result;
}

// ---------------------------------------------------------------------------

TEST(RocminfoTest, ExitsSuccessfully) {
  EXPECT_EQ(rocminfo_output().exit_code, 0)
      << "rocminfo failed with exit code " << rocminfo_output().exit_code << "\nOutput:\n"
      << rocminfo_output().output;
}

TEST(RocminfoTest, InterposerActive) {
  // The interposer is active if rocminfo sees a GPU agent. The simulated
  // GPU reports as gfx950, which only appears when the interposer is
  // providing the KFD topology.
  EXPECT_NE(rocminfo_output().output.find("gfx950"), std::string::npos)
      << "LD_PRELOAD interposer does not appear to be active (no gfx950 agent).\nOutput:\n"
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
