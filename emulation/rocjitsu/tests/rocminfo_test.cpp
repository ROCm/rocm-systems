// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rocminfo_test.cpp
/// @brief Verifies that rocminfo runs successfully against the simulated GPU
///        via the rocjitsu CLI launcher and reports expected topology.

#include <gtest/gtest.h>

#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace {

struct ProcessResult {
  std::string output;
  int exit_code;
};

struct TestPaths {
  std::string rocjitsu_bin = ROCJITSU_BIN;
  std::string config_path = RJ_CONFIG_PATH;
};

std::filesystem::path resolve_relative_to_exe(const std::filesystem::path &exe_dir,
                                              const char *path) {
  std::filesystem::path candidate(path);
  if (!candidate.is_absolute())
    candidate = exe_dir / candidate;
  return candidate.lexically_normal();
}

std::filesystem::path current_exe_dir() {
  std::error_code ec;
  std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec)
    return {};
  return exe.parent_path();
}

bool installed_paths_exist(const TestPaths &paths) {
  return std::filesystem::exists(paths.rocjitsu_bin) && std::filesystem::exists(paths.config_path);
}

TestPaths installed_paths(const std::filesystem::path &exe_dir) {
  return {
      resolve_relative_to_exe(exe_dir, RJ_INSTALLED_ROCJITSU_BIN).string(),
      resolve_relative_to_exe(exe_dir, RJ_INSTALLED_CONFIG_PATH).string(),
  };
}

const TestPaths &test_paths() {
  static TestPaths paths = [] {
    std::filesystem::path exe_dir = current_exe_dir();
    if (!exe_dir.empty()) {
      TestPaths installed = installed_paths(exe_dir);
      if (installed_paths_exist(installed))
        return installed;
    }
    return TestPaths{};
  }();
  return paths;
}

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
  const TestPaths &paths = test_paths();
  const char *rocminfo_path = std::getenv("ROCMINFO_PATH");
  static const ProcessResult result =
      run_command(paths.rocjitsu_bin + " --config " + paths.config_path + " -- " +
                  (rocminfo_path ? rocminfo_path : ROCMINFO_PATH));
  return result;
}

// Removes surrounding horizontal whitespace from a rocminfo field or line.
std::string_view trim(std::string_view value) {
  const size_t first = value.find_first_not_of(" \t\r");
  if (first == std::string_view::npos)
    return {};
  const size_t last = value.find_last_not_of(" \t\r");
  return value.substr(first, last - first + 1);
}

// Returns the value of an exactly labeled field within one rocminfo agent block.
std::optional<std::string_view> agent_field(std::string_view agent, std::string_view label) {
  size_t line_start = 0;
  while (line_start < agent.size()) {
    const size_t line_end = agent.find('\n', line_start);
    std::string_view line = trim(agent.substr(line_start, line_end - line_start));
    if (line.size() > label.size() && line.compare(0, label.size(), label) == 0 &&
        line[label.size()] == ':') {
      return trim(line.substr(label.size() + 1));
    }
    if (line_end == std::string_view::npos)
      break;
    line_start = line_end + 1;
  }
  return std::nullopt;
}

// Finds the rocminfo agent block identified as a gfx950 GPU.
std::optional<std::string_view> gfx950_gpu_agent(std::string_view output) {
  constexpr std::string_view kAgentHeading = "\nAgent ";
  size_t agent_start = output.find(kAgentHeading);
  while (agent_start != std::string_view::npos) {
    const size_t next_agent = output.find(kAgentHeading, agent_start + kAgentHeading.size());
    std::string_view agent = output.substr(agent_start, next_agent - agent_start);
    const auto name = agent_field(agent, "Name");
    const auto device_type = agent_field(agent, "Device Type");
    if (name && device_type && *name == "gfx950" && *device_type == "GPU") {
      return agent;
    }
    agent_start = next_agent;
  }
  return std::nullopt;
}

// Parses an entire rocminfo agent field as an unsigned decimal integer.
std::optional<uint64_t> decimal_agent_field(std::string_view agent, std::string_view label) {
  const auto text = agent_field(agent, label);
  if (!text)
    return std::nullopt;

  uint64_t value = 0;
  const auto [end, error] = std::from_chars(text->data(), text->data() + text->size(), value);
  if (error != std::errc{} || end != text->data() + text->size())
    return std::nullopt;
  return value;
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

TEST(RocminfoTest, ReportsMi350xActiveComputeUnits) {
  ASSERT_EQ(rocminfo_output().exit_code, 0) << rocminfo_output().output;
  const auto agent = gfx950_gpu_agent(rocminfo_output().output);
  ASSERT_TRUE(agent) << "rocminfo did not report a gfx950 GPU agent.\nOutput:\n"
                     << rocminfo_output().output;
  const auto compute_units = decimal_agent_field(*agent, "Compute Unit");
  ASSERT_TRUE(compute_units)
      << "rocminfo did not report a numeric GPU Compute Unit field.\nOutput:\n"
      << rocminfo_output().output;
  EXPECT_EQ(*compute_units, 256u);
}

TEST(RocminfoTest, ReportsMi350xMaxClockFrequency) {
  ASSERT_EQ(rocminfo_output().exit_code, 0) << rocminfo_output().output;
  const auto agent = gfx950_gpu_agent(rocminfo_output().output);
  ASSERT_TRUE(agent) << "rocminfo did not report a gfx950 GPU agent.\nOutput:\n"
                     << rocminfo_output().output;
  const auto max_clock = decimal_agent_field(*agent, "Max Clock Freq. (MHz)");
  ASSERT_TRUE(max_clock) << "rocminfo did not report a numeric GPU maximum clock field.\nOutput:\n"
                         << rocminfo_output().output;
  EXPECT_EQ(*max_clock, 2200u);
}

} // namespace
