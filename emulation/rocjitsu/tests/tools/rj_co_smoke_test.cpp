// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_co_smoke_test.cpp
/// @brief End-to-end smoke test for the rj_co command-line tool.

#include "waitcheck_fixture.h"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

std::filesystem::path g_code_object_tool;

struct TempDir {
  std::filesystem::path path;

  explicit TempDir(std::filesystem::path temp_path) : path(std::move(temp_path)) {
    std::filesystem::create_directories(path);
  }

  ~TempDir() { std::filesystem::remove_all(path); }
};

std::string shell_quote(std::string_view text) {
  std::string quoted = "'";
  for (const char ch : text) {
    if (ch == '\'')
      quoted += "'\\''";
    else
      quoted += ch;
  }
  quoted += "'";
  return quoted;
}

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool contains(std::string_view haystack, std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
}

bool command_succeeded(int status) {
  return status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

bool write_binary_file(const std::filesystem::path &path, const std::vector<uint8_t> &contents) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(contents.data()),
            static_cast<std::streamsize>(contents.size()));
  return out.good();
}

std::pair<std::string, std::string> run_tool(const std::filesystem::path &input,
                                             std::string_view options) {
  const TempDir temp_dir(std::filesystem::temp_directory_path() /
                         ("rj_co_smoke_" + std::to_string(static_cast<long long>(getpid()))));
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  const std::string command = shell_quote(g_code_object_tool.string()) + " " +
                              shell_quote(input.string()) + " " + std::string(options) + " > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());
  const int status = std::system(command.c_str());
  std::string stdout_text = read_text_file(output);
  std::string stderr_text = read_text_file(error);
  EXPECT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  return {std::move(stdout_text), std::move(stderr_text)};
}

} // namespace

TEST(RjCo, ListsKernelsAndMapsTextLocations) {
  const TempDir temp_dir(std::filesystem::temp_directory_path() /
                         ("rj_co_input_" + std::to_string(static_cast<long long>(getpid()))));
  const auto input = temp_dir.path / "missing_wait_gfx1200.co";
  ASSERT_TRUE(
      write_binary_file(input, rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));

  auto [stdout_text, stderr_text] =
      run_tool(input, "--target gfx1200 --list-kernels --map .text+0xc");

  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  const std::array<std::string_view, 6> expected = {
      "target gfx1200[0]",
      "section .text size=0x10 va=0x1100 file-offset=0x100",
      "kernel waitcheck entry=.text+0x0 va=0x1100 file-offset=0x100 size=0x10",
      ".text+0xc: section=.text section-offset=0xc va=0x110c file-offset=0x10c",
      "kernel=waitcheck+0xc",
      "gfx1200",
  };
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjCo, GroupsWaitcheckDiagnosticsByKernel) {
  const TempDir temp_dir(std::filesystem::temp_directory_path() /
                         ("rj_co_input_" + std::to_string(static_cast<long long>(getpid()))));
  const auto input = temp_dir.path / "missing_wait_gfx1200.co";
  ASSERT_TRUE(
      write_binary_file(input, rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));

  auto [stdout_text, stderr_text] = run_tool(input, "--target gfx1200 --waitcheck");

  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  const std::array<std::string_view, 4> expected = {
      "waitcheck gfx1200[0]: instructions=2 memory-events=1 diagnostics=1",
      "kernel waitcheck diagnostics=1",
      "global_load_b32 -> v_mov_b32_e32: 1",
      "diagnostics=1",
  };
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjCo, EmitsMarkdownReproForDiagnostic) {
  const TempDir temp_dir(std::filesystem::temp_directory_path() /
                         ("rj_co_input_" + std::to_string(static_cast<long long>(getpid()))));
  const auto input = temp_dir.path / "missing_wait_gfx1200.co";
  ASSERT_TRUE(
      write_binary_file(input, rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));

  auto [stdout_text, stderr_text] =
      run_tool(input, "--target gfx1200 --repro-diagnostic 0 --max-diagnostics 1");

  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  const std::array<std::string_view, 6> expected = {
      "### waitcheck diagnostic 0",
      "rj_waitcheck ",
      "--target gfx1200 --code-object-index 0",
      "Kernel: `waitcheck`",
      "producer .text+0x0: global_load_b32",
      "consumer .text+0xc: v_mov_b32_e32",
  };
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    return 2;
  }

  g_code_object_tool = argv[1];
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
