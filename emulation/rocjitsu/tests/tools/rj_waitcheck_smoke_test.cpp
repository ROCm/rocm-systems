// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_waitcheck_smoke_test.cpp
/// @brief End-to-end smoke test for the rj_waitcheck command-line tool.

#include "waitcheck_fixture.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

std::filesystem::path g_waitcheck_tool;

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

bool command_exited_with(int status, int code) {
  return status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == code;
}

bool command_succeeded(int status) { return command_exited_with(status, 0); }

bool write_binary_file(const std::filesystem::path &path, const std::vector<uint8_t> &contents) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(contents.data()),
            static_cast<std::streamsize>(contents.size()));
  return out.good();
}

} // namespace

TEST(RjWaitcheck, ReportsMissingWait) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto input = temp_dir.path / "missing_wait_gfx1200.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(
      write_binary_file(input, rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(input.string()) + " --target gfx1200 > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_exited_with(status, 4)) << "stderr:\n"
                                              << stderr_text << "\nstdout:\n"
                                              << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 4> expected = {
      "rj_waitcheck:", "diagnostics=1", "missing s_wait_loadcnt <= 0", "consumer .text+"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, AnalyzesGfx1250CodeObject) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto input = temp_dir.path / "vmax_u64_gfx1250.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(
      write_binary_file(input, rocjitsu::waitcheck_test::make_gfx1250_vmax_u64_code_object()));

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(input.string()) + " --target gfx1250 > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 2> expected = {
      "vmax_u64_gfx1250.co:gfx1250[0]: instructions=1 memory-events=0 diagnostics=0",
      "diagnostics=0"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, MaxDiagnosticsZeroKeepsHazardStatusWithoutDetails) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto input = temp_dir.path / "missing_wait_gfx1200.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(
      write_binary_file(input, rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(input.string()) +
                              " --target gfx1200 --max-diagnostics 0 --no-fail > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text, "diagnostics=>=1")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "omitted at least 1 diagnostic(s) after limit")) << stdout_text;
  EXPECT_FALSE(contains(stdout_text, "missing s_wait_loadcnt")) << stdout_text;
}

TEST(RjWaitcheck, BatchSkipsUnsupportedInputs) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto missing = temp_dir.path / "missing_wait_gfx1200.co";
  const auto correct = temp_dir.path / "correct_wait_gfx1200.co";
  const auto unsupported = temp_dir.path / "unsupported.txt";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(write_binary_file(missing,
                                rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));
  ASSERT_TRUE(write_binary_file(correct,
                                rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));
  {
    std::ofstream out(unsupported);
    out << "not an ELF";
  }

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(missing.string()) + " " + shell_quote(correct.string()) +
                              " " + shell_quote(unsupported.string()) +
                              " --all-code-objects --skip-unsupported --no-fail > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 5> expected = {
      "missing_wait_gfx1200.co:gfx1200[0]: instructions=2 memory-events=2 diagnostics=1",
      "correct_wait_gfx1200.co:gfx1200[0]: instructions=3 memory-events=2 diagnostics=0",
      "unsupported.txt: skipped: failed to parse input executable or code object",
      "missing s_wait_loadcnt <= 0",
      "rj_waitcheck: scanned inputs=3 skipped=1 code-objects=2 diagnostics=1"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, BatchSkipsUndecodableCodeObjects) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto invalid = temp_dir.path / "invalid_inst_gfx1200.co";
  const auto correct = temp_dir.path / "correct_wait_gfx1200.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(write_binary_file(
      invalid, rocjitsu::waitcheck_test::make_gfx1200_invalid_instruction_code_object()));
  ASSERT_TRUE(write_binary_file(correct,
                                rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(invalid.string()) + " " + shell_quote(correct.string()) +
                              " --all-code-objects --skip-unsupported --no-fail > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 4> expected = {
      "invalid_inst_gfx1200.co: skipped: waitcheck analysis failed for gfx1200[0]: "
      "decode failed while building CFG: Invalid instruction opcode: 800000",
      "correct_wait_gfx1200.co:gfx1200[0]: instructions=3 memory-events=2 diagnostics=0",
      "rj_waitcheck: scanned inputs=2 skipped=1 code-objects=1 diagnostics=0", "diagnostics=0"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, RecursiveDirectorySweepScansNestedCorpusFiles) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto corpus = temp_dir.path / "corpus";
  const auto nested = corpus / "nested";
  const auto missing = nested / "missing_wait_gfx1200.co";
  const auto correct = corpus / "correct_wait_gfx1200.co";
  const auto unsupported = nested / "unsupported.txt";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";
  std::filesystem::create_directories(nested);

  ASSERT_TRUE(write_binary_file(missing,
                                rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));
  ASSERT_TRUE(write_binary_file(correct,
                                rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));
  {
    std::ofstream out(unsupported);
    out << "not an ELF";
  }

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(corpus.string()) +
                              " --recursive --all-code-objects --skip-unsupported --no-fail > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 5> expected = {
      "correct_wait_gfx1200.co:gfx1200[0]: instructions=3 memory-events=2 diagnostics=0",
      "missing_wait_gfx1200.co:gfx1200[0]: instructions=2 memory-events=2 diagnostics=1",
      "unsupported.txt: skipped: failed to parse input executable or code object",
      "missing s_wait_loadcnt <= 0",
      "rj_waitcheck: scanned inputs=3 skipped=1 code-objects=2 diagnostics=1"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, SummaryOnlyRecursiveSweepReportsLowerBoundTotals) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto corpus = temp_dir.path / "corpus";
  const auto nested = corpus / "nested";
  const auto missing = nested / "missing_wait_gfx1200.co";
  const auto correct = corpus / "correct_wait_gfx1200.co";
  const auto unsupported = nested / "unsupported.txt";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";
  std::filesystem::create_directories(nested);

  ASSERT_TRUE(write_binary_file(missing,
                                rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));
  ASSERT_TRUE(write_binary_file(correct,
                                rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));
  {
    std::ofstream out(unsupported);
    out << "not an ELF";
  }

  const std::string command =
      shell_quote(g_waitcheck_tool.string()) + " " + shell_quote(corpus.string()) +
      " --recursive --all-code-objects --skip-unsupported --no-fail --max-diagnostics 0 "
      "--stop-after-first-diagnostic --summary-only > " +
      shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text,
                       "rj_waitcheck: scanned inputs=3 skipped=1 code-objects=2 diagnostics=>=1"))
      << stdout_text;
  EXPECT_FALSE(contains(stdout_text, "missing_wait_gfx1200.co:gfx1200")) << stdout_text;
  EXPECT_FALSE(contains(stdout_text, "unsupported.txt: skipped")) << stdout_text;
  EXPECT_FALSE(contains(stdout_text, "missing s_wait_loadcnt")) << stdout_text;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: rj_waitcheck_smoke_test <rj_waitcheck>\n";
    return 2;
  }

  g_waitcheck_tool = argv[1];
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
