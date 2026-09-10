// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hardware-free unit tests for the NPM (Node Power Management) helpers in
// rocm_smi_npm.cc: set_npm_board_limit() and get_npm_node_power().
//
// These functions take an arbitrary board_path string and operate on plain
// files underneath it (they do not require /sys or any specific sysfs
// layout), so we exercise them against a real temporary directory created
// via std::filesystem instead of mocking /sys. This mirrors the existing
// (unexercised prior to this change) get_npm_board_limit()/
// get_npm_board_status() logic in the same translation unit, all of which
// share the same "board_path directory + single-value file" convention.

#include "rocm_smi/rocm_smi_npm.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

// RAII helper: creates a unique temp directory for the duration of a test
// and removes it (recursively) on destruction.
class TempBoardDir {
 public:
  TempBoardDir() {
    path_ = fs::temp_directory_path() /
            fs::path("amdsmi_npm_test_" +
                     std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                     std::to_string(reinterpret_cast<uintptr_t>(this)));
    fs::create_directories(path_);
  }

  ~TempBoardDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }

  const fs::path& path() const { return path_; }

  void WriteFile(const std::string& filename, const std::string& contents) const {
    std::ofstream ofs(path_ / filename);
    ofs << contents;
  }

  std::string ReadFile(const std::string& filename) const {
    std::ifstream ifs(path_ / filename);
    std::string line;
    std::getline(ifs, line);
    return line;
  }

 private:
  fs::path path_;
};

}  // namespace

// ---------------------------------------------------------------------
// set_npm_board_limit()
// ---------------------------------------------------------------------

TEST(GpuUnit, SetNpmBoardLimitEmptyPathIsInvalidArgs) {
  EXPECT_EQ(amd::smi::set_npm_board_limit("", 100), RSMI_STATUS_INVALID_ARGS);
}

TEST(GpuUnit, SetNpmBoardLimitMissingBoardDirIsNotSupported) {
  fs::path missing = fs::temp_directory_path() / "amdsmi_npm_test_definitely_missing_dir_xyz";
  std::error_code ec;
  fs::remove_all(missing, ec);  // ensure it really doesn't exist

  EXPECT_EQ(amd::smi::set_npm_board_limit(missing.string(), 100), RSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmBoardLimitMissingLimitFileIsNotSupported) {
  TempBoardDir board;
  // Deliberately do not create cur_node_power_limit.
  EXPECT_EQ(amd::smi::set_npm_board_limit(board.path().string(), 100), RSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmBoardLimitSuccessWritesValue) {
  TempBoardDir board;
  board.WriteFile("cur_node_power_limit", "0");

  EXPECT_EQ(amd::smi::set_npm_board_limit(board.path().string(), 250), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "250");
}

TEST(GpuUnit, SetNpmBoardLimitSuccessOverwritesPreviousValue) {
  TempBoardDir board;
  board.WriteFile("cur_node_power_limit", "999");

  EXPECT_EQ(amd::smi::set_npm_board_limit(board.path().string(), 42), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "42");
}

TEST(GpuUnit, SetNpmBoardLimitDirectoryInPlaceOfFileIsNotSupported) {
  TempBoardDir board;
  // Create cur_node_power_limit as a directory, not a regular file, to
  // exercise the is_regular_file() guard.
  fs::create_directory(board.path() / "cur_node_power_limit");

  EXPECT_EQ(amd::smi::set_npm_board_limit(board.path().string(), 100), RSMI_STATUS_NOT_SUPPORTED);
}

// ---------------------------------------------------------------------
// get_npm_node_power()
// ---------------------------------------------------------------------

TEST(GpuUnit, GetNpmNodePowerNullOutPtrIsInvalidArgs) {
  TempBoardDir board;
  board.WriteFile("node_power", "123");
  EXPECT_EQ(amd::smi::get_npm_node_power(board.path().string(), nullptr), RSMI_STATUS_INVALID_ARGS);
}

TEST(GpuUnit, GetNpmNodePowerEmptyPathIsInvalidArgs) {
  uint64_t power = 0;
  EXPECT_EQ(amd::smi::get_npm_node_power("", &power), RSMI_STATUS_INVALID_ARGS);
}

TEST(GpuUnit, GetNpmNodePowerMissingBoardDirIsNotSupported) {
  fs::path missing = fs::temp_directory_path() / "amdsmi_npm_test_definitely_missing_dir_pwr";
  std::error_code ec;
  fs::remove_all(missing, ec);

  uint64_t power = 0;
  EXPECT_EQ(amd::smi::get_npm_node_power(missing.string(), &power), RSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, GetNpmNodePowerMissingFileIsNotSupported) {
  TempBoardDir board;
  // Deliberately do not create node_power.
  uint64_t power = 0;
  EXPECT_EQ(amd::smi::get_npm_node_power(board.path().string(), &power), RSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, GetNpmNodePowerSuccessReadsValue) {
  TempBoardDir board;
  board.WriteFile("node_power", "777");

  uint64_t power = 0;
  EXPECT_EQ(amd::smi::get_npm_node_power(board.path().string(), &power), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(power, 777u);
}

TEST(GpuUnit, GetNpmNodePowerNonNumericContentsIsUnexpectedData) {
  TempBoardDir board;
  board.WriteFile("node_power", "not-a-number");

  uint64_t power = 0;
  EXPECT_EQ(amd::smi::get_npm_node_power(board.path().string(), &power),
            RSMI_STATUS_UNEXPECTED_DATA);
}

// ---------------------------------------------------------------------
// get_npm_board_max_limit()
// ---------------------------------------------------------------------

TEST(GpuUnit, GetNpmBoardMaxLimitNullOutPtrIsInvalidArgs) {
  TempBoardDir board;
  board.WriteFile("max_node_power_limit", "6400");
  EXPECT_EQ(amd::smi::get_npm_board_max_limit(board.path().string(), nullptr),
            RSMI_STATUS_INVALID_ARGS);
}

TEST(GpuUnit, GetNpmBoardMaxLimitEmptyPathIsInvalidArgs) {
  uint64_t limit = 0;
  EXPECT_EQ(amd::smi::get_npm_board_max_limit("", &limit), RSMI_STATUS_INVALID_ARGS);
}

TEST(GpuUnit, GetNpmBoardMaxLimitMissingBoardDirIsNotSupported) {
  fs::path missing = fs::temp_directory_path() / "amdsmi_npm_test_definitely_missing_dir_max";
  std::error_code ec;
  fs::remove_all(missing, ec);

  uint64_t limit = 0;
  EXPECT_EQ(amd::smi::get_npm_board_max_limit(missing.string(), &limit), RSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, GetNpmBoardMaxLimitMissingFileIsNotSupported) {
  TempBoardDir board;
  // Deliberately do not create max_node_power_limit.
  uint64_t limit = 0;
  EXPECT_EQ(amd::smi::get_npm_board_max_limit(board.path().string(), &limit),
            RSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, GetNpmBoardMaxLimitSuccessReadsValue) {
  TempBoardDir board;
  board.WriteFile("max_node_power_limit", "6400");

  uint64_t limit = 0;
  EXPECT_EQ(amd::smi::get_npm_board_max_limit(board.path().string(), &limit), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(limit, 6400u);
}

TEST(GpuUnit, GetNpmBoardMaxLimitNonNumericContentsIsUnexpectedData) {
  TempBoardDir board;
  board.WriteFile("max_node_power_limit", "not-a-number");

  uint64_t limit = 0;
  EXPECT_EQ(amd::smi::get_npm_board_max_limit(board.path().string(), &limit),
            RSMI_STATUS_UNEXPECTED_DATA);
}
