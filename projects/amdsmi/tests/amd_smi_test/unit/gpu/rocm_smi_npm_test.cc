// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hardware-free unit tests for the NPM board helpers in rocm_smi_npm.cc, run
// against a temp directory standing in for the sysfs board/ directory.

#include "rocm_smi/rocm_smi_npm.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

// RAII helper: creates a private (0700) temp directory and removes it on destruction.
class TempBoardDir {
 public:
  TempBoardDir() {
    std::string tmpl = (fs::temp_directory_path() / "amdsmi_npm_test_XXXXXX").string();
    if (::mkdtemp(tmpl.data()) == nullptr) {
      ADD_FAILURE() << "mkdtemp failed: " << std::strerror(errno);
      return;
    }
    path_ = tmpl;
  }

  ~TempBoardDir() {
    if (path_.empty()) return;
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
  TempBoardDir parent;
  fs::path missing = parent.path() / "missing";

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
  TempBoardDir parent;
  fs::path missing = parent.path() / "missing";

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
  TempBoardDir parent;
  fs::path missing = parent.path() / "missing";

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

TEST(GpuUnit, GetNpmBoardMaxLimitLeadingSpaceParses) {
  TempBoardDir board;
  board.WriteFile("max_node_power_limit", " 6400");

  uint64_t limit = 0;
  EXPECT_EQ(amd::smi::get_npm_board_max_limit(board.path().string(), &limit), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(limit, 6400u);
}

// ---------------------------------------------------------------------
// validate_npm_board_limit()
// ---------------------------------------------------------------------

TEST(GpuUnit, ValidateNpmBoardLimitEnabledInRangeIsSuccess) {
  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");

  EXPECT_EQ(amd::smi::validate_npm_board_limit(board.path().string(), 250), RSMI_STATUS_SUCCESS);
}

TEST(GpuUnit, ValidateNpmBoardLimitAtMaxIsSuccess) {
  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");

  EXPECT_EQ(amd::smi::validate_npm_board_limit(board.path().string(), 6400), RSMI_STATUS_SUCCESS);
}

TEST(GpuUnit, ValidateNpmBoardLimitDisabledIsInvalidArgs) {
  TempBoardDir board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("max_node_power_limit", "6400");

  EXPECT_EQ(amd::smi::validate_npm_board_limit(board.path().string(), 250),
            RSMI_STATUS_INVALID_ARGS);
}

TEST(GpuUnit, ValidateNpmBoardLimitMissingStatusIsNotSupported) {
  TempBoardDir board;
  board.WriteFile("max_node_power_limit", "6400");

  EXPECT_EQ(amd::smi::validate_npm_board_limit(board.path().string(), 250),
            RSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, ValidateNpmBoardLimitMissingMaxIsNotSupported) {
  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");

  EXPECT_EQ(amd::smi::validate_npm_board_limit(board.path().string(), 250),
            RSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, ValidateNpmBoardLimitNegativeMaxIsUnexpectedData) {
  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "-1");

  EXPECT_EQ(amd::smi::validate_npm_board_limit(board.path().string(), 250),
            RSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, ValidateNpmBoardLimitU64MaxIsUnexpectedData) {
  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "18446744073709551615");

  EXPECT_EQ(amd::smi::validate_npm_board_limit(board.path().string(), 250),
            RSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, ValidateNpmBoardLimitZeroIsInvalidArgs) {
  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");

  EXPECT_EQ(amd::smi::validate_npm_board_limit(board.path().string(), 0), RSMI_STATUS_INVALID_ARGS);
}

TEST(GpuUnit, ValidateNpmBoardLimitOverMaxIsInvalidArgs) {
  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");

  EXPECT_EQ(amd::smi::validate_npm_board_limit(board.path().string(), 6401),
            RSMI_STATUS_INVALID_ARGS);
}
