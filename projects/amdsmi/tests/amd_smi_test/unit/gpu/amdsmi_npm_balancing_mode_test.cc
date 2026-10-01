// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for amdsmi_get_npm_balancing_mode()/amdsmi_set_npm_balancing_mode()
// (amd_smi.cc) and the underlying rsmi_dev_npm_balancing_mode_get()/set()
// (rocm_smi.cc).
//
// Like amdsmi_set_npm_limit(), these take an amdsmi_node_handle that is
// internally a `std::string*` board-path pointer normally vended by
// amdsmi_get_node_handle(). This worktree does not gate node_handle
// dereference on a registered-handle check (no
// amdsmi_test_register_node_handle()/is_registered_node_handle() exists
// here), so tests fabricate the handle directly by reinterpret_cast'ing the
// address of a local std::string, bypassing only the acquisition helper --
// not the function under test.
//
// rsmi_dev_npm_balancing_mode_set() applies REQUIRE_ROOT_ACCESS before
// touching any file, so:
//   - non-root callers always observe AMDSMI_STATUS_NO_PERM, verifiable
//     without root and asserted unconditionally below.
//   - success / not-supported branches (reached only past the root gate)
//     require root and are gated with GTEST_SKIP_ following the existing
//     is_sudo_user() convention used throughout tests/amd_smi_test/main.cc.
// Since set_npm_board_mode()/get_npm_board_mode() operate on an arbitrary
// board_path string (not hardcoded to /sys), the root-gated cases use a
// plain std::filesystem temp directory instead of real /sys mocking.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rocm_smi/rocm_smi_utils.h"

namespace fs = std::filesystem;

namespace {

// RAII helper identical in spirit to kfd_numa_node_test.cc's ScopedTempDir:
// mkdtemp()-based (atomic, 0700, fails if the name already exists) rather
// than fs::create_directories() on a predictable path, which would succeed
// silently through a pre-planted symlink.
class TempBoardDir {
 public:
  TempBoardDir() {
    std::string tmpl =
        (fs::temp_directory_path() / "amdsmi_npm_balancing_mode_test_XXXXXX").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (char* result = mkdtemp(buf.data())) {
      path_ = result;
    }
  }
  ~TempBoardDir() {
    if (!path_.empty()) {
      std::error_code ec;
      fs::remove_all(path_, ec);
    }
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

// Small RAII wrapper around amdsmi_init()/amdsmi_shut_down() so each test is
// self-contained regardless of test execution order.
class ScopedAmdSmiInit {
 public:
  ScopedAmdSmiInit() { status_ = amdsmi_init(AMDSMI_INIT_AMD_GPUS); }
  ~ScopedAmdSmiInit() {
    if (status_ == AMDSMI_STATUS_SUCCESS) {
      amdsmi_shut_down();
    }
  }
  amdsmi_status_t status() const { return status_; }

 private:
  amdsmi_status_t status_;
};

}  // namespace

// ---------------------------------------------------------------------
// amdsmi_get_npm_balancing_mode()
// ---------------------------------------------------------------------

TEST(GpuUnit, GetNpmBalancingModeNullHandleIsInval) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(nullptr, &mode), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmBalancingModeNullModeIsInval) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  std::string board_path = "/tmp/amdsmi_npm_balancing_mode_test_null_mode_probe";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);

  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmBalancingModeDisabledStillReportsLastSelectedMode) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("npm_mode", "1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_INVALID;
  // Get is never gated on NPM enablement; it reflects board/npm_mode even
  // while NPM is disabled.
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
}

TEST(GpuUnit, GetNpmBalancingModeEnabledDefaultsToPowerBalancing) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  // The platform/driver creates board/npm_mode with its default value ("1")
  // when NPM is enabled; simulate that here rather than leaving the file
  // absent, since a genuinely missing/unreadable file now reports
  // AMDSMI_STATUS_SUCCESS + AMDSMI_NPM_BALANCING_MODE_INVALID (N/A).
  board.WriteFile("npm_mode", "1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_INVALID;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
}

TEST(GpuUnit, GetNpmBalancingModeEnabledMissingModeFileIsInvalidMode) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  // Deliberately do not create board/npm_mode: the sysfs node absent/
  // unreadable must report N/A (INVALID), not a fabricated PB, and must
  // agree with amdsmi_set_npm_balancing_mode()'s NOT_SUPPORTED for the
  // identical state (see SetNpmBalancingModeRootMissingModeFileIsNotSupported).
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_INVALID);
}

TEST(GpuUnit, GetNpmBalancingModeGarbageValueIsUnexpectedData) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("npm_mode", "0");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);

  // "0" is not a valid encoding ("1"/"2" only); this is a genuine
  // read/decode error, not the "missing file" N/A case.
  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, GetNpmBalancingModeEnabledReadsFrequencyBalancing) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("npm_mode", "2");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_INVALID;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
}

// ---------------------------------------------------------------------
// amdsmi_set_npm_balancing_mode()
// ---------------------------------------------------------------------

TEST(GpuUnit, SetNpmBalancingModeNullHandleIsInval) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(nullptr, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmBalancingModeInvalidModeIsInval) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  std::string board_path = "/tmp/amdsmi_npm_balancing_mode_test_invalid_mode_probe";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, static_cast<amdsmi_npm_balancing_mode_t>(
                                                      AMDSMI_NPM_BALANCING_MODE_INVALID)),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmBalancingModeNonRootIsNoPerm) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (amd::smi::is_sudo_user()) {
    GTEST_SKIP_(
        "Running as root: REQUIRE_ROOT_ACCESS gate cannot be exercised here; "
        "see SetNpmBalancingModeRoot* tests instead");
  }

  // Any non-empty board path is sufficient: REQUIRE_ROOT_ACCESS in
  // rsmi_dev_npm_balancing_mode_set() rejects before the path is ever
  // touched.
  std::string board_path = "/tmp/amdsmi_npm_balancing_mode_test_nonroot_probe";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_NO_PERM);
}

TEST(GpuUnit, SetNpmBalancingModeRootRejectsWhenNpmDisabled) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("npm_mode", "1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);

  // Must not even open board/npm_mode for write when NPM is disabled: this is
  // AMDSMI_STATUS_NOT_SUPPORTED, a deliberate divergence from
  // amdsmi_set_npm_limit()'s own disabled-check (AMDSMI_STATUS_INVAL).
  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_NOT_SUPPORTED);
  // The file must be untouched.
  EXPECT_EQ(board.ReadFile("npm_mode"), "1");
}

TEST(GpuUnit, SetNpmBalancingModeRootRoundTrip) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("npm_mode", "1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("npm_mode"), "2");

  amdsmi_npm_balancing_mode_t readback = AMDSMI_NPM_BALANCING_MODE_INVALID;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &readback), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(readback, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("npm_mode"), "1");

  readback = AMDSMI_NPM_BALANCING_MODE_INVALID;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &readback), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(readback, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
}

TEST(GpuUnit, SetNpmBalancingModeRootMissingModeFileIsNotSupported) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  // Deliberately do not create board/npm_mode: set_npm_board_mode() requires
  // the file to already exist (it does not create it), mirroring
  // set_npm_board_limit()'s missing-file handling.
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmBalancingModeRootMissingBoardDirIsNotSupported) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  // Get a unique, mkdtemp()-allocated path (unpredictable, not plantable in
  // advance), then remove it so the path itself is missing for the test --
  // avoids a fixed/predictable name for the root-privileged remove_all().
  TempBoardDir board;
  ASSERT_FALSE(board.path().empty());
  std::string board_path = board.path().string();
  {
    std::error_code ec;
    fs::remove_all(board.path(), ec);
  }
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_NOT_SUPPORTED);
}
