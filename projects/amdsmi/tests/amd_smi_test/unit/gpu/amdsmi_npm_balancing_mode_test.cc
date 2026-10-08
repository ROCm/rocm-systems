// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for amdsmi_get_npm_balancing_mode()/amdsmi_set_npm_balancing_mode()
// (amd_smi.cc) and the underlying rsmi_dev_npm_balancing_mode_get()/set()
// (rocm_smi.cc).
//
// Like amdsmi_set_npm_limit(), these take an amdsmi_node_handle that is
// internally a `std::string*` board-path pointer normally vended by
// amdsmi_get_node_handle(). These functions only accept a node_handle that
// this library actually vended (tracked in a registry populated by
// amdsmi_get_node_handle(); see is_registered_node_handle() in amd_smi.cc).
// Since tests fabricate their node_handle directly instead of calling
// amdsmi_get_node_handle(), every fabricated handle below is registered via
// the test-only amdsmi_test_register_node_handle() hook
// (amd_smi_test_internal.h) immediately after construction, so it is treated
// as valid by that check. The *RejectsUnregisteredHandle tests below are the
// ones that specifically exercise the *absence* of that registration call,
// and must not call amdsmi_test_register_node_handle().
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
#include <memory>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_internal.h"
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
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(nullptr, &mode), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmBalancingModeNullModeIsInval) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  std::string board_path = "/tmp/amdsmi_npm_balancing_mode_test_null_mode_probe";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmBalancingModeDisabledStillReportsLastSelectedMode) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("mode", "1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_INVALID;
  // Get is never gated on NPM enablement; it reflects board/mode even
  // while NPM is disabled.
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
}

TEST(GpuUnit, GetNpmBalancingModeEnabledDefaultsToPowerBalancing) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  // The platform/driver creates board/mode with its default value ("1")
  // when NPM is enabled; simulate that here rather than leaving the file
  // absent, since a genuinely missing/unreadable file now reports
  // AMDSMI_STATUS_SUCCESS + AMDSMI_NPM_BALANCING_MODE_INVALID (N/A).
  board.WriteFile("mode", "1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_INVALID;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
}

TEST(GpuUnit, GetNpmBalancingModeEnabledMissingModeFileIsInvalidMode) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  // Deliberately do not create board/mode: the sysfs node absent/
  // unreadable must report N/A (INVALID), not a fabricated PB, and must
  // agree with amdsmi_set_npm_balancing_mode()'s NOT_SUPPORTED for the
  // identical state (see SetNpmBalancingModeRootMissingModeFileIsNotSupported).
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_INVALID);
}

TEST(GpuUnit, GetNpmBalancingModeGarbageValueIsUnexpectedData) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "0");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  // "0" is not a valid encoding ("1"/"2" only); this is a genuine
  // read/decode error, not the "missing file" N/A case.
  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, GetNpmBalancingModeEnabledReadsFrequencyBalancing) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "2");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_INVALID;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
}

// ---------------------------------------------------------------------
// amdsmi_set_npm_balancing_mode()
// ---------------------------------------------------------------------

TEST(GpuUnit, SetNpmBalancingModeNullHandleIsInval) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(nullptr, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmBalancingModeInvalidModeIsInval) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  std::string board_path = "/tmp/amdsmi_npm_balancing_mode_test_invalid_mode_probe";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, static_cast<amdsmi_npm_balancing_mode_t>(
                                                      AMDSMI_NPM_BALANCING_MODE_INVALID)),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmBalancingModeNonRootIsNoPerm) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
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
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_NO_PERM);
}

TEST(GpuUnit, SetNpmBalancingModeRootRejectsWhenNpmDisabled) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("mode", "1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  // Must not even open board/mode for write when NPM is disabled: this is
  // AMDSMI_STATUS_NOT_SUPPORTED, a deliberate divergence from
  // amdsmi_set_npm_limit()'s own disabled-check (AMDSMI_STATUS_INVAL).
  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_NOT_SUPPORTED);
  // The file must be untouched.
  EXPECT_EQ(board.ReadFile("mode"), "1");
}

TEST(GpuUnit, SetNpmBalancingModeRootRoundTrip) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("mode"), "2");

  amdsmi_npm_balancing_mode_t readback = AMDSMI_NPM_BALANCING_MODE_INVALID;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &readback), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(readback, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("mode"), "1");

  readback = AMDSMI_NPM_BALANCING_MODE_INVALID;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &readback), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(readback, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
}

TEST(GpuUnit, SetNpmBalancingModeRootMissingModeFileIsNotSupported) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  // Deliberately do not create board/mode: set_npm_board_mode() requires
  // the file to already exist (it does not create it), mirroring
  // set_npm_board_limit()'s missing-file handling.
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmBalancingModeRootMissingBoardDirIsNotSupported) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
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
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmBalancingModeRootRejectsModeAbsentFromSupportedBitmask) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "1");
  // bit 2 (FREQUENCY_BALANCING) only -- POWER_BALANCING (bit 1) is absent.
  board.WriteFile("supported_mode", "0x4");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING),
            AMDSMI_STATUS_NOT_SUPPORTED);
  // The file must be untouched.
  EXPECT_EQ(board.ReadFile("mode"), "1");
}

TEST(GpuUnit, SetNpmBalancingModeRootAllowsModePresentInSupportedBitmask) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "1");
  // bits 1+2 -- both modes supported, matching today's real platforms.
  board.WriteFile("supported_mode", "0x6");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("mode"), "2");
}

TEST(GpuUnit, SetNpmBalancingModeRootRejectsCorruptSupportedModesFile) {
  // A read failure other than NOT_SUPPORTED (e.g. corrupt/unparsable
  // content) must fail closed and reject the write, matching the
  // npm_status gate's policy -- it must not be treated like a missing file.
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "1");
  board.WriteFile("supported_mode", "-1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_NOT_SUPPORTED);
  // The file must be untouched.
  EXPECT_EQ(board.ReadFile("mode"), "1");
}

TEST(GpuUnit, SetNpmBalancingModeRootToleratesMissingSupportedModesFile) {
  // The supported_mode sysfs file is not yet implemented on all driver
  // versions; its absence must not block an otherwise-valid set.
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "1");
  // Deliberately do not create board/supported_mode.
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("mode"), "2");
}

// ---------------------------------------------------------------------
// amdsmi_get_npm_supported_balancing_modes()
// ---------------------------------------------------------------------

TEST(GpuUnit, GetNpmSupportedBalancingModesNullHandleIsInval) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(nullptr, &supported_modes),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesNullOutputIsInval) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  std::string board_path = "/tmp/amdsmi_npm_supported_balancing_modes_test_null_out_probe";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesMissingFileIsNotSupported) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  // Deliberately do not create board/supported_mode: not yet implemented
  // by the driver on all platforms.
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes),
            AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesNotGatedOnNpmEnablement) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("supported_mode", "0x6");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(supported_modes, 0x6u);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesDecodesHexBitmask) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("supported_mode", "0x2");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(supported_modes, 0x2u);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesGarbageValueIsUnexpectedData) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  board.WriteFile("supported_mode", "not-a-hex-number");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes),
            AMDSMI_STATUS_UNEXPECTED_DATA);
}

// ---------------------------------------------------------------------
// Registered-node_handle validation
//
// These tests deliberately do NOT call amdsmi_test_register_node_handle():
// the handle below points at real, validly-constructed heap memory, but it
// was never registered, which is exactly the condition
// is_registered_node_handle() targets. Asserting AMDSMI_STATUS_INVAL (rather
// than merely "did not crash") demonstrates the registry check itself is
// what stopped execution.
// ---------------------------------------------------------------------

TEST(GpuUnit, GetNpmBalancingModeRejectsUnregisteredHandle) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  // Heap-allocated so this address cannot alias a stack slot some other test
  // in this binary already passed to amdsmi_test_register_node_handle() --
  // see GetNpmInfoRejectsUnregisteredHandle in amdsmi_npm_limit_test.cc for
  // the full rationale.
  auto unregistered_board_path =
      std::make_unique<std::string>("/tmp/amdsmi_npm_balancing_mode_test_unregistered_probe1");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_INVALID;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmBalancingModeRejectsUnregisteredHandle) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  auto unregistered_board_path =
      std::make_unique<std::string>("/tmp/amdsmi_npm_balancing_mode_test_unregistered_probe2");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  // AMDSMI_STATUS_INVAL must be returned regardless of caller privilege: the
  // registry check runs before REQUIRE_ROOT_ACCESS is ever reached.
  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesRejectsUnregisteredHandle) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  auto unregistered_board_path =
      std::make_unique<std::string>("/tmp/amdsmi_npm_balancing_mode_test_unregistered_probe3");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  amdsmi_bit_field_t supported_modes = 0;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesNegativeValueIsUnexpectedData) {
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  TempBoardDir board;
  // std::stoull("-1", ..., 16) would otherwise silently parse as UINT64_MAX;
  // must be rejected as a parse error instead.
  board.WriteFile("supported_mode", "-1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes),
            AMDSMI_STATUS_UNEXPECTED_DATA);
}
