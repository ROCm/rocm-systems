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
#include "test_common.h"

#if defined(ENABLE_WSL_BACKEND)
#include "amd_smi/impl/amd_smi_wsl_device.h"
#endif  // ENABLE_WSL_BACKEND

namespace fs = std::filesystem;

// On a real WSL2 CI runner, WSLGPUBackend::IsActive() makes every NPM call
// short-circuit to AMDSMI_STATUS_NOT_SUPPORTED before ever consulting this
// test's fabricated board_path content (see the ENABLE_WSL_BACKEND guard at
// the top of each amdsmi_*npm* function in amd_smi.cc). A plain helper
// function would not work here: GTEST_SKIP_() expands to a `return`, which
// would only exit the helper, not the TEST() body -- hence a macro.
#if defined(ENABLE_WSL_BACKEND)
#define SKIP_IF_WSL_BACKEND_ACTIVE()                                   \
  do {                                                                 \
    if (amd::smi::WSLGPUBackend::IsActive()) {                         \
      GTEST_SKIP_(                                                     \
          "WSLGPUBackend is active: this NPM call short-circuits to "  \
          "AMDSMI_STATUS_NOT_SUPPORTED before this test's fabricated " \
          "board content is ever consulted");                          \
    }                                                                  \
  } while (0)
#else
#define SKIP_IF_WSL_BACKEND_ACTIVE() \
  do {                               \
  } while (0)
#endif  // ENABLE_WSL_BACKEND

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
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_balancing_mode", "GetNpmBalancingModeNullHandleIsInval",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_balancing_mode(nullptr, &mode);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmBalancingModeNullModeIsInval) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  std::string board_path = "/tmp/amdsmi_npm_balancing_mode_test_null_mode_probe";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  DISPLAY_AMDSMI_API("amdsmi_get_npm_balancing_mode", "GetNpmBalancingModeNullModeIsInval",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_balancing_mode(handle, nullptr);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmBalancingModeDisabledStillReportsLastSelectedMode) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

  TempBoardDir board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("mode", "1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_INVALID;
  // Get is never gated on NPM enablement; it reflects board/mode even
  // while NPM is disabled.
  DISPLAY_AMDSMI_API("amdsmi_get_npm_balancing_mode",
                     "GetNpmBalancingModeDisabledStillReportsLastSelectedMode",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_balancing_mode(handle, &mode);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
}

TEST(GpuUnit, GetNpmBalancingModeEnabledDefaultsToPowerBalancing) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

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
  DISPLAY_AMDSMI_API("amdsmi_get_npm_balancing_mode",
                     "GetNpmBalancingModeEnabledDefaultsToPowerBalancing", GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_balancing_mode(handle, &mode);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
}

TEST(GpuUnit, GetNpmBalancingModeEnabledMissingModeFileIsNotSupported) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  // Deliberately do not create board/mode: the sysfs node absent/unreadable
  // must report NOT_SUPPORTED, agreeing with amdsmi_set_npm_balancing_mode()
  // for the identical state (see SetNpmBalancingModeRootMissingModeFileIsNotSupported).
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_balancing_mode",
                     "GetNpmBalancingModeEnabledMissingModeFileIsNotSupported",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_balancing_mode(handle, &mode);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, GetNpmBalancingModeGarbageValueIsUnexpectedData) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "0");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  // "0" is not a valid encoding ("1"/"2" only); this is a genuine
  // read/decode error, not the "missing file" N/A case.
  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_balancing_mode",
                     "GetNpmBalancingModeGarbageValueIsUnexpectedData", GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_balancing_mode(handle, &mode);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err,
                        AMDSMI_STATUS_UNEXPECTED_DATA);
  EXPECT_EQ(err, AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, GetNpmBalancingModeEnabledReadsFrequencyBalancing) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "2");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_INVALID;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_balancing_mode",
                     "GetNpmBalancingModeEnabledReadsFrequencyBalancing", GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_balancing_mode(handle, &mode);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
}

// ---------------------------------------------------------------------
// amdsmi_set_npm_balancing_mode()
// ---------------------------------------------------------------------

TEST(GpuUnit, SetNpmBalancingModeNullHandleIsInval) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode", "SetNpmBalancingModeNullHandleIsInval",
                     GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(nullptr, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmBalancingModeInvalidModeIsInval) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  std::string board_path = "/tmp/amdsmi_npm_balancing_mode_test_invalid_mode_probe";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode", "SetNpmBalancingModeInvalidModeIsInval",
                     GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(
      handle, static_cast<amdsmi_npm_balancing_mode_t>(AMDSMI_NPM_BALANCING_MODE_INVALID));
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmBalancingModeNonRootIsNoPerm) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

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

  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode", "SetNpmBalancingModeNonRootIsNoPerm",
                     GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_NO_PERM);
  EXPECT_EQ(err, AMDSMI_STATUS_NO_PERM);
}

TEST(GpuUnit, SetNpmBalancingModeRootRejectsWhenNpmDisabled) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

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
  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode",
                     "SetNpmBalancingModeRootRejectsWhenNpmDisabled", GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
  // The file must be untouched.
  EXPECT_EQ(board.ReadFile("mode"), "1");
}

TEST(GpuUnit, SetNpmBalancingModeRootRoundTrip) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode", "SetNpmBalancingModeRootRoundTrip",
                     GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("mode"), "2");

  amdsmi_npm_balancing_mode_t readback = AMDSMI_NPM_BALANCING_MODE_INVALID;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_balancing_mode", "SetNpmBalancingModeRootRoundTrip",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_balancing_mode(handle, &readback);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(readback, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);

  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode", "SetNpmBalancingModeRootRoundTrip",
                     GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("mode"), "1");

  readback = AMDSMI_NPM_BALANCING_MODE_INVALID;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_balancing_mode", "SetNpmBalancingModeRootRoundTrip",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_balancing_mode(handle, &readback);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(readback, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
}

TEST(GpuUnit, SetNpmBalancingModeRootMissingModeFileIsNotSupported) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

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

  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode",
                     "SetNpmBalancingModeRootMissingModeFileIsNotSupported",
                     GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmBalancingModeRootMissingBoardDirIsNotSupported) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

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

  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode",
                     "SetNpmBalancingModeRootMissingBoardDirIsNotSupported",
                     GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmBalancingModeRootRejectsModeAbsentFromSupportedBitmask) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

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

  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode",
                     "SetNpmBalancingModeRootRejectsModeAbsentFromSupportedBitmask",
                     GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err,
                        AMDSMI_STATUS_SETTING_UNAVAILABLE);
  EXPECT_EQ(err, AMDSMI_STATUS_SETTING_UNAVAILABLE);
  // The file must be untouched.
  EXPECT_EQ(board.ReadFile("mode"), "1");
}

TEST(GpuUnit, SetNpmBalancingModeRootAllowsModePresentInSupportedBitmask) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

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

  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode",
                     "SetNpmBalancingModeRootAllowsModePresentInSupportedBitmask",
                     GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("mode"), "2");
}

TEST(GpuUnit, SetNpmBalancingModeRootRejectsCorruptSupportedModesFile) {
  amdsmi_status_t err;
  // A read failure other than NOT_SUPPORTED (e.g. corrupt/unparsable
  // content) must fail closed as SETTING_UNAVAILABLE and reject the write --
  // it must not be treated like a missing file.
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

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

  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode",
                     "SetNpmBalancingModeRootRejectsCorruptSupportedModesFile",
                     GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err,
                        AMDSMI_STATUS_SETTING_UNAVAILABLE);
  EXPECT_EQ(err, AMDSMI_STATUS_SETTING_UNAVAILABLE);
  // The file must be untouched.
  EXPECT_EQ(board.ReadFile("mode"), "1");
}

TEST(GpuUnit, SetNpmBalancingModeRootToleratesMissingSupportedModesFile) {
  amdsmi_status_t err;
  // The supported_mode sysfs file is not yet implemented on all driver
  // versions; its absence must not block an otherwise-valid set.
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

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

  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode",
                     "SetNpmBalancingModeRootToleratesMissingSupportedModesFile",
                     GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("mode"), "2");
}

// ---------------------------------------------------------------------
// amdsmi_get_npm_supported_balancing_modes()
// ---------------------------------------------------------------------

TEST(GpuUnit, GetNpmSupportedBalancingModesNullHandleIsInval) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_supported_balancing_modes",
                     "GetNpmSupportedBalancingModesNullHandleIsInval", GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_supported_balancing_modes(nullptr, &supported_modes);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesNullOutputIsInval) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  std::string board_path = "/tmp/amdsmi_npm_supported_balancing_modes_test_null_out_probe";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  DISPLAY_AMDSMI_API("amdsmi_get_npm_supported_balancing_modes",
                     "GetNpmSupportedBalancingModesNullOutputIsInval", GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_supported_balancing_modes(handle, nullptr);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesMissingFileIsNotSupported) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

  TempBoardDir board;
  // Deliberately do not create board/supported_mode: not yet implemented
  // by the driver on all platforms.
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_supported_balancing_modes",
                     "GetNpmSupportedBalancingModesMissingFileIsNotSupported",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesNotGatedOnNpmEnablement) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

  TempBoardDir board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("supported_mode", "0x6");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_supported_balancing_modes",
                     "GetNpmSupportedBalancingModesNotGatedOnNpmEnablement",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(supported_modes, 0x6u);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesDecodesHexBitmask) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

  TempBoardDir board;
  board.WriteFile("supported_mode", "0x2");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_supported_balancing_modes",
                     "GetNpmSupportedBalancingModesDecodesHexBitmask", GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(supported_modes, 0x2u);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesGarbageValueIsUnexpectedData) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

  TempBoardDir board;
  board.WriteFile("supported_mode", "not-a-hex-number");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_supported_balancing_modes",
                     "GetNpmSupportedBalancingModesGarbageValueIsUnexpectedData",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err,
                        AMDSMI_STATUS_UNEXPECTED_DATA);
  EXPECT_EQ(err, AMDSMI_STATUS_UNEXPECTED_DATA);
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
  amdsmi_status_t err;
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
  DISPLAY_AMDSMI_API("amdsmi_get_npm_balancing_mode",
                     "GetNpmBalancingModeRejectsUnregisteredHandle", GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_balancing_mode(handle, &mode);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmBalancingModeRejectsUnregisteredHandle) {
  amdsmi_status_t err;
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
  DISPLAY_AMDSMI_API("amdsmi_set_npm_balancing_mode",
                     "SetNpmBalancingModeRejectsUnregisteredHandle", GetTestVerbosity() >= 1);
  err = amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesRejectsUnregisteredHandle) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  auto unregistered_board_path =
      std::make_unique<std::string>("/tmp/amdsmi_npm_balancing_mode_test_unregistered_probe3");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  amdsmi_bit_field_t supported_modes = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_supported_balancing_modes",
                     "GetNpmSupportedBalancingModesRejectsUnregisteredHandle",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesNegativeValueIsUnexpectedData) {
  amdsmi_status_t err;
  ScopedAmdSmiInit init;
  if (init.status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) {
    GTEST_SKIP_("No GPU driver loaded");
  }
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);
  SKIP_IF_WSL_BACKEND_ACTIVE();

  TempBoardDir board;
  // std::stoull("-1", ..., 16) would otherwise silently parse as UINT64_MAX;
  // must be rejected as a parse error instead.
  board.WriteFile("supported_mode", "-1");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_npm_supported_balancing_modes",
                     "GetNpmSupportedBalancingModesNegativeValueIsUnexpectedData",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err,
                        AMDSMI_STATUS_UNEXPECTED_DATA);
  EXPECT_EQ(err, AMDSMI_STATUS_UNEXPECTED_DATA);
}
