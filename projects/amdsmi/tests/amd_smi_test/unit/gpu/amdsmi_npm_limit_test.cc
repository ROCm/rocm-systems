// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for amdsmi_set_npm_limit() and amdsmi_get_npm_info(). Handles come from
// amdsmi_test_get_node_handle() because amdsmi_get_node_handle() needs oam_id == 0 hardware.
// Root-only cases point the handle at a temp board directory.

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_internal.h"
#include "config/amd_smi_config.h"
#include "rocm_smi/rocm_smi_utils.h"

#if defined(ENABLE_WSL_BACKEND)
#include "amd_smi/impl/amd_smi_wsl_device.h"
#endif  // ENABLE_WSL_BACKEND

namespace fs = std::filesystem;

namespace {

// Kept local to avoid coupling unit-test translation units.
class TempBoardDir {
 public:
  TempBoardDir() {
    std::string tmpl = (fs::temp_directory_path() / "amdsmi_npm_limit_test_XXXXXX").string();
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

// Small RAII wrapper around amdsmi_init()/amdsmi_shut_down() so each test is
// self-contained regardless of test execution order (ref-counted, so nesting
// with other tests' init/shutdown pairs elsewhere in the binary is safe).
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
// amdsmi_set_npm_limit()
// ---------------------------------------------------------------------

TEST(GpuUnit, SetNpmLimitNullHandleIsInval) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(nullptr, 100), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmLimitNonRootIsNoPerm) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (amd::smi::is_sudo_user()) {
    GTEST_SKIP_(
        "Running as root: REQUIRE_ROOT_ACCESS gate cannot be exercised here; "
        "see SetNpmLimitRoot* tests instead");
  }

  // REQUIRE_ROOT_ACCESS rejects before the path is touched, so it need not exist.
  amdsmi_node_handle handle = nullptr;
  ASSERT_EQ(amdsmi_test_get_node_handle("/tmp/amdsmi_npm_limit_test_nonroot_probe", &handle),
            AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 100), AMDSMI_STATUS_NO_PERM);
}

TEST(GpuUnit, SetNpmLimitRootSuccessWritesValue) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "0");
  amdsmi_node_handle handle = nullptr;
  ASSERT_EQ(amdsmi_test_get_node_handle(board.path().string(), &handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "250");
}

// ---------------------------------------------------------------------
// amdsmi_set_npm_limit() -- validation before write
// ---------------------------------------------------------------------

TEST(GpuUnit, SetNpmLimitRootRejectsWhenNpmDisabled) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "0");
  amdsmi_node_handle handle = nullptr;
  ASSERT_EQ(amdsmi_test_get_node_handle(board.path().string(), &handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "0");
}

TEST(GpuUnit, SetNpmLimitRootRejectsZero) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "6000");
  amdsmi_node_handle handle = nullptr;
  ASSERT_EQ(amdsmi_test_get_node_handle(board.path().string(), &handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 0), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitRootRejectsOverMax) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "6000");
  amdsmi_node_handle handle = nullptr;
  ASSERT_EQ(amdsmi_test_get_node_handle(board.path().string(), &handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 6401), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitRootRejectsWhenMaxUnreadable) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("cur_node_power_limit", "6000");
  amdsmi_node_handle handle = nullptr;
  ASSERT_EQ(amdsmi_test_get_node_handle(board.path().string(), &handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitRootRejectsWhenMaxCorrupted) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  // "-1" parses to UINT64_MAX, which must not be accepted as a bound.
  board.WriteFile("max_node_power_limit", "-1");
  board.WriteFile("cur_node_power_limit", "6000");
  amdsmi_node_handle handle = nullptr;
  ASSERT_EQ(amdsmi_test_get_node_handle(board.path().string(), &handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_UNEXPECTED_DATA);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitRootAcceptsInRange) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "0");
  amdsmi_node_handle handle = nullptr;
  ASSERT_EQ(amdsmi_test_get_node_handle(board.path().string(), &handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 6400), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6400");
}

TEST(GpuUnit, SetNpmLimitRootMissingFileIsNotSupported) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");
  amdsmi_node_handle handle = nullptr;
  ASSERT_EQ(amdsmi_test_get_node_handle(board.path().string(), &handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmLimitRootMissingBoardDirIsNotSupported) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir parent;
  fs::path missing = parent.path() / "missing";
  amdsmi_node_handle handle = nullptr;
  ASSERT_EQ(amdsmi_test_get_node_handle(missing.string(), &handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_NOT_SUPPORTED);
}

// ---------------------------------------------------------------------
// Registered-node_handle validation
// ---------------------------------------------------------------------

TEST(GpuUnit, GetNpmInfoRejectsUnregisteredHandle) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  // Heap-allocated and never registered, so it cannot alias a registry-owned handle.
  auto unregistered_board_path =
      std::make_unique<std::string>("/tmp/amdsmi_npm_limit_test_unregistered_probe");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  amdsmi_npm_info_t npm_info{};
  EXPECT_EQ(amdsmi_get_npm_info(handle, &npm_info), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmLimitRejectsUnregisteredHandle) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  auto unregistered_board_path =
      std::make_unique<std::string>("/tmp/amdsmi_npm_limit_test_unregistered_probe2");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  // The registry check runs before REQUIRE_ROOT_ACCESS, so INVAL regardless of privilege.
  EXPECT_EQ(amdsmi_set_npm_limit(handle, 100), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmLimitAcceptsHandleAfterTestRegistration) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (amd::smi::is_sudo_user()) {
    GTEST_SKIP_(
        "Running as root: this test isolates the registry check from "
        "REQUIRE_ROOT_ACCESS by asserting NO_PERM (not INVAL)");
  }

  // A registered handle must get past the registry check and stop at REQUIRE_ROOT_ACCESS.
  amdsmi_node_handle handle = nullptr;
  ASSERT_EQ(amdsmi_test_get_node_handle("/tmp/amdsmi_npm_limit_test_registered_probe", &handle),
            AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 100), AMDSMI_STATUS_NO_PERM);
}

#if defined(ENABLE_WSL_BACKEND)
// Only observable on real WSL2 where WSLGPUBackend::TryPopulate() succeeded.
TEST(GpuUnit, SetNpmLimitNotSupportedWhenWslBackendActive) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::WSLGPUBackend::IsActive()) {
    GTEST_SKIP_(
        "WSLGPUBackend is not active (not a real WSL2 machine, or "
        "librocdxg/rocdxg node enumeration unavailable) -- the "
        "NOT_SUPPORTED short-circuit cannot be exercised here");
  }

  amdsmi_node_handle handle = nullptr;
  ASSERT_EQ(amdsmi_test_get_node_handle("/tmp/amdsmi_npm_limit_test_wsl_probe", &handle),
            AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 100), AMDSMI_STATUS_NOT_SUPPORTED);
}
#endif  // ENABLE_WSL_BACKEND

// ---------------------------------------------------------------------
// amdsmi_get_npm_info() -- current_node_power round trip
// ---------------------------------------------------------------------

// Uses the real amdsmi_get_node_handle() path; skips where no node handle is available.
TEST(GpuUnit, GetNpmInfoCurrentNodePowerRoundTrip) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_socket_handle sockets[16];
  uint32_t socket_count = 16;
  ASSERT_EQ(amdsmi_get_socket_handles(&socket_count, sockets), AMDSMI_STATUS_SUCCESS);
  if (socket_count == 0) {
    GTEST_SKIP_("No sockets discovered on this system");
  }

  bool exercised_at_least_one_node = false;
  constexpr uint32_t kSentinel = std::numeric_limits<uint32_t>::max();

  for (uint32_t s = 0; s < socket_count; ++s) {
    amdsmi_processor_handle procs[16];
    uint32_t proc_count = 16;
    if (amdsmi_get_processor_handles(sockets[s], &proc_count, procs) != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    for (uint32_t p = 0; p < proc_count; ++p) {
      amdsmi_node_handle node_handle = nullptr;
      if (amdsmi_get_node_handle(procs[p], &node_handle) != AMDSMI_STATUS_SUCCESS) {
        continue;
      }

      amdsmi_npm_info_t npm_info;
      std::memset(&npm_info, 0, sizeof(npm_info));
      amdsmi_status_t r = amdsmi_get_npm_info(node_handle, &npm_info);
      if (r != AMDSMI_STATUS_SUCCESS) {
        continue;
      }
      exercised_at_least_one_node = true;

      // UINT32_MAX means board/node_power was unavailable.
      if (npm_info.current_node_power != kSentinel) {
        EXPECT_LT(npm_info.current_node_power, kSentinel);
      }
    }
  }

  if (!exercised_at_least_one_node) {
    GTEST_SKIP_(
        "amdsmi_get_node_handle()/amdsmi_get_npm_info() did not succeed for any processor on "
        "this system");
  }
}
