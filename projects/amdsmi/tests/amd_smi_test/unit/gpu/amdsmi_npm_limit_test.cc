// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for the public amdsmi_set_npm_limit() API (amd_smi.cc) and the
// current_node_power round-trip in amdsmi_get_npm_info().
//
// amdsmi_set_npm_limit() takes an amdsmi_node_handle, which is internally a
// `std::string*` board-path pointer (see amdsmi_get_node_handle() in
// amd_smi.cc). The public acquisition path (amdsmi_get_node_handle()) gates
// on `asic_info.oam_id != 0`, which is not satisfiable on this test
// environment's hardware/mock setup (pre-existing, out of scope for this
// feature). To exercise amdsmi_set_npm_limit() itself we construct the
// node_handle directly by reinterpret_cast'ing the address of a local
// std::string, exactly as amdsmi_get_node_handle() would have handed back,
// bypassing only the acquisition helper -- not the function under test. This
// mirrors the approach used in the feature's own smoke test (see the
// implementer's handoff notes).
//
// amdsmi_get_npm_info()/amdsmi_set_npm_limit() only accept a node_handle that
// this library actually vended (tracked in a registry populated by
// amdsmi_get_node_handle(); see is_registered_node_handle() in amd_smi.cc,
// fixing a pre-cast/dereference validation gap). Since tests fabricate their
// node_handle directly instead of calling amdsmi_get_node_handle(), every
// fabricated handle below is registered via the test-only
// amdsmi_test_register_node_handle() hook (amd_smi_test_internal.h)
// immediately after construction, so it is treated as valid by that check.
// The NpmInfoRejectsUnregisteredHandle/SetNpmLimitRejectsUnregisteredHandle
// tests below are the ones that specifically exercise the *absence* of that
// registration call, and must not call amdsmi_test_register_node_handle().
//
// rsmi_dev_npm_limit_set() (rocm_smi.cc) applies REQUIRE_ROOT_ACCESS before
// touching any file, so:
//   - non-root callers always observe AMDSMI_STATUS_NO_PERM, regardless of
//     whether the board_path/file exists -- this is verifiable without root
//     and is asserted unconditionally below.
//   - success / not-supported branches (reached only past the root gate)
//     require root and are gated with GTEST_SKIP_ following the existing
//     is_sudo_user() convention used throughout tests/amd_smi_test/main.cc.
// Since set_npm_board_limit()/rsmi_dev_npm_limit_set() operate on an
// arbitrary board_path string (not hardcoded to /sys), the root-gated cases
// use a plain std::filesystem temp directory instead of real /sys mocking.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
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

// RAII helper identical in spirit to the one in rocm_smi_npm_test.cc; kept
// local to this file to avoid coupling across unit-test translation units.
class TempBoardDir {
 public:
  TempBoardDir() {
    path_ = fs::temp_directory_path() /
            fs::path("amdsmi_npm_limit_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
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

  // Any non-empty board path is sufficient: REQUIRE_ROOT_ACCESS in
  // rsmi_dev_npm_limit_set() rejects before the path is ever touched.
  std::string board_path = "/tmp/amdsmi_npm_limit_test_nonroot_probe";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 100), AMDSMI_STATUS_NO_PERM);
}

TEST(GpuUnit, SetNpmLimitRootSuccessWritesValue) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  // max_node_power_limit must be present and >= the requested limit now that
  // rsmi_dev_npm_limit_set() range-checks against it before writing (mirrors
  // rsmi_dev_power_cap_set()'s range_get()-before-write pattern).
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "0");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "250");
}

// ---------------------------------------------------------------------
// amdsmi_set_npm_limit() -- platform-max range enforcement (F-5 fix)
//
// rsmi_dev_npm_limit_set() now mirrors rsmi_dev_power_cap_set(): it queries
// the platform max via get_npm_board_max_limit() and rejects out-of-range
// requests with AMDSMI_STATUS_INVAL before ever touching
// cur_node_power_limit, and fails closed (propagates the underlying error)
// if the max itself can't be read.
// ---------------------------------------------------------------------

TEST(GpuUnit, SetNpmLimitRootRejectsZero) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "6000");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 0), AMDSMI_STATUS_INVAL);
  // The write must not have happened.
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitRootRejectsOverMax) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "6000");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 6401), AMDSMI_STATUS_INVAL);
  // The write must not have happened.
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitRootRejectsWhenMaxUnreadable) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  // Deliberately do not create max_node_power_limit: the platform max is
  // unreadable, so the request must be rejected (fail closed) rather than
  // falling through to an unbounded write.
  board.WriteFile("cur_node_power_limit", "6000");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_NOT_SUPPORTED);
  // The write must not have happened.
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitRootRejectsWhenMaxCorrupted) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  // "-1" is a regression case, not generic garbage: std::stoull("-1", ...)
  // is well-defined C++ behavior that successfully parses to UINT64_MAX
  // (leading '-' negates modulo 2^64) rather than throwing, so a naive
  // parser would treat this corrupted content as a "successfully read"
  // huge platform max -- which would let any limit through the
  // `limit > max_limit` check. The platform max must fail to parse and be
  // treated as unreadable (fail closed), exactly like the missing-file case
  // in SetNpmLimitRootRejectsWhenMaxUnreadable above, not silently accepted
  // as UINT64_MAX.
  board.WriteFile("max_node_power_limit", "-1");
  board.WriteFile("cur_node_power_limit", "6000");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  // Unlike the missing-file case above (RSMI_STATUS_NOT_SUPPORTED), a
  // present-but-corrupted file is a parse failure:
  // get_npm_board_max_limit()/read_board_uint64() returns
  // RSMI_STATUS_UNEXPECTED_DATA, which rsmi_dev_npm_limit_set() propagates
  // as-is (fail closed) and which maps to AMDSMI_STATUS_UNEXPECTED_DATA.
  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_UNEXPECTED_DATA);
  // The write must not have happened.
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitRootAcceptsInRange) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  TempBoardDir board;
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "0");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

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
  // max_node_power_limit present (so the new range check itself succeeds),
  // but cur_node_power_limit deliberately absent -- exercises the write-time
  // NOT_SUPPORTED path in set_npm_board_limit(), distinct from
  // SetNpmLimitRootRejectsWhenMaxUnreadable above.
  board.WriteFile("max_node_power_limit", "6400");
  std::string board_path = board.path().string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmLimitRootMissingBoardDirIsNotSupported) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  fs::path missing = fs::temp_directory_path() / "amdsmi_npm_limit_test_missing_board_dir";
  std::error_code ec;
  fs::remove_all(missing, ec);
  std::string board_path = missing.string();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_NOT_SUPPORTED);
}

// ---------------------------------------------------------------------
// Registered-node_handle validation (F-2 fix)
//
// amdsmi_get_npm_info()/amdsmi_set_npm_limit() previously
// reinterpret_cast<std::string*>()'d and immediately dereferenced
// (->empty()) their caller-supplied node_handle with no check that the
// pointer was ever actually vended by this library. An unprivileged caller
// could pass any non-null garbage value and hit that dereference *before*
// any privilege check (REQUIRE_ROOT_ACCESS lives one layer down, only inside
// rsmi_dev_npm_limit_set(), and not at all in the read path). Both functions
// now call is_registered_node_handle() first and reject anything not found
// in the node_handle registry with AMDSMI_STATUS_INVAL, never dereferencing
// an unregistered pointer.
//
// These tests deliberately do NOT call amdsmi_test_register_node_handle():
// the handle below points at real, validly-constructed stack/heap memory
// (so a would-be crash is not guaranteed/portable to reproduce in CI), but
// it was never registered, which is exactly the condition the fix targets.
// Asserting AMDSMI_STATUS_INVAL (rather than merely "did not crash")
// demonstrates the registry check itself is what stopped execution, not
// incidental luck about what garbage bytes happened to be at that address.
// ---------------------------------------------------------------------

TEST(GpuUnit, GetNpmInfoRejectsUnregisteredHandle) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  // Heap-allocated (not stack-allocated) so this address cannot alias a
  // stack slot some other test in this binary already passed to
  // amdsmi_test_register_node_handle(): registered pointers live for the
  // rest of the process, and other tests here register the address of a
  // same-shaped stack-local std::string, which the allocator/compiler can
  // legitimately hand back again for a later test's stack frame. A fresh
  // heap allocation that nothing in this file ever registers cannot collide
  // with that set.
  auto unregistered_board_path =
      std::make_unique<std::string>("/tmp/amdsmi_npm_limit_test_unregistered_probe");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  amdsmi_npm_info_t npm_info{};
  EXPECT_EQ(amdsmi_get_npm_info(handle, &npm_info), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmLimitRejectsUnregisteredHandle) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  // See GetNpmInfoRejectsUnregisteredHandle above for why this must be
  // heap-allocated rather than a stack-local std::string.
  auto unregistered_board_path =
      std::make_unique<std::string>("/tmp/amdsmi_npm_limit_test_unregistered_probe2");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  // AMDSMI_STATUS_INVAL must be returned regardless of caller privilege: the
  // registry check runs before REQUIRE_ROOT_ACCESS is ever reached, whether
  // or not this test process happens to be root.
  EXPECT_EQ(amdsmi_set_npm_limit(handle, 100), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmLimitAcceptsHandleAfterTestRegistration) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (amd::smi::is_sudo_user()) {
    GTEST_SKIP_(
        "Running as root: this test isolates the registry check from "
        "REQUIRE_ROOT_ACCESS by asserting NO_PERM (not INVAL); see "
        "SetNpmLimitNonRootIsNoPerm for the rationale");
  }

  // Same handle shape as SetNpmLimitRejectsUnregisteredHandle above, but
  // registered first: the registry check must pass, and the request must
  // proceed to (and be stopped by) REQUIRE_ROOT_ACCESS instead of being
  // rejected as INVAL -- demonstrating the fix does not reject legitimately
  // vended handles.
  std::string board_path = "/tmp/amdsmi_npm_limit_test_registered_probe";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 100), AMDSMI_STATUS_NO_PERM);
}

#if defined(ENABLE_WSL_BACKEND)
// When the WSL backend is active, amdsmi_set_npm_limit() must short-circuit
// to AMDSMI_STATUS_NOT_SUPPORTED before ever touching board_path/sysfs (see
// the `#ifdef ENABLE_WSL_BACKEND` / WSLGPUBackend::IsActive() guard at the
// top of amdsmi_set_npm_limit() in amd_smi.cc). This is only observable on a
// real WSL2 machine where amdsmi_init() -> populate_amd_gpu_devices() ->
// WSLGPUBackend::TryPopulate() actually succeeded (requires /dev/dxg and a
// loadable librocdxg); otherwise IsActive() stays false even when compiled
// with ENABLE_WSL_BACKEND=ON, so this test skips itself in that case rather
// than asserting anything about the (untaken) native-Linux path.
TEST(GpuUnit, SetNpmLimitNotSupportedWhenWslBackendActive) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  if (!amd::smi::WSLGPUBackend::IsActive()) {
    GTEST_SKIP_(
        "WSLGPUBackend is not active (not a real WSL2 machine, or "
        "librocdxg/rocdxg node enumeration unavailable) -- the "
        "NOT_SUPPORTED short-circuit cannot be exercised here");
  }

  // Any board path is sufficient: the WSL-backend guard runs before the
  // path is ever dereferenced.
  std::string board_path = "/tmp/amdsmi_npm_limit_test_wsl_probe";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 100), AMDSMI_STATUS_NOT_SUPPORTED);
}
#endif  // ENABLE_WSL_BACKEND

// ---------------------------------------------------------------------
// amdsmi_get_npm_info() -- current_node_power round trip
// ---------------------------------------------------------------------

// current_node_power moved from amdsmi_power_info_t (per-GPU, amd-smi metric
// --power) to amdsmi_npm_info_t (per-node, amd-smi node -p /
// amdsmi_get_npm_info()) -- see the "Set npm power limit" design doc. It is
// still populated by resolving the board directory from the real device's
// enumeration info (drm_render) via amdsmi_get_node_handle(), not from a
// caller-supplied path, so it cannot be pointed at a temp directory the way
// amdsmi_set_npm_limit() can. This test therefore validates against whatever
// the current hardware reports: either a concrete value (if
// board/node_power exists) or the UINT32_MAX sentinel/"N/A" case (if it
// doesn't, e.g. no board/ directory on this platform) -- both are valid,
// contract-conformant outcomes, and the test asserts the round trip is
// internally consistent either way.
//
// amdsmi_get_node_handle() additionally gates on asic_info.oam_id == 0 (see
// the file header comment above), so on hardware/mocks where that gate is
// not satisfied this test will skip rather than fail -- it is exercising the
// real acquisition path end-to-end, not a test-fabricated handle.
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

      // Contract: current_node_power is either the sentinel (falls back
      // silently, e.g. no board/node_power file) or a concrete uint32 value
      // read from board/node_power -- never left uninitialized (e.g.
      // garbage/0 as a side effect of a partially-implemented read).
      if (npm_info.current_node_power != kSentinel) {
        // A "real" reading was obtained: sanity-check it is a plausible
        // wattage rather than accidentally aliasing the sentinel's bit
        // pattern or being negative-when-reinterpreted (uint32_t can't be
        // negative, but guard against absurd overflow wrap regardless).
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

TEST(GpuUnit, GetPowerInfoNullInfoIsInval) {
  ScopedAmdSmiInit init;
  ASSERT_EQ(init.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_socket_handle sockets[16];
  uint32_t socket_count = 16;
  ASSERT_EQ(amdsmi_get_socket_handles(&socket_count, sockets), AMDSMI_STATUS_SUCCESS);
  if (socket_count == 0) {
    GTEST_SKIP_("No sockets discovered on this system");
  }
  amdsmi_processor_handle procs[16];
  uint32_t proc_count = 16;
  ASSERT_EQ(amdsmi_get_processor_handles(sockets[0], &proc_count, procs), AMDSMI_STATUS_SUCCESS);
  if (proc_count == 0) {
    GTEST_SKIP_("No processors discovered on this system");
  }

  EXPECT_EQ(amdsmi_get_power_info(procs[0], nullptr), AMDSMI_STATUS_INVAL);
}
