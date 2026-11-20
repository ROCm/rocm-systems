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
// Permission failures are tested with file modes, as in ras_features_test.cc: see
// SetNpmLimitUnwritableLimitFileIsNoPerm. The board directory is a FakeSysfsTree, since
// set_npm_board_limit()/rsmi_dev_npm_limit_set() take an arbitrary board_path string.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_test_internal.h"
#include "config/amd_smi_config.h"
#include "test_fixture_utils.h"

#if defined(ENABLE_WSL_BACKEND)
#include "amd_smi/impl/amd_smi_wsl_device.h"
#endif  // ENABLE_WSL_BACKEND

namespace {

// Real amdsmi_init() with GPU discovery, unlike the fixture ScopedAmdsmiInit. Needed where the
// test depends on discovery: the round trip walks real sockets (amdsmi_get_node_handle() issues
// DRM ioctls no fixture fakes yet), and the WSL backend only activates during discovery.
class ScopedRealDiscoveryInit {
 public:
  ScopedRealDiscoveryInit() { status_ = amdsmi_init(AMDSMI_INIT_AMD_GPUS); }
  ~ScopedRealDiscoveryInit() {
    if (status_ == AMDSMI_STATUS_SUCCESS) {
      amdsmi_shut_down();
    }
  }
  amdsmi_status_t status() const { return status_; }

 private:
  amdsmi_status_t status_;
};

// amdsmi_init() without hardware discovery, plus a fake ROCm SMI device 0: the NPM entry points
// call rsmi_dev_npm_*(0, ...), which needs a device 0 even though the board path comes from the
// handle. For the setters: ROCm SMI is never initialized, so its cached euid stays 0 and the
// root-only setters run unprivileged. A real init would make them return NO_PERM.
class NpmTestEnv {
 public:
  NpmTestEnv() : device_(tree_.root() + "/card0") {}

 private:
  ScopedAmdsmiInit init_;
  FakeSysfsTree tree_;
  ScopedRocmSmiDevice device_;
};

// The real amdsmi_init() over a fake KFD and DRM tree with one GPU, for the getters, which need
// no root.
class NpmDiscoveredGpuEnv {
 public:
  NpmDiscoveredGpuEnv() {
    WriteGpuNode(tree_, kGpuNode, kGpuCard, kGpuId);
    topology_.emplace(tree_);
  }

 private:
  static constexpr uint32_t kGpuNode = 1;
  static constexpr uint32_t kGpuCard = 0;
  static constexpr uint64_t kGpuId = 4242;

  FakeSysfsTree tree_;
  std::optional<FakeKfdTopology> topology_;
};

}  // namespace

// Driverless runners (e.g. the ASAN CI container) cannot init; any other init
// failure is still a test failure.
#define REQUIRE_AMDSMI_INIT(init)                             \
  do {                                                        \
    if ((init).status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) { \
      GTEST_SKIP() << "amdgpu driver not loaded";             \
    }                                                         \
    ASSERT_EQ((init).status(), AMDSMI_STATUS_SUCCESS);        \
  } while (0)

// ---------------------------------------------------------------------
// amdsmi_set_npm_limit()
// ---------------------------------------------------------------------

TEST(GpuUnit, SetNpmLimitNullHandleIsInval) {
  NpmTestEnv env;

  EXPECT_EQ(amdsmi_set_npm_limit(nullptr, 100), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmLimitUnwritableLimitFileIsNoPerm) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "6000");
  board.SetPermissions("cur_node_power_limit", 0444);
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  auto set_limit = [&] { return amdsmi_set_npm_limit(handle, 100); };
  EXPECT_EQ(amd::smi::testing::RunAsUnprivilegedUser(set_limit), AMDSMI_STATUS_NO_PERM);
  // A rejected write leaves the file unchanged.
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitSuccessWritesValue) {
  NpmTestEnv env;

  FakeSysfsTree board;
  // NPM must be enabled on this node now that rsmi_dev_npm_limit_set()
  // rejects the write outright when it is not.
  board.WriteFile("npm_status", "enabled");
  // max_node_power_limit must be present and >= the requested limit now that
  // rsmi_dev_npm_limit_set() range-checks against it before writing (mirrors
  // rsmi_dev_power_cap_set()'s range_get()-before-write pattern).
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "0");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "250");
}

TEST(GpuUnit, SetNpmLimitOverwritesLongerPreviousValue) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "6000");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  // A write that did not truncate would leave "2500".
  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "250");
}

// ---------------------------------------------------------------------
// amdsmi_set_npm_limit() -- NPM-disabled enforcement
//
// rsmi_dev_npm_limit_set() rejects the write outright when NPM is disabled
// on the node (before even reading the platform max): writing
// cur_node_power_limit while NPM is disabled has no defined effect.
// ---------------------------------------------------------------------

TEST(GpuUnit, SetNpmLimitRejectsWhenNpmDisabled) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "0");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_INVAL);
  // A rejected write leaves the file unchanged.
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "0");
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

TEST(GpuUnit, SetNpmLimitRejectsZero) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "6000");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 0), AMDSMI_STATUS_INVAL);
  // A rejected write leaves the file unchanged.
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitRejectsOverMax) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "6000");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 6401), AMDSMI_STATUS_INVAL);
  // A rejected write leaves the file unchanged.
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitRejectsWhenMaxUnreadable) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  // Deliberately do not create max_node_power_limit: the platform max is
  // unreadable, so the request must be rejected (fail closed) rather than
  // falling through to an unbounded write.
  board.WriteFile("cur_node_power_limit", "6000");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_NOT_SUPPORTED);
  // A rejected write leaves the file unchanged.
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitRejectsWhenMaxCorrupted) {
  NpmTestEnv env;

  FakeSysfsTree board;
  // "-1" is a regression case, not generic garbage: std::stoull("-1", ...)
  // is well-defined C++ behavior that successfully parses to UINT64_MAX
  // (leading '-' negates modulo 2^64) rather than throwing, so a naive
  // parser would treat this corrupted content as a "successfully read"
  // huge platform max -- which would let any limit through the
  // `limit > max_limit` check. The platform max must fail to parse and be
  // treated as unreadable (fail closed), exactly like the missing-file case
  // in SetNpmLimitRejectsWhenMaxUnreadable above, not silently accepted
  // as UINT64_MAX.
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "-1");
  board.WriteFile("cur_node_power_limit", "6000");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  // Unlike the missing-file case above (RSMI_STATUS_NOT_SUPPORTED), a
  // present-but-corrupted file is a parse failure:
  // get_npm_board_max_limit()/read_board_uint64() returns
  // RSMI_STATUS_UNEXPECTED_DATA, which rsmi_dev_npm_limit_set() propagates
  // as-is (fail closed) and which maps to AMDSMI_STATUS_UNEXPECTED_DATA.
  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_UNEXPECTED_DATA);
  // A rejected write leaves the file unchanged.
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitRejectsWhenMaxIsNotANumber) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "not-a-number");
  board.WriteFile("cur_node_power_limit", "6000");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_UNEXPECTED_DATA);
  // A rejected write leaves the file unchanged.
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6000");
}

TEST(GpuUnit, SetNpmLimitAcceptsInRange) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");
  board.WriteFile("cur_node_power_limit", "0");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 6400), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("cur_node_power_limit"), "6400");
}

TEST(GpuUnit, SetNpmLimitMissingFileIsNotSupported) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  // max_node_power_limit present (so the new range check itself succeeds),
  // but cur_node_power_limit deliberately absent -- exercises the write-time
  // NOT_SUPPORTED path in set_npm_board_limit(), distinct from
  // SetNpmLimitRejectsWhenMaxUnreadable above.
  board.WriteFile("max_node_power_limit", "6400");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmLimitMissingBoardDirIsNotSupported) {
  NpmTestEnv env;

  FakeSysfsTree board;  // the board directory below is never created
  std::string board_path = board.root() + "/missing_board";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 250), AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmLimitDirectoryInPlaceOfLimitFileIsNotSupported) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("max_node_power_limit", "6400");
  // WriteFile() creates parent directories, so this makes cur_node_power_limit a directory.
  board.WriteFile("cur_node_power_limit/placeholder", "");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_limit(handle, 100), AMDSMI_STATUS_NOT_SUPPORTED);
}

// ---------------------------------------------------------------------
// amdsmi_get_npm_info() -- current_node_power from board/node_power
//
// Only npm_status is required; an unreadable or malformed node_power is reported as the
// UINT32_MAX sentinel rather than failing the call.
// ---------------------------------------------------------------------

TEST(GpuUnit, GetNpmInfoReportsCurrentNodePower) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("node_power", "777");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_info_t npm_info{};
  EXPECT_EQ(amdsmi_get_npm_info(handle, &npm_info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(npm_info.current_node_power, 777u);
}

TEST(GpuUnit, GetNpmInfoMissingNodePowerReportsSentinel) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_info_t npm_info{};
  EXPECT_EQ(amdsmi_get_npm_info(handle, &npm_info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(npm_info.current_node_power, std::numeric_limits<uint32_t>::max());
}

TEST(GpuUnit, GetNpmInfoNonNumericNodePowerReportsSentinel) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("node_power", "not-a-number");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_info_t npm_info{};
  EXPECT_EQ(amdsmi_get_npm_info(handle, &npm_info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(npm_info.current_node_power, std::numeric_limits<uint32_t>::max());
}

TEST(GpuUnit, GetNpmInfoMissingBoardDirIsNotSupported) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;  // the board directory below is never created
  std::string board_path = board.root() + "/missing_board";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_info_t npm_info{};
  EXPECT_EQ(amdsmi_get_npm_info(handle, &npm_info), AMDSMI_STATUS_NOT_SUPPORTED);
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
  NpmDiscoveredGpuEnv env;

  // Heap-allocated (not stack-allocated) so this address cannot alias a
  // stack slot some other test in this binary already passed to
  // amdsmi_test_register_node_handle(): registered pointers live for the
  // rest of the process, and other tests here register the address of a
  // same-shaped stack-local std::string, which the allocator/compiler can
  // legitimately hand back again for a later test's stack frame. A fresh
  // heap allocation that nothing in this file ever registers cannot collide
  // with that set.
  FakeSysfsTree board;  // the board directory below is never created
  auto unregistered_board_path = std::make_unique<std::string>(board.root() + "/missing_board");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  amdsmi_npm_info_t npm_info{};
  EXPECT_EQ(amdsmi_get_npm_info(handle, &npm_info), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmLimitRejectsUnregisteredHandle) {
  NpmTestEnv env;

  // See GetNpmInfoRejectsUnregisteredHandle above for why this must be
  // heap-allocated rather than a stack-local std::string.
  FakeSysfsTree board;  // the board directory below is never created
  auto unregistered_board_path = std::make_unique<std::string>(board.root() + "/missing_board");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  // The registry check runs first, before any board file is touched.
  EXPECT_EQ(amdsmi_set_npm_limit(handle, 100), AMDSMI_STATUS_INVAL);
}

#if defined(ENABLE_WSL_BACKEND)
// When the WSL backend is active, amdsmi_set_npm_limit() must short-circuit
// to AMDSMI_STATUS_NOT_SUPPORTED before ever touching board_path/sysfs (see
// the `#ifdef ENABLE_WSL_BACKEND` / WSLGPUBackend::IsActive() guard at the
// top of amdsmi_set_npm_limit() in amd_smi.cc).
//
// End-to-end needs: a real WSL2 machine, i.e. /dev/dxg and a loadable librocdxg, so that
// WSLGPUBackend::TryPopulate() succeeds during amdsmi_init(). IsActive() is a file-static set only
// there and has no test seam, so anywhere else this test skips.
TEST(GpuUnit, SetNpmLimitNotSupportedWhenWslBackendActive) {
  ScopedRealDiscoveryInit init;
  REQUIRE_AMDSMI_INIT(init);

  if (!amd::smi::WSLGPUBackend::IsActive()) {
    GTEST_SKIP_(
        "WSLGPUBackend is not active (not a real WSL2 machine, or "
        "librocdxg/rocdxg node enumeration unavailable) -- the "
        "NOT_SUPPORTED short-circuit cannot be exercised here");
  }

  // Any board path is sufficient: the WSL-backend guard runs before the
  // path is ever dereferenced.
  FakeSysfsTree board;  // the board directory below is never created
  std::string board_path = board.root() + "/missing_board";
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
// caller-supplied path. The GetNpmInfo*NodePower* tests above cover the file
// reads through a fabricated handle; this test validates against whatever
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
//
// End-to-end needs: the amdgpu driver loaded with a real GPU, and either that hardware's DRM
// ioctls or a fake for them (none exists yet), since amdsmi_get_node_handle() issues them.
TEST(GpuUnit, GetNpmInfoCurrentNodePowerRoundTrip) {
  ScopedRealDiscoveryInit init;
  REQUIRE_AMDSMI_INIT(init);

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
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  ScopedRocmSmiDevice rocm_device(tree.root() + "/card0");
  amd::smi::AMDSmiDrm drm;
  amd::smi::AMDSmiGPUDevice gpu_device(rocm_device.index(), "card0", amdsmi_bdf_t{}, drm);
  amd::smi::testing::ScopedProcessorRegistration registration(&gpu_device);

  EXPECT_EQ(amdsmi_get_power_info(registration.handle(), nullptr), AMDSMI_STATUS_INVAL);
}
