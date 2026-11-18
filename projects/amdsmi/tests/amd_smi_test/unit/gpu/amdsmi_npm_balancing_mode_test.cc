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
// Permission failures are tested with file modes, as in ras_features_test.cc: see
// SetNpmBalancingModeUnwritableModeFileIsNoPerm. The board directory is a FakeSysfsTree, since
// set_npm_board_mode()/get_npm_board_mode() take an arbitrary board_path string.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_internal.h"
#include "test_fixture_utils.h"

namespace {

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

// ---------------------------------------------------------------------
// amdsmi_get_npm_balancing_mode()
// ---------------------------------------------------------------------

TEST(GpuUnit, GetNpmBalancingModeNullHandleIsInval) {
  NpmDiscoveredGpuEnv env;

  amdsmi_npm_balancing_mode_t mode;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(nullptr, &mode), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmBalancingModeNullModeIsInval) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;  // the board directory below is never created
  std::string board_path = board.root() + "/missing_board";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmBalancingModeDisabledStillReportsLastSelectedMode) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("mode", "1");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_INVALID;
  // Get is never gated on NPM enablement; it reflects board/mode even
  // while NPM is disabled.
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
}

TEST(GpuUnit, GetNpmBalancingModeEnabledDefaultsToPowerBalancing) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  // The platform/driver creates board/mode with its default value ("1")
  // when NPM is enabled; simulate that here rather than leaving the file
  // absent, since a genuinely missing/unreadable file now reports
  // AMDSMI_STATUS_SUCCESS + AMDSMI_NPM_BALANCING_MODE_INVALID (N/A).
  board.WriteFile("mode", "1");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_INVALID;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(mode, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING);
}

TEST(GpuUnit, GetNpmBalancingModeEnabledMissingModeFileIsNotSupported) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  // Deliberately do not create board/mode: the sysfs node absent/unreadable
  // must report NOT_SUPPORTED, agreeing with amdsmi_set_npm_balancing_mode()
  // for the identical state (see SetNpmBalancingModeMissingModeFileIsNotSupported).
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, GetNpmBalancingModeGarbageValueIsUnexpectedData) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "0");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  // "0" is not a valid encoding ("1"/"2" only); this is a genuine
  // read/decode error, not the "missing file" N/A case.
  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, GetNpmBalancingModeEnabledReadsFrequencyBalancing) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "2");
  std::string board_path = board.root();
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
  NpmTestEnv env;

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(nullptr, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmBalancingModeInvalidModeIsInval) {
  NpmTestEnv env;

  FakeSysfsTree board;  // the board directory below is never created
  std::string board_path = board.root() + "/missing_board";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, static_cast<amdsmi_npm_balancing_mode_t>(
                                                      AMDSMI_NPM_BALANCING_MODE_INVALID)),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmBalancingModeUnwritableModeFileIsNoPerm) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "1");
  board.SetPermissions("mode", 0444);
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  auto set_mode = [&] {
    return amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING);
  };
  EXPECT_EQ(amd::smi::testing::RunAsUnprivilegedUser(set_mode), AMDSMI_STATUS_NO_PERM);
  // A rejected write leaves the file unchanged.
  EXPECT_EQ(board.ReadFile("mode"), "1");
}

TEST(GpuUnit, SetNpmBalancingModeRejectsWhenNpmDisabled) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("mode", "1");
  std::string board_path = board.root();
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

TEST(GpuUnit, SetNpmBalancingModeRoundTrip) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "1");
  std::string board_path = board.root();
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

TEST(GpuUnit, SetNpmBalancingModeMissingModeFileIsNotSupported) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  // Deliberately do not create board/mode: set_npm_board_mode() requires
  // the file to already exist (it does not create it), mirroring
  // set_npm_board_limit()'s missing-file handling.
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmBalancingModeMissingBoardDirIsNotSupported) {
  NpmTestEnv env;

  FakeSysfsTree board;  // the board directory below is never created
  std::string board_path = board.root() + "/missing_board";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, SetNpmBalancingModeRejectsModeAbsentFromSupportedBitmask) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "1");
  // bit 2 (FREQUENCY_BALANCING) only -- POWER_BALANCING (bit 1) is absent.
  board.WriteFile("supported_mode", "0x4");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_POWER_BALANCING),
            AMDSMI_STATUS_SETTING_UNAVAILABLE);
  // The file must be untouched.
  EXPECT_EQ(board.ReadFile("mode"), "1");
}

TEST(GpuUnit, SetNpmBalancingModeAllowsModePresentInSupportedBitmask) {
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "1");
  // bits 1+2 -- both modes supported, matching today's real platforms.
  board.WriteFile("supported_mode", "0x6");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(board.ReadFile("mode"), "2");
}

TEST(GpuUnit, SetNpmBalancingModeRejectsCorruptSupportedModesFile) {
  // A read failure other than NOT_SUPPORTED (e.g. corrupt/unparsable
  // content) must fail closed as SETTING_UNAVAILABLE and reject the write --
  // it must not be treated like a missing file.
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "1");
  board.WriteFile("supported_mode", "-1");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_SETTING_UNAVAILABLE);
  // The file must be untouched.
  EXPECT_EQ(board.ReadFile("mode"), "1");
}

TEST(GpuUnit, SetNpmBalancingModeToleratesMissingSupportedModesFile) {
  // The supported_mode sysfs file is not yet implemented on all driver
  // versions; its absence must not block an otherwise-valid set.
  NpmTestEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "enabled");
  board.WriteFile("mode", "1");
  // Deliberately do not create board/supported_mode.
  std::string board_path = board.root();
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
  NpmDiscoveredGpuEnv env;

  amdsmi_bit_field_t supported_modes;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(nullptr, &supported_modes),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesNullOutputIsInval) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;  // the board directory below is never created
  std::string board_path = board.root() + "/missing_board";
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesMissingFileIsNotSupported) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  // Deliberately do not create board/supported_mode: not yet implemented
  // by the driver on all platforms.
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes),
            AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesNotGatedOnNpmEnablement) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  board.WriteFile("npm_status", "disabled");
  board.WriteFile("supported_mode", "0x6");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(supported_modes, 0x6u);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesDecodesHexBitmask) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  board.WriteFile("supported_mode", "0x2");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(supported_modes, 0x2u);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesGarbageValueIsUnexpectedData) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  board.WriteFile("supported_mode", "not-a-hex-number");
  std::string board_path = board.root();
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
  NpmDiscoveredGpuEnv env;

  // Heap-allocated so this address cannot alias a stack slot some other test
  // in this binary already passed to amdsmi_test_register_node_handle() --
  // see GetNpmInfoRejectsUnregisteredHandle in amdsmi_npm_limit_test.cc for
  // the full rationale.
  FakeSysfsTree board;  // the board directory below is never created
  auto unregistered_board_path = std::make_unique<std::string>(board.root() + "/missing_board");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  amdsmi_npm_balancing_mode_t mode = AMDSMI_NPM_BALANCING_MODE_INVALID;
  EXPECT_EQ(amdsmi_get_npm_balancing_mode(handle, &mode), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, SetNpmBalancingModeRejectsUnregisteredHandle) {
  NpmTestEnv env;

  FakeSysfsTree board;  // the board directory below is never created
  auto unregistered_board_path = std::make_unique<std::string>(board.root() + "/missing_board");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  // The registry check runs first, before any board file is touched.
  EXPECT_EQ(amdsmi_set_npm_balancing_mode(handle, AMDSMI_NPM_BALANCING_MODE_FREQUENCY_BALANCING),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesRejectsUnregisteredHandle) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;  // the board directory below is never created
  auto unregistered_board_path = std::make_unique<std::string>(board.root() + "/missing_board");
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(unregistered_board_path.get());

  amdsmi_bit_field_t supported_modes = 0;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetNpmSupportedBalancingModesNegativeValueIsUnexpectedData) {
  NpmDiscoveredGpuEnv env;

  FakeSysfsTree board;
  // std::stoull("-1", ..., 16) would otherwise silently parse as UINT64_MAX;
  // must be rejected as a parse error instead.
  board.WriteFile("supported_mode", "-1");
  std::string board_path = board.root();
  amdsmi_node_handle handle = reinterpret_cast<amdsmi_node_handle>(&board_path);
  ASSERT_EQ(amdsmi_test_register_node_handle(handle), AMDSMI_STATUS_SUCCESS);

  amdsmi_bit_field_t supported_modes = 0;
  EXPECT_EQ(amdsmi_get_npm_supported_balancing_modes(handle, &supported_modes),
            AMDSMI_STATUS_UNEXPECTED_DATA);
}
