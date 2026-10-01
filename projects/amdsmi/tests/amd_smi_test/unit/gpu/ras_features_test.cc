// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Regression tests for amdsmi_get_gpu_total_ecc_count()'s RAS enabled-blocks
// mask read, driven through the real, unmodified public API with a synthetic
// GPU registered via AMDSmiSystem and a real (hardware-inert) rocm_smi::Device
// -- no GPU required. See docs/design/faking-external-interfaces.md for the
// general pattern this instantiates.
//
// Guards the regression: a failed/malformed ras/features read previously fell
// through to AMDSMI_STATUS_SUCCESS with every ecc count left at 0,
// indistinguishable from a GPU that genuinely has no errors. It now propagates
// the failure so callers can tell "RAS state unknown" apart from a real zero.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiDrm;
using amd::smi::AMDSmiGPUDevice;
using amd::smi::testing::ScopedOverride;
using amd::smi::testing::ScopedProcessorRegistration;

// Fixture content for a real, arbitrary-but-valid RAS features mask, shared
// by every test that just needs "a mask that parses" rather than a specific
// bit pattern.
constexpr char kValidFeatureMaskLine[] = "feature mask: 0x1abf877\n";
constexpr char kZeroFeatureMaskLine[] = "feature mask: 0x0\n";
constexpr char kAllOnesFeatureMaskLine[] = "feature mask: 0xffffffffffffffff\n";

// Constructs the full synthetic device stack (rocm_smi::Device + AMDSmiDrm +
// AMDSmiGPUDevice + AMDSmiSystem registration) and calls the real public
// amdsmi_get_gpu_total_ecc_count() against it.
//
// `tree` supplies the SAME root to two independent things, which is why the
// production code and the test end up reading the exact same file:
//   1. The sysfs-root override the caller already pointed at tree.root() via
//      ScopedOverride, so smi_amdgpu_get_enabled_blocks() builds its path as
//      tree.root() + "/" + card + "/device/ras/features" -- e.g.
//      "/tmp/amdsmi_fake_sysfs_XXXXXX/card0/device/ras/features"
//      (production default, no override active: "/sys/class/drm/card0/device/ras/features").
//   2. The fake rocm_smi::Device's own path (tree.root() + "/" + card), used
//      only to satisfy SMIGPUDEVICE_MUTEX -- not read from in these tests.
amdsmi_status_t CallTotalEccCount(const FakeSysfsTree& tree, const std::string& card,
                                  amdsmi_error_count_t* ec) {
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + card);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), card, amdsmi_bdf_t{}, drm);
  ScopedProcessorRegistration registration(&gpu_device);

  return amdsmi_get_gpu_total_ecc_count(registration.handle(), ec);
}

TEST(GpuUnit, RAS_FEATURES__ValidMaskSucceeds) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  tree.WriteFile("card0/device/ras/features", kValidFeatureMaskLine);
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  amdsmi_error_count_t ec = {};
  EXPECT_EQ(CallTotalEccCount(tree, "card0", &ec), AMDSMI_STATUS_SUCCESS);
}

// Regression test: the accumulation loop only adds to *ec via +=; if the function
// didn't zero *ec up front, a caller-supplied struct holding stale values (e.g.
// reused across calls, as a real caller might) would leak straight through on
// AMDSMI_STATUS_SUCCESS. Writes a real per-block (UMC) count file so the final
// values are checked against the actual parsed counts, not just against zero --
// proving both that the stale sentinel didn't leak AND that accumulation works.
TEST(GpuUnit, RAS_FEATURES__ValidMaskOverwritesStaleOutput) {
  constexpr uint64_t kSentinelEccCount = 4242;
  constexpr uint64_t kUmcUncorrectable = 5;
  constexpr uint64_t kUmcCorrectable = 3;

  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  tree.WriteFile("card0/device/ras/features", kValidFeatureMaskLine);
  // kValidFeatureMaskLine enables the UMC block (bit 0); this is the file
  // amdsmi_get_gpu_ecc_count() reads for it.
  tree.WriteFile("card0/device/ras/umc_err_count", "ue: " + std::to_string(kUmcUncorrectable) +
                                                       "\nce: " + std::to_string(kUmcCorrectable) +
                                                       "\n");
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  amdsmi_error_count_t ec = {};
  ec.correctable_count = ec.uncorrectable_count = ec.deferred_count = kSentinelEccCount;
  EXPECT_EQ(CallTotalEccCount(tree, "card0", &ec), AMDSMI_STATUS_SUCCESS);
  // Neither the sentinel nor sentinel+real (accumulated onto stale data) --
  // exactly the real UMC counts, proving the reset happened before summing.
  constexpr const char* kLeakHint =
      "a result of sentinel + real count means *ec wasn't zeroed before accumulating";
  EXPECT_EQ(ec.correctable_count, kUmcCorrectable) << kLeakHint;
  EXPECT_EQ(ec.uncorrectable_count, kUmcUncorrectable) << kLeakHint;
  EXPECT_EQ(ec.deferred_count, 0) << kLeakHint;
}

TEST(GpuUnit, RAS_FEATURES__ZeroMaskFails) {
  // Non-zero sentinel to prove the failure path leaves ec untouched rather
  // than falling through to a misleading zero total.
  constexpr uint64_t kSentinelEccCount = 4242;

  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  tree.WriteFile("card0/device/ras/features", kZeroFeatureMaskLine);
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  amdsmi_error_count_t ec = {};
  ec.correctable_count = ec.uncorrectable_count = ec.deferred_count = kSentinelEccCount;
  EXPECT_EQ(CallTotalEccCount(tree, "card0", &ec), AMDSMI_STATUS_API_FAILED);
  // The failure must propagate instead of masking as a misleading zero total.
  EXPECT_EQ(ec.correctable_count, kSentinelEccCount);
}

TEST(GpuUnit, RAS_FEATURES__UlongMaxMaskFails) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  tree.WriteFile("card0/device/ras/features", kAllOnesFeatureMaskLine);
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  amdsmi_error_count_t ec = {};
  EXPECT_EQ(CallTotalEccCount(tree, "card0", &ec), AMDSMI_STATUS_API_FAILED);
}

TEST(GpuUnit, RAS_FEATURES__EmptyFileFails) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  tree.WriteFile("card0/device/ras/features", "");
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  amdsmi_error_count_t ec = {};
  EXPECT_EQ(CallTotalEccCount(tree, "card0", &ec), AMDSMI_STATUS_API_FAILED);
}

TEST(GpuUnit, RAS_FEATURES__MissingValueTokenFails) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  tree.WriteFile("card0/device/ras/features", "feature mask:\n");
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  amdsmi_error_count_t ec = {};
  EXPECT_EQ(CallTotalEccCount(tree, "card0", &ec), AMDSMI_STATUS_API_FAILED);
}

TEST(GpuUnit, RAS_FEATURES__NonHexTokenFails) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  tree.WriteFile("card0/device/ras/features", "feature mask: garbage\n");
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  amdsmi_error_count_t ec = {};
  EXPECT_EQ(CallTotalEccCount(tree, "card0", &ec), AMDSMI_STATUS_API_FAILED);
}

// No fixture file written at all: the real ifstream-open-fail branch, through
// the real default sysfs root (no guard active).
TEST(GpuUnit, RAS_FEATURES__MissingFileNeverWrittenIsNotSupported) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;  // card0/device/ras/features intentionally never written
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  amdsmi_error_count_t ec = {};
  EXPECT_EQ(CallTotalEccCount(tree, "card0", &ec), AMDSMI_STATUS_NOT_SUPPORTED);
}

// File present, then removed mid-scenario: same NOT_SUPPORTED path, reached a
// second, distinct way.
TEST(GpuUnit, RAS_FEATURES__FileRemovedAfterCreationIsNotSupported) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  tree.WriteFile("card0/device/ras/features", kValidFeatureMaskLine);
  tree.RemoveFile("card0/device/ras/features");
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  amdsmi_error_count_t ec = {};
  EXPECT_EQ(CallTotalEccCount(tree, "card0", &ec), AMDSMI_STATUS_NOT_SUPPORTED);
}

// Permission-denied read (EACCES). Real root bypasses file-mode checks via
// CAP_DAC_OVERRIDE, so under root this forks a child that permanently drops
// to an unprivileged UID (kUnprivilegedId, "nobody") before repeating the
// same call -- forcing a genuine permission check regardless of who runs the
// suite. The child never rejoins gtest bookkeeping (per
// cross_process_serialization_test.cc's precedent: gtest state doesn't cross
// fork()); it reports its raw amdsmi_status_t back to the parent over a pipe,
// and only the parent asserts.
TEST(GpuUnit, RAS_FEATURES__PermissionDeniedIsNotSupported) {
  // Conventional "nobody" uid/gid on Linux; any real unprivileged account
  // would do, but this one is guaranteed to exist.
  constexpr uid_t kUnprivilegedId = 65534;
  constexpr mode_t kNoPermissions = 0000;
  constexpr mode_t kTraversableDirMode = 0755;
  constexpr int kReadEnd = 0;
  constexpr int kWriteEnd = 1;

  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  tree.WriteFile("card0/device/ras/features", kValidFeatureMaskLine);
  tree.SetPermissions("card0/device/ras/features", kNoPermissions);
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  if (geteuid() != 0) {
    // Non-root: the 0000 mode above already forces a real EACCES, no fork needed.
    amdsmi_error_count_t ec = {};
    EXPECT_EQ(CallTotalEccCount(tree, "card0", &ec), AMDSMI_STATUS_NOT_SUPPORTED);
  } else {
    // mkdtemp() makes tree.root() itself 0700 (owner-only); relax it so the
    // unprivileged child below can still traverse down to the 0000 leaf file --
    // otherwise it would fail on directory traversal instead of the file mode.
    ASSERT_EQ(chmod(tree.root().c_str(), kTraversableDirMode), 0);

    int result_pipe[2];
    ASSERT_EQ(pipe(result_pipe), 0);
    pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      close(result_pipe[kReadEnd]);
      amdsmi_status_t status = AMDSMI_STATUS_UNKNOWN_ERROR;
      // Order matters: dropping uid before gid would forfeit permission to
      // change gid. Both are permanent for this process -- it only ever exits.
      // Logged directly (not ADD_FAILURE()): gtest state doesn't cross fork(),
      // so this is the only way a failure here becomes visible in test output.
      if (setgid(kUnprivilegedId) != 0) {
        std::cerr << "setgid(" << kUnprivilegedId << ") failed: " << strerror(errno) << std::endl;
      } else if (setuid(kUnprivilegedId) != 0) {
        std::cerr << "setuid(" << kUnprivilegedId << ") failed: " << strerror(errno) << std::endl;
      } else {
        amdsmi_error_count_t ec = {};
        status = CallTotalEccCount(tree, "card0", &ec);
      }
      ssize_t bytes_written = write(result_pipe[kWriteEnd], &status, sizeof(status));
      (void)bytes_written;
      close(result_pipe[kWriteEnd]);
      _exit(0);
    }
    close(result_pipe[kWriteEnd]);
    amdsmi_status_t child_status = AMDSMI_STATUS_UNKNOWN_ERROR;
    ssize_t bytes_read = read(result_pipe[kReadEnd], &child_status, sizeof(child_status));
    close(result_pipe[kReadEnd]);
    waitpid(child, nullptr, 0);

    ASSERT_EQ(bytes_read, static_cast<ssize_t>(sizeof(child_status)));
    EXPECT_EQ(child_status, AMDSMI_STATUS_NOT_SUPPORTED);
  }
}

}  // namespace
