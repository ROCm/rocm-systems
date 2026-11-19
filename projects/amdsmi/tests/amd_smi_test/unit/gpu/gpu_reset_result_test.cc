// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// amdsmi_reset_gpu() against a fake amdgpu_gpu_recover debugfs file. The kernel performs the
// recovery inside the read and returns its result as the file's value, so a nonzero value must
// fail the call instead of being reported as a successful reset.

#include <gtest/gtest.h>
#include <unistd.h>

#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "test_common.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiDrm;
using amd::smi::AMDSmiGPUDevice;
using amd::smi::testing::ScopedOverride;
using amd::smi::testing::ScopedProcessorRegistration;
namespace test_hooks = amd::smi::test_hooks;

// ScopedRocmSmiDevice leaves the card index at 0, so the device's debugfs directory is "0".
constexpr char kGpuRecoverFile[] = "debugfs/0/amdgpu_gpu_recover";

amdsmi_status_t ResetWithRecoverValue(const std::string& recover_value) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  tree.WriteFile(kGpuRecoverFile, recover_value + "\n");
  ScopedOverride<std::string> root_guard(test_hooks::DebugfsDriRoot, test_hooks::SetDebugfsDriRoot,
                                         tree.root() + "/debugfs");

  ScopedRocmSmiDevice rocm_device(tree.root() + "/card0");
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), "card0", amdsmi_bdf_t{}, drm);
  ScopedProcessorRegistration registration(&gpu_device);

  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;
  DISPLAY_AMDSMI_API("amdsmi_reset_gpu", "amdgpu_gpu_recover=" + recover_value, verbose);
  return amdsmi_reset_gpu(registration.handle());
}

TEST(GpuUnit, GPU_RESET__ZeroRecoverResultSucceeds) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "amdsmi_reset_gpu requires root";
  }
  EXPECT_EQ(ResetWithRecoverValue("0"), AMDSMI_STATUS_SUCCESS);
}

TEST(GpuUnit, GPU_RESET__FailedRecoveryIsReported) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "amdsmi_reset_gpu requires root";
  }
  EXPECT_EQ(ResetWithRecoverValue("-110"), AMDSMI_STATUS_AMDGPU_RESTART_ERR);
}

// Older kernels print "gpu recover" with no result; that must keep reporting success.
TEST(GpuUnit, GPU_RESET__LegacyTextOutputSucceeds) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "amdsmi_reset_gpu requires root";
  }
  EXPECT_EQ(ResetWithRecoverValue("gpu recover"), AMDSMI_STATUS_SUCCESS);
}

}  // namespace
