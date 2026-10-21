// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
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

// Board fields must come from the overridden DRM root, not the host's sysfs.
// Distinct fixture values detect paths that still use the hardcoded root.
TEST(GpuUnit, SysfsDrmRootBoardInfoUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // Ordinary temporary files stand in for driver-provided sysfs attributes.
  // FakeSysfsTree removes them when the test ends.
  tree.WriteFile("card0/device/product_number", "MODEL_TEST_123\n");
  tree.WriteFile("card0/device/serial_number", "SERIAL_TEST_456\n");
  tree.WriteFile("card0/device/fru_id", "FRU_TEST_789\n");
  tree.WriteFile("card0/device/manufacturer", "TEST_MANUFACTURER\n");
  tree.WriteFile("card0/device/product_name", "TEST_GPU\n");

  // The root includes its trailing slash; the guard restores the previous
  // root on scope exit, including an early return from the test.
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  // Supply the device and mutex backing needed by the real public API
  // without requiring a physical GPU.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/card0");
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), "card0", amdsmi_bdf_t{}, drm);

  // Public API handle validation requires a registered processor.
  // The guard unregisters it before the synthetic device is destroyed.
  ScopedProcessorRegistration registration(&gpu_device);

  amdsmi_board_info_t board_info{};

  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_board_info", "card=card0", verbose);
  amdsmi_status_t status = amdsmi_get_gpu_board_info(registration.handle(), &board_info);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);

  // SUCCESS alone does not prove the individual board files were read.
  // Check every field so fallback values cannot hide an unmigrated path.
  EXPECT_STREQ(board_info.model_number, "MODEL_TEST_123");
  EXPECT_STREQ(board_info.product_serial, "SERIAL_TEST_456");
  EXPECT_STREQ(board_info.fru_id, "FRU_TEST_789");
  EXPECT_STREQ(board_info.manufacturer_name, "TEST_MANUFACTURER");
  EXPECT_STREQ(board_info.product_name, "TEST_GPU");
}

}  // namespace

// VCN activity must be read from the overridden DRM root,
// not from the host's sysfs directory.
TEST(GpuUnit, SysfsDrmRootVcnBusyUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // Use a temporary file so the expected value does not depend on GPU activity.
  tree.WriteFile("card0/device/vcn_busy_percent", "37\n");

  // Point the root helper at our fixture and restore its previous value
  // automatically when the test leaves this scope.
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  // Supply the device and mutex needed by the real API without a physical GPU.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/card0");
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), "card0", amdsmi_bdf_t{}, drm);

  // Register the synthetic GPU so public API handle validation accepts it.
  ScopedProcessorRegistration registration(&gpu_device);

  uint32_t busy_percent = 0;
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  DISPLAY_AMDSMI_API("amdsmi_get_vcn_busy_percent", "card=card0", verbose);
  amdsmi_status_t status = amdsmi_get_vcn_busy_percent(registration.handle(), &busy_percent);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  // Check the file's value as well as the status; SUCCESS alone is insufficient.
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(busy_percent, 37u);
}  // namespace
