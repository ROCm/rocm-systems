// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for sysfs DRM root overrides, exercising real public APIs and
// internal helpers with synthetic GPU devices and temporary sysfs files
// -- no GPU required. See docs/design/faking-external-interfaces.md for the
// fixture pattern.
//
// Guards against regressions that bypass the shared root helper with
// hardcoded "/sys/class/drm/" paths. Exact fixture values and write-back
// checks verify that reads and writes reach the temporary tree; SUCCESS
// alone cannot prove the path used. Scoped guards restore the root and
// device registrations, and temporary files are removed after each test.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_gpu_metrics.h"
#include "rocm_smi/rocm_smi_main.h"
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

// VCN activity must be read from the overridden DRM root,
// not from the host's sysfs directory.
TEST(GpuUnit, SysfsDrmRootVcnBusyUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // Prevent a hardcoded root from accidentally reading matching host GPU activity.
  const std::string gpu_path = "card_test_vcn";

  // Use a temporary file so the expected value does not depend on GPU activity.
  tree.WriteFile(gpu_path + "/device/vcn_busy_percent", "37\n");

  // Point the root helper at our fixture and restore its previous value
  // automatically when the test leaves this scope.
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  // Supply the device and mutex needed by the real API without a physical GPU.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);

  // Register the synthetic GPU so public API handle validation accepts it.
  ScopedProcessorRegistration registration(&gpu_device);

  uint32_t busy_percent = 0;
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  DISPLAY_AMDSMI_API("amdsmi_get_vcn_busy_percent", "card=" + gpu_path, verbose);
  amdsmi_status_t status = amdsmi_get_vcn_busy_percent(registration.handle(), &busy_percent);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  // Check the file's value as well as the status; SUCCESS alone is insufficient.
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(busy_percent, 37u);
}

// Hwmon discovery must use the overridden DRM root, not the host's sysfs.
TEST(GpuUnit, SysfsDrmRootHwmonDirUsesOverride) {
  // Initialize library state without discovering physical hardware.
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // Discovery requires an AMD vendor ID before it scans the hwmon directory.
  tree.WriteFile("card0/device/vendor", "0x1002\n");

  // Writing this file creates the parent hwmon42 directory for discovery.
  // The helper finds the directory; it does not read the name file.
  tree.WriteFile("card0/device/hwmon/hwmon42/name", "amdgpu\n");

  // The helper expects a trailing slash. The guard restores the previous root
  // when the test exits, so later tests are unaffected.
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  // Supply the device and mutex required by the real helper without a GPU.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/card0");
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), "card0", amdsmi_bdf_t{}, drm);

  std::string hwmon_path;
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  // This internal helper takes a device pointer, so handle registration is unnecessary.
  DISPLAY_AMDSMI_API("smi_amdgpu_find_hwmon_dir", "card=card0", verbose);
  amdsmi_status_t status = smi_amdgpu_find_hwmon_dir(&gpu_device, &hwmon_path);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  // SUCCESS alone could hide discovery under the host's sysfs.
  // Require the exact fixture path to prove the root override was used.
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(hwmon_path, tree.root() + "/card0/device/hwmon/hwmon42");
}

// Bad-page records must come from the overridden DRM root, not the host's sysfs.
TEST(GpuUnit, SysfsDrmRootBadPageInfoUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // A synthetic device name prevents the old hardcoded path from finding
  // a real GPU's bad-page records during failure verification.
  const std::string gpu_path = "card_test_bad_pages";

  // Records contain a hexadecimal address, hexadecimal size, and status.
  // 'R' identifies a page that has been reserved.
  tree.WriteFile(gpu_path + "/device/ras/gpu_vram_bad_pages", "0x12345000 : 0x1000 : R\n");

  // Keep the trailing slash and restore the previous root when the test exits.
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  // Supply a real device object and mutex without discovering physical hardware.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);
  ScopedProcessorRegistration registration(&gpu_device);

  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;
  uint32_t num_pages = 0;

  // Query the count before supplying storage for the single fixture record.
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_info", "card=" + gpu_path, verbose);
  amdsmi_status_t status = amdsmi_get_gpu_bad_page_info(registration.handle(), &num_pages, nullptr);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_pages, 1u);

  amdsmi_retired_page_record_t page{};
  status = amdsmi_get_gpu_bad_page_info(registration.handle(), &num_pages, &page);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_pages, 1u);

  // Check the parsed fixture values, not just SUCCESS or the record count.
  EXPECT_EQ(page.page_address, 0x12345000ULL);
  EXPECT_EQ(page.page_size, 0x1000ULL);
  EXPECT_EQ(page.status, AMDSMI_MEM_PAGE_STATUS_RESERVED);
}

TEST(GpuUnit, SysfsDrmRootEccErrorCountUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // Avoid reading a real GPU's counters if the hardcoded root returns.
  const std::string gpu_path = "card_test_ecc";

  // Different fallback values prove the primary file retains precedence.
  tree.WriteFile(gpu_path + "/device/ras/umc_err_count", "ue: 7\nce: 13\n");
  tree.WriteFile(gpu_path + "/device/ras/aca_umc", "ue: 19\nce: 23\n");

  // Restore the previous root automatically when the test exits.
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  // Supply the device and mutex required by the helper without physical hardware.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);

  amdsmi_error_count_t error_count{};
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  DISPLAY_AMDSMI_API("smi_amdgpu_get_ecc_error_count", "card=" + gpu_path, verbose);
  amdsmi_status_t status = smi_amdgpu_get_ecc_error_count(&gpu_device, &error_count);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

  // Exact counters prove that the primary fixture was read and parsed.
  EXPECT_EQ(error_count.uncorrectable_count, 7u);
  EXPECT_EQ(error_count.correctable_count, 13u);
}

TEST(GpuUnit, SysfsDrmRootEccErrorCountFallbackUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  const std::string gpu_path = "card_test_ecc";

  // Leave umc_err_count absent to force the existing ACA fallback path.
  tree.WriteFile(gpu_path + "/device/ras/aca_umc", "ue: 19\nce: 23\n");

  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);

  amdsmi_error_count_t error_count{};
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  DISPLAY_AMDSMI_API("smi_amdgpu_get_ecc_error_count", "card=" + gpu_path, verbose);
  amdsmi_status_t status = smi_amdgpu_get_ecc_error_count(&gpu_device, &error_count);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

  // The fallback must read from the same overridden root as the primary file.
  EXPECT_EQ(error_count.uncorrectable_count, 19u);
  EXPECT_EQ(error_count.correctable_count, 23u);
}

TEST(GpuUnit, SysfsDrmRootPowerManagementEnabledUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // Prevent the hardcoded root from accidentally finding a physical GPU.
  const std::string gpu_path = "card_test_power_management";

  tree.WriteFile(gpu_path + "/device/pp_features",
                 "0. FEATURE_A (0) disabled\n"
                 "1. FEATURE_B (1) enabled\n");

  // Keep the required trailing slash; the guard restores the root on exit.
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  // Supply the device and mutex without requiring physical hardware.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);

  // Start false so success without updating the output cannot pass.
  bool enabled = false;
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  // The internal helper accepts a device pointer, so no handle registration is needed.
  DISPLAY_AMDSMI_API("smi_amdgpu_is_gpu_power_management_enabled", "card=" + gpu_path, verbose);
  amdsmi_status_t status = smi_amdgpu_is_gpu_power_management_enabled(&gpu_device, &enabled);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  // Both assertions are required to prove the overridden fixture was read.
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(enabled);
}

TEST(GpuUnit, SysfsDrmRootClockRangesUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // Prevent an unmigrated path from reading a physical GPU's clock data.
  const std::string gpu_path = "card_test_clock_ranges";

  // DPM levels supply the highest level index and the sleep frequency.
  tree.WriteFile(gpu_path + "/device/pp_dpm_sclk",
                 "S: 100Mhz\n"
                 "0: 600Mhz\n"
                 "1: 1200Mhz *\n"
                 "2: 1800Mhz\n");

  // Distinct OD limits prove this file was read instead of falling back
  // to the DPM range of 600-1800 MHz.
  tree.WriteFile(gpu_path + "/device/pp_od_clk_voltage",
                 "OD_SCLK:\n"
                 "0: 500Mhz\n"
                 "1: 2100Mhz\n");

  // Keep the required trailing slash; restore the previous root on exit.
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  // Supply the device and mutex without requiring physical hardware.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);

  int max_freq = 0;
  int min_freq = 0;
  int num_dpm = 0;
  int sleep_state_freq = 0;
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  // The internal helper accepts a device pointer; no handle registration is needed.
  DISPLAY_AMDSMI_API("smi_amdgpu_get_ranges", "card=" + gpu_path + ", domain=GFX", verbose);
  amdsmi_status_t status = smi_amdgpu_get_ranges(&gpu_device, AMDSMI_CLK_TYPE_GFX, &max_freq,
                                                 &min_freq, &num_dpm, &sleep_state_freq);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

  // Exact values verify that both files came from the overridden root.
  EXPECT_EQ(max_freq, 2100);
  EXPECT_EQ(min_freq, 500);

  // Despite its name, num_dpm returns the highest level index, not the count.
  EXPECT_EQ(num_dpm, 2);
  EXPECT_EQ(sleep_state_freq, 100);
}

TEST(GpuUnit, SysfsDrmRootPcieInfoUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // Prevent either hardcoded path from accidentally reading a physical GPU.
  const std::string gpu_path = "card_test_pcie";

  tree.WriteFile(gpu_path + "/device/max_link_width", "16\n");
  tree.WriteFile(gpu_path + "/device/max_link_speed", "32.0 GT/s PCIe\n");

  // The API reads GPU metrics after the static link attributes.
  // Supply a valid v1.0 table so the entire call can succeed without hardware.
  amd::smi::AMDGpuMetrics_v10_t metrics{};
  metrics.m_common_header.m_structure_size = static_cast<uint16_t>(sizeof(metrics));
  metrics.m_common_header.m_format_revision = 1;
  metrics.m_common_header.m_content_revision = 0;

  // Different current and maximum values distinguish the two data sources.
  metrics.m_pcie_link_width = 8;
  metrics.m_pcie_link_speed = 160;  // Metrics use 0.1 GT/s: 160 means 16 GT/s.

  // An explicit length preserves embedded null bytes in the binary table.
  tree.WriteFile(gpu_path + "/device/gpu_metrics",
                 std::string(reinterpret_cast<const char*>(&metrics), sizeof(metrics)));

  // Redirect both static link reads and restore the previous root on exit.
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  // ROCm SMI reads gpu_metrics through this device path, independently of
  // the root helper used by the two static link attributes.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);

  // Public API handle validation requires a registered processor.
  ScopedProcessorRegistration registration(&gpu_device);

  amdsmi_pcie_info_t info{};
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  DISPLAY_AMDSMI_API("amdsmi_get_pcie_info", "card=" + gpu_path, verbose);
  amdsmi_status_t status = amdsmi_get_pcie_info(registration.handle(), &info);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

  // Exact static values prove that both overridden link files were read.
  // The API converts 32 GT/s to 32000 MT/s and identifies PCIe generation 5.
  EXPECT_EQ(info.pcie_static.max_pcie_width, 16u);
  EXPECT_EQ(info.pcie_static.max_pcie_speed, 32000u);
  EXPECT_EQ(info.pcie_static.pcie_interface_version, 5u);

  // Confirm the metrics fixture also reached the public API successfully.
  EXPECT_EQ(info.pcie_metric.pcie_width, 8u);
  EXPECT_EQ(info.pcie_metric.pcie_speed, 16000u);
}

TEST(GpuUnit, SysfsDrmRootUmaCarveoutInfoUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // A synthetic card name prevents reads from a physical GPU and keeps
  // ASIC detection from selecting the platform-wide fwupd APU path.
  const std::string gpu_path = "card_test_uma_carveout";

  // Match sysfs's newline-terminated value. Without trailing whitespace,
  // integer extraction can set eofbit, which the current reader rejects.
  tree.WriteFile(gpu_path + "/device/uma/carveout", "1\n");

  // Distinct descriptions let us verify the options file was actually read,
  // not merely that the current-index file was found.
  tree.WriteFile(gpu_path + "/device/uma/carveout_options",
                 "0: Minimum (512 MB)\n"
                 "1: Medium (1 GB)\n"
                 "2: Maximum (2 GB)\n");

  // Keep the required trailing slash; restore the previous root even if
  // an ASSERT below exits the test early.
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  // Provide the real device and mutex backing without hardware discovery.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);

  // The public getter validates its handle before calling the static helper.
  ScopedProcessorRegistration registration(&gpu_device);

  amdsmi_uma_carveout_info_t info{};
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  DISPLAY_AMDSMI_API("amdsmi_get_gpu_uma_carveout_info", "card=" + gpu_path, verbose);
  amdsmi_status_t status = amdsmi_get_gpu_uma_carveout_info(registration.handle(), &info);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

  // A nonzero current index proves the first file populated the output.
  EXPECT_EQ(info.current_index, 1u);

  // Verify every fixture option to prove the second path uses the same root.
  ASSERT_EQ(info.num_options, 3u);
  EXPECT_EQ(info.options[0].index, 0u);
  EXPECT_STREQ(info.options[0].description, "Minimum (512 MB)");
  EXPECT_EQ(info.options[1].index, 1u);
  EXPECT_STREQ(info.options[1].description, "Medium (1 GB)");
  EXPECT_EQ(info.options[2].index, 2u);
  EXPECT_STREQ(info.options[2].description, "Maximum (2 GB)");
}

TEST(GpuUnit, SysfsDrmRootUmaCarveoutSetUsesOverride) {
  // This test must exercise the write, not the dry-run SUCCESS path.
  // Check the environment without modifying global state for other tests.
  const char* dry_run = std::getenv("AMDSMI_DRY_RUN");
  ASSERT_TRUE(dry_run == nullptr || std::string(dry_run) != "1")
      << "Run this test with AMDSMI_DRY_RUN=0 to verify the actual file write.";

  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // Avoid real GPU paths and keep ASIC detection away from the fwupd APU path.
  const std::string gpu_path = "card_test_uma_carveout_set";
  const std::string carveout_relative_path = gpu_path + "/device/uma/carveout";

  // Start with a different index so returning SUCCESS without writing fails.
  // The newline also matches the format expected by the internal getter.
  tree.WriteFile(carveout_relative_path, "0\n");

  // The setter calls the internal getter to validate the requested index.
  tree.WriteFile(gpu_path + "/device/uma/carveout_options",
                 "0: Minimum (512 MB)\n"
                 "1: Medium (1 GB)\n"
                 "2: Maximum (2 GB)\n");

  // Redirect the setter and its validation reads to the same temporary tree.
  // The guard restores the previous root even when an assertion exits early.
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  // Supply real library device objects and mutex backing without a real GPU.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);

  // Public API handle validation requires the synthetic device to be registered.
  ScopedProcessorRegistration registration(&gpu_device);

  const uint32_t requested_index = 1;
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  DISPLAY_AMDSMI_API("amdsmi_set_gpu_uma_carveout", "card=" + gpu_path + ", option_index=1",
                     verbose);
  amdsmi_status_t status = amdsmi_set_gpu_uma_carveout(registration.handle(), requested_index);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

  // Read the fixture directly rather than using the getter: this independently
  // verifies the write destination and avoids the getter's newline expectation.
  std::ifstream carveout_file(tree.root() + "/" + carveout_relative_path);
  ASSERT_TRUE(carveout_file.is_open());

  std::string written_value;
  ASSERT_TRUE(static_cast<bool>(std::getline(carveout_file, written_value)));

  // SUCCESS alone is insufficient; the original "0" must have changed to "1".
  EXPECT_EQ(written_value, "1");
}

TEST(GpuUnit, SysfsDrmRootPpDpmClockFrequenciesUseRenderNode) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // Keep the GPU path distinct from the render node. This catches a refactor
  // that incorrectly replaces renderD<N> addressing with get_gpu_path().
  const std::string gpu_path = "card_test_pp_dpm";
  const uint32_t render_minor = 4242;
  const std::string render_path = "renderD" + std::to_string(render_minor);
  const char* pp_dpm_file = "pp_dpm_vclk";

  // The middle entry is current; distinct frequencies verify the parsed data
  // and the MHz-to-Hz conversion, not merely a successful file open.
  tree.WriteFile(render_path + "/device/" + pp_dpm_file,
                 "0: 300Mhz\n"
                 "1: 700Mhz *\n"
                 "2: 1100Mhz\n");

  // A readable decoy makes incorrect GPU-path addressing return different
  // data, so SUCCESS alone cannot hide selection of the wrong directory.
  tree.WriteFile(gpu_path + "/device/" + pp_dpm_file, "0: 99Mhz *\n");

  // Redirect the root while preserving the helper's render-node construction.
  // The guard restores the original root even if an assertion exits early.
  ScopedOverride<std::string> root_guard(
      smi_amdgpu_sysfs_drm_root, smi_amdgpu_set_sysfs_drm_root_for_testing, tree.root() + "/");

  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);

  // get_drm_render_minor() queries ROCm SMI device identifiers. Populate the
  // synthetic backing device rather than assuming its constructor sets this.
  auto& devices = amd::smi::RocmSMI::getInstance().devices();
  ASSERT_LT(rocm_device.index(), devices.size());
  ASSERT_NE(devices[rocm_device.index()], nullptr);
  devices[rocm_device.index()]->set_drm_render_minor(render_minor);

  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);

  // Verify fixture setup independently before exercising the file reader.
  ASSERT_EQ(gpu_device.get_drm_render_minor(), render_minor);

  amdsmi_frequencies_t frequencies{};
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  // This internal helper accepts a device pointer, so processor-handle
  // registration is unnecessary.
  DISPLAY_AMDSMI_API("smi_amdgpu_read_clk_freq_from_pp_dpm",
                     "render=" + render_path + ", file=" + pp_dpm_file, verbose);
  amdsmi_status_t status =
      smi_amdgpu_read_clk_freq_from_pp_dpm(&gpu_device, pp_dpm_file, &frequencies);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

  // These values can only come from the render-node fixture, not the decoy.
  ASSERT_EQ(frequencies.num_supported, 3u);
  EXPECT_EQ(frequencies.current, 1u);
  EXPECT_EQ(frequencies.frequency[0], 300000000ULL);
  EXPECT_EQ(frequencies.frequency[1], 700000000ULL);
  EXPECT_EQ(frequencies.frequency[2], 1100000000ULL);
  EXPECT_EQ(frequencies.has_deep_sleep, 0u);
}

}  // namespace
