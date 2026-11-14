// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for sysfs DRM root overrides, exercising the real public APIs
// with synthetic GPU devices and temporary sysfs files -- no GPU required. See
// docs/design/faking-external-interfaces.md for the fixture pattern.
//
// Guards against regressions that bypass the shared root helper with
// hardcoded "/sys/class/drm/" paths. Exact fixture values and write-back
// checks verify that reads and writes reach the temporary tree; SUCCESS
// alone cannot prove the path used. Scoped guards restore the root and
// device registrations, and temporary files are removed after each test.

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "rocm_smi/rocm_smi_gpu_metrics.h"
#include "test_common.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiDrm;
using amd::smi::AMDSmiGPUDevice;
using amd::smi::test_hooks::SetSysfsDrmRoot;
using amd::smi::test_hooks::SysfsDrmRoot;
using amd::smi::testing::ScopedEnvVar;
using amd::smi::testing::ScopedOverride;
using amd::smi::testing::ScopedProcessorRegistration;

// A valid, otherwise-zero v1.0 gpu_metrics table; tests set only the fields they check.
amd::smi::AMDGpuMetrics_v10_t MakeGpuMetricsV10() {
  amd::smi::AMDGpuMetrics_v10_t metrics{};
  metrics.m_common_header.m_structure_size = static_cast<uint16_t>(sizeof(metrics));
  metrics.m_common_header.m_format_revision = 1;
  metrics.m_common_header.m_content_revision = 0;
  return metrics;
}

// An explicit length preserves embedded null bytes in the binary table.
void WriteGpuMetrics(FakeSysfsTree& tree, const std::string& gpu_path,
                     const amd::smi::AMDGpuMetrics_v10_t& metrics) {
  tree.WriteFile(gpu_path + "/device/gpu_metrics",
                 std::string(reinterpret_cast<const char*>(&metrics), sizeof(metrics)));
}

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

  // The guard restores the previous
  // root on scope exit, including an early return from the test.
  ScopedOverride<std::string> root_guard(SysfsDrmRoot, SetSysfsDrmRoot, tree.root());

  // Supply the device and mutex backing needed by the real public API without a physical GPU.
  // Back it with a separate empty directory: the API falls back to ROCm SMI for the serial number
  // and product name, and ROCm SMI reading the fixture files would hide a hardcoded path.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/rocm_smi_backing");
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
  ScopedOverride<std::string> root_guard(SysfsDrmRoot, SetSysfsDrmRoot, tree.root());

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

// The power cap is found and read under the overridden DRM root, not the host's sysfs.
TEST(GpuUnit, SysfsDrmRootPowerLimitUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // A synthetic card name keeps an unmigrated path from reading a physical GPU's hwmon.
  const std::string gpu_path = "card_test_power_limit";

  // Hwmon discovery requires an AMD vendor ID before it scans the hwmon directory.
  tree.WriteFile(gpu_path + "/device/vendor", "0x1002\n");

  // The hwmon directory name is discovered; the distinct cap value proves which tree was read.
  tree.WriteFile(gpu_path + "/device/hwmon/hwmon42/power1_cap", "123456\n");

  // The API reads gpu_metrics before the power cap, so supply a valid table.
  WriteGpuMetrics(tree, gpu_path, MakeGpuMetricsV10());

  // The guard restores the previous root on exit.
  ScopedOverride<std::string> root_guard(SysfsDrmRoot, SetSysfsDrmRoot, tree.root());

  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);
  ScopedProcessorRegistration registration(&gpu_device);

  amdsmi_power_info_t info{};
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  DISPLAY_AMDSMI_API("amdsmi_get_power_info", "card=" + gpu_path, verbose);
  amdsmi_status_t status = amdsmi_get_power_info(registration.handle(), &info);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

  // The value can only come from the fixture's power1_cap file.
  EXPECT_EQ(info.power_limit, 123456u);
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

  // Restore the previous root when the test exits.
  ScopedOverride<std::string> root_guard(SysfsDrmRoot, SetSysfsDrmRoot, tree.root());

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

TEST(GpuUnit, SysfsDrmRootPowerManagementEnabledUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // Prevent the hardcoded root from accidentally finding a physical GPU.
  const std::string gpu_path = "card_test_power_management";

  tree.WriteFile(gpu_path + "/device/pp_features",
                 "0. FEATURE_A (0) disabled\n"
                 "1. FEATURE_B (1) enabled\n");

  // The guard restores the root on exit.
  ScopedOverride<std::string> root_guard(SysfsDrmRoot, SetSysfsDrmRoot, tree.root());

  // Supply the device and mutex without requiring physical hardware.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);
  ScopedProcessorRegistration registration(&gpu_device);

  // Start false so success without updating the output cannot pass.
  bool enabled = false;
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  DISPLAY_AMDSMI_API("amdsmi_is_gpu_power_management_enabled", "card=" + gpu_path, verbose);
  amdsmi_status_t status = amdsmi_is_gpu_power_management_enabled(registration.handle(), &enabled);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  // Both assertions are required to prove the overridden fixture was read.
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(enabled);
}

// The clock range comes from the overridden DRM root; the current clock comes from gpu_metrics.
TEST(GpuUnit, SysfsDrmRootClockInfoUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  // Prevent an unmigrated path from reading a physical GPU's clock data.
  const std::string gpu_path = "card_test_clock_info";

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

  // The API reads gpu_metrics before the ranges, so supply a valid table with a known clock.
  auto metrics = MakeGpuMetricsV10();
  metrics.m_current_gfxclk = 1500;
  WriteGpuMetrics(tree, gpu_path, metrics);

  // Restore the previous root on exit.
  ScopedOverride<std::string> root_guard(SysfsDrmRoot, SetSysfsDrmRoot, tree.root());

  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);
  ScopedProcessorRegistration registration(&gpu_device);

  amdsmi_clk_info_t info{};
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  DISPLAY_AMDSMI_API("amdsmi_get_clock_info", "card=" + gpu_path + ", domain=GFX", verbose);
  amdsmi_status_t status = amdsmi_get_clock_info(registration.handle(), AMDSMI_CLK_TYPE_GFX, &info);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

  // Exact values verify that both sysfs files came from the overridden root.
  EXPECT_EQ(info.max_clk, 2100u);
  EXPECT_EQ(info.min_clk, 500u);
  EXPECT_EQ(info.clk_deep_sleep, 100u);

  // The current clock comes from the metrics fixture, not sysfs.
  EXPECT_EQ(info.clk, 1500u);
}

// dpm_cap is the highest DPM level index, read from the overridden DRM root.
TEST(GpuUnit, SysfsDrmRootPowerCapDpmLevelUsesOverride) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  const std::string gpu_path = "card_test_dpm_level";

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

  ScopedOverride<std::string> root_guard(SysfsDrmRoot, SetSysfsDrmRoot, tree.root());

  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);
  ScopedProcessorRegistration registration(&gpu_device);

  amdsmi_power_cap_info_t info{};
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  // The API returns the status of the power-cap read, which this fake device does not provide,
  // and fills dpm_cap regardless. Check the value, not the status.
  DISPLAY_AMDSMI_API("amdsmi_get_power_cap_info", "card=" + gpu_path, verbose);
  static_cast<void>(amdsmi_get_power_cap_info(registration.handle(), 0, &info));

  // Despite the name, dpm_cap is the highest level index (levels 0..2), not a count.
  EXPECT_EQ(info.dpm_cap, 2u);
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
  auto metrics = MakeGpuMetricsV10();

  // Different current and maximum values distinguish the two data sources.
  metrics.m_pcie_link_width = 8;
  metrics.m_pcie_link_speed = 160;  // Metrics use 0.1 GT/s: 160 means 16 GT/s.
  WriteGpuMetrics(tree, gpu_path, metrics);

  // Redirect both static link reads and restore the previous root on exit.
  ScopedOverride<std::string> root_guard(SysfsDrmRoot, SetSysfsDrmRoot, tree.root());

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

  // Restore the previous root even if
  // an ASSERT below exits the test early.
  ScopedOverride<std::string> root_guard(SysfsDrmRoot, SetSysfsDrmRoot, tree.root());

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
  ScopedAmdsmiInit smi_init;

  // Dry-run mode skips the write this test verifies. Turn it off for this test only; the guard
  // restores the previous value, or unsets it, when the test ends, even on an early exit.
  ScopedEnvVar dry_run("AMDSMI_DRY_RUN", "0");
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
  ScopedOverride<std::string> root_guard(SysfsDrmRoot, SetSysfsDrmRoot, tree.root());

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

// Clock levels are read through the device's render node, under the overridden DRM root.
TEST(GpuUnit, SysfsDrmRootPpDpmClockFrequenciesUseRenderNode) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  const std::string gpu_path = "card_test_pp_dpm";

  // The middle entry is current; distinct frequencies verify the parsed data
  // and the MHz-to-Hz conversion, not merely a successful file open.
  tree.WriteFile(gpu_path + "/device/pp_dpm_vclk",
                 "0: 300Mhz\n"
                 "1: 700Mhz *\n"
                 "2: 1100Mhz\n");

  // The guard restores the original root even if an assertion exits early.
  ScopedOverride<std::string> root_guard(SysfsDrmRoot, SetSysfsDrmRoot, tree.root());

  // The fixture also creates the device's renderD<N> link to the card directory.
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + gpu_path);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), gpu_path, amdsmi_bdf_t{}, drm);

  ScopedProcessorRegistration registration(&gpu_device);

  amdsmi_frequencies_t frequencies{};
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;

  DISPLAY_AMDSMI_API("amdsmi_get_clk_freq", "card=" + gpu_path + ", clk=VCLK0", verbose);
  amdsmi_status_t status =
      amdsmi_get_clk_freq(registration.handle(), AMDSMI_CLK_TYPE_VCLK0, &frequencies);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

  // These values can only come from the fixture file.
  ASSERT_EQ(frequencies.num_supported, 3u);
  EXPECT_EQ(frequencies.current, 1u);
  EXPECT_EQ(frequencies.frequency[0], 300000000ULL);
  EXPECT_EQ(frequencies.frequency[1], 700000000ULL);
  EXPECT_EQ(frequencies.frequency[2], 1100000000ULL);
  EXPECT_EQ(frequencies.has_deep_sleep, 0u);
}

// Socket-scope CPU affinity is read from the overridden DRM root, not the host's sysfs.
TEST(GpuUnit, SysfsDrmRootLocalCpulistUsesOverride) {
  constexpr uint32_t kCpuNode = 0;
  constexpr uint32_t kGpuNode = 1;
  constexpr uint32_t kLinkIndex = 0;
  constexpr uint64_t kGpuId = 4242;
  constexpr uint64_t kPcieWeight = 20;
  // A card index no host has, so a hardcoded root cannot read a matching file.
  constexpr uint32_t kDrmCard = 9876;
  constexpr uint32_t kCpuSetSize = 2;
  const std::string card_dir = "card" + std::to_string(kDrmCard) + "/device/";

  FakeSysfsTree tree;
  WriteCpuNode(tree, kCpuNode);
  WriteGpuNode(tree, kGpuNode, kDrmCard, kGpuId);
  WriteIoLink(tree, kGpuNode, kLinkIndex, kCpuNode, amd::smi::IOLINK_TYPE_PCIEXPRESS, kPcieWeight);
  tree.WriteFile(card_dir + "numa_node", "0\n");
  tree.WriteFile(card_dir + "local_cpulist", "0-3,64\n");
  FakeKfdTopology topology(tree);

  uint64_t cpu_set[kCpuSetSize] = {};
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_affinity_with_scope",
                     "card=card" + std::to_string(kDrmCard) + ", scope=SOCKET", verbose);
  const amdsmi_status_t status = amdsmi_get_cpu_affinity_with_scope(
      topology.handle(kGpuNode), kCpuSetSize, cpu_set, AMDSMI_AFFINITY_SCOPE_SOCKET);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(cpu_set[0], 0xFULL);  // CPUs 0-3
  EXPECT_EQ(cpu_set[1], 0x1ULL);  // CPU 64
}

}  // namespace
