// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "rocm_smi/rocm_smi_gpu_metrics.h"
#include "test_fixture_utils.h"

namespace {

// Keep a real, aligned object, not a cast from a byte buffer. Never call a
// getter with a smaller object than the type it was built to accept.
template <typename T, size_t ShippedSize>
struct ShippedOutput {
  static_assert(sizeof(T) == ShippedSize, "output type differs from the shipped caller's size");
  std::array<unsigned char, 16> before;
  T value{};
  std::array<unsigned char, 16> after;

  ShippedOutput() {
    static_assert(offsetof(ShippedOutput, value) == sizeof(before));
    static_assert(offsetof(ShippedOutput, after) == sizeof(before) + ShippedSize);
    before.fill(0xa5);
    after.fill(0x5a);
  }

  void CheckGuards() const {
    EXPECT_TRUE(std::all_of(before.begin(), before.end(), [](auto b) { return b == 0xa5; }));
    EXPECT_TRUE(std::all_of(after.begin(), after.end(), [](auto b) { return b == 0x5a; }));
  }
};

// Public handle dispatch and a real RSMI device, without discovering hardware.
struct TestGpu {
  static constexpr char kCard[] = "card42";
  ScopedAmdsmiInit init;
  FakeSysfsTree tree;
  amd::smi::testing::ScopedOverride<std::string> drm_root{
      amd::smi::test_hooks::SysfsDrmRoot, amd::smi::test_hooks::SetSysfsDrmRoot, tree.root()};
  ScopedRocmSmiDevice rsmi{tree.root() + "/" + kCard};
  amd::smi::AMDSmiDrm drm;
  amd::smi::AMDSmiGPUDevice gpu{rsmi.index(), kCard, amdsmi_bdf_t{}, drm};
  amd::smi::testing::ScopedProcessorRegistration registration{&gpu};

  void WriteAttribute(const std::string& attr, const std::string& data) {
    tree.WriteFile(std::string(kCard) + "/device/" + attr, data);
  }
  void WriteMetrics() {
    amd::smi::AMDGpuMetrics_v10_t table{};
    table.m_common_header = {sizeof(table), 1, 0};
    table.m_temperature_edge = 37;
    table.m_pcie_link_width = 8;
    WriteAttribute("gpu_metrics",
                   std::string(reinterpret_cast<const char*>(&table), sizeof(table)));
  }
};

TEST(SystemUnit, ShippedOutputFabricVersionIsNotAnInput) {
  TestGpu gpu;
  gpu.WriteAttribute("ualink/accel_id", "5\n");
  std::optional<uint32_t> reported_version;
  for (uint32_t initial : {0u, 0xa5a5a5a5u}) {
    SCOPED_TRACE(initial);
    ShippedOutput<amdsmi_fabric_info_t, SHIPPED_fabric_info_SIZE> output;
    output.value.fabric_version = initial;
    const auto status = amdsmi_get_gpu_fabric_info(gpu.registration.handle(), &output.value);
    output.CheckGuards();
    ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(output.value.fabric_info.v1.accelerator_id, 5u);
    if (reported_version) EXPECT_EQ(output.value.fabric_version, *reported_version);
    reported_version = output.value.fabric_version;
  }
}

TEST(SystemUnit, ShippedOutputDriverInfoWithoutDrm) {
  TestGpu gpu;
  ShippedOutput<amdsmi_driver_info_t, SHIPPED_driver_info_SIZE> output;
  std::memset(&output.value, 0xa5, sizeof(output.value));
  const auto status = amdsmi_get_gpu_driver_info(gpu.registration.handle(), &output.value);
  output.CheckGuards();
  // No DRM node. The driver-version string is filled before the DRM-only
  // fields are attempted, even when /sys/module/amdgpu/version is absent.
  ASSERT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(output.value.driver_version[0], '\0');
  EXPECT_NE(std::memchr(output.value.driver_version, '\0', sizeof(output.value.driver_version)),
            nullptr);
}

TEST(SystemUnit, ShippedOutputLibVersion) {
  ShippedOutput<amdsmi_version_t, SHIPPED_version_SIZE> output;
  output.value = {UINT32_MAX, UINT32_MAX, UINT32_MAX, nullptr};
  const auto status = amdsmi_get_lib_version(&output.value);
  output.CheckGuards();
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(output.value.major, AMDSMI_LIB_VERSION_MAJOR);
  EXPECT_EQ(output.value.minor, AMDSMI_LIB_VERSION_MINOR);
  EXPECT_EQ(output.value.release, AMDSMI_LIB_VERSION_RELEASE);
  EXPECT_STREQ(output.value.build, AMDSMI_LIB_VERSION_STRING);
}

TEST(SystemUnit, ShippedOutputEnumerationInfo) {
  FakeSysfsTree tree;
  WriteGpuNode(tree, 1, 0, 4242);
  FakeKfdTopology topology(tree);
  ShippedOutput<amdsmi_enumeration_info_t, SHIPPED_enumeration_info_SIZE> output;
  output.value.physical_acc_id = 7u;
  const auto status = amdsmi_get_gpu_enumeration_info(topology.handle(1), &output.value);
  output.CheckGuards();
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(output.value.hsa_id, 1u);
  EXPECT_EQ(output.value.physical_acc_id, UINT32_MAX);  // No IFoE identity on this GPU.
}

TEST(SystemUnit, ShippedOutputGpuMetrics) {
  TestGpu gpu;
  gpu.WriteMetrics();
  ShippedOutput<amdsmi_gpu_metrics_t, SHIPPED_gpu_metrics_SIZE> output;
  const auto status = amdsmi_get_gpu_metrics_info(gpu.registration.handle(), &output.value);
  output.CheckGuards();
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(output.value.temperature_edge, 37u);
}

TEST(SystemUnit, ShippedOutputBoardInfo) {
  TestGpu gpu;
  gpu.WriteAttribute("product_number", "TEST42\n");
  ShippedOutput<amdsmi_board_info_t, SHIPPED_board_info_SIZE> output;
  const auto status = amdsmi_get_gpu_board_info(gpu.registration.handle(), &output.value);
  output.CheckGuards();
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(output.value.model_number, "TEST42");
}

TEST(SystemUnit, ShippedOutputPowerInfo) {
  TestGpu gpu;
  gpu.WriteMetrics();
  gpu.WriteAttribute("vendor", "0x1002\n");
  gpu.WriteAttribute("hwmon/hwmon42/power1_cap", "123456\n");
  ShippedOutput<amdsmi_power_info_t, SHIPPED_power_info_SIZE> output;
  const auto status = amdsmi_get_power_info(gpu.registration.handle(), &output.value);
  output.CheckGuards();
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(output.value.power_limit, 123456u);
}

TEST(SystemUnit, ShippedOutputPcieInfo) {
  TestGpu gpu;
  gpu.WriteMetrics();
  gpu.WriteAttribute("max_link_width", "16\n");
  gpu.WriteAttribute("max_link_speed", "32.0 GT/s PCIe\n");
  ShippedOutput<amdsmi_pcie_info_t, SHIPPED_pcie_info_SIZE> output;
  const auto status = amdsmi_get_pcie_info(gpu.registration.handle(), &output.value);
  output.CheckGuards();
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(output.value.pcie_static.max_pcie_width, 16u);
  EXPECT_EQ(output.value.pcie_metric.pcie_width, 8u);
}

TEST(SystemUnit, ShippedOutputErrorCount) {
  TestGpu gpu;
  gpu.WriteAttribute("ras/features", "feature mask: 0x1\n");
  gpu.WriteAttribute("ras/umc_err_count", "ue: 5\nce: 3\n");
  ShippedOutput<amdsmi_error_count_t, SHIPPED_error_count_SIZE> output;
  const auto status = amdsmi_get_gpu_total_ecc_count(gpu.registration.handle(), &output.value);
  output.CheckGuards();
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(output.value.correctable_count, 3u);
}

TEST(SystemUnit, ShippedOutputVramInfo) {
  TestGpu gpu;
  gpu.WriteAttribute("mem_info_vram_total", "3145728\n");
  ShippedOutput<amdsmi_vram_info_t, SHIPPED_vram_info_SIZE> output;
  const auto status = amdsmi_get_gpu_vram_info(gpu.registration.handle(), &output.value);
  output.CheckGuards();
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(output.value.vram_size, 3u);
}

TEST(SystemUnit, ShippedOutputTopologyNearest) {
  FakeSysfsTree tree;
  WriteGpuNode(tree, 1, 0, 4242);
  FakeKfdTopology topology(tree);
  ShippedOutput<amdsmi_topology_nearest_t, SHIPPED_topology_nearest_SIZE> output;
  output.value.count = UINT32_MAX;
  const auto status =
      amdsmi_get_link_topology_nearest(topology.handle(1), AMDSMI_LINK_TYPE_XGMI, &output.value);
  output.CheckGuards();
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(output.value.count, 0u);
}

}  // namespace
