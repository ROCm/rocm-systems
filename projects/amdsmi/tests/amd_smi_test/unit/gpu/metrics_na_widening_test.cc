// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <amd_smi_test/test_base.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi_gpu_metrics.h"
#include "test_common.h"

namespace amd::smi {

GpuMetricsBasePtr amdgpu_metrics_factory(AMDGpuMetricVersionFlags_t gpu_metric_version,
                                         bool is_partition_metrics, const std::string& file_path);

}  // namespace amd::smi

namespace {

constexpr uint16_t kU16Max = std::numeric_limits<uint16_t>::max();
constexpr uint32_t kU32Max = std::numeric_limits<uint32_t>::max();

void CopyApuV24(uint16_t socket_power, uint16_t gfx_power, rsmi_apu_metrics_t* out) {
  constexpr auto kV24 = amd::smi::AMDGpuMetricVersionFlags_t::kApuMetricV24;
  auto metrics = amd::smi::amdgpu_metrics_factory(kV24, false, "");
  ASSERT_NE(metrics, nullptr);
  ASSERT_EQ(metrics->get_gpu_metrics_version_used(), kV24);
  auto* tbl = static_cast<amd::smi::AMDApuMetrics_v24_t*>(metrics->get_metrics_table().get());
  ASSERT_NE(tbl, nullptr);
  *tbl = amd::smi::AMDApuMetrics_v24_t{};
  tbl->m_average_socket_power = socket_power;
  tbl->m_average_gfx_power = gfx_power;

  auto [status, pub] = metrics->copy_internal_to_external_metrics();
  ASSERT_EQ(status, RSMI_STATUS_SUCCESS);
  ASSERT_NE(pub.apu_metrics, nullptr);
  *out = *pub.apu_metrics;
}

void CopyGpuV10(uint32_t energy, uint8_t link_width, uint8_t link_speed, rsmi_gpu_metrics_t* out) {
  constexpr auto kV10 = amd::smi::AMDGpuMetricVersionFlags_t::kGpuMetricV10;
  auto metrics = amd::smi::amdgpu_metrics_factory(kV10, false, "");
  ASSERT_NE(metrics, nullptr);
  ASSERT_EQ(metrics->get_gpu_metrics_version_used(), kV10);
  auto* tbl = static_cast<amd::smi::AMDGpuMetrics_v10_t*>(metrics->get_metrics_table().get());
  ASSERT_NE(tbl, nullptr);
  *tbl = amd::smi::AMDGpuMetrics_v10_t{};
  tbl->m_energy_accumulator = energy;
  tbl->m_pcie_link_width = link_width;
  tbl->m_pcie_link_speed = link_speed;

  auto [status, pub] = metrics->copy_internal_to_external_metrics();
  ASSERT_EQ(status, RSMI_STATUS_SUCCESS);
  *out = pub;
}

}  // namespace

// v1.0 stores these narrower than the public fields, so the N/A value must widen.
TEST(GpuUnit, GpuV10NaWidensToPublicMax) {
  PRINT_VERBOSITY();
  rsmi_gpu_metrics_t gpu{};
  ASSERT_NO_FATAL_FAILURE(CopyGpuV10(kU32Max, UINT8_MAX, UINT8_MAX, &gpu));
  EXPECT_EQ(gpu.energy_accumulator, UINT64_MAX);
  EXPECT_EQ(gpu.pcie_link_width, kU16Max);
  EXPECT_EQ(gpu.pcie_link_speed, kU16Max);
}

TEST(GpuUnit, GpuV10ValidValuesPassThrough) {
  PRINT_VERBOSITY();
  rsmi_gpu_metrics_t gpu{};
  ASSERT_NO_FATAL_FAILURE(CopyGpuV10(kU32Max - 1, 16, UINT8_MAX - 1, &gpu));
  EXPECT_EQ(gpu.energy_accumulator, kU32Max - 1);
  EXPECT_EQ(gpu.pcie_link_width, 16);
  EXPECT_EQ(gpu.pcie_link_speed, UINT8_MAX - 1);
}

// v2.4 stores these as uint16 but the public fields are uint32, so the N/A value must widen.
TEST(GpuUnit, ApuV24PowerNaWidensToUint32Max) {
  PRINT_VERBOSITY();
  rsmi_apu_metrics_t apu{};
  ASSERT_NO_FATAL_FAILURE(CopyApuV24(kU16Max, kU16Max, &apu));
  EXPECT_EQ(apu.average_socket_power, kU32Max);
  EXPECT_EQ(apu.average_gfx_power, kU32Max);
}

TEST(GpuUnit, ApuV24PowerValidValuesPassThrough) {
  PRINT_VERBOSITY();
  for (uint16_t v : {uint16_t{0}, uint16_t{15000}, static_cast<uint16_t>(kU16Max - 1)}) {
    rsmi_apu_metrics_t apu{};
    ASSERT_NO_FATAL_FAILURE(CopyApuV24(v, v, &apu));
    EXPECT_EQ(apu.average_socket_power, v);
    EXPECT_EQ(apu.average_gfx_power, v);
  }
}

// amdsmi_get_gpu_activity widens the uint16 activity metrics into uint32 fields with this helper.
TEST(GpuUnit, TranslateUmaxWidensUint16Na) {
  PRINT_VERBOSITY();
  EXPECT_EQ(translate_umax_or_assign_value<uint32_t>(kU16Max, kU16Max), kU32Max);
  for (uint16_t v : {uint16_t{0}, uint16_t{100}, static_cast<uint16_t>(kU16Max - 1)}) {
    EXPECT_EQ(translate_umax_or_assign_value<uint32_t>(v, v), v);
  }
}
