// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for the APU gpu_metrics v3.0 temperature, VCN activity and per-core fields that
// reach amdsmi_apu_metrics_t. The table is filled in memory -- no GPU required.
// Some APUs (e.g. Strix Halo) report 0 instead of the 0xFFFF sentinel for core
// and skin temperatures that are not measured; those must surface as not
// available rather than as 0 C.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>

#include "rocm_smi/rocm_smi_gpu_metrics.h"

namespace {

using amd::smi::AMDApuMetrics_v30_t;
using amd::smi::ApuMetricsBase_v30_t;

constexpr uint16_t kNotAvailable = std::numeric_limits<uint16_t>::max();

// Start from the kernel's view of an unfilled v3.0 table: every field is the sentinel.
AMDApuMetrics_v30_t* ResetV30Table(ApuMetricsBase_v30_t& apu_metrics) {
  auto* table = static_cast<AMDApuMetrics_v30_t*>(apu_metrics.get_metrics_table().get());
  std::memset(table, 0xFF, sizeof(*table));
  table->m_common_header.m_structure_size = static_cast<uint16_t>(sizeof(*table));
  table->m_common_header.m_format_revision = 3;
  table->m_common_header.m_content_revision = 0;
  return table;
}

}  // namespace

TEST(GpuUnit, ApuMetricsV30ZeroTemperatureIsNotAvailable) {
  ApuMetricsBase_v30_t apu_metrics;
  auto* table = ResetV30Table(apu_metrics);
  std::fill(std::begin(table->m_temperature_core), std::end(table->m_temperature_core),
            uint16_t{0});
  table->m_temperature_skin = 0;
  table->m_temperature_gfx = 7238;
  table->m_temperature_soc = 6900;
  table->m_average_core_c0_activity[0] = 0;  // an idle core: 0 is a real reading here

  const auto [status, metrics] = apu_metrics.copy_internal_to_external_metrics();
  ASSERT_EQ(status, RSMI_STATUS_SUCCESS);
  ASSERT_NE(metrics.apu_metrics, nullptr);
  for (const auto temp : metrics.apu_metrics->temperature_core) {
    EXPECT_EQ(temp, kNotAvailable);
  }
  EXPECT_EQ(metrics.apu_metrics->temperature_skin, kNotAvailable);
  EXPECT_EQ(metrics.apu_metrics->temperature_gfx, 7238);
  EXPECT_EQ(metrics.apu_metrics->temperature_soc, 6900);
  EXPECT_EQ(metrics.apu_metrics->average_core_c0_activity[0], 0);
}

TEST(GpuUnit, ApuMetricsV30NonZeroTemperatureIsKept) {
  ApuMetricsBase_v30_t apu_metrics;
  auto* table = ResetV30Table(apu_metrics);
  // Measured cores, a core at 0, near-zero readings, and absent slots 8-15 left at 0xFFFF.
  const uint16_t cores[] = {2, 0, 1, 2979, 3027, 0, 3012, 3027};
  std::copy(std::begin(cores), std::end(cores), table->m_temperature_core);
  table->m_temperature_skin = 3500;

  const auto [status, metrics] = apu_metrics.copy_internal_to_external_metrics();
  ASSERT_EQ(status, RSMI_STATUS_SUCCESS);
  ASSERT_NE(metrics.apu_metrics, nullptr);
  const uint16_t expected[] = {2, kNotAvailable, 1, 2979, 3027, kNotAvailable, 3012, 3027};
  for (size_t i = 0; i < std::size(expected); ++i) {
    EXPECT_EQ(metrics.apu_metrics->temperature_core[i], expected[i]) << "core " << i;
  }
  for (size_t i = std::size(expected); i < RSMI_APU_MAX_CORES; ++i) {
    EXPECT_EQ(metrics.apu_metrics->temperature_core[i], kNotAvailable) << "core " << i;
  }
  EXPECT_EQ(metrics.apu_metrics->temperature_skin, 3500);
}

TEST(GpuUnit, ApuMetricsV30VcnActivityIsWholePercent) {
  ApuMetricsBase_v30_t apu_metrics;
  auto* table = ResetV30Table(apu_metrics);
  // Strix Halo during a 4K encode that DRM fdinfo measured at 58.6 % busy.
  table->m_average_vcn_activity = 6067;
  table->m_average_gfx_activity = 1;

  const auto [status, metrics] = apu_metrics.copy_internal_to_external_metrics();
  ASSERT_EQ(status, RSMI_STATUS_SUCCESS);
  ASSERT_NE(metrics.apu_metrics, nullptr);
  EXPECT_EQ(metrics.apu_metrics->average_vcn_activity, 61);
  EXPECT_EQ(metrics.apu_metrics->average_gfx_activity, 1);
}

TEST(GpuUnit, ApuMetricsV30VcnActivityNotAvailableIsKept) {
  ApuMetricsBase_v30_t apu_metrics;
  ResetV30Table(apu_metrics);

  const auto [status, metrics] = apu_metrics.copy_internal_to_external_metrics();
  ASSERT_EQ(status, RSMI_STATUS_SUCCESS);
  ASSERT_NE(metrics.apu_metrics, nullptr);
  EXPECT_EQ(metrics.apu_metrics->average_vcn_activity, kNotAvailable);
}

TEST(GpuUnit, ApuMetricsV30CoreSlotWithZeroClockIsNotAvailable) {
  ApuMetricsBase_v30_t apu_metrics;
  auto* table = ResetV30Table(apu_metrics);
  // 12-core Strix Halo at idle: slots 6, 7, 14 and 15 have no core, while cores 3 and 11-13
  // are present at 0 % C0.
  const uint16_t c0[] = {3, 1, 1, 0, 2, 0, 0, 0, 1, 5, 1, 0, 0, 0, 0, 0};
  const uint16_t power[] = {134, 55, 12, 12, 12, 12, 0, 0, 29, 30, 6, 6, 6, 6, 0, 0};
  const uint16_t clock[] = {4960, 3178, 3178, 3178, 3178, 3178, 0, 0,
                            3190, 3190, 3190, 3190, 3190, 3190, 0, 0};
  std::copy(std::begin(c0), std::end(c0), table->m_average_core_c0_activity);
  std::copy(std::begin(power), std::end(power), table->m_average_core_power);
  std::copy(std::begin(clock), std::end(clock), table->m_current_coreclk);

  const auto [status, metrics] = apu_metrics.copy_internal_to_external_metrics();
  ASSERT_EQ(status, RSMI_STATUS_SUCCESS);
  ASSERT_NE(metrics.apu_metrics, nullptr);
  const auto& apu = *metrics.apu_metrics;
  for (size_t i = 0; i < RSMI_APU_MAX_CORES; ++i) {
    const bool present = clock[i] != 0;
    EXPECT_EQ(apu.average_core_c0_activity[i], present ? c0[i] : kNotAvailable) << "core " << i;
    EXPECT_EQ(apu.average_core_power[i], present ? power[i] : kNotAvailable) << "core " << i;
    EXPECT_EQ(apu.current_coreclk[i], present ? clock[i] : kNotAvailable) << "core " << i;
  }
}
