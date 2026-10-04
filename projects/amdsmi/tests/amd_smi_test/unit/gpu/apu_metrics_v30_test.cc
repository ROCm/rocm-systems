// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for the APU gpu_metrics v3.0 temperature fields that reach
// amdsmi_apu_metrics_t. The table is filled in memory -- no GPU required.
// Some PMFW builds (e.g. Strix Halo) report 0 instead of the 0xFFFF sentinel
// for core and skin temperatures they do not measure; those must surface as
// not available rather than as 0 C.

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
