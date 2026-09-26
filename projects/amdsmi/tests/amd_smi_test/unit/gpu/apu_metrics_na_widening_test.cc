// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <amd_smi_test/test_base.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

#include "rocm_smi/rocm_smi_gpu_metrics.h"
#include "test_common.h"

namespace amd::smi {

GpuMetricsBasePtr amdgpu_metrics_factory(AMDGpuMetricVersionFlags_t gpu_metric_version,
                                         bool is_partition_metrics, const std::string& file_path);

}  // namespace amd::smi

namespace {

constexpr uint16_t kU16Max = std::numeric_limits<uint16_t>::max();
constexpr uint32_t kU32Max = std::numeric_limits<uint32_t>::max();

rsmi_apu_metrics_t CopyApuV24(uint16_t socket_power, uint16_t gfx_power) {
  auto metrics = amd::smi::amdgpu_metrics_factory(
      amd::smi::AMDGpuMetricVersionFlags_t::kApuMetricV24, false, "");
  EXPECT_NE(metrics, nullptr);
  auto* tbl = static_cast<amd::smi::AMDApuMetrics_v24_t*>(metrics->get_metrics_table().get());
  *tbl = amd::smi::AMDApuMetrics_v24_t{};
  tbl->m_average_socket_power = socket_power;
  tbl->m_average_gfx_power = gfx_power;

  auto [status, pub] = metrics->copy_internal_to_external_metrics();
  EXPECT_EQ(status, RSMI_STATUS_SUCCESS);
  EXPECT_NE(pub.apu_metrics, nullptr);
  return *pub.apu_metrics;
}

}  // namespace

// v2.4 stores these as uint16 but the public fields are uint32, so the N/A value must widen.
TEST(GpuUnit, ApuV24PowerNaWidensToUint32Max) {
  PRINT_VERBOSITY();
  const auto apu = CopyApuV24(kU16Max, kU16Max);
  EXPECT_EQ(apu.average_socket_power, kU32Max);
  EXPECT_EQ(apu.average_gfx_power, kU32Max);
}

TEST(GpuUnit, ApuV24PowerValidValuesPassThrough) {
  PRINT_VERBOSITY();
  for (uint16_t v : {uint16_t{0}, uint16_t{15000}, static_cast<uint16_t>(kU16Max - 1)}) {
    const auto apu = CopyApuV24(v, v);
    EXPECT_EQ(apu.average_socket_power, v);
    EXPECT_EQ(apu.average_gfx_power, v);
  }
}
