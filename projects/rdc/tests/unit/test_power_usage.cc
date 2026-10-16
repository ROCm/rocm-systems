/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// RDC_FI_POWER_USAGE reports the socket power from amdsmi_get_power_info() in
// microwatts. amd-smi reports a power it cannot read as UINT32_MAX, and MI300
// series and later GPUs have no average socket power, so rdcd reported
// UINT32_MAX * 10^6 for them instead of their current socket power.
//
// librdc cannot be loaded without a GPU (it initializes amdsmi when loaded), so
// this binary builds the metric fetcher itself. The amdsmi calls the tests reach
// are replaced by the fakes below; the binary links libamd_smi only for the
// calls they never make.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/impl/RdcMetricFetcherImpl.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace {

constexpr uint32_t kU32NotAvailable = std::numeric_limits<uint32_t>::max();
constexpr uint16_t kU16NotAvailable = std::numeric_limits<uint16_t>::max();

// The only GPU, alone on the only socket. Its processor handle points here.
struct FakeGpu {
  amdsmi_power_info_t power_info;
  amdsmi_gpu_metrics_t metrics;
};
FakeGpu g_gpu;
int g_socket;

// The tests share one fetcher and never destroy it: ~RdcMetricFetcherImpl() can
// fail to wake its worker thread and then wait for the thread forever. The
// fetcher keeps no state for the field these tests fetch.
amd::rdc::RdcMetricFetcherImpl& fetcher() {
  static auto* const instance = new amd::rdc::RdcMetricFetcherImpl();
  return *instance;
}

class PowerUsageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_gpu = {};
    amd::rdc::reset_flat_gpu_table();
  }

  // Fetches RDC_FI_POWER_USAGE the way rdcd does when bulk fetch is off.
  rdc_status_t fetch(rdc_field_value* value) {
    *value = {};
    return fetcher().fetch_smi_field(0, RDC_FI_POWER_USAGE, value);
  }

  // Fetches RDC_FI_POWER_USAGE the way rdcd does with RDC_BULK_FETCH_ENABLED.
  // Returns the values the bulk fetch produced; a field it skips is fetched
  // one by one instead.
  std::vector<rdc_gpu_field_value_t> bulk_fetch() {
    rdc_gpu_field_t field = {0, RDC_FI_POWER_USAGE};
    std::vector<rdc_gpu_field_value_t> results;
    EXPECT_EQ(fetcher().bulk_fetch_smi_fields(&field, 1, results), RDC_ST_OK);
    return results;
  }
};

}  // namespace

// Fakes for the amdsmi calls the tests reach.
amdsmi_status_t amdsmi_get_socket_handles(uint32_t* socket_count,
                                          amdsmi_socket_handle* socket_handles) {
  if (socket_handles != nullptr && *socket_count > 0) {
    socket_handles[0] = &g_socket;
  }
  *socket_count = 1;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle, uint32_t* processor_count,
                                             amdsmi_processor_handle* processor_handles) {
  if (processor_handles != nullptr && *processor_count > 0) {
    processor_handles[0] = &g_gpu;
  }
  *processor_count = 1;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle,
                                          amdsmi_processor_type_t* processor_type) {
  *processor_type = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_power_info(amdsmi_processor_handle processor_handle,
                                      amdsmi_power_info_t* info) {
  *info = static_cast<FakeGpu*>(processor_handle)->power_info;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_metrics_info(amdsmi_processor_handle processor_handle,
                                            amdsmi_gpu_metrics_t* pgpu_metrics) {
  *pgpu_metrics = static_cast<FakeGpu*>(processor_handle)->metrics;
  return AMDSMI_STATUS_SUCCESS;
}

// MI300 series and later GPUs: amd-smi has no average socket power for them.
TEST_F(PowerUsageTest, ReportsCurrentSocketPowerWithoutAverage) {
  g_gpu.power_info.average_socket_power = kU32NotAvailable;
  g_gpu.power_info.current_socket_power = 140;

  rdc_field_value value;
  EXPECT_EQ(fetch(&value), RDC_ST_OK);
  EXPECT_EQ(value.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(value.type, INTEGER);
  EXPECT_EQ(value.value.l_int, 140000000);
}

// Older amd-smi releases reported a power they could not read as UINT16_MAX.
TEST_F(PowerUsageTest, ReportsCurrentSocketPowerWithoutAverageFromOlderAmdSmi) {
  g_gpu.power_info.average_socket_power = kU16NotAvailable;
  g_gpu.power_info.current_socket_power = 140;

  rdc_field_value value;
  EXPECT_EQ(fetch(&value), RDC_ST_OK);
  EXPECT_EQ(value.value.l_int, 140000000);
}

TEST_F(PowerUsageTest, PrefersAverageSocketPower) {
  g_gpu.power_info.average_socket_power = 250;
  g_gpu.power_info.current_socket_power = 260;

  rdc_field_value value;
  EXPECT_EQ(fetch(&value), RDC_ST_OK);
  EXPECT_EQ(value.value.l_int, 250000000);
}

// As for the secondary partitions of a partitioned GPU, which have no socket power.
TEST_F(PowerUsageTest, NotAvailableWithoutSocketPower) {
  g_gpu.power_info.average_socket_power = kU32NotAvailable;
  g_gpu.power_info.current_socket_power = kU32NotAvailable;

  rdc_field_value value;
  EXPECT_EQ(fetch(&value), RDC_ST_SMI_ERROR);
  EXPECT_EQ(value.field_id, RDC_FI_POWER_USAGE);
  EXPECT_EQ(value.type, INTEGER);
  EXPECT_NE(value.status, AMDSMI_STATUS_SUCCESS);
}

TEST_F(PowerUsageTest, NotAvailableWithoutSocketPowerFromOlderAmdSmi) {
  g_gpu.power_info.average_socket_power = kU16NotAvailable;
  g_gpu.power_info.current_socket_power = kU16NotAvailable;

  rdc_field_value value;
  EXPECT_EQ(fetch(&value), RDC_ST_SMI_ERROR);
  EXPECT_EQ(value.field_id, RDC_FI_POWER_USAGE);
  EXPECT_NE(value.status, AMDSMI_STATUS_SUCCESS);
}

TEST_F(PowerUsageTest, BulkFetchReportsAverageSocketPower) {
  g_gpu.metrics.average_socket_power = 250;
  g_gpu.metrics.current_socket_power = 260;

  auto results = bulk_fetch();
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].field_value.value.l_int, 250000000);
}

TEST_F(PowerUsageTest, BulkFetchReportsCurrentSocketPowerWithoutAverage) {
  g_gpu.metrics.average_socket_power = kU16NotAvailable;
  g_gpu.metrics.current_socket_power = 140;

  auto results = bulk_fetch();
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].field_value.field_id, RDC_FI_POWER_USAGE);
  EXPECT_EQ(results[0].field_value.value.l_int, 140000000);
}

// The bulk fetch skips the field, so rdcd fetches it one by one, which reports it
// as not available.
TEST_F(PowerUsageTest, BulkFetchSkipsGpuWithoutSocketPower) {
  g_gpu.metrics.average_socket_power = kU16NotAvailable;
  g_gpu.metrics.current_socket_power = kU16NotAvailable;

  EXPECT_TRUE(bulk_fetch().empty());
}

TEST_F(PowerUsageTest, BulkFetchSkipsGpuWithZeroSocketPower) {
  g_gpu.metrics.average_socket_power = 0;
  g_gpu.metrics.current_socket_power = 0;

  EXPECT_TRUE(bulk_fetch().empty());
}
