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

// ~RdcMetricFetcherImpl() stops the fetcher's worker thread. It stored the stop
// flag without the worker's mutex, so a worker that had checked the flag but not
// yet started to wait missed the wake-up, and the destructor waited for it
// forever. rdcd and embedded RDC destroy their fetcher when they stop.
//
// librdc cannot be loaded without a GPU (it initializes amdsmi when loaded), so
// this binary builds the metric fetcher itself. Building and destroying a
// fetcher makes no amdsmi calls. The amdsmi calls the other tests reach are
// replaced by the fakes below; the binary links libamd_smi only for the calls
// no test makes.

#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/impl/RdcMetricFetcherImpl.h"
#include "rdc_lib/impl/SmiUtils.h"

// Several threads build and destroy fetchers at once, which hits that window
// within seconds. If the destructor can hang again, this test hangs until the
// ctest or CI runner timeout stops it.
TEST(MetricFetcherLifetimeTest, DestroyingFetchersDoesNotHang) {
  constexpr int kThreads = 16;
  constexpr int kFetchersPerThread = 1000;

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; t++) {
    threads.emplace_back([] {
      for (int f = 0; f < kFetchersPerThread; f++) {
        amd::rdc::RdcMetricFetcherImpl fetcher;
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

// RDC_FI_GPU_MEMORY_CUR_BANDWIDTH. Without an instantaneous memory activity, rdcd
// derives the activity from the memory-activity accumulator of two samples. The
// bandwidth is then N/A without a rate: for the first sample, and when firmware
// time stalled or went back, the accumulator went back, or more than 60 s passed.
namespace {

// The only GPU, alone on the only socket. Its processor handle points here.
struct FakeGpu {
  amdsmi_gpu_metrics_t metrics;
  amdsmi_vram_info_t vram_info;
};
FakeGpu g_gpu;
int g_socket;
int g_placeholder;                 // a processor whose type AMD SMI cannot report
bool g_with_placeholder = false;   // listed after the GPU: the GPU table retries at every use
int g_listings = 0;                // listings of the sockets so far
int g_gpu_listed_until = INT_MAX;  // the listing from which on the GPU is no longer listed

constexpr uint64_t kFirmwareMs = 100000;  // firmware timestamps are in 10 ns units

class MemoryBandwidthTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_gpu = {};
    g_gpu.metrics.average_umc_activity = std::numeric_limits<uint16_t>::max();
    g_gpu.metrics.firmware_timestamp = 1000 * kFirmwareMs;
    g_gpu.metrics.mem_activity_acc = 1000;
    g_gpu.vram_info.vram_max_bandwidth = 2000;
    amd::rdc::reset_flat_gpu_table();
  }

  rdc_status_t fetch(rdc_field_value* value) {
    *value = {};
    return fetcher_.fetch_smi_field(0, RDC_FI_GPU_MEMORY_CUR_BANDWIDTH, value);
  }

  // Fetches a first sample, which has no rate yet, then one at firmware_ms with
  // the accumulator at mem_activity_acc.
  rdc_status_t fetch_second(uint64_t firmware_ms, uint64_t mem_activity_acc,
                            rdc_field_value* value) {
    rdc_field_value first;
    EXPECT_EQ(fetch(&first), RDC_ST_SMI_ERROR);
    EXPECT_EQ(first.status, AMDSMI_STATUS_NOT_SUPPORTED);
    g_gpu.metrics.firmware_timestamp = firmware_ms * kFirmwareMs;
    g_gpu.metrics.mem_activity_acc = mem_activity_acc;
    return fetch(value);
  }

  amd::rdc::RdcMetricFetcherImpl fetcher_;
};

}  // namespace

// Fakes for the amdsmi calls the tests reach.
amdsmi_status_t amdsmi_get_socket_handles(uint32_t* socket_count,
                                          amdsmi_socket_handle* socket_handles) {
  if (socket_handles != nullptr && *socket_count > 0) {
    socket_handles[0] = &g_socket;
    g_listings++;
  }
  *socket_count = 1;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle, uint32_t* processor_count,
                                             amdsmi_processor_handle* processor_handles) {
  std::vector<amdsmi_processor_handle> listed;
  if (g_listings < g_gpu_listed_until) {
    listed.push_back(&g_gpu);
  }
  if (g_with_placeholder) {
    listed.push_back(&g_placeholder);
  }
  if (processor_handles != nullptr) {
    for (uint32_t p = 0; p < *processor_count && p < listed.size(); p++) {
      processor_handles[p] = listed[p];
    }
  }
  *processor_count = static_cast<uint32_t>(listed.size());
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle processor_handle,
                                          amdsmi_processor_type_t* processor_type) {
  if (processor_handle == &g_placeholder) {
    return AMDSMI_STATUS_UNKNOWN_ERROR;
  }
  *processor_type = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_metrics_info(amdsmi_processor_handle processor_handle,
                                            amdsmi_gpu_metrics_t* pgpu_metrics) {
  *pgpu_metrics = static_cast<FakeGpu*>(processor_handle)->metrics;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_vram_info(amdsmi_processor_handle processor_handle,
                                         amdsmi_vram_info_t* info) {
  *info = static_cast<FakeGpu*>(processor_handle)->vram_info;
  return AMDSMI_STATUS_SUCCESS;
}

// The accumulator grows by the activity percentage every firmware millisecond.
TEST_F(MemoryBandwidthTest, ReportsTheAccumulatorRate) {
  rdc_field_value value;
  EXPECT_EQ(fetch_second(2000, 1000 + 25 * 1000, &value), RDC_ST_OK);  // 25% for 1 s
  EXPECT_EQ(value.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(value.value.l_int, 500);
}

TEST_F(MemoryBandwidthTest, ReportsZeroWithoutMemoryTraffic) {
  rdc_field_value value;
  EXPECT_EQ(fetch_second(2000, 1000, &value), RDC_ST_OK);
  EXPECT_EQ(value.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(value.value.l_int, 0);
}

// With an instantaneous memory activity, a first sample has a bandwidth too.
TEST_F(MemoryBandwidthTest, ReportsTheInstantaneousActivity) {
  g_gpu.metrics.average_umc_activity = 40;
  rdc_field_value value;
  EXPECT_EQ(fetch(&value), RDC_ST_OK);
  EXPECT_EQ(value.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(value.value.l_int, 800);  // 40% of 2000
}

TEST_F(MemoryBandwidthTest, NotAvailableWhenFirmwareTimeStalls) {
  rdc_field_value value;
  EXPECT_EQ(fetch_second(1000, 1000, &value), RDC_ST_SMI_ERROR);
  EXPECT_EQ(value.status, AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST_F(MemoryBandwidthTest, NotAvailableWhenFirmwareTimeGoesBack) {
  rdc_field_value value;
  EXPECT_EQ(fetch_second(900, 2000, &value), RDC_ST_SMI_ERROR);
  EXPECT_EQ(value.status, AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST_F(MemoryBandwidthTest, NotAvailableWhenTheAccumulatorGoesBack) {
  rdc_field_value value;
  EXPECT_EQ(fetch_second(2000, 0, &value), RDC_ST_SMI_ERROR);
  EXPECT_EQ(value.status, AMDSMI_STATUS_NOT_SUPPORTED);
}

// The ECC getters look the GPU up again after fetch_smi_field() did. A GPU that is no longer
// listed in between reads as not found, which the getter returns instead of calling AMD SMI
// with no handle.
class EccLookupTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_with_placeholder = true;
    amd::rdc::set_flat_gpu_table_retry_period(std::chrono::milliseconds(0));
  }
  void TearDown() override {
    g_with_placeholder = false;
    g_gpu_listed_until = INT_MAX;
    amd::rdc::set_flat_gpu_table_retry_period(std::chrono::milliseconds(5000));
    amd::rdc::reset_flat_gpu_table();
  }
  amd::rdc::RdcMetricFetcherImpl fetcher_;
};

TEST_F(EccLookupTest, AGpuThatIsNoLongerListedReadsNotFound) {
  // RDC_FI_ECC_UMC_CE goes through get_ecc(), RDC_FI_ECC_UMC_DE through get_ecc_deferred(),
  // RDC_FI_ECC_CORRECT_TOTAL through get_ecc_total() and RDC_FI_ECC_DEFERRED_TOTAL through
  // get_ecc_deferred_total().
  for (rdc_field_t field : {RDC_FI_ECC_UMC_CE, RDC_FI_ECC_UMC_DE, RDC_FI_ECC_CORRECT_TOTAL,
                            RDC_FI_ECC_DEFERRED_TOTAL}) {
    g_gpu_listed_until = INT_MAX;
    amd::rdc::reset_flat_gpu_table();
    ASSERT_EQ(amd::rdc::get_flat_gpu_table()->size(), 2u);  // the GPU and the placeholder
    // fetch_smi_field() lists the processors once and finds the GPU; the getter's own lookup
    // lists them again, without the GPU.
    g_gpu_listed_until = g_listings + 2;
    rdc_field_value value = {};
    fetcher_.fetch_smi_field(0, field, &value);
    EXPECT_EQ(value.status, AMDSMI_STATUS_NOT_FOUND) << "field " << field;
    EXPECT_EQ(g_listings, g_gpu_listed_until) << "field " << field << ": not two lookups";
  }
}

TEST_F(MemoryBandwidthTest, NotAvailableWhenSamplesAreMoreThanAMinuteApart) {
  rdc_field_value value;
  EXPECT_EQ(fetch_second(1000 + 61000, 1000 + 61000 * 10, &value), RDC_ST_SMI_ERROR);
  EXPECT_EQ(value.status, AMDSMI_STATUS_NOT_SUPPORTED);
}
