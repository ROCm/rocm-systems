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

// RDC_FI_GPU_PARTITION_COUNT reports how many compute partitions a GPU is split
// into. It came from the gpu_metrics partition count, which only some drivers
// fill (none that publish gpu_metrics v1.9), so rdcd reported 65535, the value
// amd-smi uses for "not available", in every partition mode.
//
// librdc cannot be loaded without a GPU (it initializes amdsmi when loaded), so
// this binary builds the metric fetcher itself. The amdsmi calls the tests reach
// are replaced by the fakes below; the binary links libamd_smi only for the
// calls they never make.

#include <gtest/gtest.h>

#include <cstdint>
#include <deque>
#include <limits>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/impl/RdcMetricFetcherImpl.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace {

// What the fake amd-smi reports for one processor. amd-smi lists each
// partition of a partitioned GPU as a processor on that GPU's socket.
struct FakeProcessor {
  amdsmi_status_t profile_status;
  amdsmi_accelerator_partition_profile_t profile;
  uint32_t partition_id;
};

// g_sockets[s] holds the processors of GPU s. A handle points at its entry; a
// deque keeps those addresses valid while the test adds GPUs.
std::deque<std::vector<FakeProcessor>> g_sockets;

// Adds a GPU split into num_partitions partitions of the given type and returns
// its processors.
std::vector<FakeProcessor>& add_gpu(amdsmi_accelerator_partition_type_t type,
                                    uint32_t num_partitions) {
  std::vector<FakeProcessor> processors(num_partitions);
  for (uint32_t p = 0; p < num_partitions; p++) {
    processors[p].profile_status = AMDSMI_STATUS_SUCCESS;
    processors[p].profile = {};
    processors[p].profile.profile_type = type;
    processors[p].profile.num_partitions = num_partitions;
    processors[p].partition_id = p;
  }
  g_sockets.push_back(processors);
  return g_sockets.back();
}

// Adds a GPU whose partition profile amd-smi reports with the given status.
void add_gpu_with_profile_status(amdsmi_status_t status) {
  FakeProcessor processor = {};
  processor.profile_status = status;
  g_sockets.push_back({processor});
}

// The tests share one fetcher and never destroy it: ~RdcMetricFetcherImpl() can
// fail to wake its worker thread and then wait for the thread forever. The
// fetcher keeps no state for the field these tests fetch.
amd::rdc::RdcMetricFetcherImpl& fetcher() {
  static auto* const instance = new amd::rdc::RdcMetricFetcherImpl();
  return *instance;
}

class PartitionCountTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_sockets.clear();
    amd::rdc::reset_flat_gpu_table();
  }

  // Fetches RDC_FI_GPU_PARTITION_COUNT for the GPU at RDC index gpu_index.
  rdc_status_t fetch(uint32_t gpu_index, rdc_field_value* value) {
    *value = {};
    return fetcher().fetch_smi_field(gpu_index, RDC_FI_GPU_PARTITION_COUNT, value);
  }
};

}  // namespace

// Fakes for the amdsmi calls the tests reach.
amdsmi_status_t amdsmi_get_socket_handles(uint32_t* socket_count,
                                          amdsmi_socket_handle* socket_handles) {
  if (socket_handles != nullptr) {
    for (uint32_t s = 0; s < *socket_count && s < g_sockets.size(); s++) {
      socket_handles[s] = &g_sockets[s];
    }
  }
  *socket_count = static_cast<uint32_t>(g_sockets.size());
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle socket_handle,
                                             uint32_t* processor_count,
                                             amdsmi_processor_handle* processor_handles) {
  auto* processors = static_cast<std::vector<FakeProcessor>*>(socket_handle);
  if (processor_handles != nullptr) {
    for (uint32_t p = 0; p < *processor_count && p < processors->size(); p++) {
      processor_handles[p] = &(*processors)[p];
    }
  }
  *processor_count = static_cast<uint32_t>(processors->size());
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle,
                                          amdsmi_processor_type_t* processor_type) {
  *processor_type = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_accelerator_partition_profile(
    amdsmi_processor_handle processor_handle, amdsmi_accelerator_partition_profile_t* profile,
    uint32_t* partition_id) {
  const auto* processor = static_cast<const FakeProcessor*>(processor_handle);
  if (processor->profile_status == AMDSMI_STATUS_SUCCESS) {
    *profile = processor->profile;
    *partition_id = processor->partition_id;
  }
  return processor->profile_status;
}

// gpu_metrics as drivers that publish v1.9 report it: no partition count. RDC
// took field 9 from here before, so the tests fail if it does again.
amdsmi_status_t amdsmi_get_gpu_metrics_info(amdsmi_processor_handle,
                                            amdsmi_gpu_metrics_t* pgpu_metrics) {
  *pgpu_metrics = {};
  pgpu_metrics->num_partition = std::numeric_limits<uint16_t>::max();
  return AMDSMI_STATUS_SUCCESS;
}

TEST_F(PartitionCountTest, SpxGpuHasOnePartition) {
  add_gpu(AMDSMI_ACCELERATOR_PARTITION_SPX, 1);

  rdc_field_value value;
  EXPECT_EQ(fetch(0, &value), RDC_ST_OK);
  EXPECT_EQ(value.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(value.type, INTEGER);
  EXPECT_EQ(value.value.l_int, 1);
}

// Every RDC GPU index of a partitioned GPU reports that GPU's count.
TEST_F(PartitionCountTest, ReportsTheCountOfEachGpu) {
  add_gpu(AMDSMI_ACCELERATOR_PARTITION_SPX, 1);  // RDC GPU 0
  add_gpu(AMDSMI_ACCELERATOR_PARTITION_DPX, 2);  // RDC GPUs 1 and 2
  add_gpu(AMDSMI_ACCELERATOR_PARTITION_CPX, 8);  // RDC GPUs 3 to 10
  const std::vector<int64_t> expected = {1, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8};

  for (uint32_t gpu = 0; gpu < expected.size(); gpu++) {
    rdc_field_value value;
    EXPECT_EQ(fetch(gpu, &value), RDC_ST_OK) << "GPU " << gpu;
    EXPECT_EQ(value.value.l_int, expected[gpu]) << "GPU " << gpu;
  }
}

// amd-smi has no partition profile for GPUs that cannot be partitioned.
TEST_F(PartitionCountTest, UnpartitionableGpuHasOnePartition) {
  add_gpu_with_profile_status(AMDSMI_STATUS_NOT_SUPPORTED);

  rdc_field_value value;
  EXPECT_EQ(fetch(0, &value), RDC_ST_OK);
  EXPECT_EQ(value.value.l_int, 1);
}

// A profile without a usable count gives the number of processors amd-smi lists
// for the GPU.
TEST_F(PartitionCountTest, CountsProcessorsWhenProfileHasNoCount) {
  for (auto& processor : add_gpu(AMDSMI_ACCELERATOR_PARTITION_QPX, 4)) {
    processor.profile.num_partitions = std::numeric_limits<uint32_t>::max();
  }

  for (uint32_t gpu = 0; gpu < 4; gpu++) {
    rdc_field_value value;
    EXPECT_EQ(fetch(gpu, &value), RDC_ST_OK) << "GPU " << gpu;
    EXPECT_EQ(value.value.l_int, 4) << "GPU " << gpu;
  }
}

TEST_F(PartitionCountTest, NotAvailableWhenProfileQueryFails) {
  add_gpu_with_profile_status(AMDSMI_STATUS_UNKNOWN_ERROR);

  rdc_field_value value;
  EXPECT_EQ(fetch(0, &value), RDC_ST_SMI_ERROR);
  EXPECT_EQ(value.field_id, RDC_FI_GPU_PARTITION_COUNT);
  EXPECT_EQ(value.status, AMDSMI_STATUS_UNKNOWN_ERROR);
}

// The count is read from the GPU's first processor, which must be partition 0.
TEST_F(PartitionCountTest, NotAvailableWhenFirstProcessorIsNotPartitionZero) {
  auto& processors = add_gpu(AMDSMI_ACCELERATOR_PARTITION_DPX, 2);
  processors[0].partition_id = 1;
  processors[1].partition_id = 0;

  rdc_field_value value;
  EXPECT_EQ(fetch(0, &value), RDC_ST_SMI_ERROR);
  EXPECT_EQ(value.status, AMDSMI_STATUS_UNEXPECTED_DATA);
}
