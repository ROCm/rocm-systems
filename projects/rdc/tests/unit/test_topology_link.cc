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

// rdc_device_topology_get(gpu_index) reports the links from gpu_index to every
// GPU: rdci topo prints link_infos[i] as the link to GPU i. This test builds
// RdcTopologyLinkImpl.cc against the fake AMD SMI calls below, so it needs no GPU.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/impl/RdcTopologyLinkImpl.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace {

// Fake system: GPUs 0-3 and GPUs 4-7 are two XGMI hives joined by PCIe. Every
// ordered GPU pair gets its own weight and bandwidth, so a link reported for the
// wrong pair shows up.
constexpr uint32_t kGpusPerHive = 4;
constexpr uint32_t kNoGpu = std::numeric_limits<uint32_t>::max();
constexpr uint64_t kNotAvailable = std::numeric_limits<uint64_t>::max();

uint32_t gpu_count = 8;
uint32_t missing_gpu = kNoGpu;   // get_processor_handle_from_id() fails for this GPU
uint32_t failing_peer = kNoGpu;  // AMD SMI link queries to this GPU fail

amdsmi_processor_handle handle_of(uint32_t index) {
  return reinterpret_cast<amdsmi_processor_handle>(static_cast<uintptr_t>(index) + 1);
}

uint32_t index_of(amdsmi_processor_handle handle) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(handle) - 1);
}

bool same_hive(uint32_t a, uint32_t b) { return a / kGpusPerHive == b / kGpusPerHive; }

uint64_t weight_of(uint32_t src, uint32_t dst) { return 1000 * (src + 1) + dst; }

void ExpectNotAvailable(const rdc_topology_link_info_t& link) {
  EXPECT_EQ(link.weight, kNotAvailable);
  EXPECT_EQ(link.min_bandwidth, kNotAvailable);
  EXPECT_EQ(link.max_bandwidth, kNotAvailable);
  EXPECT_EQ(link.hops, kNotAvailable);
  EXPECT_FALSE(link.is_p2p_accessible);
}

class FakeMetricFetcher : public amd::rdc::RdcMetricFetcher {
 public:
  rdc_status_t acquire_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t delete_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t fetch_smi_field(uint32_t, rdc_field_t field_id, rdc_field_value* value) override {
    if (field_id != RDC_FI_GPU_COUNT) return RDC_ST_NOT_SUPPORTED;
    value->type = INTEGER;
    value->status = AMDSMI_STATUS_SUCCESS;
    value->value.l_int = gpu_count;
    return RDC_ST_OK;
  }
  rdc_status_t bulk_fetch_smi_fields(rdc_gpu_field_t*, uint32_t,
                                     std::vector<rdc_gpu_field_value_t>&) override {
    return RDC_ST_NOT_SUPPORTED;
  }
};

class RdcTopologyLinkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    gpu_count = 8;
    missing_gpu = kNoGpu;
    failing_peer = kNoGpu;
  }

  amd::rdc::RdcTopologyLinkImpl topology_{nullptr, std::make_shared<FakeMetricFetcher>()};
};

}  // namespace

// Fakes for the AMD SMI lookups made by RdcTopologyLinkImpl.cc.
namespace amd {
namespace rdc {
amdsmi_status_t get_processor_handle_from_id(uint32_t gpu_id,
                                             amdsmi_processor_handle* processor_handle) {
  if (gpu_id >= gpu_count || gpu_id == missing_gpu) return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  *processor_handle = handle_of(gpu_id);
  return AMDSMI_STATUS_SUCCESS;
}
}  // namespace rdc
}  // namespace amd

extern "C" {
amdsmi_status_t amdsmi_topo_get_numa_node_number(amdsmi_processor_handle processor_handle,
                                                 uint32_t* numa_node) {
  *numa_node = index_of(processor_handle) / kGpusPerHive;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_topo_get_link_weight(amdsmi_processor_handle src,
                                            amdsmi_processor_handle dst, uint64_t* weight) {
  if (index_of(dst) == failing_peer) return AMDSMI_STATUS_NOT_SUPPORTED;
  *weight = weight_of(index_of(src), index_of(dst));
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_minmax_bandwidth_between_processors(amdsmi_processor_handle src,
                                                               amdsmi_processor_handle dst,
                                                               uint64_t* min_bandwidth,
                                                               uint64_t* max_bandwidth) {
  if (index_of(dst) == failing_peer) return AMDSMI_STATUS_NOT_SUPPORTED;
  *min_bandwidth = 10 * weight_of(index_of(src), index_of(dst));
  *max_bandwidth = 20 * weight_of(index_of(src), index_of(dst));
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_topo_get_link_type(amdsmi_processor_handle src, amdsmi_processor_handle dst,
                                          uint64_t* hops, amdsmi_link_type_t* type) {
  if (index_of(dst) == failing_peer) return AMDSMI_STATUS_NOT_SUPPORTED;
  const bool xgmi = same_hive(index_of(src), index_of(dst));
  *hops = xgmi ? 1 : 2;
  *type = xgmi ? AMDSMI_LINK_TYPE_XGMI : AMDSMI_LINK_TYPE_PCIE;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_is_P2P_accessible(amdsmi_processor_handle src, amdsmi_processor_handle dst,
                                         bool* accessible) {
  if (index_of(dst) == failing_peer) return AMDSMI_STATUS_NOT_SUPPORTED;
  *accessible = same_hive(index_of(src), index_of(dst));
  return AMDSMI_STATUS_SUCCESS;
}

// Used by rdc_link_status_get(), which these tests do not call.
amdsmi_status_t amdsmi_get_gpu_xgmi_link_status(amdsmi_processor_handle,
                                                amdsmi_xgmi_link_status_t*) {
  return AMDSMI_STATUS_NOT_SUPPORTED;
}
}  // extern "C"

TEST_F(RdcTopologyLinkTest, ReportsLinksOfTheRequestedGpu) {
  for (uint32_t gpu = 0; gpu < gpu_count; ++gpu) {
    SCOPED_TRACE("gpu " + std::to_string(gpu));
    rdc_device_topology_t topo;
    ASSERT_EQ(topology_.rdc_device_topology_get(gpu, &topo), RDC_ST_OK);
    ASSERT_EQ(topo.num_of_gpus, gpu_count);
    EXPECT_EQ(topo.numa_node, gpu / kGpusPerHive);

    for (uint32_t peer = 0; peer < gpu_count; ++peer) {
      SCOPED_TRACE("peer " + std::to_string(peer));
      const rdc_topology_link_info_t& link = topo.link_infos[peer];
      EXPECT_EQ(link.gpu_index, peer);
      if (peer == gpu) {
        // A GPU has no link to itself.
        ExpectNotAvailable(link);
        EXPECT_EQ(link.link_type, RDC_IOLINK_TYPE_UNDEFINED);
        continue;
      }
      const bool xgmi = same_hive(gpu, peer);
      EXPECT_EQ(link.weight, weight_of(gpu, peer));
      EXPECT_EQ(link.min_bandwidth, 10 * weight_of(gpu, peer));
      EXPECT_EQ(link.max_bandwidth, 20 * weight_of(gpu, peer));
      EXPECT_EQ(link.hops, xgmi ? 1u : 2u);
      EXPECT_EQ(link.link_type, xgmi ? RDC_IOLINK_TYPE_XGMI : RDC_IOLINK_TYPE_PCIEXPRESS);
      EXPECT_EQ(link.is_p2p_accessible, xgmi);
    }
  }
}

TEST_F(RdcTopologyLinkTest, ReportsFailedLookups) {
  rdc_device_topology_t topo;

  // A failed link query leaves that link N/A and still reports the others.
  failing_peer = 5;
  ASSERT_EQ(topology_.rdc_device_topology_get(0, &topo), RDC_ST_OK);
  ExpectNotAvailable(topo.link_infos[5]);
  EXPECT_EQ(topo.link_infos[4].weight, weight_of(0, 4));
  EXPECT_EQ(topo.link_infos[6].weight, weight_of(0, 6));

  // A GPU that cannot be looked up fails the whole query.
  failing_peer = kNoGpu;
  missing_gpu = 5;
  EXPECT_EQ(topology_.rdc_device_topology_get(0, &topo), RDC_ST_NOT_FOUND);
}

TEST_F(RdcTopologyLinkTest, RejectsUnknownGpu) {
  rdc_device_topology_t topo;
  EXPECT_EQ(topology_.rdc_device_topology_get(gpu_count, &topo), RDC_ST_NOT_FOUND);
}

TEST_F(RdcTopologyLinkTest, RejectsMoreGpusThanTheResultHolds) {
  rdc_device_topology_t topo;

  gpu_count = RDC_MAX_NUM_DEVICES;
  ASSERT_EQ(topology_.rdc_device_topology_get(0, &topo), RDC_ST_OK);
  EXPECT_EQ(topo.num_of_gpus, RDC_MAX_NUM_DEVICES);
  EXPECT_EQ(topo.link_infos[RDC_MAX_NUM_DEVICES - 1].weight, weight_of(0, RDC_MAX_NUM_DEVICES - 1));

  gpu_count = RDC_MAX_NUM_DEVICES + 1;
  EXPECT_EQ(topology_.rdc_device_topology_get(0, &topo), RDC_ST_MAX_LIMIT);
}
