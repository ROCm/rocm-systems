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

// rdc_link_status_get() copies each GPU's XGMI link states from
// amdsmi_get_gpu_xgmi_link_status(). A failed call can leave that output unset
// or partly set. rdcd used to copy it anyway: it reported the previous GPU's
// links, or a garbage link count wrote past the link_states array. Both
// rdc_link_status_get() and rdc_device_topology_get() must also go on past a
// GPU that AMD SMI cannot identify.
//
// This binary builds RdcTopologyLinkImpl.cc itself, with the amdsmi calls that
// file makes replaced by the fakes below, so it needs neither a GPU nor librdc.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/impl/RdcTopologyLinkImpl.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace {

// What the fake amdsmi_get_gpu_xgmi_link_status() does for one GPU.
struct LinkQuery {
  amdsmi_status_t status;
  bool writes_output;
  amdsmi_xgmi_link_status_t output;
  // false: a GPU AMD SMI cannot identify, whose handle lookup fails with NOT_FOUND.
  bool available = true;
};

// One entry per GPU. The fake processor handle of GPU i points at entry i.
std::vector<LinkQuery> g_queries;

amdsmi_xgmi_link_status_t make_links(std::initializer_list<amdsmi_xgmi_link_status_type_t> states) {
  amdsmi_xgmi_link_status_t links = {};
  EXPECT_LE(states.size(), static_cast<size_t>(AMDSMI_MAX_NUM_XGMI_LINKS));
  for (auto state : states) {
    if (links.total_links == AMDSMI_MAX_NUM_XGMI_LINKS) {
      break;
    }
    links.status[links.total_links++] = state;
  }
  return links;
}

// rdc_link_status_get() only asks the fetcher for the GPU count.
class FakeMetricFetcher : public amd::rdc::RdcMetricFetcher {
 public:
  rdc_status_t acquire_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t delete_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t fetch_smi_field(uint32_t, rdc_field_t field_id, rdc_field_value* value) override {
    if (field_id != RDC_FI_GPU_COUNT) {
      return RDC_ST_NOT_SUPPORTED;
    }
    value->field_id = field_id;
    value->status = RDC_ST_OK;
    value->type = INTEGER;
    value->value.l_int = static_cast<int64_t>(g_queries.size());
    return RDC_ST_OK;
  }
  rdc_status_t bulk_fetch_smi_fields(rdc_gpu_field_t*, uint32_t,
                                     std::vector<rdc_gpu_field_value_t>&) override {
    return RDC_ST_NOT_SUPPORTED;
  }
};

// Runs rdc_link_status_get() over g_queries. The result starts out as junk, so
// every field the test checks must have been written by the call.
std::unique_ptr<rdc_link_status_t> get_link_status() {
  amd::rdc::RdcTopologyLinkImpl impl(nullptr, std::make_shared<FakeMetricFetcher>());
  auto results = std::make_unique<rdc_link_status_t>();
  std::memset(results.get(), 0xA5, sizeof(*results));
  EXPECT_EQ(impl.rdc_link_status_get(results.get()), RDC_ST_OK);
  return results;
}

}  // namespace

// Fakes for the amdsmi calls in RdcTopologyLinkImpl.cc.
namespace amd {
namespace rdc {
amdsmi_status_t get_processor_handle_from_id(uint32_t gpu_id,
                                             amdsmi_processor_handle* processor_handle) {
  if (gpu_id >= g_queries.size() || !g_queries[gpu_id].available) {
    return AMDSMI_STATUS_NOT_FOUND;
  }
  *processor_handle = &g_queries[gpu_id];
  return AMDSMI_STATUS_SUCCESS;
}
}  // namespace rdc
}  // namespace amd

amdsmi_status_t amdsmi_get_gpu_xgmi_link_status(amdsmi_processor_handle processor_handle,
                                                amdsmi_xgmi_link_status_t* link_status) {
  const auto* query = static_cast<const LinkQuery*>(processor_handle);
  if (query->writes_output) {
    *link_status = query->output;
  }
  return query->status;
}

// Only rdc_device_topology_get() calls these: every two GPUs are one XGMI hop apart.
amdsmi_status_t amdsmi_topo_get_numa_node_number(amdsmi_processor_handle, uint32_t* numa_node) {
  *numa_node = 0;
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t amdsmi_topo_get_link_weight(amdsmi_processor_handle, amdsmi_processor_handle,
                                            uint64_t* weight) {
  *weight = 15;
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t amdsmi_get_minmax_bandwidth_between_processors(amdsmi_processor_handle,
                                                               amdsmi_processor_handle,
                                                               uint64_t* min_bandwidth,
                                                               uint64_t* max_bandwidth) {
  *min_bandwidth = 1;
  *max_bandwidth = 2;
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t amdsmi_topo_get_link_type(amdsmi_processor_handle, amdsmi_processor_handle,
                                          uint64_t* hops, amdsmi_link_type_t* type) {
  *hops = 1;
  *type = AMDSMI_LINK_TYPE_XGMI;
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t amdsmi_is_P2P_accessible(amdsmi_processor_handle, amdsmi_processor_handle,
                                         bool* accessible) {
  *accessible = true;
  return AMDSMI_STATUS_SUCCESS;
}

TEST(RdcLinkStatus, CopiesLinkStates) {
  g_queries = {
      {AMDSMI_STATUS_SUCCESS, true,
       make_links({AMDSMI_XGMI_LINK_DISABLE, AMDSMI_XGMI_LINK_UP, AMDSMI_XGMI_LINK_DOWN})}};
  auto results = get_link_status();

  ASSERT_EQ(results->num_of_gpus, 1);
  const rdc_gpu_link_status_t& gpu = results->gpus[0];
  EXPECT_EQ(gpu.gpu_index, 0u);
  EXPECT_EQ(gpu.link_types, RDC_IOLINK_TYPE_XGMI);
  ASSERT_EQ(gpu.num_of_links, 3u);
  EXPECT_EQ(gpu.link_states[0], RDC_LINK_STATE_DISABLED);
  EXPECT_EQ(gpu.link_states[1], RDC_LINK_STATE_UP);
  EXPECT_EQ(gpu.link_states[2], RDC_LINK_STATE_DOWN);
}

TEST(RdcLinkStatus, FailedQueryReportsNoLinks) {
  // Junk such as an uninitialized output holds: a link count past both arrays.
  amdsmi_xgmi_link_status_t junk;
  std::memset(&junk, 0x5A, sizeof(junk));
  junk.total_links = 40;
  // amdsmi sets the link count, then stops at the first unexpected value.
  amdsmi_xgmi_link_status_t partial = make_links({AMDSMI_XGMI_LINK_UP});
  partial.total_links = AMDSMI_MAX_NUM_XGMI_LINKS;

  g_queries = {{AMDSMI_STATUS_API_FAILED, true, junk},
               {AMDSMI_STATUS_UNEXPECTED_DATA, true, partial},
               {AMDSMI_STATUS_NO_PERM, false, {}},
               {AMDSMI_STATUS_SUCCESS, true, make_links({AMDSMI_XGMI_LINK_UP})}};
  auto results = get_link_status();

  ASSERT_EQ(results->num_of_gpus, 4);
  for (uint32_t i = 0; i < 4; i++) {
    SCOPED_TRACE(i);
    EXPECT_EQ(results->gpus[i].gpu_index, i);
    EXPECT_EQ(results->gpus[i].link_types, RDC_IOLINK_TYPE_XGMI);
  }
  EXPECT_EQ(results->gpus[0].num_of_links, 0u);
  EXPECT_EQ(results->gpus[1].num_of_links, 0u);
  EXPECT_EQ(results->gpus[2].num_of_links, 0u);
  ASSERT_EQ(results->gpus[3].num_of_links, 1u);
  EXPECT_EQ(results->gpus[3].link_states[0], RDC_LINK_STATE_UP);
}

TEST(RdcLinkStatus, UnavailableGpuReportsNoLinks) {
  // GPU 1 keeps its index while AMD SMI cannot identify it; the GPUs after it
  // still report their links.
  g_queries = {{AMDSMI_STATUS_SUCCESS, true, make_links({AMDSMI_XGMI_LINK_UP})},
               {AMDSMI_STATUS_SUCCESS, true, make_links({AMDSMI_XGMI_LINK_UP}), false},
               {AMDSMI_STATUS_SUCCESS, true, make_links({AMDSMI_XGMI_LINK_DOWN})}};
  auto results = get_link_status();

  ASSERT_EQ(results->num_of_gpus, 3);
  for (uint32_t i = 0; i < 3; i++) {
    SCOPED_TRACE(i);
    EXPECT_EQ(results->gpus[i].gpu_index, i);
    EXPECT_EQ(results->gpus[i].link_types, RDC_IOLINK_TYPE_XGMI);
  }
  ASSERT_EQ(results->gpus[0].num_of_links, 1u);
  EXPECT_EQ(results->gpus[0].link_states[0], RDC_LINK_STATE_UP);
  EXPECT_EQ(results->gpus[1].num_of_links, 0u);
  ASSERT_EQ(results->gpus[2].num_of_links, 1u);
  EXPECT_EQ(results->gpus[2].link_states[0], RDC_LINK_STATE_DOWN);
}

TEST(RdcDeviceTopology, UnavailableGpuReadsNotAvailable) {
  // GPU 1 keeps its index while AMD SMI cannot identify it: its entry reads N/A
  // (the largest values and an undefined link type), and GPU 0's link to the
  // GPU after it still reads. The call zeroes the result first, so the link type
  // and P2P checks below hold either way; the others need the N/A entry.
  g_queries = {{AMDSMI_STATUS_SUCCESS, true, {}},
               {AMDSMI_STATUS_SUCCESS, true, {}, false},
               {AMDSMI_STATUS_SUCCESS, true, {}}};
  amd::rdc::RdcTopologyLinkImpl impl(nullptr, std::make_shared<FakeMetricFetcher>());
  auto results = std::make_unique<rdc_device_topology_t>();
  ASSERT_EQ(impl.rdc_device_topology_get(0, results.get()), RDC_ST_OK);

  ASSERT_EQ(results->num_of_gpus, 3u);
  const rdc_topology_link_info_t& unavailable = results->link_infos[1];
  EXPECT_EQ(unavailable.gpu_index, 1u);
  EXPECT_EQ(unavailable.weight, UINT64_MAX);
  EXPECT_EQ(unavailable.min_bandwidth, UINT64_MAX);
  EXPECT_EQ(unavailable.max_bandwidth, UINT64_MAX);
  EXPECT_EQ(unavailable.hops, UINT64_MAX);
  EXPECT_EQ(unavailable.link_type, RDC_IOLINK_TYPE_UNDEFINED);
  EXPECT_FALSE(unavailable.is_p2p_accessible);
  // link_infos[i] is the link from GPU 0 to GPU i; GPU 0's own entry reads N/A.
  EXPECT_EQ(results->link_infos[0].gpu_index, 0u);
  EXPECT_EQ(results->link_infos[0].weight, UINT64_MAX);
  const rdc_topology_link_info_t& after = results->link_infos[2];
  EXPECT_EQ(after.gpu_index, 2u);
  EXPECT_EQ(after.weight, 15u);
  EXPECT_EQ(after.hops, 1u);
  EXPECT_EQ(after.link_type, RDC_IOLINK_TYPE_XGMI);
  EXPECT_TRUE(after.is_p2p_accessible);
}

TEST(RdcLinkStatus, NotSupportedKeepsDisabledLinks) {
  // amdsmi returns NOT_SUPPORTED after marking every link disabled, and also
  // when it fails before writing anything.
  g_queries = {{AMDSMI_STATUS_NOT_SUPPORTED, true,
                make_links({AMDSMI_XGMI_LINK_DISABLE, AMDSMI_XGMI_LINK_DISABLE})},
               {AMDSMI_STATUS_NOT_SUPPORTED, false, {}}};
  auto results = get_link_status();

  ASSERT_EQ(results->num_of_gpus, 2);
  ASSERT_EQ(results->gpus[0].num_of_links, 2u);
  EXPECT_EQ(results->gpus[0].link_states[0], RDC_LINK_STATE_DISABLED);
  EXPECT_EQ(results->gpus[0].link_states[1], RDC_LINK_STATE_DISABLED);
  EXPECT_EQ(results->gpus[1].num_of_links, 0u);
}

TEST(RdcLinkStatus, CapsLinkCountAtArraySize) {
  amdsmi_xgmi_link_status_t links = make_links(
      {AMDSMI_XGMI_LINK_UP, AMDSMI_XGMI_LINK_UP, AMDSMI_XGMI_LINK_UP, AMDSMI_XGMI_LINK_UP,
       AMDSMI_XGMI_LINK_UP, AMDSMI_XGMI_LINK_UP, AMDSMI_XGMI_LINK_UP, AMDSMI_XGMI_LINK_UP});
  links.total_links = 2 * RDC_MAX_NUM_OF_LINKS;
  g_queries = {{AMDSMI_STATUS_SUCCESS, true, links}};
  auto results = get_link_status();

  const uint32_t max_links = std::min<uint32_t>(AMDSMI_MAX_NUM_XGMI_LINKS, RDC_MAX_NUM_OF_LINKS);
  ASSERT_EQ(results->num_of_gpus, 1);
  ASSERT_EQ(results->gpus[0].num_of_links, max_links);
  for (uint32_t n = 0; n < max_links; n++) {
    EXPECT_EQ(results->gpus[0].link_states[n], RDC_LINK_STATE_UP) << "link " << n;
  }
}
