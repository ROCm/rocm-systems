/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "rdc_lib/impl/RdcTopologyLinkImpl.h"

#include <assert.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <ctime>
#include <map>
#include <sstream>
#include <unordered_map>

#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/SmiUtils.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdcTopologyLinkImpl::RdcTopologyLinkImpl(const RdcGroupSettingsPtr& group_settings,
                                         RdcMetricFetcherPtr metric_fetcher)
    : group_settings_(group_settings), metric_fetcher_(metric_fetcher) {}

RdcTopologyLinkImpl::~RdcTopologyLinkImpl() {}

rdc_status_t RdcTopologyLinkImpl::rdc_device_topology_get(uint32_t gpu_index,
                                                          rdc_device_topology_t* results) {
  rdc_status_t status = RDC_ST_NOT_FOUND;
  amdsmi_status_t err = AMDSMI_STATUS_SUCCESS;
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];

  amdsmi_processor_handle processor_handle;
  err = get_processor_handle_from_id(gpu_index, &processor_handle);
  if (err != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_INFO, "Fail to get process GPUs processor handle information: " << err);
    return status;
  }

  uint32_t numa_node = 0;
  err = amdsmi_topo_get_numa_node_number(processor_handle, &numa_node);
  if (err != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_INFO, "Fail to get process GPUs numa_node information: " << err);
    return status;
  }

  uint32_t count = 0;
  rdc_field_value device_count;
  status = metric_fetcher_->fetch_smi_field(0, RDC_FI_GPU_COUNT, &device_count);
  if (status != RDC_ST_OK) {
    return status;
  }

  // Assign the index to the index list
  count = device_count.value.l_int;
  assert(count <= RDC_MAX_NUM_DEVICES);
  for (uint32_t i = 0; i < count; i++) {
    gpu_index_list[i] = i;
  }

  results->num_of_gpus = count;
  results->numa_node = numa_node;

  for (uint32_t i = 0; i < count; i++) {
    std::pair<amdsmi_processor_handle, amdsmi_processor_handle> ph;
    err = get_processor_handle_from_id(gpu_index_list[i], &ph.first);
    if (err != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_INFO, "Fail to get process GPUs processor handle information: " << err);
      return status;
    }
    for (uint32_t j = 0; j < count; j++) {
      if (gpu_index_list[i] == gpu_index_list[j]) continue;
      err = get_processor_handle_from_id(gpu_index_list[j], &ph.second);
      if (err != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_INFO, "Fail to get process GPUs processor handle information: " << err);
        return status;
      }

      uint64_t weight = std::numeric_limits<uint64_t>::max();
      err = amdsmi_topo_get_link_weight(ph.first, ph.second, &weight);
      if (err != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_INFO, "Fail to get process GPUs weight information: " << err);
      }

      uint64_t min_bandwidth = std::numeric_limits<uint64_t>::max();
      uint64_t max_bandwidth = std::numeric_limits<uint64_t>::max();
      err = amdsmi_get_minmax_bandwidth_between_processors(ph.first, ph.second, &min_bandwidth,
                                                           &max_bandwidth);
      if (err != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_INFO, "Fail to get process GPUs detail information: " << err);
      }

      uint64_t hops = std::numeric_limits<uint64_t>::max();
      amdsmi_link_type_t type = AMDSMI_LINK_TYPE_UNKNOWN;
      err = amdsmi_topo_get_link_type(ph.first, ph.second, &hops, &type);
      if (err != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_INFO, "Fail to get process GPUs hops and type information: " << err);
      }

      bool accessible = false;
      err = amdsmi_is_P2P_accessible(ph.first, ph.second, &accessible);
      if (err != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_INFO, "Fail to get process GPUs P2P accessible information: " << err);
      }

      results->link_infos[i].gpu_index = gpu_index_list[i];
      results->link_infos[i].weight = weight;
      results->link_infos[i].min_bandwidth = min_bandwidth;
      results->link_infos[i].max_bandwidth = max_bandwidth;
      results->link_infos[i].hops = hops;
      results->link_infos[i].link_type = static_cast<rdc_topology_link_type_t>(type);
      results->link_infos[i].is_p2p_accessible = accessible;
    }
  }
  return RDC_ST_OK;
}

rdc_status_t RdcTopologyLinkImpl::rdc_link_status_get(rdc_link_status_t* results) {
  rdc_status_t status = RDC_ST_NOT_FOUND;
  amdsmi_status_t err = AMDSMI_STATUS_SUCCESS;
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];

  uint32_t count = 0;
  rdc_field_value device_count;
  status = metric_fetcher_->fetch_smi_field(0, RDC_FI_GPU_COUNT, &device_count);
  if (status != RDC_ST_OK) {
    return status;
  }

  // Assign the index to the index list
  count = device_count.value.l_int;
  assert(count <= RDC_MAX_NUM_DEVICES);
  for (uint32_t i = 0; i < count; i++) {
    gpu_index_list[i] = i;
  }
  results->num_of_gpus = count;

  for (uint32_t i = 0; i < count; i++) {
    amdsmi_processor_handle processor_handle;
    err = get_processor_handle_from_id(gpu_index_list[i], &processor_handle);
    if (err != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_INFO, "Fail to get process GPUs processor handle information: " << err);
      return status;
    }

    amdsmi_xgmi_link_status_t link_status;
    err = amdsmi_get_gpu_xgmi_link_status(processor_handle, &link_status);
    if (err != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_INFO, "Fail to get process GPUs xgmi link information: " << err);
    }
    results->gpus[i].gpu_index = gpu_index_list[i];
    results->gpus[i].num_of_links = link_status.total_links;
    for (uint32_t n = 0; n < link_status.total_links; n++) {
      results->gpus[i].link_states[n] = static_cast<rdc_link_state_t>(link_status.status[n]);
    }

    results->gpus[i].link_types = RDC_IOLINK_TYPE_XGMI;
  }

  return status;
}

}  // namespace rdc
}  // namespace amd
