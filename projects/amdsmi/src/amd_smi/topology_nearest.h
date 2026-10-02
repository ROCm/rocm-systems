// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef SRC_AMD_SMI_TOPOLOGY_NEAREST_H_
#define SRC_AMD_SMI_TOPOLOGY_NEAREST_H_

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <queue>
#include <vector>

#include "amd_smi/amdsmi.h"

namespace amd::smi {

template <typename GetSockets = decltype(&amdsmi_get_socket_handles),
          typename GetProcessors = decltype(&amdsmi_get_processor_handles),
          typename IsAccessible = decltype(&amdsmi_is_P2P_accessible),
          typename GetLinkType = decltype(&amdsmi_topo_get_link_type),
          typename GetLinkWeight = decltype(&amdsmi_topo_get_link_weight)>
static amdsmi_status_t get_link_topology_nearest(
    amdsmi_processor_handle processor_handle, amdsmi_link_type_t link_type,
    amdsmi_topology_nearest_t* topology_nearest_info,
    GetSockets get_sockets = amdsmi_get_socket_handles,
    GetProcessors get_processors = amdsmi_get_processor_handles,
    IsAccessible is_p2p_accessible = amdsmi_is_P2P_accessible,
    GetLinkType get_link_type = amdsmi_topo_get_link_type,
    GetLinkWeight get_link_weight = amdsmi_topo_get_link_weight) {
  struct LinkTopologyInfo {
    amdsmi_processor_handle target_processor_handle;
    uint64_t num_hops;
    uint64_t link_weight;
  };

  struct LinkTopologyOrderCmp {
    constexpr bool operator()(const LinkTopologyInfo& left,
                              const LinkTopologyInfo& right) const noexcept {
      if (left.num_hops == right.num_hops) {
        return left.link_weight > right.link_weight;
      } else {
        return left.num_hops > right.num_hops;
      }
    }
  };
  std::priority_queue<LinkTopologyInfo, std::vector<LinkTopologyInfo>, LinkTopologyOrderCmp>
      link_topology_order{};

  auto socket_counter = uint32_t(0);
  if (auto api_status = get_sockets(&socket_counter, nullptr);
      api_status != AMDSMI_STATUS_SUCCESS) {
    return api_status;
  }

  std::vector<amdsmi_socket_handle> socket_list(socket_counter);
  if (auto api_status = get_sockets(&socket_counter, socket_list.data());
      api_status != AMDSMI_STATUS_SUCCESS) {
    return api_status;
  }

  amdsmi_processor_handle device_list[AMDSMI_MAX_DEVICES * AMDSMI_MAX_NUM_XCP];
  for (auto socket_idx = uint32_t(0); socket_idx < socket_counter; ++socket_idx) {
    uint32_t device_counter(AMDSMI_MAX_DEVICES * AMDSMI_MAX_NUM_XCP);
    if (auto api_status = get_processors(socket_list[socket_idx], &device_counter, device_list);
        api_status != AMDSMI_STATUS_SUCCESS) {
      return api_status;
    }

    for (auto device_idx = uint32_t(0); device_idx < device_counter; ++device_idx) {
      if (processor_handle != device_list[device_idx]) {
        auto is_accessible(false);
        if (auto api_status =
                is_p2p_accessible(processor_handle, device_list[device_idx], &is_accessible);
            api_status != AMDSMI_STATUS_SUCCESS || !is_accessible) {
          continue;
        }

        auto link_type_new = link_type;
        auto num_hops = uint64_t(0);
        if (auto api_status =
                get_link_type(processor_handle, device_list[device_idx], &num_hops, &link_type_new);
            api_status != AMDSMI_STATUS_SUCCESS || link_type_new != link_type) {
          continue;
        }

        auto link_weight = uint64_t(0);
        if (auto api_status =
                get_link_weight(processor_handle, device_list[device_idx], &link_weight);
            api_status != AMDSMI_STATUS_SUCCESS) {
          continue;
        }

        link_topology_order.push({device_list[device_idx], num_hops, link_weight});
      }
    }
  }

  std::fill(std::begin(topology_nearest_info->processor_list),
            std::end(topology_nearest_info->processor_list), nullptr);
  auto topology_nearest_counter = uint32_t(0);
  while (!link_topology_order.empty()) {
    auto link_info = link_topology_order.top();
    link_topology_order.pop();

    if (topology_nearest_counter < (AMDSMI_MAX_DEVICES * AMDSMI_MAX_NUM_XCP)) {
      topology_nearest_info->processor_list[topology_nearest_counter++] =
          link_info.target_processor_handle;
    }
  }
  topology_nearest_info->count = topology_nearest_counter;

  return AMDSMI_STATUS_SUCCESS;
}

}  // namespace amd::smi

#endif  // SRC_AMD_SMI_TOPOLOGY_NEAREST_H_
