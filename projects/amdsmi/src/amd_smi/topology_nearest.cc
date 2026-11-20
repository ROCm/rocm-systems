// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <iterator>
#include <queue>
#include <vector>

#include "topology_nearest_internal.h"

namespace amd {
namespace smi {
namespace detail {

amdsmi_status_t get_link_topology_nearest(amdsmi_processor_handle processor_handle,
                                          amdsmi_link_type_t link_type,
                                          amdsmi_topology_nearest_t* topology_nearest_info,
                                          const TopologyDeps& deps) {
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
  if (auto api_status = deps.get_sockets(&socket_counter, nullptr);
      api_status != AMDSMI_STATUS_SUCCESS) {
    return api_status;
  }

  std::vector<amdsmi_socket_handle> socket_list(socket_counter);
  if (socket_counter != 0) {
    if (auto api_status = deps.get_sockets(&socket_counter, socket_list.data());
        api_status != AMDSMI_STATUS_SUCCESS) {
      return api_status;
    }
    if (socket_counter > socket_list.size()) {
      return AMDSMI_STATUS_UNEXPECTED_SIZE;
    }
  }

  amdsmi_processor_handle device_list[AMDSMI_MAX_DEVICES * AMDSMI_MAX_NUM_XCP];
  for (auto socket_idx = uint32_t(0); socket_idx < socket_counter; ++socket_idx) {
    uint32_t device_counter(AMDSMI_MAX_DEVICES * AMDSMI_MAX_NUM_XCP);
    if (auto api_status =
            deps.get_processors(socket_list[socket_idx], &device_counter, device_list);
        api_status != AMDSMI_STATUS_SUCCESS) {
      return api_status;
    }
    if (device_counter > std::size(device_list)) {
      return AMDSMI_STATUS_UNEXPECTED_SIZE;
    }

    for (auto device_idx = uint32_t(0); device_idx < device_counter; ++device_idx) {
      if (processor_handle != device_list[device_idx]) {
        auto is_accessible(false);
        if (auto api_status =
                deps.is_p2p_accessible(processor_handle, device_list[device_idx], &is_accessible);
            api_status != AMDSMI_STATUS_SUCCESS || !is_accessible) {
          continue;
        }

        auto link_type_new = link_type;
        auto num_hops = uint64_t(0);
        if (auto api_status = deps.get_link_type(processor_handle, device_list[device_idx],
                                                 &num_hops, &link_type_new);
            api_status != AMDSMI_STATUS_SUCCESS || link_type_new != link_type) {
          continue;
        }

        auto link_weight = uint64_t(0);
        if (auto api_status =
                deps.get_link_weight(processor_handle, device_list[device_idx], &link_weight);
            api_status != AMDSMI_STATUS_SUCCESS) {
          continue;
        }

        link_topology_order.push({device_list[device_idx], num_hops, link_weight});
      }
    }
  }

  *topology_nearest_info = {};
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

}  // namespace detail
}  // namespace smi
}  // namespace amd
