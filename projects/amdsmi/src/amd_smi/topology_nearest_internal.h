// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_TOPOLOGY_NEAREST_INTERNAL_H_
#define AMD_SMI_TOPOLOGY_NEAREST_INTERNAL_H_

#include "amd_smi/amdsmi.h"

namespace amd {
namespace smi {
namespace detail {

struct TopologyDeps {
  decltype(&amdsmi_get_socket_handles) get_sockets;
  decltype(&amdsmi_get_processor_handles) get_processors;
  decltype(&amdsmi_is_P2P_accessible) is_p2p_accessible;
  decltype(&amdsmi_topo_get_link_type) get_link_type;
  decltype(&amdsmi_topo_get_link_weight) get_link_weight;
};

amdsmi_status_t get_link_topology_nearest(amdsmi_processor_handle processor_handle,
                                          amdsmi_link_type_t link_type,
                                          amdsmi_topology_nearest_t* topology_nearest_info,
                                          const TopologyDeps& deps);

}  // namespace detail
}  // namespace smi
}  // namespace amd

#endif  // AMD_SMI_TOPOLOGY_NEAREST_INTERNAL_H_
