// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_AMD_SMI_RDMA_PORT_H_
#define AMD_SMI_INCLUDE_AMD_SMI_RDMA_PORT_H_

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "amd_smi/amdsmi.h"

namespace amd::smi {

// Position of one RDMA port in the unified NIC library's indexing.
struct RdmaPortRef_t {
  uint32_t port_index;
  uint32_t ib_index;
  uint32_t rdma_port_index;
};

/**
 * Resolves the flat RDMA port index of the public API to the unified library's (netdev port,
 * InfiniBand device, RDMA port) triple. The flat index counts each device's ports in
 * amdsmi_nic_rdma_devices_info_t order. A device's InfiniBand index is its position among the
 * devices on the same netdev port.
 */
inline auto was_rdma_port_resolved(const amdsmi_nic_rdma_devices_info_t& rdma,
                                   const amdsmi_nic_port_info_t& ports, uint32_t flat_index,
                                   RdmaPortRef_t* ref) -> bool {
  if (ref == nullptr) {
    return false;
  }

  const auto is_same_netdev = [](const char* lhs, const char* rhs) -> bool {
    return (std::strncmp(lhs, rhs, AMDSMI_MAX_STRING_LENGTH) == 0);
  };
  const auto num_devs = std::min<uint32_t>(rdma.num_rdma_dev, AMDSMI_MAX_NIC_RDMA_DEV);
  const auto num_netdevs = std::min<uint32_t>(ports.num_ports, AMDSMI_MAX_NIC_PORTS);

  auto first_flat_index = uint32_t{0};
  for (auto dev = uint32_t{0}; dev < num_devs; ++dev) {
    const auto& dev_info = rdma.rdma_dev_info[dev];
    const auto dev_ports = std::min<uint32_t>(dev_info.num_rdma_ports, AMDSMI_MAX_NIC_PORTS);
    if (flat_index >= (first_flat_index + dev_ports)) {
      first_flat_index += dev_ports;
      continue;
    }

    // Every port of a device reports the same netdev.
    const auto* netdev = dev_info.rdma_port_info[0].netdev;
    auto port_index = uint32_t{0};
    while ((port_index < num_netdevs) && !is_same_netdev(netdev, ports.ports[port_index].netdev)) {
      ++port_index;
    }
    if (port_index == num_netdevs) {
      return false;
    }

    auto ib_index = uint32_t{0};
    for (auto earlier = uint32_t{0}; earlier < dev; ++earlier) {
      if (is_same_netdev(rdma.rdma_dev_info[earlier].rdma_port_info[0].netdev, netdev)) {
        ++ib_index;
      }
    }

    *ref = RdmaPortRef_t{port_index, ib_index, (flat_index - first_flat_index)};
    return true;
  }
  return false;
}

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_RDMA_PORT_H_
