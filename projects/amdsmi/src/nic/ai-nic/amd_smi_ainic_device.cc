// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/nic/amd_smi_ainic_device.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

#include "amd_smi/impl/amd_smi_nic_testing.h"
#include "amd_smi/impl/amd_smi_system.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace amd::smi {
amdsmi_status_t AMDSmiAINICDevice::amd_query_nic_info(AINICInfo& info) const {
  const auto lock = std::lock_guard<std::mutex>{m_rdma_mtu_mutex};
  fill_rdma_mtu_once_locked();
  info = ai_nic_info_;
  return AMDSMI_STATUS_SUCCESS;
}

static void copy_port_mtu(const amdsmi_nic_rdma_dev_info_t& from, amdsmi_nic_rdma_dev_info_t& to) {
  const auto num_from = std::min<size_t>(from.num_rdma_ports, std::size(from.rdma_port_info));
  const auto num_to = std::min<size_t>(to.num_rdma_ports, std::size(to.rdma_port_info));
  for (size_t i = 0; i < num_to; i++) {
    auto& port = to.rdma_port_info[i];
    for (size_t j = 0; j < num_from; j++) {
      if (from.rdma_port_info[j].rdma_port == port.rdma_port) {
        port.max_mtu = from.rdma_port_info[j].max_mtu;
        port.active_mtu = from.rdma_port_info[j].active_mtu;
        break;
      }
    }
  }
}

void AMDSmiAINICDevice::copy_rdma_mtu(amdsmi_nic_rdma_devices_info_t& rdma_dev) const {
  const auto lock = std::lock_guard<std::mutex>{m_rdma_mtu_mutex};
  fill_rdma_mtu_once_locked();
  const auto& probed = ai_nic_info_.rdma_dev;
  const auto num_probed = std::min<size_t>(probed.num_rdma_dev, std::size(probed.rdma_dev_info));
  const auto num_read = std::min<size_t>(rdma_dev.num_rdma_dev, std::size(rdma_dev.rdma_dev_info));
  for (size_t i = 0; i < num_read; i++) {
    auto& device = rdma_dev.rdma_dev_info[i];
    for (size_t j = 0; j < num_probed; j++) {
      const auto& known = probed.rdma_dev_info[j];
      if (std::strncmp(device.rdma_dev, known.rdma_dev, sizeof(device.rdma_dev)) == 0) {
        copy_port_mtu(known, device);
        break;
      }
    }
  }
}

// Holds only this device's mutex across the probe, never the NIC context lock. A context-level
// failure is retried by the next read; a port the probe could not read stays unknown.
void AMDSmiAINICDevice::fill_rdma_mtu_once_locked() const {
  if (m_was_rdma_mtu_filled) {
    return;
  }
  if (ai_nic_info_.rdma_dev.num_rdma_dev == 0) {
    m_was_rdma_mtu_filled = true;
    return;
  }
  const auto status = fill_ainic_rdma_mtu(AMDSmiSystem::getInstance().get_ainic_ctx(), bdf(),
                                          ai_nic_info_.rdma_dev);
  m_was_rdma_mtu_filled = (status == SMI_NIC_STATUS_SUCCESS);
  if (!m_was_rdma_mtu_filled) {
    std::ostringstream ss;
    ss << __func__ << ": RDMA MTU unavailable for BDF=0x" << std::hex << bdf() << std::dec
       << " smi_nic_status_t: " << status << " - reporting it as unknown";
    LOG_INFO(ss);
  }
}
}  // namespace amd::smi
