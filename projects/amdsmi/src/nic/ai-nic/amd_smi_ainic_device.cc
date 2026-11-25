// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/nic/amd_smi_ainic_device.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
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

void AMDSmiAINICDevice::get_rdma_dev(amdsmi_nic_rdma_devices_info_t& rdma_dev) const {
  const auto lock = std::lock_guard<std::mutex>{m_rdma_mtu_mutex};
  fill_rdma_mtu_once_locked();
  rdma_dev = ai_nic_info_.rdma_dev;
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
