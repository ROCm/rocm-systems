// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <mutex>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_processor.h"

// User could to get the AI_NIC processor using below existing function
// ret = amdsmi_get_processor_handles_by_type(sockets[i], AMDSMI_PROCESSOR_TYPE_AMD_NIC, nullptr,
// &tmp_device_count); Get the ai nic information using the nic handle

namespace amd::smi {

class AMDSmiAINICDevice : public AMDSmiProcessor {
 public:
  /**
   * @brief Main NIC Information
   *
   * @cond @tag{gpu_bm_linux} @endcond
   */
  struct AINICInfo {
    amdsmi_nic_asic_info_t asic;
    amdsmi_nic_bus_info_t bus;
    amdsmi_nic_driver_info_t driver;
    amdsmi_nic_numa_info_t numa;
    amdsmi_nic_fw_entry_t versions;
    amdsmi_nic_port_info_t port;
    amdsmi_nic_rdma_devices_info_t rdma_dev;
    uint8_t type;         //!< SmiNicKind value
    uint32_t capability;  //!< amdsmi_nic_capability_bits_t bitmask
  };

  explicit AMDSmiAINICDevice(const AINICInfo& ai_nic_info)
      : AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_NIC), ai_nic_info_(ai_nic_info) {}
  ~AMDSmiAINICDevice() = default;
  amdsmi_status_t amd_query_nic_info(AINICInfo& info) const;

  // Single-field reads that avoid copying the whole AINICInfo, which is over 600 KB.
  const amdsmi_nic_asic_info_t& asic() const { return ai_nic_info_.asic; }
  const amdsmi_nic_bus_info_t& bus() const { return ai_nic_info_.bus; }
  const amdsmi_nic_driver_info_t& driver() const { return ai_nic_info_.driver; }
  const amdsmi_nic_numa_info_t& numa() const { return ai_nic_info_.numa; }
  const amdsmi_nic_port_info_t& port() const { return ai_nic_info_.port; }
  // The RDMA MTU is probed on first use (the probe forks per port); every read that returns an
  // MTU goes through get_rdma_dev() or amd_query_nic_info(), which copy under the fill's lock.
  void get_rdma_dev(amdsmi_nic_rdma_devices_info_t& rdma_dev) const;
  // For readers that only need which RDMA devices and ports exist, not their MTU.
  const amdsmi_nic_rdma_devices_info_t& rdma_dev_without_mtu() const {
    return ai_nic_info_.rdma_dev;
  }
  uint8_t nic_type() const { return ai_nic_info_.type; }
  uint32_t capability() const { return ai_nic_info_.capability; }
  uint64_t bdf() const { return ai_nic_info_.bus.bdf.as_uint; }

 private:
  void fill_rdma_mtu_once_locked() const;

  // Mutable: only the RDMA MTU fields change after construction, under m_rdma_mtu_mutex.
  mutable AINICInfo ai_nic_info_;
  mutable std::mutex m_rdma_mtu_mutex;
  mutable bool m_was_rdma_mtu_filled = false;
};

}  // namespace amd::smi

amdsmi_status_t amdsmi_get_ainic_info(amdsmi_processor_handle processor_handle,
                                      amd::smi::AMDSmiAINICDevice::AINICInfo* info);
