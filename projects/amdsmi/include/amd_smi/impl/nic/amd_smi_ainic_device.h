// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

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
  };

  AMDSmiAINICDevice(uint32_t nic_idx, const amdsmi_bdf_t& bdf, const AINICInfo& ai_nic_info)
      : AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_NIC),
        nic_idx_(nic_idx),
        bdf_(bdf),
        ai_nic_info_(ai_nic_info) {}
  ~AMDSmiAINICDevice() = default;
  amdsmi_status_t amd_query_nic_info(AINICInfo& info) const;

 private:
  uint32_t nic_idx_;
  amdsmi_bdf_t bdf_;
  AINICInfo ai_nic_info_;
};

}  // namespace amd::smi

amdsmi_status_t amdsmi_get_ainic_info(amdsmi_processor_handle processor_handle,
                                      amd::smi::AMDSmiAINICDevice::AINICInfo* info);
