// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/amd_smi_socket.h"

namespace amd::smi {

AMDSmiSocket::~AMDSmiSocket() {
  for (uint32_t i = 0; i < processors_.size(); i++) {
    delete processors_[i];
  }
  processors_.clear();
  for (uint32_t i = 0; i < cpu_processors_.size(); i++) {
    delete cpu_processors_[i];
  }
  cpu_processors_.clear();
  for (uint32_t i = 0; i < cpu_core_processors_.size(); i++) {
    delete cpu_core_processors_[i];
  }
  cpu_core_processors_.clear();
  for (uint32_t i = 0; i < nic_processors_.size(); i++) {
    delete nic_processors_[i];
  }
  nic_processors_.clear();

  for (uint32_t i = 0; i < switch_processors_.size(); i++) {
    delete switch_processors_[i];
  }
  switch_processors_.clear();
}

amdsmi_status_t AMDSmiSocket::get_processor_count(uint32_t* processor_count) const {
  *processor_count = static_cast<uint32_t>(processors_.size());
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiSocket::get_processor_count(amdsmi_processor_type_t type,
                                                  uint32_t* processor_count) const {
  amdsmi_status_t ret = AMDSMI_STATUS_SUCCESS;
  switch (type) {
    case AMDSMI_PROCESSOR_TYPE_AMD_GPU:
      *processor_count = static_cast<uint32_t>(processors_.size());
      break;
    case AMDSMI_PROCESSOR_TYPE_AMD_CPU:
      *processor_count = static_cast<uint32_t>(cpu_processors_.size());
      break;
    case AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE:
      *processor_count = static_cast<uint32_t>(cpu_core_processors_.size());
      break;
    case AMDSMI_PROCESSOR_TYPE_AMD_NIC:
      *processor_count = static_cast<uint32_t>(ainic_processors_.size());
      break;
    case AMDSMI_PROCESSOR_TYPE_BRCM_NIC:
      *processor_count = static_cast<uint32_t>(nic_processors_.size());
      break;
    case AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH:
      *processor_count = static_cast<uint32_t>(switch_processors_.size());
      break;
    default:
      *processor_count = 0;
      ret = AMDSMI_STATUS_INVAL;
      break;
  }
  return ret;
}

}  // namespace amd::smi
