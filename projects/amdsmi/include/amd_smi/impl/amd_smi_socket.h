// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_AMD_SMI_SOCKET_H_
#define AMD_SMI_INCLUDE_AMD_SMI_SOCKET_H_

#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_processor.h"

namespace amd::smi {

class AMDSmiSocket {
 public:
  explicit AMDSmiSocket(const std::string& id) : socket_identifier_(id) {}
  explicit AMDSmiSocket(uint32_t index) : sindex_(index) {}
  ~AMDSmiSocket();
  const std::string& get_socket_id() const { return socket_identifier_; }
  uint32_t get_socket_index() { return sindex_; }
  void add_processor(AMDSmiProcessor* processor) {
    switch (processor->get_processor_type()) {
      case AMDSMI_PROCESSOR_TYPE_AMD_GPU:
        processors_.push_back(processor);
        break;
      case AMDSMI_PROCESSOR_TYPE_AMD_CPU:
        cpu_processors_.push_back(processor);
        break;
      case AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE:
        cpu_core_processors_.push_back(processor);
        break;
      case AMDSMI_PROCESSOR_TYPE_BRCM_NIC:
        nic_processors_.push_back(processor);
        break;
      case AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH:
        switch_processors_.push_back(processor);
        break;
      case AMDSMI_PROCESSOR_TYPE_AMD_NIC:
        ainic_processors_.push_back(processor);
        break;
      default:
        break;
    }
  }
  std::vector<AMDSmiProcessor*>& get_processors() { return processors_; }
  std::vector<AMDSmiProcessor*>& get_processors(amdsmi_processor_type_t type) {
    switch (type) {
      case AMDSMI_PROCESSOR_TYPE_AMD_GPU:
        return processors_;
      case AMDSMI_PROCESSOR_TYPE_AMD_CPU:
        return cpu_processors_;
      case AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE:
        return cpu_core_processors_;
      case AMDSMI_PROCESSOR_TYPE_AMD_NIC:
        return ainic_processors_;
      case AMDSMI_PROCESSOR_TYPE_BRCM_NIC:
        return nic_processors_;
      case AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH:
        return switch_processors_;
      default:
        return processors_;
    }
  }
  amdsmi_status_t get_processor_count(uint32_t* processor_count) const;
  amdsmi_status_t get_processor_count(amdsmi_processor_type_t type,
                                      uint32_t* processor_count) const;

 private:
  uint32_t sindex_;
  std::string socket_identifier_;
  std::vector<AMDSmiProcessor*> processors_;
  std::vector<AMDSmiProcessor*> cpu_processors_;
  std::vector<AMDSmiProcessor*> cpu_core_processors_;
  std::vector<AMDSmiProcessor*> ainic_processors_;
  std::vector<AMDSmiProcessor*> nic_processors_;
  std::vector<AMDSmiProcessor*> switch_processors_;
};

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_SOCKET_H_
