// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_AMD_SMI_PROCESSOR_H_
#define AMD_SMI_INCLUDE_AMD_SMI_PROCESSOR_H_

#include <cstdint>
#include <string>

#include "amd_smi/amdsmi.h"

namespace amd::smi {

class AMDSmiProcessor {
 public:
  explicit AMDSmiProcessor(amdsmi_processor_type_t type) : processor_type_(type) {}
  explicit AMDSmiProcessor(amdsmi_processor_type_t type, uint32_t index)
      : processor_type_(type), pindex_(index) {}
  // The public processor index remains the ordinal within the socket. ESMI
  // addresses a physical core by its first Linux logical CPU instead.
  AMDSmiProcessor(uint32_t local_core, uint32_t esmi_cpu, uint32_t socket)
      : processor_type_(AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE),
        pindex_(local_core),
        esmi_cpu_(esmi_cpu),
        cpu_socket_(socket) {}
  explicit AMDSmiProcessor(const std::string& id) : processor_identifier_(id) {}
  virtual ~AMDSmiProcessor() {}
  amdsmi_processor_type_t get_processor_type() const { return processor_type_; }
  uint32_t esmi_cpu() const { return esmi_cpu_; }
  uint32_t cpu_socket() const { return cpu_socket_; }
  const std::string& get_processor_id() const { return processor_identifier_; }
  uint32_t get_processor_index() const { return pindex_; }

 private:
  amdsmi_processor_type_t processor_type_;
  // Out-of-range sentinel so the CPU and Core APIs can reject an unset index.
  uint32_t pindex_{UINT32_MAX};
  uint32_t esmi_cpu_{UINT32_MAX};
  uint32_t cpu_socket_{UINT32_MAX};
  std::string processor_identifier_;
};
}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_PROCESSOR_H_
