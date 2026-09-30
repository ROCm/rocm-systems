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
  explicit AMDSmiProcessor(const std::string& id) : processor_identifier_(id) {}
  virtual ~AMDSmiProcessor() {}
  amdsmi_processor_type_t get_processor_type() const { return processor_type_; }
  const std::string& get_processor_id() const { return processor_identifier_; }
  uint32_t get_processor_index() const { return pindex_; }

 private:
  amdsmi_processor_type_t processor_type_;
  // Out-of-range sentinel so the CPU and Core APIs can reject an unset index.
  uint32_t pindex_{UINT32_MAX};
  std::string processor_identifier_;
};
}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_PROCESSOR_H_
