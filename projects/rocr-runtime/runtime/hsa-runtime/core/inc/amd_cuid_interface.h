/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_INC_AMD_CUID_INTERFACE_H_
#define HSA_RUNTIME_CORE_INC_AMD_CUID_INTERFACE_H_

#include "hsa.h"
#include <cstdint>
#include <string>
#include <array>

namespace rocr {
namespace core {

class CuidInterface {
 public:
  static constexpr size_t kCuidLength = 16;  // derived CUID length in bytes

  // Query Secondary CUID for a GPU device.
  // CUID must point to at least kCuidLength bytes.
  static hsa_status_t QueryGpuCuid(const std::string& device_path, uint8_t* cuid,
                                   uint32_t* cuid_length);
};

}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_CUID_INTERFACE_H_