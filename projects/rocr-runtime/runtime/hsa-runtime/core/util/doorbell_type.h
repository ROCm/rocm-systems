// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef HSA_RUNTIME_CORE_UTIL_DOORBELL_TYPE_H_
#define HSA_RUNTIME_CORE_UTIL_DOORBELL_TYPE_H_

#include <cstdint>

namespace rocr {
namespace AMD {

inline constexpr unsigned int kDoorbellTypePre1_0 = 0;
inline constexpr unsigned int kDoorbellType1_0 = 1;
inline constexpr unsigned int kDoorbellType2_0 = 2;
inline constexpr unsigned int kDoorbellTypeReserved = 3;

// DoorbellType occupies bits 12-13 of the KFD capability field.
inline constexpr unsigned int ExtractDoorbellType(uint32_t capability) {
  return (capability >> 12) & 0x3;
}

inline constexpr uint32_t MakeCapabilityWithDoorbell(unsigned int doorbell_type) {
  return (doorbell_type & 0x3) << 12;
}

// ROCr supports the Vega-and-newer 2.0 doorbell protocol only.
inline constexpr bool IsDoorbellTypeSupported(unsigned int doorbell_type) {
  return doorbell_type == kDoorbellType2_0;
}

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_UTIL_DOORBELL_TYPE_H_
