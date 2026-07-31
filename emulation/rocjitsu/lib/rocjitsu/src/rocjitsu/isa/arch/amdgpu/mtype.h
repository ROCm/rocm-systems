// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_MTYPE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_MTYPE_H_

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief AMD memory type derived from instruction encoding bits.
enum class Mtype : uint8_t {
  UC = 0, ///< Uncacheable: bypass all caches.
  CC = 1, ///< Coherently cacheable.
  RW = 2, ///< Read-write cached in L1/L2.
  WB = 3, ///< Write-back; equivalent to RW for the simulator.
  NT = 4, ///< Non-temporal: bypass L1 and use L2.
};

/// @brief Combine instruction- and page-level MTYPE restrictions.
inline constexpr Mtype effective_mtype(Mtype instruction_mtype, Mtype pte_mtype) {
  if (pte_mtype == Mtype::RW || pte_mtype == Mtype::WB)
    return instruction_mtype;
  if (instruction_mtype == Mtype::NT)
    return (pte_mtype < Mtype::RW) ? pte_mtype : Mtype::NT;
  return (pte_mtype < instruction_mtype) ? pte_mtype : instruction_mtype;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_MTYPE_H_
