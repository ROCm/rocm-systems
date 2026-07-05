// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_MTYPE_H_
#define ROCJITSU_VM_AMDGPU_MTYPE_H_

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief AMD Memory Type (MType) derived from instruction encoding bits.
///
/// @details For more information about the AMD GPU memory model and MTypes
/// see the user guide for the [AMDGPU backend](https://llvm.org/docs/AMDGPUUsage.html).
///
/// Caching behavior derived from the sc1, sc0, and nt instruction bits:
///
///   | sc1 | sc0 | nt | Mtype | Behavior                              |
///   |-----|-----|----|-------|---------------------------------------|
///   |  0  |  0  |  0 | RW    | L1+L2 cached, write-back              |
///   |  0  |  1  |  0 | CC    | Coherent, L1 write-through, L2 WB     |
///   |  1  |  0  |  0 | UC    | Bypass all caches                     |
///   |  0  |  0  |  1 | RW+NT | Bypass L1, L2 only (non-temporal)     |
enum class Mtype : uint8_t {
  UC = 0, ///< Uncacheable - bypass all caches, go directly to memory.
  CC = 1, ///< Coherently Cacheable - MOESI coherence, visible to all agents.
  RW = 2, ///< Read-Write - cached in L1/L2, not coherent with CPU, write-back.
  WB = 3, ///< Write-Back - same behavior as RW.
  NT = 4, ///< Non-temporal - bypass L1, L2 only, evict soon.
};

/// @brief Combine instruction-level and PTE-level MTYPE, matching the GPU MMU.
///
/// @details Real AMDGPU hardware stores MTYPE bits in each PTE (bits 57:55 on
/// GFX9+). The GPU MMU combines the PTE MTYPE with the instruction's requested
/// MTYPE — the more restrictive (less cached) one wins. This prevents
/// instructions from requesting more caching than the page mapping allows.
///
/// Restrictiveness order: UC (0) < CC (1) < RW (2). NT is handled specially:
/// if the PTE forces UC or CC, the NT instruction is downgraded accordingly.
inline constexpr Mtype effective_mtype(Mtype instruction_mtype, Mtype pte_mtype) {
  if (pte_mtype == Mtype::RW || pte_mtype == Mtype::WB)
    return instruction_mtype;
  if (instruction_mtype == Mtype::NT)
    return (pte_mtype < Mtype::RW) ? pte_mtype : Mtype::NT;
  return (pte_mtype < instruction_mtype) ? pte_mtype : instruction_mtype;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_MTYPE_H_
