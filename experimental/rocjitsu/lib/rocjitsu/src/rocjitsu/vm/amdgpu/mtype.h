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

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_MTYPE_H_
