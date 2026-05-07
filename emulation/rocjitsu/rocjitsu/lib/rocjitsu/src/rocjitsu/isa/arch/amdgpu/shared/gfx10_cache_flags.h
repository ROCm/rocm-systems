// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx10_cache_flags.h
/// @brief GFX10 (RDNA1/2) memory coherency flag → Mtype mapping.
///
/// RDNA1 and RDNA2 use three independent coherency bits: GLC, DLC, SLC.
/// On GFX10, GLC controls L1 write-through/coherency, DLC controls L2
/// data-level coherency (cross-CU visibility), and SLC controls cache
/// line eviction policy.
///
/// Simplified mapping for the simulator:
/// | GLC | DLC | SLC | Mtype | Behavior                            |
/// |-----|-----|-----|-------|-------------------------------------|
/// |  0  |  0  |  0 | RW    | L1+L2 cached, write-back            |
/// |  1  |  0  |  0 | CC    | L1 write-through, coherent at L2    |
/// |  0  |  1  |  0 | CC    | L2 data-level coherent              |
/// |  1  |  1  |  0 | UC    | Bypass L1, L2 coherent              |
/// |  x  |  x  |  1 | NT    | Non-temporal (evict soon)           |

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_GFX10_CACHE_FLAGS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_GFX10_CACHE_FLAGS_H_

#include "rocjitsu/vm/amdgpu/mtype.h"

namespace rocjitsu {
namespace amdgpu {

/// @brief Derive Mtype from GFX10 (RDNA1/2) coherency flags.
/// @param glc Global Level Coherent.
/// @param dlc Data Level Coherent.
/// @param slc System Level Coherent (used as non-temporal hint).
[[nodiscard]] inline constexpr Mtype mtype_from_flags_gfx10(bool glc, bool dlc, bool slc) {
  if (slc)
    return Mtype::NT;
  if (glc && dlc)
    return Mtype::UC;
  if (glc || dlc)
    return Mtype::CC;
  return Mtype::RW;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_GFX10_CACHE_FLAGS_H_
