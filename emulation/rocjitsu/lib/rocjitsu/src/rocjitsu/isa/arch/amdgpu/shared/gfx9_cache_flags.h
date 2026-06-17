// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx9_cache_flags.h
/// @brief GFX9 (CDNA1/2) memory coherency flag → Mtype mapping.
///
/// CDNA1 and CDNA2 use GLC as the sole coherency control for all memory
/// instructions.  SLC is a cache-line eviction hint but does not change
/// the Mtype from the simulator's perspective.
///
/// | GLC | Mtype | Behavior                                    |
/// |-----|-------|---------------------------------------------|
/// |  0  | RW    | L1+L2 cached, write-back, not coherent      |
/// |  1  | CC    | Coherent — L1 write-through, L2 write-back  |

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_GFX9_CACHE_FLAGS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_GFX9_CACHE_FLAGS_H_

#include "rocjitsu/vm/amdgpu/mtype.h"

namespace rocjitsu {
namespace amdgpu {

/// @brief Derive Mtype from GFX9 (CDNA1/2) coherency flags.
/// @param glc Global Level Coherent bit.
/// @returns Mtype::CC if glc is set, Mtype::RW otherwise.
[[nodiscard]] inline constexpr Mtype mtype_from_flags_gfx9(bool glc) {
  return glc ? Mtype::CC : Mtype::RW;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_GFX9_CACHE_FLAGS_H_
