// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx940_cache_flags.h
/// @brief GFX940 (CDNA3/4) vector memory coherency flag → Mtype mapping.
///
/// CDNA3 and CDNA4 vector memory instructions (FLAT, GLOBAL, SCRATCH, MUBUF,
/// MTBUF) use SC0/SC1 as a 2-bit scope field and NT as a non-temporal hint.
/// SMEM on CDNA3/4 retains the GFX9 GLC-only model (use gfx9_cache_flags.h).
///
/// | SC1 | SC0 | NT | Mtype | Scope    |
/// |-----|-----|----|-------|----------|
/// |  0  |  0  |  0 | RW    | CU       |
/// |  0  |  1  |  0 | CC    | DEVICE   |
/// |  1  |  0  |  0 | UC    | SYSTEM   |
/// |  1  |  1  |  0 | UC    | SYSTEM   |
/// |  x  |  x  |  1 | NT    | (any)    |

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_GFX940_CACHE_FLAGS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_GFX940_CACHE_FLAGS_H_

#include "rocjitsu/vm/amdgpu/mtype.h"

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief Derive Mtype from a scope value and non-temporal hint.
///
/// Shared helper used by GFX940, GFX11, and GFX12 families.  The scope
/// encoding is: 0 = WAVE/CU, 1 = DEVICE, 2+ = SYSTEM.
/// @param scope 2-bit scope value (SC1:SC0 or SCOPE field).
/// @param nt    True if the non-temporal hint is set.
[[nodiscard]] inline constexpr Mtype mtype_from_scope_nt(uint8_t scope, bool nt) {
  if (nt)
    return Mtype::NT;
  if (scope >= 2)
    return Mtype::UC;
  if (scope >= 1)
    return Mtype::CC;
  return Mtype::RW;
}

/// @brief Derive Mtype from GFX940 (CDNA3/4) vector memory coherency flags.
/// @param sc0 Scope bit 0.
/// @param sc1 Scope bit 1.
/// @param nt  Non-temporal hint.
[[nodiscard]] inline constexpr Mtype mtype_from_flags_gfx940(bool sc0, bool sc1, bool nt) {
  return mtype_from_scope_nt(static_cast<uint8_t>((sc1 << 1) | sc0), nt);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_GFX940_CACHE_FLAGS_H_
