// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx12_cache_flags.h
/// @brief GFX12 (RDNA4) memory coherency flag → Mtype mapping.
///
/// RDNA4 uses a packed SCOPE field (2 bits) and a TH (temporal hint)
/// field (2–3 bits depending on instruction format).  There are no
/// GLC/DLC/SLC fields.
///
/// SCOPE encoding:
///   0 = CU, 1 = SE (Shader Engine), 2 = DEVICE, 3 = SYSTEM
///
/// TH encoding (simplified for simulator):
///   0 = regular, 1 = non-temporal (NT), 2+ = format-specific hints
///
/// | SCOPE | TH  | Mtype | Behavior                             |
/// |-------|-----|-------|--------------------------------------|
/// |   0   |  0  | RW    | L1+L2 cached, CU scope               |
/// |   1   |  0  | RW    | L1+L2 cached, SE scope               |
/// |   2   |  0  | CC    | Coherent at device scope              |
/// |   3   |  0  | UC    | System scope, bypass caches           |
/// |   x   |  1  | NT    | Non-temporal hint                     |

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_GFX12_CACHE_FLAGS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_GFX12_CACHE_FLAGS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/gfx940_cache_flags.h"

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief Non-temporal TH value for GFX12.
inline constexpr uint8_t GFX12_TH_NT = 1;

/// @brief Derive Mtype from GFX12 (RDNA4) coherency flags.
/// @param scope_val 2-bit SCOPE field value.
/// @param th        Temporal hint field value (1 = non-temporal).
[[nodiscard]] inline constexpr Mtype mtype_from_flags_gfx12(uint8_t scope_val, uint8_t th) {
  return mtype_from_scope_nt(scope_val, th == GFX12_TH_NT);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_GFX12_CACHE_FLAGS_H_
