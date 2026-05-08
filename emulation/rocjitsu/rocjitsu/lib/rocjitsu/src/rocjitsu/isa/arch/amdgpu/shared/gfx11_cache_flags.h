// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx11_cache_flags.h
/// @brief GFX11 (RDNA3/3.5) memory coherency flag → Mtype mapping.
///
/// RDNA3 and RDNA3.5 use GLC and SLC as 2-bit scope fields (semantically
/// equivalent to SC0/SC1 on CDNA3/4).  DLC serves as a non-temporal hint
/// on some instruction formats.  The generated machine_insts.h retains
/// the GLC/DLC/SLC field names from the XML spec; this function accepts
/// those names and interprets them with GFX11 semantics.
///
/// Scope encoding: scope = (SLC << 1) | GLC
/// | SLC | GLC | DLC | Mtype | Scope    |
/// |-----|-----|-----|-------|----------|
/// |  0  |  0  |  0  | RW    | CU       |
/// |  0  |  1  |  0  | CC    | DEVICE   |
/// |  1  |  0  |  0  | UC    | SYSTEM   |
/// |  1  |  1  |  0  | UC    | SYSTEM   |
/// |  x  |  x  |  1  | NT    | (any)    |

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_GFX11_CACHE_FLAGS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_GFX11_CACHE_FLAGS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/gfx940_cache_flags.h"

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief Derive Mtype from GFX11 (RDNA3/3.5) coherency flags.
///
/// On GFX11, the XML-named fields map to scope bits as follows:
///   GLC → scope bit 0 (equivalent to SC0 on CDNA3/4)
///   SLC → scope bit 1 (equivalent to SC1 on CDNA3/4)
///   DLC → non-temporal hint
///
/// @param glc Scope bit 0 (field named "glc" in generated machine_insts.h).
/// @param dlc Non-temporal hint (field named "dlc" in generated machine_insts.h).
/// @param slc Scope bit 1 (field named "slc" in generated machine_insts.h).
[[nodiscard]] inline constexpr Mtype mtype_from_flags_gfx11(bool glc, bool dlc, bool slc) {
  return mtype_from_scope_nt(static_cast<uint8_t>((slc << 1) | glc), dlc);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_GFX11_CACHE_FLAGS_H_
