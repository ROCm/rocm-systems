// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx11_cache_flags.h
/// @brief GFX11 (RDNA3/3.5) memory coherency flag → Mtype mapping.
///
/// Buffer, scalar and FLAT transactions use apply_gfx11_cache_policy. Image
/// transactions still use the legacy approximation in mtype_from_flags_gfx11.

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_GFX11_CACHE_FLAGS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_GFX11_CACHE_FLAGS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/gfx940_cache_flags.h"

#include "rocjitsu/vm/amdgpu/atomic_op.h"

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// Legacy MIMG approximation. Its SLC-to-scope mapping describes the
/// simulator's image cache model, not a hardware system-scope operation.
[[nodiscard]] inline constexpr Mtype mtype_from_flags_gfx11(bool glc, bool dlc, bool slc) {
  return mtype_from_scope_nt(static_cast<uint8_t>((slc << 1) | glc), dlc);
}

/// RDNA3/3.5 cache policy: load GLC bypasses the client cache and still uses L2.
/// Stores and atomics have device scope regardless of GLC. SLC and DLC are
/// replacement hints for graphics caches and MALL; they do not select system scope.
template <typename MemState> inline void apply_gfx11_cache_policy(MemState &state, bool glc) {
  const bool load = state.is_load && state.atomic_op == AtomicOp::NONE;
  state.request_force_l1_bypass = load && glc;
  state.mtype = load ? Mtype::RW : Mtype::CC;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_GFX11_CACHE_FLAGS_H_
