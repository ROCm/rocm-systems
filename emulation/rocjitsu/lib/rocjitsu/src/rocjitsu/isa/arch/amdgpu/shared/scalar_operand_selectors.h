// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_SELECTORS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_SELECTORS_H_

namespace rocjitsu::amdgpu {

/// @brief Return whether a scalar source selector names the low word of a
/// 64-bit register pair.
///
/// @details This is the register-backed subset of resolve_src_scalar64(). It
/// includes ordinary SGPR pairs, architecture-specific aliases in that range,
/// VCC, TTMP/TBA/TMA pairs, EXEC, and the GFX11+ FLAT_SCRATCH_BASE selector.
/// Single-word sources such as M0 and inline constants are deliberately
/// excluded. The amdisa generator validates these shared values against every
/// ISA's OPR_SSRC table.
[[nodiscard]] inline constexpr bool is_src_scalar_register_pair(int ev) {
  return (ev >= 0 && ev <= 106) || (ev >= 108 && ev <= 122) || ev == 126 || ev == 230;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_SELECTORS_H_
