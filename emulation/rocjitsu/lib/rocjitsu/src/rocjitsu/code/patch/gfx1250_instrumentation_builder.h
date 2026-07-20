// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_instrumentation_builder.h
/// @brief gfx1250 instruction encoders used by DBI instrumentation.

#pragma once

#include "rocjitsu/code/patch/instruction_builder.h"

namespace rocjitsu {

/// @brief Encode gfx1250 `s_set_vgpr_msb simm16`.
///
/// Instrumentation uses this to select the low 256-register window while it
/// executes its own vector instructions, then restores the guest's operand
/// bank selection before executing displaced code.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_gfx1250_s_set_vgpr_msb(uint16_t mode, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250)
    return std::nullopt;
  return 0xBF860000u | mode;
}

/// @brief Encode gfx1250 `s_call_i64 s[sdst:sdst+1], simm16`.
///
/// The immediate is a signed dword offset from the instruction following the
/// call. The instruction writes that following PC to the destination pair.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_gfx1250_s_call_i64(uint16_t sdst, int16_t simm16, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || sdst > 104 || sdst % 2u != 0u)
    return std::nullopt;
  return pack_sopk(gfx1250::kSCallI64Sopk, sdst, static_cast<uint16_t>(simm16));
}

/// @brief Encode gfx1250 `v_cmp_ne_u16_e32 vcc_lo, src0, vsrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_gfx1250_v_cmp_ne_u16_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return gfx1250::build_vopc(gfx1250::kVCmpNeU16Vopc,
                             {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1)})[0];
}

} // namespace rocjitsu
