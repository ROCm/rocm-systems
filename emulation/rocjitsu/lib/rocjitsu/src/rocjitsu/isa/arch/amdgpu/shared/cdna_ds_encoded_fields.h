// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu::amdgpu {

/// @brief Replace the encoded address VGPR of a gfx942/gfx950 DS write.
/// @details Accepts exactly two words encoding ds_write_b32 or ds_write_b128.
/// The result preserves every bit except the eight-bit address field, including
/// GDS and ACC; preserving a bit does not establish that a particular modifier
/// or source instruction is legal on hardware. The register number is encoded,
/// and the caller is responsible for admission, register state and allocation.
[[nodiscard]] inline std::optional<std::array<uint32_t, 2>>
replace_cdna_ds_write_address(rj_code_target_id_t target, std::span<const uint32_t> source,
                              uint32_t encoded_vgpr) {
  // Bits 16 and 25 are GDS and ACC, respectively; neither is part of the
  // opcode or encoding identifier and both must survive a field-only update.
  constexpr uint32_t kFormatOpcodeMask = 0xfdfe0000u;
  constexpr uint32_t kDsWriteB32 = 0xd81a0000u;
  constexpr uint32_t kDsWriteB128 = 0xd9be0000u;
  if ((target != ROCJITSU_CODE_TARGET_GFX942 && target != ROCJITSU_CODE_TARGET_GFX950) ||
      source.size() != 2 || encoded_vgpr > 255)
    return std::nullopt;
  const uint32_t format_opcode = source[0] & kFormatOpcodeMask;
  if (format_opcode != kDsWriteB32 && format_opcode != kDsWriteB128)
    return std::nullopt;
  return std::array<uint32_t, 2>{source[0], (source[1] & ~0xffu) | encoded_vgpr};
}

} // namespace rocjitsu::amdgpu
