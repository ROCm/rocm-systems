// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu::amdgpu {

/// @brief Replace the raw ADDR field of a concrete target's DS/VDS encoding.
/// @details Accepts exactly two words for gfx942/gfx950 ENC_DS or
/// gfx1200/gfx1201/gfx1250 ENC_VDS, and an eight-bit encoded VGPR selector. Preserves
/// all other bits, including opcode, offsets, operands, modifiers and reserved
/// bits. Only the format prefix is checked: this does not validate an opcode,
/// prove that it has an address operand, or establish instruction legality.
/// Callers must decode and qualify the specific operation and check register
/// bank, allocation, and runtime state separately.
[[nodiscard]] inline std::optional<std::array<uint32_t, 2>>
replace_ds_address_field(rj_code_target_id_t target, std::span<const uint32_t> source,
                         uint32_t encoded_vgpr) {
  constexpr uint32_t kFormatMask = 0xfc000000u;
  constexpr uint32_t kDsFormat = 0xd8000000u;
  if ((target != ROCJITSU_CODE_TARGET_GFX942 && target != ROCJITSU_CODE_TARGET_GFX950 &&
       target != ROCJITSU_CODE_TARGET_GFX1200 && target != ROCJITSU_CODE_TARGET_GFX1201 &&
       target != ROCJITSU_CODE_TARGET_GFX1250) ||
      source.size() != 2 || encoded_vgpr > 255 || (source[0] & kFormatMask) != kDsFormat) {
    return std::nullopt;
  }
  return std::array<uint32_t, 2>{source[0], (source[1] & ~0xffu) | encoded_vgpr};
}

} // namespace rocjitsu::amdgpu
