// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu::amdgpu {

/// @brief Read the signed 24-bit byte offset of a modern VFLAT/VGLOBAL/VBUFFER encoding.
/// @details Accepts exactly three words for gfx1200/gfx1201/gfx1250/gfx1251 and
/// checks only the format prefix, not opcode legality or operand semantics.
/// VSCRATCH and other formats are unsupported. Callers must decode and qualify
/// the operation separately. The immediate is a byte offset even when CDNA5's
/// scale_offset modifier scales the vector offset.
[[nodiscard]] inline std::optional<int32_t>
read_vmem_signed24_offset(rj_code_target_id_t target, std::span<const uint32_t> source) {
  if ((target != ROCJITSU_CODE_TARGET_GFX1200 && target != ROCJITSU_CODE_TARGET_GFX1201 &&
       target != ROCJITSU_CODE_TARGET_GFX1250 && target != ROCJITSU_CODE_TARGET_GFX1251) ||
      source.size() != 3)
    return std::nullopt;
  const uint32_t prefix = source[0] & 0xff000000u;
  if (prefix != 0xec000000u && prefix != 0xee000000u && (source[0] & 0xfc000000u) != 0xc4000000u)
    return std::nullopt;
  const int32_t encoded = static_cast<int32_t>(source[2] >> 8);
  return encoded < 0x800000 ? encoded : encoded - 0x1000000;
}

/// @brief Replace a modern VFLAT/VGLOBAL/VBUFFER signed 24-bit byte offset.
/// @details Uses the same raw-format contract as read_vmem_signed24_offset.
/// Returns a complete three-word encoding preserving every bit outside the
/// immediate, including operands, padding, scope, TH and CDNA5 scale_offset.
/// Values outside [-0x800000, 0x7fffff] fail without modifying the source or
/// returning a partial encoding. No-op and unaligned values are valid field
/// operations; fault delta, alignment and semantic admission belong to callers.
[[nodiscard]] inline std::optional<std::array<uint32_t, 3>>
replace_vmem_signed24_offset(rj_code_target_id_t target, std::span<const uint32_t> source,
                             int64_t byte_offset) {
  if (byte_offset < -0x800000 || byte_offset > 0x7fffff ||
      !read_vmem_signed24_offset(target, source))
    return std::nullopt;
  const uint32_t encoded = static_cast<uint32_t>(byte_offset) & 0xffffffu;
  return std::array<uint32_t, 3>{source[0], source[1], (source[2] & 0xffu) | (encoded << 8)};
}

} // namespace rocjitsu::amdgpu
