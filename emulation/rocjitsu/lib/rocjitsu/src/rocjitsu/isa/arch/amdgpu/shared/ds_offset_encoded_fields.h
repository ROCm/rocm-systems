// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu::amdgpu {

/// @brief Read the raw eight-bit offset0 field of a concrete DS/VDS encoding.
/// @details Accepts exactly two words for gfx942/gfx950/gfx1100 ENC_DS or
/// gfx1200/gfx1201/gfx1250/gfx1251 ENC_VDS. Only the format prefix is checked:
/// this does not validate an opcode, prove that it uses offset0, or establish
/// instruction legality. The caller must decode and qualify the operation,
/// including its offset units and whether offset1 forms a combined offset.
[[nodiscard]] inline std::optional<uint32_t>
read_ds_offset0_field(rj_code_target_id_t target, std::span<const uint32_t> source) {
  constexpr uint32_t kFormatMask = 0xfc000000u;
  constexpr uint32_t kDsFormat = 0xd8000000u;
  if ((target != ROCJITSU_CODE_TARGET_GFX942 && target != ROCJITSU_CODE_TARGET_GFX950 &&
       target != ROCJITSU_CODE_TARGET_GFX1100 && target != ROCJITSU_CODE_TARGET_GFX1200 &&
       target != ROCJITSU_CODE_TARGET_GFX1201 && target != ROCJITSU_CODE_TARGET_GFX1250 &&
       target != ROCJITSU_CODE_TARGET_GFX1251) ||
      source.size() != 2 || (source[0] & kFormatMask) != kDsFormat)
    return std::nullopt;
  return source[0] & 0xffu;
}

/// @brief Replace the raw offset0 field while preserving every other source bit.
/// @details Uses read_ds_offset0_field's raw-format contract and accepts values
/// from zero through 255 without truncation. The source remains unchanged on
/// success and failure. In particular, offset1, opcode, register operands,
/// modifiers and reserved bits are preserved. This is not a combined-offset
/// operation and does not impose an operation's alignment or no-op policy.
[[nodiscard]] inline std::optional<std::array<uint32_t, 2>>
replace_ds_offset0_field(rj_code_target_id_t target, std::span<const uint32_t> source,
                         uint32_t raw_offset0) {
  if (raw_offset0 > 255 || !read_ds_offset0_field(target, source))
    return std::nullopt;
  return std::array<uint32_t, 2>{(source[0] & ~0xffu) | raw_offset0, source[1]};
}

} // namespace rocjitsu::amdgpu
