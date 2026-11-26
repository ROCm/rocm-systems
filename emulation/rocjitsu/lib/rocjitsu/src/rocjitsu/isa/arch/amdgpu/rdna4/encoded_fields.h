// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu::rdna4 {

/// @brief Replace the encoded address VGPR of a gfx1200 or gfx1201 ds_store_b32.
/// @details The input must contain exactly two 32-bit words. The returned
/// encoding preserves every bit except the eight-bit address field. The
/// register number is encoded, not a physical bank-relative register; the
/// caller is responsible for register state and allocation checks.
[[nodiscard]] inline std::optional<std::array<uint32_t, 2>>
replace_ds_store_b32_address(rj_code_target_id_t target, std::span<const uint32_t> source,
                             uint32_t encoded_vgpr) {
  constexpr uint32_t kOpcodeMask = 0xfffc0000u;
  constexpr uint32_t kDsStoreB32 = 0xd8340000u;
  if ((target != ROCJITSU_CODE_TARGET_GFX1200 && target != ROCJITSU_CODE_TARGET_GFX1201) ||
      source.size() != 2 || encoded_vgpr > 255 || (source[0] & kOpcodeMask) != kDsStoreB32) {
    return std::nullopt;
  }
  return std::array<uint32_t, 2>{source[0], (source[1] & ~0xffu) | encoded_vgpr};
}

/// @brief Replace the encoded address VGPR of a gfx1200 or gfx1201 ds_load_b32.
/// @details Requires exactly two words and preserves all bits outside the
/// eight-bit address field. The caller is responsible for register state and
/// allocation checks; two-address forms are not supported.
[[nodiscard]] inline std::optional<std::array<uint32_t, 2>>
replace_ds_load_b32_address(rj_code_target_id_t target, std::span<const uint32_t> source,
                            uint32_t encoded_vgpr) {
  constexpr uint32_t kOpcodeMask = 0xfffc0000u;
  constexpr uint32_t kDsLoadB32 = 0xd8d80000u;
  if ((target != ROCJITSU_CODE_TARGET_GFX1200 && target != ROCJITSU_CODE_TARGET_GFX1201) ||
      source.size() != 2 || encoded_vgpr > 255 || (source[0] & kOpcodeMask) != kDsLoadB32) {
    return std::nullopt;
  }
  return std::array<uint32_t, 2>{source[0], (source[1] & ~0xffu) | encoded_vgpr};
}

} // namespace rocjitsu::rdna4
