// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu::cdna5 {

/// @brief Replace the encoded address VGPR of gfx1250 ds_store_b32 or ds_load_b32.
/// @details The input must contain exactly two 32-bit words. All bits except the
/// eight-bit address selector are preserved, including padding and unused fields.
/// The selector is encoded, not a physical bank-relative register; the caller
/// must check bank state and register allocation. Two-address forms are unsupported.
[[nodiscard]] inline std::optional<std::array<uint32_t, 2>>
replace_ds_b32_address(rj_code_target_id_t target, std::span<const uint32_t> source,
                       uint32_t encoded_vgpr) {
  constexpr uint32_t kOpcodeMask = 0xfffc0000u;
  constexpr uint32_t kDsStoreB32 = 0xd8340000u;
  constexpr uint32_t kDsLoadB32 = 0xd8d80000u;
  if (target != ROCJITSU_CODE_TARGET_GFX1250 || source.size() != 2 || encoded_vgpr > 255)
    return std::nullopt;
  const uint32_t opcode = source[0] & kOpcodeMask;
  if (opcode != kDsStoreB32 && opcode != kDsLoadB32)
    return std::nullopt;
  return std::array<uint32_t, 2>{source[0], (source[1] & ~0xffu) | encoded_vgpr};
}

} // namespace rocjitsu::cdna5
