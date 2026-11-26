// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu::amdgpu {

/// @brief Read a legacy FLAT/GLOBAL/MUBUF immediate as a byte offset.
/// @details Accepts exactly two words for gfx942/gfx950/gfx1100 and checks
/// raw format and segment only, not opcode legality or operand semantics.
/// FLAT and MUBUF use unsigned12, while GLOBAL uses signed13 on all three
/// targets. FLAT ignores bit12, which is preserved by replacement. SCRATCH and the
/// reserved FLAT segment are unsupported. Callers must supply the correct
/// concrete source target and separately decode and qualify the operation.
[[nodiscard]] inline std::optional<int32_t>
read_legacy_memory_byte_offset(rj_code_target_id_t target, std::span<const uint32_t> source) {
  const bool cdna = target == ROCJITSU_CODE_TARGET_GFX942 || target == ROCJITSU_CODE_TARGET_GFX950;
  if ((!cdna && target != ROCJITSU_CODE_TARGET_GFX1100) || source.size() != 2)
    return std::nullopt;
  const uint32_t prefix = source[0] & 0xfc000000u;
  if (prefix == 0xe0000000u)
    return static_cast<int32_t>(source[0] & 0xfffu);
  if (prefix != 0xdc000000u)
    return std::nullopt;
  // Cache flags occupy bits 16/17 on CDNA and 14/15 on RDNA3, so the
  // segment must be interpreted using the selected concrete target.
  const uint32_t segment = (source[0] >> (cdna ? 14 : 16)) & 3u;
  if (segment != 0 && segment != 2)
    return std::nullopt;
  if (segment == 0)
    return static_cast<int32_t>(source[0] & 0xfffu);
  const int32_t encoded = static_cast<int32_t>(source[0] & 0x1fffu);
  return encoded < 0x1000 ? encoded : encoded - 0x2000;
}

/// @brief Replace a legacy memory byte offset, preserving every other source bit.
/// @details Uses read_legacy_memory_byte_offset's raw-format contract and
/// returns the complete two-word encoding. Signed13 accepts [-4096, 4095];
/// unsigned12 accepts [0, 4095]. Out-of-range values fail before narrowing,
/// without changing the source or returning a partial encoding. In particular,
/// FLAT's ignored bit12, segment, cache/return flags and all operands
/// remain unchanged. No-op and unaligned values are valid field operations;
/// fault delta, alignment and semantic admission belong to callers.
[[nodiscard]] inline std::optional<std::array<uint32_t, 2>>
replace_legacy_memory_byte_offset(rj_code_target_id_t target, std::span<const uint32_t> source,
                                  int64_t byte_offset) {
  if (!read_legacy_memory_byte_offset(target, source))
    return std::nullopt;
  const uint32_t segment_shift = target == ROCJITSU_CODE_TARGET_GFX1100 ? 16u : 14u;
  const bool signed_offset =
      (source[0] & 0xfc000000u) == 0xdc000000u && ((source[0] >> segment_shift) & 3u) == 2u;
  const int64_t minimum = signed_offset ? -4096 : 0;
  if (byte_offset < minimum || byte_offset > 4095)
    return std::nullopt;
  const uint32_t mask = signed_offset ? 0x1fffu : 0xfffu;
  const uint32_t encoded = static_cast<uint32_t>(byte_offset) & mask;
  return std::array<uint32_t, 2>{(source[0] & ~mask) | encoded, source[1]};
}

} // namespace rocjitsu::amdgpu
