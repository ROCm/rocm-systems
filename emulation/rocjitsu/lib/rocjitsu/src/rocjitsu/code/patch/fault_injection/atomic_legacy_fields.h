// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/legacy_memory_displacement_encoded_fields.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace rocjitsu::fault_mutation_detail {

inline bool legacy_atomic_target(rj_code_target_id_t target) {
  return target == ROCJITSU_CODE_TARGET_GFX950 || target == ROCJITSU_CODE_TARGET_GFX942 ||
         target == ROCJITSU_CODE_TARGET_GFX1100;
}

// The caller supplies the mnemonic decoded from these complete source words.
// B2 owns raw field access; FI additionally binds each mnemonic to its segment.
inline std::optional<int32_t> read_legacy_atomic_byte_offset(rj_code_target_id_t target,
                                                             std::string_view mnemonic,
                                                             std::span<const uint32_t> source) {
  if (!legacy_atomic_target(target) || source.size() != 2)
    return std::nullopt;
  const uint32_t prefix = source[0] & 0xfc000000u;
  if (prefix == 0xe0000000u) {
    if (!mnemonic.starts_with("buffer_atomic"))
      return std::nullopt;
  } else if (prefix == 0xdc000000u) {
    const uint32_t shift = target == ROCJITSU_CODE_TARGET_GFX1100 ? 16u : 14u;
    const uint32_t segment = (source[0] >> shift) & 3u;
    if (!((segment == 0 && mnemonic.starts_with("flat_atomic")) ||
          (segment == 2 && mnemonic.starts_with("global_atomic"))))
      return std::nullopt;
  } else {
    return std::nullopt;
  }
  return amdgpu::read_legacy_memory_byte_offset(target, source);
}

// Request delta and alignment policy remain in the common atomic recipe. The
// final byte offset is checked by B2 before any narrowing or source modification.
inline std::optional<std::array<uint32_t, 2>>
replace_legacy_atomic_byte_offset(rj_code_target_id_t target, std::string_view mnemonic,
                                  std::span<const uint32_t> source, int64_t final_offset) {
  if (!read_legacy_atomic_byte_offset(target, mnemonic, source))
    return std::nullopt;
  return amdgpu::replace_legacy_memory_byte_offset(target, source, final_offset);
}

} // namespace rocjitsu::fault_mutation_detail
