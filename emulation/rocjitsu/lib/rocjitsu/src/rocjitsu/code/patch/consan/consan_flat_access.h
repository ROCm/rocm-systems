// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_flat_access.h
/// @brief Architecture-neutral semantic classification for ConSan FLAT accesses.

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace rocjitsu {

enum class ConSanFlatSubwordPlacement : uint8_t {
  Low16,
  High16,
};

struct ConSanFlatLoadSubwordSemantics {
  uint32_t memory_width_bits = 0;
  ConSanFlatSubwordPlacement placement = ConSanFlatSubwordPlacement::Low16;
};

/// Return the memory width and destination-half placement for FLAT loads that
/// write an 8- or 16-bit result into one half of a VGPR.
[[nodiscard]] inline constexpr std::optional<ConSanFlatLoadSubwordSemantics>
consan_flat_load_subword_semantics(std::string_view mnemonic) {
  if (mnemonic == "flat_load_d16_u8" || mnemonic == "flat_load_d16_i8" ||
      mnemonic == "flat_load_ubyte_d16" || mnemonic == "flat_load_sbyte_d16")
    return ConSanFlatLoadSubwordSemantics{8u, ConSanFlatSubwordPlacement::Low16};

  if (mnemonic == "flat_load_d16_b16" || mnemonic == "flat_load_short_d16")
    return ConSanFlatLoadSubwordSemantics{16u, ConSanFlatSubwordPlacement::Low16};

  if (mnemonic == "flat_load_d16_hi_u8" || mnemonic == "flat_load_d16_hi_i8" ||
      mnemonic == "flat_load_ubyte_d16_hi" || mnemonic == "flat_load_sbyte_d16_hi")
    return ConSanFlatLoadSubwordSemantics{8u, ConSanFlatSubwordPlacement::High16};

  if (mnemonic == "flat_load_d16_hi_b16" || mnemonic == "flat_load_short_d16_hi")
    return ConSanFlatLoadSubwordSemantics{16u, ConSanFlatSubwordPlacement::High16};

  return std::nullopt;
}

} // namespace rocjitsu
