// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace rocjitsu::consan::test {

// The original source VGPR is 0x89abcdef. A high-byte store extracts bits 23:16.
inline std::array<uint32_t, 4> d16_output(std::string_view target, uint32_t wave, bool halfword,
                                          bool signed_byte) {
  const uint32_t result = halfword ? 0x89abu : signed_byte ? 0xffabu : 0x00abu;
  const bool preserve =
      target == "gfx1100" || target == "gfx1201"; // SRAM ECC clears the other half on CDNA.
  const uint32_t address = 1012 + 4 * wave;
  return {(preserve ? 0xdead0000u : 0u) | result, (result << 16) | (preserve ? address : 0u),
          halfword ? 0x89aba5a5u : 0xaba5a5a5u, 0xa5a5a5a5u};
}

} // namespace rocjitsu::consan::test
