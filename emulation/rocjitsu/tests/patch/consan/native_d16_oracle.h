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

inline std::array<uint32_t, 4> uniform_subword_output(std::string_view mode,
                                                      std::string_view target) {
  const bool high = mode.find("_high_") != std::string_view::npos;
  const bool varying = mode.ends_with("_varying");
  const bool load_high = high == varying;
  const bool preserve = target == "gfx1100" || target == "gfx1201";
  const uint32_t untouched = preserve ? (load_high ? 0x596cu : 0x31420000u) : 0;
  const uint32_t mask = mode.starts_with("uniform_b8_") ? 0xffu : 0xffffu;
  std::array<uint32_t, 4> expected = {};
  for (uint32_t lane = 0; lane < 2; ++lane) {
    const uint32_t source = untouched | ((0x7280u + lane) << (load_high ? 16 : 0));
    expected[2 * lane] = source;
    const uint32_t selected = (source >> (high ? 16 : 0)) & mask;
    expected[2 * lane + 1] = varying ? selected & ~1u : selected;
  }
  return expected;
}

} // namespace rocjitsu::consan::test
