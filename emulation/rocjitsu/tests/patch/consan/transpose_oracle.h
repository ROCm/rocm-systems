// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace rocjitsu::consan::test {

inline uint32_t transpose_words(uint32_t bits, bool cdna4) {
  return bits == 6 ? 3 : bits == 16 && !cdna4 ? 4 : 2;
}

// Invert the tile layout: each destination element selects one source row,
// and the destination lane selects a column in that row. Read the initialized
// LDS bits directly rather than calling the simulator's response permutation.
inline std::array<uint32_t, 4> transpose_output(uint32_t lane, uint32_t bits, bool cdna4,
                                                std::string_view scenario) {
  std::array<uint32_t, 4> result{16 * lane, 0x12345678u, 0x9abcdef0u, 0xfedcba98u};
  if (scenario == "overlap" || scenario == "adjacent") {
    result[0] = lane;
    return result;
  }
  if (cdna4 && scenario == "empty")
    return result;
  const uint32_t words = transpose_words(bits, cdna4);
  const uint32_t columns = words * 32 / bits;
  const uint32_t column = lane % columns;
  constexpr std::array<uint32_t, 4> seeds{0x10305070, 0x21416181, 0x32527292, 0x436383a3};
  for (uint32_t word = 0; word < words; ++word) {
    result[word] = 0;
    for (uint32_t bit = 0; bit < 32; ++bit) {
      const uint32_t element = (32 * word + bit) / bits;
      uint32_t source;
      if (bits == 4)
        source =
            cdna4 ? (lane / 16) * 16 + element : (lane / 16) * 8 + (element / 8) * 16 + element % 8;
      else if (bits == 6)
        source =
            cdna4 ? (lane / 16) * 16 + element : (lane / 16) * 4 + (element / 4) * 8 + element % 4;
      else if (bits == 8)
        source =
            (lane / 16) * 16 + (cdna4 ? (lane % 16) / 8 + 2 * element
                                      : 4 * ((lane % 16) / 8) + 8 * (element / 4) + element % 4);
      else
        source =
            cdna4 ? (lane / 16) * 16 + (lane % 16) / 4 + 4 * element : (lane / 8) * 8 + element;
      if (scenario == "short" && source >= 4)
        source = 0;
      const uint32_t source_bit = column * bits + (32 * word + bit) % bits;
      const uint32_t data = seeds[source_bit / 32] + 0x01010101u * source;
      result[word] |= ((data >> (source_bit % 32)) & 1u) << bit;
    }
  }
  return result;
}

} // namespace rocjitsu::consan::test
