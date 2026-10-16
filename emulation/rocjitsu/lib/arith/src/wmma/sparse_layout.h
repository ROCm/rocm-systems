// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Raw gfx12 sparse floating-point operand layouts, shared by exact and loose paths.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Identify one packed field in a VGPR array.
struct SparseField {
  int reg, lane, shift;
  // Return the selected field and any higher bits; the caller narrows its type.
  uint32_t read(const uint32_t *const *v) const { return v[reg][lane] >> shift; }
};

// Locate compressed A element ck in [0,16), before source modifiers.
template <int Bits, int WaveSize> constexpr SparseField swmmac_a_field(int row, int ck) {
  static_assert((Bits == 8 || Bits == 16) && (WaveSize == 32 || WaveSize == 64));
  if constexpr (Bits == 16) {
    return {WaveSize == 64 ? (ck % 4) / 2 : 2 * (ck / 8) + ((ck / 2) & 1),
            row + 16 * (WaveSize == 64 ? ck / 4 : (ck / 4) & 1), 16 * (ck % 2)};
  } else {
    return {(ck % (WaveSize == 64 ? 4 : 8)) / 4,
            row + 16 * (ck / 8) + (WaveSize == 64 ? 32 * ((ck / 4) % 2) : 0), 8 * (ck % 4)};
  }
}

// Decode the dense B coordinate selected by A's compressed slot and metadata key.
// Metadata pair ordering is a caller precondition, not validated here.
template <int Bits, int WaveSize>
inline int swmmac_selected_k(const uint32_t *const *index, int row, int ck, unsigned key) {
  auto field = swmmac_a_field<Bits, WaveSize>(row, ck);
  int slot = WaveSize == 64 ? ck % 4 : Bits == 16 ? (ck % 4) + 4 * (ck / 8) : ck % 8;
  return 4 * (ck / 2) + ((index[0][field.lane] >> (key + 2 * slot)) & 3);
}

// Read one raw dense B value at k in [0,32), before source modifiers.
template <int Bits, int WaveSize>
inline uint32_t swmmac_b_bits(const uint32_t *const *b, int col, int k) {
  if constexpr (Bits == 16) {
    int lane = col + 16 * (WaveSize == 64 ? k / 8 : (k / 8) & 1);
    int slot = WaveSize == 64 ? k % 8 : 8 * (k / 16) + k % 8;
    return (b[slot / 2][lane] >> (16 * (slot % 2))) & 0xffff;
  } else {
    int lane = col + 16 * (k / 16) + (WaveSize == 64 ? 32 * ((k / 8) % 2) : 0);
    return (b[(k % (WaveSize == 64 ? 8 : 16)) / 4][lane] >> (8 * (k % 4))) & 0xff;
  }
}

} // namespace rocjitsu::arith
