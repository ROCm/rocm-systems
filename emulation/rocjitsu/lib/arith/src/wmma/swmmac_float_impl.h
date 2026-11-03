// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local sparse operand preparation and result layout helpers.
#pragma once

#include "common/float_formats.h"
#include "common/internal.h"
#include "common/isa.h"
#include "rj_arith/rj_arith.h"
#include "wmma/sparse_layout.h"
#include "wmma/swmmac_float.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Snapshot factors, sparse indices, and tied D into logical matrices, applying
// A/C modifiers. D uses 32-lane sections even for Wave64; metadata pairs must
// contain two increasing positions in [0,3]. B modifiers remain unapplied.
template <bool Bf16, int WaveSize, bool Packed>
inline void swmmac16_prepare(SwmmacFloatInputs &input, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *index) {
  // Widen one raw FP16/BF16 value without applying operand modifiers.
  auto decode = [](uint16_t x) { return Bf16 ? bf16_to_float(x) : f16_to_float(x); };
  int key = rocjitsu::arith::SwmmacIndex<16, 32, WaveSize>::shift(mode);

  // Snapshot compressed A and map each retained factor to its dense K position.
  for (int row = 0; row < 16; ++row) {
    for (int ck = 0; ck < 16; ++ck) {
      int slot = ck % 2;
      uint16_t value = uint16_t(swmmac_a_field<16, WaveSize>(row, ck).read(a));
      if (mode & (slot ? RJ_ARITH_WMMA_NEG_HI_A : RJ_ARITH_WMMA_NEG_LO_A)) {
        value ^= 0x8000;
      }
      input.a[row][ck] = decode(value);
      input.selected[row][ck] = uint8_t(swmmac_selected_k<16, WaveSize>(index, row, ck, key));
    }
  }

  // Decode dense B separately: its sign modifier depends on compressed position.
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      input.b[k][col] = decode(uint16_t(swmmac_b_bits<16, WaveSize>(b, col, k)));
    }
  }

  // D is tied to C; read and modify the initial accumulator before any stores.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int reg = row % 8, lane = col + 16 * (row / 8);
      if constexpr (Packed) {
        input.acc[row][col] = decode(uint16_t(d[reg / 2][lane] >> (16 * (reg % 2))));
      } else {
        input.acc[row][col] = as_float(d[reg][lane]);
      }
      uint32_t acc = as_bits(input.acc[row][col]);
      if (mode & RJ_ARITH_WMMA_ABS_C) {
        acc &= 0x7fffffffU;
      }
      if (mode & RJ_ARITH_WMMA_NEG_C) {
        acc ^= 0x80000000U;
      }
      input.acc[row][col] = as_float(acc);
    }
  }
}

// Fill result with a complete tile in 32-lane destination sections. Packed
// BF16 truncates; packed FP16 rounds, saturating finite overflow if requested.
template <bool Bf16, bool Packed>
inline void swmmac_float_pack(uint32_t (&result)[Packed ? 4 : 8][32], const float (&acc)[16][16],
                              bool saturate) {
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int reg = row % 8, lane = col + 16 * (row / 8);
      if constexpr (Packed) {
        uint16_t value =
            Bf16 ? uint16_t(as_bits(acc[row][col]) >> 16) : float_to_f16(acc[row][col], saturate);
        if (reg % 2) {
          result[reg / 2][lane] |= uint32_t(value) << 16;
        } else {
          result[reg / 2][lane] = value;
        }
      } else {
        result[reg][lane] = as_bits(acc[row][col]);
      }
    }
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
