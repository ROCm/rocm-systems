// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 floating SWMMAC layouts, tied accumulators and final nearest-even packing.
#pragma once

#include "mfma/sparse_float.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Locate a source in the inherited split-halfwave layout (the block size is in elements).
template <int Bits, int Block> int swmmac1250_input_index(int row, int k) {
  constexpr int per_word = 32 / Bits;
  int element = (k / (2 * Block)) * Block + k % Block;
  int lane = row + 16 * (k / Block % 2);
  return (element / per_word * 32 + lane) * per_word + element % per_word;
}

// Narrow once after accumulation; FP16_OVFL saturates infinities as well as finite overflow.
template <bool Bf16> uint16_t swmmac1250_narrow(float value, bool saturate) {
  uint16_t result;
  if constexpr (Bf16) {
    // Retain util's discarded-payload marker when narrowing an inherited NaN.
    uint32_t bits = as_bits(value);
    if ((bits & 0x7f800000U) != 0x7f800000U) {
      bits += 0x7fffU + ((bits >> 16) & 1);
    } else if (bits & 0xffffU) {
      bits |= 0x10000U;
    }
    result = uint16_t(bits >> 16);
  } else {
    result = float_to_f16(value, saturate);
  }
  constexpr uint16_t infinity = Bf16 ? 0x7f80 : 0x7c00;
  if (saturate && (result & 0x7fff) == infinity) {
    result = (result & 0x8000) | (infinity - 1);
  }
  return result;
}

// Form: 0=F32, 1=F16, 2=BF16, 3=BF16 result with F32 accumulator.
// Read all eight mixed-width accumulator registers before writing only the lower four.
template <int Bits, bool BfA, bool BfB, int Form>
void swmmac1250_float_kernel(uint64_t flags, uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b, const uint32_t *const *index) {
  constexpr int K = Bits == 16 ? 64 : 128;
  constexpr int entries = K / 4;
  alignas(64) float left[16 * K / 2], raw_right[16 * K], right[K][16], result[16][16];
  smfmac_decode<16 * K / 2, Bits, BfA, false, 32>(a, left);
  smfmac_decode<16 * K, Bits, BfB, false, 32>(b, raw_right);
  for (int k = 0; k < K; ++k) {
    for (int col = 0; col < 16; ++col) {
      // Sparse FP8 B uses 32-element blocks, rather than dense WMMA's 16.
      right[k][col] = raw_right[swmmac1250_input_index<Bits, K / 4>(col, k)];
    }
  }
  for (int row = 0; row < 16; ++row) {
    float values[K / 2];
    int selected[K / 2];
    for (int t = 0; t < K / 2; ++t) {
      values[t] = left[swmmac1250_input_index<Bits, K / 8>(row, t)];
      int lane = row + 16 * (t / entries), local = t % entries;
      selected[t] = 4 * (t / 2) + int(index[local / 16][lane] >> (2 * (local % 16)) & 3);
    }
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8);
      if constexpr (Form == 1 || Form == 2) {
        uint16_t half = uint16_t(d[row % 8 / 2][lane] >> (16 * (row % 2)));
        result[row][col] = Form == 1 ? f16_to_float(half) : bf16_to_float(half);
      } else {
        result[row][col] = as_float(d[row % 8][lane]);
      }
    }
    sparse_float_row<16, K / 2>(values, right, selected, result[row]);
  }
  for (int reg = 0; reg < (Form == 0 ? 8 : 4); ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      int row = (Form == 0 ? reg : 2 * reg) + 8 * (lane / 16), col = lane % 16;
      if constexpr (Form == 0) {
        d[reg][lane] = as_bits(result[row][col]);
      } else {
        bool saturate = flags & RJ_ARITH_FP16_OVFL;
        d[reg][lane] = uint32_t(swmmac1250_narrow<(Form >= 2)>(result[row][col], saturate)) |
                       uint32_t(swmmac1250_narrow<(Form >= 2)>(result[row + 1][col], saturate))
                           << 16;
      }
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
