// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512BW OCP FP8/BF8 widening through 256-byte magnitude tables.
// Shared FMA and native BF16 pair-dot input decoding, including special encodings.

#pragma once

#include "common/isa.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Positive-magnitude encodings; vector decode applies the input sign separately.
struct Fp8Bf16LutTable {
  uint16_t entries[128];
};

// Build the 128-entry magnitude LUT mapping OCP FP8 (E4M3FN) or BF8 (E5M2)
// magnitudes [0..127] to exact positive BF16 encodings. Sign bit 7 is applied
// separately via bitwise OR into bit 15, covering all 256 byte values
// (subnormals, normals, zeros, infinities, and signed quiet NaNs).
template <bool Bf8> constexpr Fp8Bf16LutTable make_fp8_bf16_lut() {
  Fp8Bf16LutTable lut{};
  constexpr int frac_bits = Bf8 ? 2 : 3;
  constexpr int exp_bias_delta = Bf8 ? 112 : 120;
  for (int m = 0; m < 128; ++m) {
    if (m == 0) {
      lut.entries[m] = 0;
    } else if (m < (1 << frac_bits)) {
      int lead = 0;
      for (int bit = frac_bits - 1; bit >= 0; --bit) {
        if (m & (1 << bit)) {
          lead = bit;
          break;
        }
      }
      int shift = frac_bits - lead;
      int bf16_exp = exp_bias_delta + 1 - shift;
      int bf16_frac = ((m << shift) & ((1 << frac_bits) - 1)) << (7 - frac_bits);
      lut.entries[m] = uint16_t((bf16_exp << 7) | bf16_frac);
    } else if constexpr (Bf8) {
      if (m == 124) {
        lut.entries[m] = 0x7f80;
      } else if (m > 124) {
        lut.entries[m] = 0x7fc0;
      } else {
        lut.entries[m] = uint16_t((m << 5) + (exp_bias_delta << 7));
      }
    } else {
      if (m == 127) {
        lut.entries[m] = 0x7fc0;
      } else {
        lut.entries[m] = uint16_t((m << 4) + (exp_bias_delta << 7));
      }
    }
  }
  return lut;
}

// Four-ZMM 128-entry BF16 LUT for branchless 32-byte-at-a-time FP8/BF8 decode
// on AVX-512BW. Load once outside operand loops to reuse across K slices.
template <bool Bf8> struct Fp8Bf16Lut {
  __m512i t0, t1, t2, t3;

  [[gnu::always_inline]] Fp8Bf16Lut() {
    alignas(64) static constexpr Fp8Bf16LutTable table = make_fp8_bf16_lut<Bf8>();
    t0 = _mm512_load_si512(table.entries);
    t1 = _mm512_load_si512(table.entries + 32);
    t2 = _mm512_load_si512(table.entries + 64);
    t3 = _mm512_load_si512(table.entries + 96);
  }

  // Convert 32 FP8/BF8 bytes into 32 exact BF16 encodings preserving byte order.
  [[gnu::always_inline]] __m512i decode_bf16(__m256i bytes32) const {
    const __m512i raw = _mm512_cvtepu8_epi16(bytes32);
    const __mmask32 hi_mask = _mm512_test_epi16_mask(raw, _mm512_set1_epi16(0x0040));
    const __m512i lo = _mm512_permutex2var_epi16(t0, raw, t1);
    const __m512i hi = _mm512_permutex2var_epi16(t2, raw, t3);
    const __m512i mag = _mm512_mask_mov_epi16(lo, hi_mask, hi);
    const __m512i sign = _mm512_slli_epi16(raw, 8);
    // mag | (sign & 0x8000): ternary logic 0xf8 = A | (B & C).
    return _mm512_ternarylogic_epi32(mag, sign, _mm512_set1_epi16(short(0x8000)), 0xf8);
  }

  // Convert two adjacent FP8/BF8 factors per 32-bit lane into BF16 pair-dot
  // order (low byte placed in high 16-bit half for DPBF16's high-first order).
  [[gnu::always_inline]] __m512i decode_bf16_pairs(__m512i words, int half) const {
    const __m256i packed = _mm512_cvtepi32_epi16(half ? _mm512_srli_epi32(words, 16) : words);
    return _mm512_rol_epi32(decode_bf16(packed), 16);
  }
};

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
