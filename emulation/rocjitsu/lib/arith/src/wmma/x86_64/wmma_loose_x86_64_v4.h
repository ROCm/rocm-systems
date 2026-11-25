// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 row layout and packed conversions for the loose FMA and BF16 pair-dot kernels.
#pragma once

#include "common/compiler.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/layout.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

template <bool Bf16> inline __m512 wmma_widen_half(__m512i words, int shift) {
  if constexpr (Bf16) {
    return _mm512_castsi512_ps(shift ? _mm512_and_si512(words, _mm512_set1_epi32(0xffff0000U))
                                     : _mm512_slli_epi32(words, 16));
  } else {
    return _mm512_cvt_roundph_ps(_mm512_cvtepi32_epi16(_mm512_srli_epi32(words, shift)),
                                 _MM_FROUND_NO_EXC);
  }
}

// RNE preserves subnormals. Saturating finite overflow replaces a rounded
// infinity with the adjacent maximum finite half of the same sign.
template <bool Saturate> inline __m256i wmma_narrow_f16(__m512 x) {
  auto half = _mm512_cvt_roundps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  if constexpr (Saturate) {
    auto magnitude = _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(INT32_MAX));
    auto finite = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
    auto inf = _mm256_cmpeq_epi16_mask(_mm256_and_si256(half, _mm256_set1_epi16(0x7fff)),
                                       _mm256_set1_epi16(0x7c00));
    half = _mm256_mask_sub_epi16(half, finite & inf, half, _mm256_set1_epi16(1));
  }
  return half;
}

// Snapshot C before any D store, including the selected gfx11 half.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
[[gnu::always_inline]] inline void wmma_load_acc(__m512 (&acc)[16], uint32_t mods,
                                                 const uint32_t *const *c) {
  auto keep = _mm512_set1_epi32(mods & RJ_ARITH_WMMA_ABS_C ? INT32_MAX : -1);
  auto sign = _mm512_set1_epi32(mods & RJ_ARITH_WMMA_NEG_C ? INT32_MIN : 0);
  // Constant row indices keep the sixteen accumulators in vector registers.
  RJ_ARITH_UNROLL(16)
  for (int row = 0; row < 16; ++row) {
    auto src = wmma_half_output<Lanes, Rdna3>(row);
    int reg = Packed && !Rdna3 ? src.reg / 2 : src.reg;
    auto bits = _mm512_loadu_si512(c[reg] + src.lane);
    if constexpr (Packed) {
      int shift = Rdna3 ? (mods & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0) : 16 * (src.reg % 2);
      bits = _mm512_castps_si512(wmma_widen_half<Bf16>(bits, shift));
    }
    acc[row] = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_and_si512(bits, keep), sign));
  }
}

// Materialize rows in ascending destination-register order. gfx12 packs
// adjacent rows into a word. gfx11 replaces the same half of every D word.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed, bool Saturate>
[[gnu::always_inline]] inline void wmma_store_acc(const __m512 (&acc)[16], uint32_t mods,
                                                  uint32_t *const *d) {
  constexpr int Registers = (256 / Lanes) / (Packed && !Rdna3 ? 2 : 1);
  auto encode = [](__m512 value) {
    if constexpr (!Packed) {
      return _mm512_castps_si512(value);
    } else if constexpr (Bf16) {
      return _mm512_srli_epi32(_mm512_castps_si512(value), 16);
    } else {
      return _mm512_cvtepu16_epi32(wmma_narrow_f16<Saturate>(value));
    }
  };
  for (int reg = 0; reg < Registers; ++reg) {
    for (int group = 0; group < Lanes / 16; ++group) {
      int row = Rdna3 ? reg * (Lanes / 16) + group
                      : reg * (Packed ? 2 : 1) + 8 * (group % 2) + 4 * (group / 2);
      auto value = encode(acc[row]);
      if constexpr (Packed && Rdna3) {
        auto old = _mm512_loadu_si512(d[reg] + 16 * group);
        value = mods & RJ_ARITH_WMMA_HIGH_C_D
                    ? _mm512_mask_blend_epi16(0xaaaaaaaa, old, _mm512_slli_epi32(value, 16))
                    : _mm512_mask_blend_epi16(0x55555555, old, value);
      } else if constexpr (Packed) {
        value = _mm512_or_si512(value, _mm512_slli_epi32(encode(acc[row + 1]), 16));
      }
      _mm512_storeu_si512(d[reg] + 16 * group, value);
    }
  }
}

#if defined(__AVX512BF16__)
// Convert two adjacent FP8 factors per word to the BF16 pair-dot order.
template <bool Bf8> inline __m512i wmma_fp8_bf16_pairs(__m512i words, int half) {
  auto packed = _mm512_cvtepi32_epi16(_mm512_srli_epi32(words, half * 16));
  auto raw = _mm512_cvtepu8_epi16(packed);
  auto magnitude = _mm512_and_si512(raw, _mm512_set1_epi16(127));
  // Rebase normal exponents directly in the BF16 encoding.
  auto encoded = _mm512_add_epi16(_mm512_slli_epi16(magnitude, Bf8 ? 5 : 4),
                                  _mm512_set1_epi16((Bf8 ? 112 : 120) << 7));
  // Positive subnormal FP8 encodings indexed by their fraction.
  alignas(64) static constexpr uint16_t Bf8Subnormals[32] = {0, 0x3780, 0x3800, 0x3840};
  alignas(64) static constexpr uint16_t Fp8Subnormals[32] = {0,      0x3b00, 0x3b80, 0x3bc0,
                                                             0x3c00, 0x3c20, 0x3c40, 0x3c60};
  auto subnormal_values = _mm512_load_si512(Bf8 ? Bf8Subnormals : Fp8Subnormals);
  encoded = _mm512_mask_permutexvar_epi16(
      encoded, _mm512_cmplt_epi16_mask(magnitude, _mm512_set1_epi16(Bf8 ? 4 : 8)), magnitude,
      subnormal_values);
  if constexpr (Bf8) {
    encoded = _mm512_mask_set1_epi16(
        encoded, _mm512_cmpeq_epi16_mask(magnitude, _mm512_set1_epi16(124)), 0x7f80);
  }
  encoded = _mm512_mask_set1_epi16(
      encoded, _mm512_cmpgt_epi16_mask(magnitude, _mm512_set1_epi16(Bf8 ? 124 : 126)), 0x7fc0);
  encoded = _mm512_or_si512(
      encoded, _mm512_and_si512(_mm512_slli_epi16(raw, 8), _mm512_set1_epi16(INT16_MIN)));
  return _mm512_rol_epi32(encoded, 16);
}

// DPBF16 flushes a nonzero subnormal input accumulator even with DAZ disabled.
inline __mmask16 wmma_f32_subnormal(__m512i bits) {
  auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
  return _mm512_cmplt_epu32_mask(_mm512_sub_epi32(magnitude, _mm512_set1_epi32(1)),
                                 _mm512_set1_epi32(0x7fffff));
}

// Both inputs contain adjacent BF16 K pairs with their halves swapped:
// DPBF16 consumes the high half first. Keep one accumulator per output row.
template <bool Packed, bool Rdna3>
[[gnu::always_inline]] inline void
wmma_bf16_pairs(__m512 (&sums)[16], const uint32_t (&left)[8][16], const __m512i (&right)[8]) {
  for (int step = 0; step < 8; ++step) {
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      auto factor = _mm512_broadcastd_epi32(_mm_loadu_si32(&left[step][row]));
      sums[row] = _mm512_dpbf16_ps(sums[row], (__m512bh)right[step], (__m512bh)factor);
      if constexpr (Packed) {
        if (step < 7 && (Rdna3 || step % 2)) {
          sums[row] = _mm512_castsi512_ps(
              _mm512_and_si512(_mm512_castps_si512(sums[row]), _mm512_set1_epi32(0xffff0000U)));
        }
      }
    }
  }
}
#endif

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
