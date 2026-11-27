// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 byte-input WMMA with SIMD widening and shared row FMAs.
#pragma once
#include "common/compiler.h"
#include "wmma/gfx1250_fp8.h"
#include "wmma/gfx1250_half.h"
#include "wmma/x86_64/gfx1250_accumulator.h"
#include "wmma/x86_64/gfx1250_bf8.h"
#include "wmma/x86_64/gfx1250_fp8_decode.h"
#include "wmma/x86_64/gfx1250_reduction.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Canonicalize E5M2 NaN encodings to signed quiet 0x7e/0xfe before F16C widening.
inline Gfx1250PackedWords gfx1250_canonicalize_bf8(Gfx1250PackedWords words) {
  using Bytes [[gnu::vector_size(sizeof(Gfx1250PackedWords))]] = uint8_t;
  const auto bytes = bit_cast<Bytes>(words);
  const auto canonical = (bytes & 0x80u) | 0x7eu;
  return bit_cast<Gfx1250PackedWords>(((bytes & 0x7fu) > 0x7cu) ? canonical : bytes);
}

#if defined(RJ_ARITH_TARGET_X86_64_V4)
template <bool Bf8>
inline void gfx1250_stage_byte_quad_v4(const uint32_t *source, float (*tile)[16]) {
  __m512i words = _mm512_loadu_si512(source);
  if constexpr (Bf8) {
    const __m512i mag = _mm512_and_si512(words, _mm512_set1_epi8(0x7f));
    const __mmask64 nan = _mm512_cmpgt_epi8_mask(mag, _mm512_set1_epi8(0x7c));
    if (nan) [[unlikely]] {
      const __m512i canon =
          _mm512_or_si512(_mm512_and_si512(words, _mm512_set1_epi8(static_cast<int8_t>(0x80))),
                          _mm512_set1_epi8(0x7e));
      words = _mm512_mask_mov_epi8(words, nan, canon);
    }
    const __m256i lo = _mm512_cvtepi32_epi16(words);
    const __m256i hi = _mm512_cvtepi32_epi16(_mm512_srli_epi32(words, 16));
    const __m256i hi_mask = _mm256_set1_epi16(static_cast<int16_t>(0xff00));
    _mm512_store_ps(tile[0], _mm512_cvtph_ps(_mm256_slli_epi16(lo, 8)));
    _mm512_store_ps(tile[1], _mm512_cvtph_ps(_mm256_and_si256(lo, hi_mask)));
    _mm512_store_ps(tile[2], _mm512_cvtph_ps(_mm256_slli_epi16(hi, 8)));
    _mm512_store_ps(tile[3], _mm512_cvtph_ps(_mm256_and_si256(hi, hi_mask)));
  } else {
    const __m512i lo_bf16 = gfx1250_fp8_to_bf16_pairs<false>(_mm512_cvtepi32_epi16(words));
    const __m512i hi_bf16 =
        gfx1250_fp8_to_bf16_pairs<false>(_mm512_cvtepi32_epi16(_mm512_srli_epi32(words, 16)));
    const __m512i hi_mask = _mm512_set1_epi32(int(0xffff0000U));
    _mm512_store_ps(tile[0], _mm512_castsi512_ps(_mm512_slli_epi32(lo_bf16, 16)));
    _mm512_store_ps(tile[1], _mm512_castsi512_ps(_mm512_and_si512(lo_bf16, hi_mask)));
    _mm512_store_ps(tile[2], _mm512_castsi512_ps(_mm512_slli_epi32(hi_bf16, 16)));
    _mm512_store_ps(tile[3], _mm512_castsi512_ps(_mm512_and_si512(hi_bf16, hi_mask)));
  }
}

// Pre-decode A/B in a tight F16C pass (no store-to-load forwarding stalls),
// then keep all 16 accumulators register-resident through reduction and store.
template <int K, bool Bf8A, bool Bf8B, bool Packed>
void wmma_gfx1250_fp8_kernel(uint64_t flags, uint64_t controls, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  constexpr Wmma1250Form form = Packed ? Wmma1250Form::F16 : Wmma1250Form::F32F16;
  alignas(64) float left[K][16], right[K][16];
  for (int k = 0; k < K; k += 4) {
    const int reg = 4 * (k / 32) + (k % 16) / 4;
    const int lane = 16 * ((k / 16) % 2);
    gfx1250_stage_byte_quad_v4<Bf8A>(a[reg] + lane, left + k);
    gfx1250_stage_byte_quad_v4<Bf8B>(b[reg] + lane, right + k);
  }
  __m512 acc[16];
  gfx1250_load_acc_zmm<form>(c, controls, acc);
  for (int k = 0; k < K; ++k) {
    const __m512 column = _mm512_load_ps(right[k]);
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(left[k][row]), column, acc[row]);
    }
  }
  gfx1250_store_acc_zmm<form>(flags, d, acc, flags & RJ_ARITH_FP16_OVFL);
}
#else
template <bool Bf8>
inline void gfx1250_stage_byte_word(const uint32_t *source, float (*tile)[16], int lane) {
  __m256i word = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source));
  if constexpr (Bf8) {
    const __m256i mag = _mm256_and_si256(word, _mm256_set1_epi8(0x7f));
    const __m256i nan = _mm256_cmpgt_epi8(mag, _mm256_set1_epi8(0x7c));
    if (_mm256_movemask_epi8(nan)) [[unlikely]] {
      word = bit_cast<__m256i>(gfx1250_canonicalize_bf8(bit_cast<Gfx1250PackedWords>(word)));
    }
    for (int part = 0; part < 4; part += 2) {
      __m256 even, odd;
      gfx1250_widen_bf8_pair(bit_cast<Gfx1250PackedWords>(word), even, odd);
      _mm256_store_ps(tile[part] + lane, even);
      _mm256_store_ps(tile[part + 1] + lane, odd);
      word = _mm256_srli_epi32(word, 16);
    }
  } else {
    __m256 k0, k1, k2, k3;
    gfx1250_widen_fp8_quad_v3(word, k0, k1, k2, k3);
    _mm256_store_ps(tile[0] + lane, k0);
    _mm256_store_ps(tile[1] + lane, k1);
    _mm256_store_ps(tile[2] + lane, k2);
    _mm256_store_ps(tile[3] + lane, k3);
  }
}

// Stage every source before publication, preserving whole-register operand overlap.
template <int K, bool Bf8A, bool Bf8B, bool Packed>
void wmma_gfx1250_fp8_kernel(uint64_t flags, uint64_t controls, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  constexpr auto form = Packed ? Wmma1250Form::F16 : Wmma1250Form::F32F16;
  alignas(64) float left[K][16], right[K][16], result[16][16];
  // Four successive K elements occupy one word in each of 16 adjacent lanes, which are the
  // rows of A and the columns of B: widen both directly into left[k][row], right[k][column].
  for (int k = 0; k < K; k += 4) {
    const int reg = 4 * (k / 32) + (k % 16) / 4;
    const int lane = 16 * ((k / 16) % 2);
    for (int offset = 0; offset < 16; offset += 8) {
      gfx1250_stage_byte_word<Bf8A>(a[reg] + lane + offset, left + k, offset);
      gfx1250_stage_byte_word<Bf8B>(b[reg] + lane + offset, right + k, offset);
    }
  }
  gfx1250_load_accumulators<form>(c, controls, result);
  gfx1250_reduce_interleaved<K, false>(left, right, result);
  gfx1250_store_results<form>(flags, d, result, flags & RJ_ARITH_FP16_OVFL);
}
#endif
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
