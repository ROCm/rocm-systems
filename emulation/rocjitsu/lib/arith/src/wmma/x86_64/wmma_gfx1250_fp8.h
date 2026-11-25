// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 byte-input WMMA with SIMD widening and shared row FMAs.
#pragma once
#include "common/x86_64/fp8.h"
#include "wmma/gfx1250_fp8.h"
#include "wmma/gfx1250_half.h"
#include "wmma/x86_64/gfx1250_accumulator.h"
#include "wmma/x86_64/gfx1250_bf8.h"
#include "wmma/x86_64/gfx1250_reduction.h"
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Canonicalize the four packed E5M2 inputs before exposing them to F16C.
// All three NaN encodings of a given sign widen to the same quiet NaN.
inline Gfx1250PackedWords gfx1250_canonicalize_bf8(Gfx1250PackedWords words) {
  using Bytes [[gnu::vector_size(sizeof(Gfx1250PackedWords))]] = uint8_t;
  const auto bytes = bit_cast<Bytes>(words);
  const auto canonical = (bytes & 0x80u) | 0x7eu;
  return bit_cast<Gfx1250PackedWords>(((bytes & 0x7fu) > 0x7cu) ? canonical : bytes);
}

template <bool Bf8>
inline void gfx1250_stage_byte_word(const uint32_t *source, float (*tile)[16], int lane,
                                    bool nonfinite) {
  Gfx1250PackedWords word;
  std::memcpy(&word, source, sizeof(word));
  if constexpr (Bf8) {
    if (nonfinite) {
      word = gfx1250_canonicalize_bf8(word);
    }
    for (int part = 0; part < 4; part += 2) {
      Gfx1250PackedFloat even, odd;
      gfx1250_widen_bf8_pair(word, even, odd);
      std::memcpy(tile[part] + lane, &even, sizeof(even));
      std::memcpy(tile[part + 1] + lane, &odd, sizeof(odd));
      word >>= 16;
    }
  } else {
    for (int part = 0; part < 4; ++part) {
#if defined(__AVX512F__)
      auto value = widen_fp8<false>(bit_cast<__m512i>(word));
#else
      auto value = widen_fp8<false>(bit_cast<__m256i>(word));
#endif
      std::memcpy(tile[part] + lane, &value, sizeof(value));
      word >>= 8;
    }
  }
}
// Stage every source before publication, preserving whole-register operand overlap.
template <int K, bool Bf8A, bool Bf8B, bool Packed>
void wmma_gfx1250_fp8_kernel(uint64_t flags, uint64_t controls, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  constexpr auto form = Packed ? Wmma1250Form::F16 : Wmma1250Form::F32F16;
  alignas(64) float left[K][16], right[K][16], result[16][16];
  constexpr int registers = K / 8;
  constexpr int a_exponent = Bf8A ? 0x7c : 0x7f;
  constexpr int b_exponent = Bf8B ? 0x7c : 0x7f;
  const bool nonfinite = gfx1250_has_nonfinite<registers, true, a_exponent, b_exponent>(a, b);
  // Four successive K elements occupy one word in each of 16 adjacent lanes, which are the
  // rows of A and the columns of B: widen both directly into left[k][row], right[k][column].
  for (int k = 0; k < K; k += 4) {
    const int byte_index = gfx1250_fp8_index(0, k);
    const int reg = byte_index / 128;
    const int lane = byte_index / 4 % 32;
    constexpr int width = sizeof(Gfx1250PackedFloat) / sizeof(float);
    for (int offset = 0; offset < 16; offset += width) {
      gfx1250_stage_byte_word<Bf8A>(a[reg] + lane + offset, left + k, offset, nonfinite);
      gfx1250_stage_byte_word<Bf8B>(b[reg] + lane + offset, right + k, offset, nonfinite);
    }
  }
  gfx1250_load_accumulators<form>(c, controls, result);
  if (nonfinite) {
    gfx1250_reduce_rows<K, false, true>(left, right, result);
  } else {
    gfx1250_reduce_interleaved<K, false>(left, right, result);
  }
  gfx1250_store_results<form>(flags, d, result, flags & RJ_ARITH_FP16_OVFL);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
