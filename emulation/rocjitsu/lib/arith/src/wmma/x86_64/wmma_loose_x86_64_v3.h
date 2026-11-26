// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared AVX2 (x86-64-v3) half-precision and OCP FP8/BF8 conversion helpers
// for loose WMMA and SWMMAC kernels. Family-specific packing, rescaling, and
// NaN-scan policies stay in each caller.
#pragma once

#include "common/compiler.h"
#include "common/isa.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode 8 packed FP16/BF16 dwords into two FP32 vectors (even = low half, odd = high half).
template <bool Bf16>
[[gnu::always_inline]] inline void wmma_widen_half_pair_v3(__m256i words, __m256 &even,
                                                           __m256 &odd) {
  if constexpr (Bf16) {
    even = _mm256_castsi256_ps(_mm256_slli_epi32(words, 16));
    odd = _mm256_castsi256_ps(_mm256_and_si256(words, _mm256_set1_epi32(int(0xffff0000U))));
  } else {
    // Collect low halves in lane 0 (128b) and high halves in lane 1 (128b).
    const __m256i order = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0,
                                           1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    const __m256i split = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(words, order), 0xd8);
    even = _mm256_cvtph_ps(_mm256_castsi256_si128(split));
    odd = _mm256_cvtph_ps(_mm256_extracti128_si256(split, 1));
  }
}

// Widen one selected half (low or high) of 8 packed FP16/BF16 dwords to FP32.
template <bool Bf16>
[[gnu::always_inline]] inline __m256 wmma_widen_half_v3(__m256i words, bool high) {
  if constexpr (Bf16) {
    return _mm256_castsi256_ps(high ? _mm256_and_si256(words, _mm256_set1_epi32(int(0xffff0000U)))
                                    : _mm256_slli_epi32(words, 16));
  } else {
    const __m256i lo_mask =
        _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9,
                         12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i hi_mask =
        _mm256_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 6, 7, 10,
                         11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i compacted =
        _mm256_permute4x64_epi64(_mm256_shuffle_epi8(words, high ? hi_mask : lo_mask), 0x08);
    return _mm256_cvtph_ps(_mm256_castsi256_si128(compacted));
  }
}

// Narrow 8 FP32 values to 8 packed FP16 encodings (RNE) with optional finite-overflow saturation.
template <bool Saturate> [[gnu::always_inline]] inline __m128i wmma_narrow_f16_v3(__m256 x) {
  __m128i half = _mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  if constexpr (Saturate) {
    const __m256i mag = _mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32(INT32_MAX));
    const __m256i finite32 = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000), mag);
    const __m128i finite16 =
        _mm_packs_epi32(_mm256_castsi256_si128(finite32), _mm256_extracti128_si256(finite32, 1));
    const __m128i inf16 =
        _mm_cmpeq_epi16(_mm_and_si128(half, _mm_set1_epi16(0x7fff)), _mm_set1_epi16(0x7c00));
    // finite16 & inf16 is -1 for finite overflow lanes, replacing +/-Inf with +/-0x7bff.
    half = _mm_add_epi16(half, _mm_and_si128(finite16, inf16));
  }
  return half;
}

// Round an intermediate DOT2/DOT4 accumulator stage to packed BF16 or FP16 precision.
template <bool Bf16, bool Saturate>
[[gnu::always_inline]] inline __m256 wmma_round_stage_v3(__m256 x) {
  if constexpr (Bf16) {
    return _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(int(0xffff0000U))));
  } else {
    return _mm256_cvtph_ps(wmma_narrow_f16_v3<Saturate>(x));
  }
}

// Pack two adjacent FP32 rows into gfx12 interleaved 16-bit halfwords (even low, odd high).
template <bool Bf16, bool Saturate>
[[gnu::always_inline]] inline __m256i wmma_pack_half_pair_v3(__m256 even, __m256 odd) {
  if constexpr (Bf16) {
    return _mm256_blend_epi16(_mm256_srli_epi32(_mm256_castps_si256(even), 16),
                              _mm256_castps_si256(odd), 0xaa);
  } else {
    const __m128i e = wmma_narrow_f16_v3<Saturate>(even);
    const __m128i o = wmma_narrow_f16_v3<Saturate>(odd);
    return _mm256_set_m128i(_mm_unpackhi_epi16(e, o), _mm_unpacklo_epi16(e, o));
  }
}

// Canonicalize OCP BF8 (E5M2) NaN bytes (magnitude > 0x7c) to quiet NaN 0x7e
// preserving sign, so shifting left by 8 produces an exact binary16 encoding.
[[gnu::always_inline]] inline __m256i wmma_bf8_canon_v3(__m256i words) {
  const __m256i byte_abs = _mm256_set1_epi8(0x7f);
  const __m256i limit = _mm256_set1_epi8(0x7c);
  const __m256i sign = _mm256_set1_epi8(char(0x80));
  const __m256i qnan = _mm256_set1_epi8(0x7e);
  const __m256i nan = _mm256_cmpgt_epi8(_mm256_and_si256(words, byte_abs), limit);
  const __m256i canon = _mm256_or_si256(_mm256_and_si256(words, sign), qnan);
  return _mm256_blendv_epi8(words, canon, nan);
}

// Widen one byte pair (bytes 0,1 when HighPair=false; bytes 2,3 when HighPair=true)
// from 8 packed dwords into two FP32 vectors via F16C.
// - Bf8=true expects BF8-canonicalized input and produces exact FP32 values.
// - Bf8=false widens OCP E4M3FN with a 2^-8 scale factor (exact, including
//   subnormals and NaNs when CheckNan=true); callers apply *256 or deferred C-add
//   scaling and may set CheckNan=false after verifying no 0x7f/0xff bytes exist.
template <bool Bf8, bool HighPair, bool CheckNan = true>
[[gnu::always_inline]] inline void wmma_widen_fp8_pair_v3(__m256i words, __m256 &even,
                                                          __m256 &odd) {
  const __m256i mask =
      HighPair ? _mm256_setr_epi8(-1, 2, -1, 6, -1, 10, -1, 14, -1, 3, -1, 7, -1, 11, -1, 15, -1, 2,
                                  -1, 6, -1, 10, -1, 14, -1, 3, -1, 7, -1, 11, -1, 15)
               : _mm256_setr_epi8(-1, 0, -1, 4, -1, 8, -1, 12, -1, 1, -1, 5, -1, 9, -1, 13, -1, 0,
                                  -1, 4, -1, 8, -1, 12, -1, 1, -1, 5, -1, 9, -1, 13);
  __m256i p = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(words, mask), 0xd8);
  if constexpr (!Bf8) {
    const __m256i f16_sign = _mm256_set1_epi16(short(0x8000));
    const __m256i f16_body = _mm256_set1_epi16(0x3f80);
    const __m256i sign = _mm256_and_si256(p, f16_sign);
    const __m256i body = _mm256_and_si256(_mm256_srli_epi16(p, 1), f16_body);
    if constexpr (CheckNan) {
      const __m256i f16_nan_bit = _mm256_set1_epi16(0x4000);
      const __m256i nan = _mm256_cmpeq_epi16(body, f16_body);
      p = _mm256_or_si256(sign, _mm256_or_si256(body, _mm256_and_si256(nan, f16_nan_bit)));
    } else {
      p = _mm256_or_si256(sign, body);
    }
  }
  even = _mm256_cvtph_ps(_mm256_castsi256_si128(p));
  odd = _mm256_cvtph_ps(_mm256_extracti128_si256(p, 1));
}

// Decode all 4 OCP FP8/BF8 bytes from 8 packed dwords into four FP32 vectors
// (exact for Bf8=true; 2^-8-scaled for Bf8=false).
template <bool Bf8, bool CheckNan = true>
[[gnu::always_inline]] inline void wmma_decode_fp8_word_v3(__m256i words, __m256 &f0, __m256 &f1,
                                                           __m256 &f2, __m256 &f3) {
  if constexpr (Bf8) {
    words = wmma_bf8_canon_v3(words);
  }
  wmma_widen_fp8_pair_v3<Bf8, false, CheckNan>(words, f0, f1);
  wmma_widen_fp8_pair_v3<Bf8, true, CheckNan>(words, f2, f3);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
