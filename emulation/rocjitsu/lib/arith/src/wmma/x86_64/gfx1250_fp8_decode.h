// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Fast x86-64 SIMD widening of OCP FP8 (E4M3FN) and OCP BF8 (E5M2) inputs.
// F16C widening converts finite FP8 via x = w << 7; f16 = x + (x & 0x4000); val = 256 * F16(f16),
// requiring zero LUT registers. AVX-512BW vpermi2w LUTs convert FP8/BF8 directly to BF16 pairs
// for AVX512_BF16 vdpbf16ps backends.
#pragma once

#include "common/bits.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

#if defined(RJ_ARITH_TARGET_X86_64_V4)
// 128-entry BF16 magnitude table for OCP FP8 (E4M3FN) or OCP BF8 (E5M2).
// Every finite encoding is exact in BF16; all NaNs map to canonical +QNaN (0x7fc0).
template <bool Bf8> struct Gfx1250Fp8Bf16Table {
  uint16_t entries[128]{};
  constexpr Gfx1250Fp8Bf16Table() {
    for (int i = 0; i < 128; ++i) {
      if constexpr (Bf8) {
        if (i == 0) {
          entries[i] = 0x0000;
        } else if (i < 4) {
          entries[i] = i == 1 ? 0x3780 : i == 2 ? 0x3800 : 0x3840;
        } else if (i < 124) {
          entries[i] = uint16_t(((i >> 2) + 112) << 7 | (i & 3) << 5);
        } else if (i == 124) {
          entries[i] = 0x7f80;
        } else {
          entries[i] = 0x7fc0;
        }
      } else {
        if (i == 0) {
          entries[i] = 0x0000;
        } else if (i < 8) {
          constexpr uint16_t sub[8] = {0, 0x3b00, 0x3b80, 0x3bc0, 0x3c00, 0x3c20, 0x3c40, 0x3c60};
          entries[i] = sub[i];
        } else if (i < 127) {
          entries[i] = uint16_t(((i >> 3) + 120) << 7 | (i & 7) << 4);
        } else {
          entries[i] = 0x7fc0;
        }
      }
    }
  }
};

template <bool Bf8> alignas(64) inline constexpr Gfx1250Fp8Bf16Table<Bf8> kGfx1250Fp8Bf16Table{};

// Convert 32 OCP FP8/BF8 bytes (16 adjacent K pairs) into 16 packed BF16 pairs in a ZMM.
template <bool Bf8> inline __m512i gfx1250_fp8_to_bf16_pairs(__m256i packed_bytes) {
  const uint16_t *tbl = kGfx1250Fp8Bf16Table<Bf8>.entries;
  const __m512i t0 = _mm512_load_si512(tbl);
  const __m512i t1 = _mm512_load_si512(tbl + 32);
  const __m512i t2 = _mm512_load_si512(tbl + 64);
  const __m512i t3 = _mm512_load_si512(tbl + 96);
  const __m512i b = _mm512_cvtepu8_epi16(packed_bytes);
  const __m512i lo = _mm512_permutex2var_epi16(t0, b, t1);
  const __m512i hi = _mm512_permutex2var_epi16(t2, b, t3);
  const __mmask32 hi_mask = _mm512_test_epi16_mask(b, _mm512_set1_epi16(64));
  const __m512i mag = _mm512_mask_mov_epi16(lo, hi_mask, hi);
  // 0xf8 = A | (B & C): combine 15-bit BF16 magnitude with sign bit shifted from bit 7 to bit 15.
  return _mm512_ternarylogic_epi32(mag, _mm512_slli_epi16(b, 8),
                                   _mm512_set1_epi16(static_cast<int16_t>(0x8000)), 0xf8);
}

// Convert 16 OCP FP8 (E4M3FN) bytes in an XMM to 16 exact FP32 values via F16C (256 * F16).
inline __m512 gfx1250_widen_fp8_16_v4(__m128i bytes) {
  const __m256i b = _mm256_cvtepu8_epi16(bytes);
  const __m256i x = _mm256_slli_epi16(b, 7);
  __m256i f16 = _mm256_add_epi16(x, _mm256_and_si256(x, _mm256_set1_epi16(0x4000)));
  // For NaN (0x7f/0xff), f16 is 0x3f80/0xbf80; adding 0x3e80 yields 0x7e00/0xfe00 (+/-qNaN).
  const __mmask16 nan = _mm256_cmpeq_epi16_mask(_mm256_and_si256(b, _mm256_set1_epi16(0x7f)),
                                                _mm256_set1_epi16(0x7f));
  f16 = _mm256_mask_add_epi16(f16, nan, f16, _mm256_set1_epi16(0x3e80));
  return _mm512_mul_ps(_mm512_cvtph_ps(f16), _mm512_set1_ps(256.0f));
}
#endif

// Convert 16 OCP FP8 bytes in an XMM (first 8 = even, last 8 = odd) to two exact FP32 YMMs.
inline void gfx1250_widen_fp8_16_split_v3(__m128i split_bytes, __m256 &first, __m256 &second,
                                          bool check_nan) {
  const __m256i b = _mm256_cvtepu8_epi16(split_bytes);
  const __m256i x = _mm256_slli_epi16(b, 7);
  __m256i f16 = _mm256_add_epi16(x, _mm256_and_si256(x, _mm256_set1_epi16(0x4000)));
  if (check_nan) [[unlikely]] {
    const __m256i nan =
        _mm256_cmpeq_epi16(_mm256_and_si256(b, _mm256_set1_epi16(0x7f)), _mm256_set1_epi16(0x7f));
    f16 = _mm256_xor_si256(f16, _mm256_and_si256(nan, _mm256_set1_epi16(0x4180)));
  }
  const __m256 scale = _mm256_set1_ps(256.0f);
  first = _mm256_mul_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(f16)), scale);
  second = _mm256_mul_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(f16, 1)), scale);
}

// Widen 16 OCP FP8 bytes stored in the low 16 bits of 8 dwords to two FP32 YMMs.
inline void gfx1250_widen_fp8_pair_v3(__m256i packed, __m256 &first, __m256 &second) {
  const __m256i raw = _mm256_and_si256(packed, _mm256_set1_epi32(0xffff));
  const __m128i pair16 =
      _mm_packus_epi32(_mm256_castsi256_si128(raw), _mm256_extracti128_si256(raw, 1));
  const __m128i deinterleave = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
  const __m128i split = _mm_shuffle_epi8(pair16, deinterleave);
  const bool has_nan = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(split, _mm_set1_epi8(0x7f)),
                                                        _mm_set1_epi8(0x7f))) != 0;
  gfx1250_widen_fp8_16_split_v3(split, first, second, has_nan);
}

// Widen all 32 OCP FP8 bytes (8 lanes x 4 K-elements in 1 YMM) to four FP32 YMMs.
inline void gfx1250_widen_fp8_quad_v3(__m256i words, __m256 &k0, __m256 &k1, __m256 &k2,
                                      __m256 &k3) {
  // Single in-lane transpose groups bytes [k0*4, k1*4, k2*4, k3*4] in each 128-bit half.
  const __m256i transpose = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                             0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
  const bool has_nan =
      _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(words, _mm256_set1_epi8(0x7f)),
                                             _mm256_set1_epi8(0x7f))) != 0;
  const __m256i t = _mm256_shuffle_epi8(words, transpose);
  const __m128i lo128 = _mm256_castsi256_si128(t);
  const __m128i hi128 = _mm256_extracti128_si256(t, 1);
  // Combine lane 0..3 and lane 4..7 for (k0, k1) and (k2, k3).
  const __m128i k01 = _mm_unpacklo_epi32(lo128, hi128);
  const __m128i k23 = _mm_unpackhi_epi32(lo128, hi128);
  gfx1250_widen_fp8_16_split_v3(k01, k0, k1, has_nan);
  gfx1250_widen_fp8_16_split_v3(k23, k2, k3, has_nan);
}

// Widen 8 contiguous OCP FP8 (E4M3FN) bytes in the low 64 bits of an XMM to 8 FP32 values.
inline __m256 gfx1250_widen_fp8_8_v3(__m128i bytes) {
  const __m128i b = _mm_cvtepu8_epi16(bytes);
  const __m128i x = _mm_slli_epi16(b, 7);
  __m128i f16 = _mm_add_epi16(x, _mm_and_si128(x, _mm_set1_epi16(0x4000)));
  const __m128i nan = _mm_cmpeq_epi16(_mm_and_si128(b, _mm_set1_epi16(0x7f)), _mm_set1_epi16(0x7f));
  if (_mm_movemask_epi8(nan)) [[unlikely]] {
    f16 = _mm_xor_si128(f16, _mm_and_si128(nan, _mm_set1_epi16(0x4180)));
  }
  return _mm256_mul_ps(_mm256_cvtph_ps(f16), _mm256_set1_ps(256.0f));
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
