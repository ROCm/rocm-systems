// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Bulk CDNA FP8/BF8 (OCP and FNUZ) widening to deinterleaved K-major planes.
#pragma once
#include "common/x86_64/fp8.h"
#include "mfma/x86_64/reduction.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode eight zero-extended source bytes in integer lanes (also used by sparse_float.h).
template <bool Bf8, bool Fnuz> inline __m256 mfma_widen_fp8(__m256i bytes) {
  return widen_fp8<Bf8, false, Fnuz>(bytes);
}

#if defined(RJ_ARITH_TARGET_X86_64_V4)
// Compile-time 128-entry BF16 magnitude table for OCP or FNUZ FP8/BF8.
// Every finite, infinite, and NaN FP8/BF8 magnitude is exact in BF16.
template <bool Bf8, bool Fnuz> struct MfmaFp8Bf16Table {
  uint16_t mag[128]{};
  constexpr MfmaFp8Bf16Table() {
    constexpr int frac_bits = Bf8 ? 2 : 3;
    constexpr int bias = (Bf8 ? 15 : 7) + int(Fnuz);
    for (int m = 0; m < 128; ++m) {
      const int exp = m >> frac_bits;
      const int frac = m & ((1 << frac_bits) - 1);
      if constexpr (!Fnuz && Bf8) {
        if (exp == 31) {
          mag[m] = frac == 0 ? 0x7f80u : 0x7fc0u;
          continue;
        }
      } else if constexpr (!Fnuz && !Bf8) {
        if (m == 127) {
          mag[m] = 0x7fc0u;
          continue;
        }
      }
      if (exp == 0) {
        if (frac == 0) {
          mag[m] = 0;
        } else {
          int msb = (frac >= 4) ? 2 : (frac >= 2) ? 1 : 0;
          int bf_exp = 127 + (1 - bias - frac_bits) + msb;
          int bf_frac = (frac - (1 << msb)) << (7 - msb);
          mag[m] = uint16_t((bf_exp << 7) | bf_frac);
        }
      } else {
        int bf_exp = exp + 127 - bias;
        int bf_frac = frac << (7 - frac_bits);
        mag[m] = uint16_t((bf_exp << 7) | bf_frac);
      }
    }
  }
};

// Convert two packed FP8/BF8 bytes per 32-bit lane (half=0: bytes 0,1; half=1: bytes 2,3)
// into 16 packed BF16 pairs in a single ZMM using a 128-entry vpermi2w LUT.
template <bool Bf8, bool Fnuz> inline __m512i mfma_fp8_to_bf16_pairs(__m512i words, int half) {
  alignas(64) static constexpr MfmaFp8Bf16Table<Bf8, Fnuz> table;
  const __m512i t0 = _mm512_load_si512(table.mag + 0);
  const __m512i t1 = _mm512_load_si512(table.mag + 32);
  const __m512i t2 = _mm512_load_si512(table.mag + 64);
  const __m512i t3 = _mm512_load_si512(table.mag + 96);
  const __m256i packed16 = _mm512_cvtepi32_epi16(half ? _mm512_srli_epi32(words, 16) : words);
  const __m512i raw = _mm512_cvtepu8_epi16(packed16);
  const __mmask32 hi_mask = _mm512_test_epi16_mask(raw, _mm512_set1_epi16(0x40));
  const __m512i lo_val = _mm512_permutex2var_epi16(t0, raw, t1);
  const __m512i hi_val = _mm512_permutex2var_epi16(t2, raw, t3);
  __m512i bf16 = _mm512_mask_mov_epi16(lo_val, hi_mask, hi_val);
  bf16 = _mm512_ternarylogic_epi32(bf16, _mm512_slli_epi16(raw, 8),
                                   _mm512_set1_epi16(short(0x8000)), 0xf8);
  if constexpr (Fnuz) {
    const __mmask32 nan_mask = _mm512_cmpeq_epi16_mask(raw, _mm512_set1_epi16(0x80));
    bf16 = _mm512_mask_set1_epi16(bf16, nan_mask, short(0xffc0));
  }
  return bf16;
}

// Unpack one Wave64 packed FP8/BF8 VGPR into four 64-lane FP32 planes.
template <bool Bf8, bool Fnuz>
inline void mfma_unpack_fp8_vgpr(const uint32_t *src, float (*out)[64]) {
  const __m512i hi_mask = _mm512_set1_epi32(int(0xffff0000U));
  for (int lane = 0; lane < 64; lane += 16) {
    const __m512i words = _mm512_loadu_si512(src + lane);
    const __m512i pair01 = mfma_fp8_to_bf16_pairs<Bf8, Fnuz>(words, 0);
    const __m512i pair23 = mfma_fp8_to_bf16_pairs<Bf8, Fnuz>(words, 1);
    _mm512_store_ps(out[0] + lane, _mm512_castsi512_ps(_mm512_slli_epi32(pair01, 16)));
    _mm512_store_ps(out[1] + lane, _mm512_castsi512_ps(_mm512_and_si512(pair01, hi_mask)));
    _mm512_store_ps(out[2] + lane, _mm512_castsi512_ps(_mm512_slli_epi32(pair23, 16)));
    _mm512_store_ps(out[3] + lane, _mm512_castsi512_ps(_mm512_and_si512(pair23, hi_mask)));
  }
}
#else
// Widen 16 zero-extended FP8/BF8 bytes in 16-bit lanes into two 8-lane FP32 vectors via F16C.
template <bool Bf8, bool Fnuz>
inline void mfma_widen_fp8_pair_v3(__m256i bytes16, float *out0, float *out1) {
  if constexpr (Bf8 && !Fnuz) {
    // OCP BF8 (E5M2, bias 15) is the exact high byte of IEEE FP16 for all 256 codes.
    const __m256i h16 = _mm256_slli_epi16(bytes16, 8);
    _mm256_store_ps(out0, _mm256_cvtph_ps(_mm256_castsi256_si128(h16)));
    _mm256_store_ps(out1, _mm256_cvtph_ps(_mm256_extracti128_si256(h16, 1)));
  } else if constexpr (!Bf8) {
    // E4M3 (OCP bias 7 -> *256; FNUZ bias 8 -> *128): magnitude<<7 stays in FP16 exp 0..15.
    const __m256i mag = _mm256_and_si256(bytes16, _mm256_set1_epi16(0x7f));
    const __m256i sign =
        _mm256_and_si256(_mm256_slli_epi16(bytes16, 8), _mm256_set1_epi16(short(0x8000)));
    __m256i h16 = _mm256_or_si256(_mm256_slli_epi16(mag, 7), sign);
    const __m256i nan_mask = Fnuz ? _mm256_cmpeq_epi16(bytes16, _mm256_set1_epi16(0x80))
                                  : _mm256_cmpeq_epi16(mag, _mm256_set1_epi16(0x7f));
    h16 = _mm256_or_si256(h16, _mm256_and_si256(nan_mask, _mm256_set1_epi16(0x7e00)));
    const __m256 scale = _mm256_set1_ps(Fnuz ? 128.0f : 256.0f);
    _mm256_store_ps(out0, _mm256_mul_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(h16)), scale));
    _mm256_store_ps(out1, _mm256_mul_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(h16, 1)), scale));
  } else {
    // FNUZ BF8 (E5M2, bias 16, 0x80=NaN, mag 124..127 finite):
    // Shift exp=31 down to 30 before F16C and scale those lanes by 1.0f instead of 0.5f.
    const __m256i mag = _mm256_and_si256(bytes16, _mm256_set1_epi16(0x7f));
    const __m256i high = _mm256_cmpgt_epi16(mag, _mm256_set1_epi16(123));
    const __m256i nan_mask = _mm256_cmpeq_epi16(bytes16, _mm256_set1_epi16(0x80));
    __m256i h16 = _mm256_sub_epi16(_mm256_slli_epi16(bytes16, 8),
                                   _mm256_and_si256(high, _mm256_set1_epi16(0x0400)));
    h16 = _mm256_or_si256(h16, _mm256_and_si256(nan_mask, _mm256_set1_epi16(0x7e00)));
    const __m256i half_bits = _mm256_set1_epi32(0x3f000000);
    const __m256i exp_bump = _mm256_set1_epi32(0x00800000);
    const __m256i high_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(high));
    const __m256i high_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(high, 1));
    const __m256 scale0 =
        _mm256_castsi256_ps(_mm256_or_si256(half_bits, _mm256_and_si256(high_lo, exp_bump)));
    const __m256 scale1 =
        _mm256_castsi256_ps(_mm256_or_si256(half_bits, _mm256_and_si256(high_hi, exp_bump)));
    _mm256_store_ps(out0, _mm256_mul_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(h16)), scale0));
    _mm256_store_ps(out1, _mm256_mul_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(h16, 1)), scale1));
  }
}

// Unpack one Wave64 packed FP8/BF8 VGPR into four 64-lane FP32 planes via AVX2+F16C.
template <bool Bf8, bool Fnuz>
inline void mfma_unpack_fp8_vgpr(const uint32_t *src, float (*out)[64]) {
  const __m256i gather = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
  for (int lane = 0; lane < 64; lane += 8) {
    const __m256i words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + lane));
    const __m256i grouped = _mm256_shuffle_epi8(words, gather);
    const __m128i lo128 = _mm256_castsi256_si128(grouped);
    const __m128i hi128 = _mm256_extracti128_si256(grouped, 1);
    const __m256i b01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi32(lo128, hi128));
    const __m256i b23 = _mm256_cvtepu8_epi16(_mm_unpackhi_epi32(lo128, hi128));
    mfma_widen_fp8_pair_v3<Bf8, Fnuz>(b01, out[0] + lane, out[1] + lane);
    mfma_widen_fp8_pair_v3<Bf8, Fnuz>(b23, out[2] + lane, out[3] + lane);
  }
}
#endif

// Decode both packed factors once into K-major planes, then run the tiled SIMD reduction.
template <int M, int K, bool Bf8A, bool Bf8B, bool Fnuz>
void mfma_fp8_kernel(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int per_lane = K / (64 / M);
  constexpr int regs = per_lane / 4;
  alignas(64) float left[per_lane][64], right[per_lane][64];
  for (int reg = 0; reg < regs; ++reg) {
    mfma_unpack_fp8_vgpr<Bf8A, Fnuz>(a[reg], left + 4 * reg);
    mfma_unpack_fp8_vgpr<Bf8B, Fnuz>(b[reg], right + 4 * reg);
  }
  mfma_reduce_planes<M, K, 1>(controls, d, left, right, c);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
