// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Deinterleave packed MFMA and WMMA operands into K-major SIMD rows.
#pragma once

#include "common/compiler.h"
#include "common/x86_64/fp8.h"
#include "mfma/mixed_formats.h"
#include "mfma/x86_64/fp8.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <cstring>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::mixed_x86 {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
using Native = __m512i;
constexpr int Width = 16;
#else
using Native = __m256i;
constexpr int Width = 8;
#endif
using Words [[gnu::vector_size(Width * 4)]] = uint32_t;
using Values [[gnu::vector_size(Width * 4)]] = float;

// Load adjacent wave lanes without assuming that the VGPR is SIMD-aligned.
inline Words load_words(const uint32_t *p) {
  Words v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

// Compile-time 32-entry BF16 magnitude LUT for FP6/BF6 on AVX-512.
template <int Format> struct SubbyteBf16Table {
  uint16_t bits[32]{};
  constexpr SubbyteBf16Table() {
    for (int i = 0; i < 32; ++i) {
      bits[i] = uint16_t(bit_cast<uint32_t>(matrix_decode_subbyte(i, Format)) >> 16);
    }
  }
};

// Widen one 6-bit or 4-bit code vector when single-slice decode is used.
template <int Format> inline Values widen(Words code) {
  if constexpr (Format <= RJ_ARITH_MATRIX_FORMAT_BF8) {
    return widen_fp8<Format == RJ_ARITH_MATRIX_FORMAT_BF8>(bit_cast<Native>(code));
  } else if constexpr (Format == RJ_ARITH_MATRIX_FORMAT_FP4) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
    const auto values =
        _mm512_setr_ps(0, 0.5f, 1, 1.5, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5, -2, -3, -4, -6);
    return _mm512_permutexvar_ps(bit_cast<Native>(code), values);
#else
    const auto values = _mm256_setr_ps(0, 0.5f, 1, 1.5, 2, 3, 4, 6);
    auto mag = _mm256_permutevar8x32_ps(values, bit_cast<Native>(code));
    return bit_cast<Values>(bit_cast<Words>(mag) | ((code & 8U) << 28));
#endif
  } else {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
    alignas(64) static constexpr SubbyteBf16Table<Format> table;
    auto index = bit_cast<Native>(code);
    auto halves = _mm512_permutexvar_epi16(index, _mm512_load_si512(table.bits));
    return bit_cast<Values>((bit_cast<Words>(halves) << 16) | ((code & 32u) << 26));
#else
    constexpr int fraction = Format == RJ_ARITH_MATRIX_FORMAT_FP6 ? 3 : 2;
    constexpr int bias = Format == RJ_ARITH_MATRIX_FORMAT_FP6 ? 1 : 3;
    Words magnitude = code & 31u;
    Words normal = (magnitude << (23 - fraction)) + ((127u - bias) << 23);
    Values small = __builtin_convertvector(magnitude, Values) *
                   (Format == RJ_ARITH_MATRIX_FORMAT_FP6 ? 0x1p-3f : 0x1p-4f);
    Words value = magnitude < (1u << fraction) ? bit_cast<Words>(small) : normal;
    return bit_cast<Values>(value | ((code & 32u) << 26));
#endif
  }
}

// Consecutive lanes hold independent matrix rows. Decode directly into K-by-row
// storage, reusing each loaded VGPR across all packed elements in that word.
template <int Format, int Block, int Groups, int RowsPerGroup, int Rows, int K>
void unpack(const uint32_t *const *input, float (&out)[K][Rows]) {
  constexpr int Bits = matrix_format_bits(Format);
  static_assert(RowsPerGroup % Width == 0 && Rows % RowsPerGroup == 0 && K % (Block * Groups) == 0);
  for (int k = 0; k < K; k += Block) {
    const int lane = (k / Block % Groups) * RowsPerGroup;
    const int first = (k / (Block * Groups)) * Block;
    for (int row = 0; row < Rows; row += Width) {
      // Wide gfx1250 FP4 uses a second bank of eight VGPRs for rows 16..31.
      const auto regs = input + (row / RowsPerGroup) * 8;
      const int first_lane = lane + row % RowsPerGroup;
      if constexpr (Format <= RJ_ARITH_MATRIX_FORMAT_BF8) {
        constexpr bool Bf8 = (Format == RJ_ARITH_MATRIX_FORMAT_BF8);
        static_assert(Block % 4 == 0);
#if defined(RJ_ARITH_TARGET_X86_64_V4)
        const __m512i hi_mask = _mm512_set1_epi32(int(0xffff0000U));
        for (int j = 0; j < Block; j += 4) {
          const int reg = (first + j) / 4;
          const __m512i words = _mm512_loadu_si512(regs[reg] + first_lane);
          const __m512i p01 = mfma_fp8_to_bf16_pairs<Bf8, false>(words, 0);
          const __m512i p23 = mfma_fp8_to_bf16_pairs<Bf8, false>(words, 1);
          _mm512_store_ps(out[k + j + 0] + row, _mm512_castsi512_ps(_mm512_slli_epi32(p01, 16)));
          _mm512_store_ps(out[k + j + 1] + row,
                          _mm512_castsi512_ps(_mm512_and_si512(p01, hi_mask)));
          _mm512_store_ps(out[k + j + 2] + row, _mm512_castsi512_ps(_mm512_slli_epi32(p23, 16)));
          _mm512_store_ps(out[k + j + 3] + row,
                          _mm512_castsi512_ps(_mm512_and_si512(p23, hi_mask)));
        }
#else
        const __m256i gather = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
        for (int j = 0; j < Block; j += 4) {
          const int reg = (first + j) / 4;
          const __m256i words =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(regs[reg] + first_lane));
          const __m256i grouped = _mm256_shuffle_epi8(words, gather);
          const __m128i lo128 = _mm256_castsi256_si128(grouped);
          const __m128i hi128 = _mm256_extracti128_si256(grouped, 1);
          const __m256i b01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi32(lo128, hi128));
          const __m256i b23 = _mm256_cvtepu8_epi16(_mm_unpackhi_epi32(lo128, hi128));
          mfma_widen_fp8_pair_v3<Bf8, false>(b01, out[k + j + 0] + row, out[k + j + 1] + row);
          mfma_widen_fp8_pair_v3<Bf8, false>(b23, out[k + j + 2] + row, out[k + j + 3] + row);
        }
#endif
      } else if constexpr (Bits == 6) {
        static_assert(Block % 2 == 0);
        // Load only the VGPRs touched by this block once, then extract pairs of 6-bit codes.
        constexpr int first_reg_max = (Block * 6 + 31) / 32;
        const int base_reg = (first * 6) / 32;
        Words cached[first_reg_max + 1];
        const int num_regs = ((first + Block) * 6 + 31) / 32 - base_reg;
        for (int r = 0; r < num_regs; ++r) {
          cached[r] = load_words(regs[base_reg + r] + first_lane);
        }
        auto extract6 = [&](int j) {
          const int bit = (first + j) * 6;
          const int r = bit / 32 - base_reg;
          const int shift = bit % 32;
          Words code = cached[r] >> shift;
          if (shift > 26) {
            code |= cached[r + 1] << (32 - shift);
          }
          return code;
        };
#if defined(RJ_ARITH_TARGET_X86_64_V4)
        alignas(64) static constexpr SubbyteBf16Table<Format> table;
        const __m512i lut = _mm512_load_si512(table.bits);
        const __m512i hi_mask = _mm512_set1_epi32(int(0xffff0000U));
        for (int j = 0; j < Block; j += 2) {
          const Words c0 = extract6(j);
          const Words c1 = extract6(j + 1);
          const Words idx = (c0 & 31u) | ((c1 & 31u) << 16);
          const __m512i halves = _mm512_permutexvar_epi16(bit_cast<Native>(idx), lut);
          const __m512i v0 =
              _mm512_or_si512(_mm512_slli_epi32(halves, 16), bit_cast<Native>((c0 & 32u) << 26));
          const __m512i v1 = _mm512_or_si512(_mm512_and_si512(halves, hi_mask),
                                             bit_cast<Native>((c1 & 32u) << 26));
          _mm512_store_ps(out[k + j + 0] + row, _mm512_castsi512_ps(v0));
          _mm512_store_ps(out[k + j + 1] + row, _mm512_castsi512_ps(v1));
        }
#else
        // F16C widening for FP6 (mag<<7, *2^14) and BF6 (mag<<8, *2^12), exact for all 64 codes.
        constexpr int mag_shift = (Format == RJ_ARITH_MATRIX_FORMAT_FP6) ? 7 : 8;
        const __m256 scale =
            _mm256_set1_ps(Format == RJ_ARITH_MATRIX_FORMAT_FP6 ? 16384.0f : 4096.0f);
        const __m256i mask31 = _mm256_set1_epi16(31);
        const __m256i mask32 = _mm256_set1_epi16(32);
        for (int j = 0; j < Block; j += 2) {
          const auto c0 = bit_cast<__m256i>(extract6(j) & 63u);
          const auto c1 = bit_cast<__m256i>(extract6(j + 1) & 63u);
          const __m256i c16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(c0, c1), 0xd8);
          const __m256i h16 =
              _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(c16, mask31), mag_shift),
                              _mm256_slli_epi16(_mm256_and_si256(c16, mask32), 10));
          _mm256_store_ps(out[k + j + 0] + row,
                          _mm256_mul_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(h16)), scale));
          _mm256_store_ps(out[k + j + 1] + row,
                          _mm256_mul_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(h16, 1)), scale));
        }
#endif
      } else {
        // FP4 (Bits == 4): load each VGPR once for up to 8 elements.
        for (int j = 0; j < Block;) {
          const int reg = (first + j) / 8;
          const int start_nib = (first + j) % 8;
          const int chunk = (Block - j < 8 - start_nib) ? (Block - j) : (8 - start_nib);
          const Words word = load_words(regs[reg] + first_lane);
          for (int s = 0; s < chunk; ++s) {
            const Values values = widen<RJ_ARITH_MATRIX_FORMAT_FP4>(word >> (4 * (start_nib + s)));
            std::memcpy(out[k + j + s] + row, &values, sizeof(values));
          }
          j += chunk;
        }
      }
    }
  }
}

// Each lane supplies 16 byte-format elements or 32 subbyte elements at a time.
// Unscaled gfx1250 subbyte pairs instead interleave groups of four.
template <int Groups, int RowsPerGroup, int Rows, int K>
void decode(const uint32_t *const *input, int format, bool interleave_four, float (&out)[K][Rows]) {
  auto decode_format = [&]<int Format>() {
    if constexpr (Format <= RJ_ARITH_MATRIX_FORMAT_BF8) {
      unpack<Format, 16, Groups, RowsPerGroup>(input, out);
    } else if (interleave_four) {
      unpack<Format, 4, Groups, RowsPerGroup>(input, out);
    } else {
      unpack<Format, 32, Groups, RowsPerGroup>(input, out);
    }
  };
  switch (format) {
  case RJ_ARITH_MATRIX_FORMAT_FP8:
    return decode_format.template operator()<RJ_ARITH_MATRIX_FORMAT_FP8>();
  case RJ_ARITH_MATRIX_FORMAT_BF8:
    return decode_format.template operator()<RJ_ARITH_MATRIX_FORMAT_BF8>();
  case RJ_ARITH_MATRIX_FORMAT_FP6:
    return decode_format.template operator()<RJ_ARITH_MATRIX_FORMAT_FP6>();
  case RJ_ARITH_MATRIX_FORMAT_BF6:
    return decode_format.template operator()<RJ_ARITH_MATRIX_FORMAT_BF6>();
  default:
    return decode_format.template operator()<RJ_ARITH_MATRIX_FORMAT_FP4>();
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::mixed_x86
