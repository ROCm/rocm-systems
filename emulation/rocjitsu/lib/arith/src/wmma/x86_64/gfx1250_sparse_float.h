// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 floating sparse WMMA. SIMD lanes hold output rows. Each packed
// B vector has a four-element K group for each of two or four columns.
#pragma once
#include "common/compiler.h"
#include "common/x86_64/fp8.h"
#include "wmma/gfx1250_sparse_float.h"
#include "wmma/x86_64/gfx1250_accumulator.h"
#include "wmma/x86_64/gfx1250_bf8.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
#if defined(__AVX512F__)
using Swmmac1250Int = __m512i;
#else
using Swmmac1250Int = __m256i;
#endif
using Swmmac1250Vec = Gfx1250PackedFloat;
constexpr int swmmac1250_width = sizeof(Swmmac1250Vec) / sizeof(float);

inline void swmmac1250_transpose(float dst[16][16], const float src[16][16]) {
  for (int row = 0; row < 16; row += 4) {
    for (int col = 0; col < 16; col += 4) {
      auto a = _mm_load_ps(src[row] + col), b = _mm_load_ps(src[row + 1] + col);
      auto c = _mm_load_ps(src[row + 2] + col), d = _mm_load_ps(src[row + 3] + col);
      _MM_TRANSPOSE4_PS(a, b, c, d);
      _mm_store_ps(dst[col] + row, a);
      _mm_store_ps(dst[col + 1] + row, b);
      _mm_store_ps(dst[col + 2] + row, c);
      _mm_store_ps(dst[col + 3] + row, d);
    }
  }
}

// A holds two selected terms per K group, independent of their selectors.
template <int Bits, bool Bf>
inline void swmmac1250_read_a(const uint32_t *words, int group, Swmmac1250Vec &first,
                              Swmmac1250Vec &second) {
  Gfx1250PackedWords packed;
  std::memcpy(&packed, words, sizeof(packed));
  if constexpr (Bits == 8) {
    packed >>= 16 * (group % 2);
    if constexpr (Bf) {
      // E5M2 occupies the high byte of a binary16. F16C can change its NaN
      // payload, which the selected-operand repair recovers before publication.
      gfx1250_widen_bf8_pair(packed, first, second);
    } else {
      first = widen_fp8<false>(bit_cast<Swmmac1250Int>(packed));
      second = widen_fp8<false>(bit_cast<Swmmac1250Int>(packed >> 8));
    }
  } else if constexpr (Bf) {
    first = bit_cast<Swmmac1250Vec>(packed << 16);
    second = bit_cast<Swmmac1250Vec>(packed & 0xffff0000u);
  } else {
    auto raw = bit_cast<Swmmac1250Int>(packed);
#if defined(__AVX512F__)
    first = half_input(raw, 0, 0);
    second = half_input(raw, 16, 0);
#else
    const auto order = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0, 1,
                                        4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    const auto split = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(raw, order), 0xd8);
    first = _mm256_cvtph_ps(_mm256_castsi256_si128(split));
    second = _mm256_cvtph_ps(_mm256_extracti128_si256(split, 1));
#endif
  }
}

// Each lane's four B bytes occupy a single VGPR. A four-half group spans two.
template <int Bits, bool Bf>
inline Swmmac1250Vec swmmac1250_read_b(const uint32_t *lo, const uint32_t *hi) {
#if defined(__AVX512F__)
  auto x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lo));
  if constexpr (Bits == 8) {
    if constexpr (Bf) {
      return _mm512_cvtph_ps(_mm256_slli_epi16(_mm256_cvtepu8_epi16(x), 8));
    }
    return widen_fp8<false>(_mm512_cvtepu8_epi32(x));
  } else {
    auto y = _mm_loadu_si128(reinterpret_cast<const __m128i *>(hi));
    auto raw = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi32(x, y)),
                                       _mm_unpackhi_epi32(x, y), 1);
    return Bf ? _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16))
              : _mm512_cvtph_ps(raw);
  }
#else
  auto x = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(lo));
  if constexpr (Bits == 8) {
    if constexpr (Bf) {
      return _mm256_cvtph_ps(_mm_slli_epi16(_mm_cvtepu8_epi16(x), 8));
    }
    return widen_fp8<false>(_mm256_cvtepu8_epi32(x));
  } else {
    auto y = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(hi));
    auto raw = _mm_unpacklo_epi32(x, y);
    return Bf ? _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16))
              : _mm256_cvtph_ps(raw);
  }
#endif
}

inline Swmmac1250Vec swmmac1250_fma(Swmmac1250Vec a, Swmmac1250Vec b, Gfx1250PackedWords selector,
                                    Swmmac1250Vec acc) {
#if defined(__AVX512F__)
  return _mm512_fmadd_ps(a, _mm512_permutexvar_ps(bit_cast<__m512i>(selector), b), acc);
#else
  return _mm256_fmadd_ps(a, _mm256_permutevar8x32_ps(b, bit_cast<__m256i>(selector)), acc);
#endif
}

template <int Bits, bool BfA, bool BfB>
[[gnu::noinline]] bool swmmac1250_reduce_simd(const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *index, float result[16][16]) {
  constexpr int K = Bits == 16 ? 64 : 128, per_word = 32 / Bits, width = swmmac1250_width;
  constexpr int group_columns = width / 4;
  constexpr bool staged = Bits == 8 && width == 8;
  // AVX2 visits each byte-input tile in four blocks. Widen it once for reuse.
  alignas(64) float left[staged ? K / 2 : 1][16], right[staged ? K / 4 : 1][64];
  if constexpr (staged) {
    for (int group = 0; group < K / 4; group++) {
      int aword = swmmac1250_input_index<Bits, K / 8>(0, 2 * group) / per_word;
      int bword = swmmac1250_input_index<Bits, K / 4>(0, 4 * group) / per_word;
      for (int row = 0; row < 16; row += width) {
        Swmmac1250Vec x, y;
        swmmac1250_read_a<Bits, BfA>(a[aword / 32] + aword % 32 + row, group, x, y);
        std::memcpy(left[2 * group] + row, &x, sizeof(x));
        std::memcpy(left[2 * group + 1] + row, &y, sizeof(y));
      }
      for (int col = 0; col < 16; col += group_columns) {
        auto p = b[bword / 32] + bword % 32 + col;
        auto value = swmmac1250_read_b<Bits, BfB>(p, p);
        std::memcpy(right[group] + 4 * col, &value, sizeof(value));
      }
    }
  }
  alignas(64) float columns[16][16];
  swmmac1250_transpose(columns, result);
  unsigned nan = 0;
  for (int row = 0; row < 16; row += width) {
    for (int start = 0; start < 16; start += width) {
      Swmmac1250Vec acc[width];
      for (int j = 0; j < width; j++) {
        std::memcpy(&acc[j], columns[start + j] + row, sizeof(acc[j]));
      }
      for (int block = 0; block < K / 4; block += 8) {
        Gfx1250PackedWords metadata;
        std::memcpy(&metadata, index[(block % (K / 8)) / 8] + 16 * (block / (K / 8)) + row,
                    sizeof(metadata));
        RJ_ARITH_UNROLL(2)
        for (int group = block; group < block + 8; group++) {
          Swmmac1250Vec x, y;
          if constexpr (staged) {
            std::memcpy(&x, left[2 * group] + row, sizeof(x));
            std::memcpy(&y, left[2 * group + 1] + row, sizeof(y));
          } else {
            int word = swmmac1250_input_index<Bits, K / 8>(0, 2 * group) / per_word;
            swmmac1250_read_a<Bits, BfA>(a[word / 32] + word % 32 + row, group, x, y);
          }
          auto first = metadata & 3u, second = (metadata >> 2) & 3u;
          metadata >>= 4;
          RJ_ARITH_UNROLL(8)
          for (int j = 0; j < width; j += group_columns) {
            Swmmac1250Vec bv;
            if constexpr (staged) {
              std::memcpy(&bv, right[group] + 4 * (start + j), sizeof(bv));
            } else {
              int word = swmmac1250_input_index<Bits, K / 4>(0, 4 * group) / per_word;
              auto *lo = b[word / 32] + word % 32 + start + j;
              const uint32_t *hi = lo;
              if constexpr (Bits == 16) {
                hi = b[word / 32 + 1] + word % 32 + start + j;
              }
              bv = swmmac1250_read_b<Bits, BfB>(lo, hi);
            }
            RJ_ARITH_UNROLL(4)
            for (int c = 0; c < group_columns; c++) {
              acc[j + c] = swmmac1250_fma(x, bv, first + uint32_t(4 * c), acc[j + c]);
              acc[j + c] = swmmac1250_fma(y, bv, second + uint32_t(4 * c), acc[j + c]);
            }
          }
        }
      }
      for (int j = 0; j < width; j++) {
#if defined(__AVX512F__)
        nan |= _mm512_cmp_ps_mask(acc[j], acc[j], _CMP_UNORD_Q);
#else
        nan |= unsigned(_mm256_movemask_ps(_mm256_cmp_ps(acc[j], acc[j], _CMP_UNORD_Q)));
#endif
        std::memcpy(columns[start + j] + row, &acc[j], sizeof(acc[j]));
      }
    }
  }
  swmmac1250_transpose(result, columns);
  return nan != 0;
}

// Once an accumulator becomes NaN, only a later selected NaN in A or B can
// change its payload. At the same term, A has priority over B. A NaN made
// solely by arithmetic has the matrix FMA's canonical negative encoding.
template <int Bits>
[[gnu::noinline]] void swmmac1250_repair_nan(const float *left, const float *right,
                                             const uint32_t *const *index,
                                             const float initial[16][16], float result[16][16]) {
  constexpr int K = Bits == 16 ? 64 : 128;
  for (int row = 0; row < 16; ++row) {
    uint32_t values[K / 2];
    int selected[K / 2];
    for (int t = 0; t < K / 2; ++t) {
      values[t] = as_bits(left[swmmac1250_input_index<Bits, K / 8>(row, t)]);
      const int lane = row + 16 * (t / (K / 4)), local = t % (K / 4);
      const int k = 4 * (t / 2) + int(index[local / 16][lane] >> (2 * (local % 16)) & 3);
      selected[t] = swmmac1250_input_index<Bits, K / 4>(0, k);
    }
    for (int col = 0; col < 16; ++col) {
      if ((as_bits(result[row][col]) & 0x7fffffffU) <= 0x7f800000U) {
        continue;
      }
      uint32_t bits = as_bits(initial[row][col]);
      if ((bits & 0x7fffffffU) <= 0x7f800000U) {
        bits = 0xffc00000U;
      }
      for (int t = K / 2 - 1; t >= 0; --t) {
        if ((values[t] & 0x7fffffffU) > 0x7f800000U) {
          bits = values[t];
          break;
        }
        const uint32_t b = as_bits(right[selected[t] + col * (32 / Bits)]);
        if ((b & 0x7fffffffU) > 0x7f800000U) {
          bits = b;
          break;
        }
      }
      result[row][col] = as_float(bits | 0x00400000U);
    }
  }
}

template <int Bits, bool BfA, bool BfB>
[[gnu::noinline]] void swmmac1250_repair_inputs(const uint32_t *const *a, const uint32_t *const *b,
                                                const uint32_t *const *index,
                                                const float initial[16][16], float result[16][16]) {
  constexpr int K = Bits == 16 ? 64 : 128;
  alignas(64) float left[16 * K / 2], right[16 * K];
  smfmac_decode<16 * K / 2, Bits, BfA, false, 32>(a, left);
  smfmac_decode<16 * K, Bits, BfB, false, 32>(b, right);
  swmmac1250_repair_nan<Bits>(left, right, index, initial, result);
}

template <int Bits, bool BfA, bool BfB, int Form>
void swmmac1250_float_vector(uint64_t flags, uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b, const uint32_t *const *index) {
  constexpr auto form = Form == 1   ? Wmma1250Form::F16
                        : Form == 2 ? Wmma1250Form::Bf16
                        : Form == 3 ? Wmma1250Form::Bf16F32
                                    : Wmma1250Form::F32F16;
  alignas(64) float result[16][16];
  gfx1250_load_accumulators<form>(d, 0, result);
  if (swmmac1250_reduce_simd<Bits, BfA, BfB>(a, b, index, result)) {
    alignas(64) float initial[16][16];
    gfx1250_load_accumulators<form>(d, 0, initial);
    swmmac1250_repair_inputs<Bits, BfA, BfB>(a, b, index, initial, result);
  }
  if constexpr (Form == 0) {
    for (int reg = 0; reg < 8; reg++) {
      std::memcpy(d[reg], result[reg], sizeof(result[reg]));
      std::memcpy(d[reg] + 16, result[reg + 8], sizeof(result[reg + 8]));
    }
  } else {
    gfx1250_store_results<form, true>(flags, d, result);
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
