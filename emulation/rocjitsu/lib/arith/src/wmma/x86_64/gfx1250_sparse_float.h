// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 floating sparse WMMA. SIMD lanes hold output rows. Each packed
// B vector has a four-element K group for each of two or four columns.
#pragma once
#include "common/compiler.h"
#include "wmma/gfx1250_sparse_float.h"
#include "wmma/x86_64/gfx1250_accumulator.h"
#include "wmma/x86_64/gfx1250_bf8.h"
#include "wmma/x86_64/gfx1250_fp8_decode.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
using Swmmac1250Int = __m512i;
#else
using Swmmac1250Int = __m256i;
#endif
using Swmmac1250Vec = Gfx1250PackedFloat;
constexpr int swmmac1250_width = sizeof(Swmmac1250Vec) / sizeof(float);

#if defined(RJ_ARITH_TARGET_X86_64_V4)
// Transpose 16 row-major ZMM registers into 16 column-major ZMM registers in place.
[[gnu::always_inline]] inline void swmmac1250_transpose_regs(const __m512 src[16], __m512 dst[16]) {
  __m512 t[16], u[16];
  RJ_ARITH_UNROLL(8)
  for (int i = 0; i < 8; ++i) {
    t[2 * i] = _mm512_unpacklo_ps(src[2 * i], src[2 * i + 1]);
    t[2 * i + 1] = _mm512_unpackhi_ps(src[2 * i], src[2 * i + 1]);
  }
  RJ_ARITH_UNROLL(4)
  for (int g = 0; g < 4; ++g) {
    u[4 * g + 0] = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(t[4 * g]), _mm512_castps_pd(t[4 * g + 2])));
    u[4 * g + 1] = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(t[4 * g]), _mm512_castps_pd(t[4 * g + 2])));
    u[4 * g + 2] = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(t[4 * g + 1]), _mm512_castps_pd(t[4 * g + 3])));
    u[4 * g + 3] = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(t[4 * g + 1]), _mm512_castps_pd(t[4 * g + 3])));
  }
  RJ_ARITH_UNROLL(4)
  for (int j = 0; j < 4; ++j) {
    __m512 v0 = _mm512_shuffle_f32x4(u[j], u[4 + j], 0x88);
    __m512 v1 = _mm512_shuffle_f32x4(u[j], u[4 + j], 0xDD);
    __m512 w0 = _mm512_shuffle_f32x4(u[8 + j], u[12 + j], 0x88);
    __m512 w1 = _mm512_shuffle_f32x4(u[8 + j], u[12 + j], 0xDD);
    dst[j] = _mm512_shuffle_f32x4(v0, w0, 0x88);
    dst[8 + j] = _mm512_shuffle_f32x4(v0, w0, 0xDD);
    dst[4 + j] = _mm512_shuffle_f32x4(v1, w1, 0x88);
    dst[12 + j] = _mm512_shuffle_f32x4(v1, w1, 0xDD);
  }
}
#else
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
#endif

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
#if defined(RJ_ARITH_TARGET_X86_64_V4)
      const __m512i bf16 =
          gfx1250_fp8_to_bf16_pairs<false>(_mm512_cvtepi32_epi16(bit_cast<__m512i>(packed)));
      first = _mm512_castsi512_ps(_mm512_slli_epi32(bf16, 16));
      second = _mm512_castsi512_ps(_mm512_and_si512(bf16, _mm512_set1_epi32(int(0xffff0000U))));
#else
      gfx1250_widen_fp8_pair_v3(bit_cast<__m256i>(packed), first, second);
#endif
    }
  } else if constexpr (Bf) {
    first = bit_cast<Swmmac1250Vec>(packed << 16);
    second = bit_cast<Swmmac1250Vec>(packed & 0xffff0000u);
  } else {
    auto raw = bit_cast<Swmmac1250Int>(packed);
#if defined(RJ_ARITH_TARGET_X86_64_V4)
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
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  auto x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lo));
  if constexpr (Bits == 8) {
    if constexpr (Bf) {
      return _mm512_cvtph_ps(_mm256_slli_epi16(_mm256_cvtepu8_epi16(x), 8));
    }
    return gfx1250_widen_fp8_16_v4(x);
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
    return gfx1250_widen_fp8_8_v3(x);
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
#if defined(RJ_ARITH_TARGET_X86_64_V4)
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
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  __m512 rows[16], acc[16];
  RJ_ARITH_UNROLL(16)
  for (int r = 0; r < 16; ++r) {
    rows[r] = _mm512_load_ps(result[r]);
  }
  swmmac1250_transpose_regs(rows, acc);
  for (int block = 0; block < K / 4; block += 8) {
    Gfx1250PackedWords metadata;
    std::memcpy(&metadata, index[(block % (K / 8)) / 8] + 16 * (block / (K / 8)), sizeof(metadata));
    RJ_ARITH_UNROLL(2)
    for (int group = block; group < block + 8; group++) {
      Swmmac1250Vec x, y;
      int aword = swmmac1250_input_index<Bits, K / 8>(0, 2 * group) / per_word;
      swmmac1250_read_a<Bits, BfA>(a[aword / 32] + aword % 32, group, x, y);
      const Gfx1250PackedWords first = metadata & 3u;
      const Gfx1250PackedWords second = (metadata >> 2) & 3u;
      metadata >>= 4;
      const int bword = swmmac1250_input_index<Bits, K / 4>(0, 4 * group) / per_word;
      if constexpr (Bits == 8 && !BfB) {
        // Widen 8 columns (32 FP8 bytes) per vpermi2w LUT pass, then unpack to two ZMMs.
        RJ_ARITH_UNROLL(2)
        for (int j = 0; j < 16; j += 8) {
          const uint32_t *lo = b[bword / 32] + bword % 32 + j;
          const __m512i bf16 = gfx1250_fp8_to_bf16_pairs<false>(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lo)));
          const __m512 bv0 = _mm512_castsi512_ps(
              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(bf16)), 16));
          const __m512 bv1 = _mm512_castsi512_ps(
              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(bf16, 1)), 16));
          RJ_ARITH_UNROLL(4)
          for (int c = 0; c < 4; ++c) {
            acc[j + c] = swmmac1250_fma(x, bv0, first + static_cast<uint32_t>(4 * c), acc[j + c]);
            acc[j + c] = swmmac1250_fma(y, bv0, second + static_cast<uint32_t>(4 * c), acc[j + c]);
          }
          RJ_ARITH_UNROLL(4)
          for (int c = 0; c < 4; ++c) {
            acc[j + 4 + c] =
                swmmac1250_fma(x, bv1, first + static_cast<uint32_t>(4 * c), acc[j + 4 + c]);
            acc[j + 4 + c] =
                swmmac1250_fma(y, bv1, second + static_cast<uint32_t>(4 * c), acc[j + 4 + c]);
          }
        }
      } else {
        RJ_ARITH_UNROLL(4)
        for (int j = 0; j < 16; j += 4) {
          const uint32_t *lo = b[bword / 32] + bword % 32 + j;
          const uint32_t *hi = lo;
          if constexpr (Bits == 16) {
            hi = b[bword / 32 + 1] + bword % 32 + j;
          }
          const __m512 bv = swmmac1250_read_b<Bits, BfB>(lo, hi);
          RJ_ARITH_UNROLL(4)
          for (int c = 0; c < 4; ++c) {
            acc[j + c] = swmmac1250_fma(x, bv, first + static_cast<uint32_t>(4 * c), acc[j + c]);
            acc[j + c] = swmmac1250_fma(y, bv, second + static_cast<uint32_t>(4 * c), acc[j + c]);
          }
        }
      }
    }
  }
  unsigned nan = 0;
  RJ_ARITH_UNROLL(16)
  for (int j = 0; j < 16; ++j) {
    nan |= _mm512_cmp_ps_mask(acc[j], acc[j], _CMP_UNORD_Q);
  }
  swmmac1250_transpose_regs(acc, rows);
  RJ_ARITH_UNROLL(16)
  for (int r = 0; r < 16; ++r) {
    _mm512_store_ps(result[r], rows[r]);
  }
  return nan != 0;
#else
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
          if constexpr (staged) {
            // Split the two selected terms across the 8 columns so live registers fit
            // in 16 YMMs without spilling acc[7] on GCC 13.
            Swmmac1250Vec x;
            std::memcpy(&x, left[2 * group] + row, sizeof(x));
            const Gfx1250PackedWords first = metadata & 3u;
            const Gfx1250PackedWords first4 = first + 4u;
            RJ_ARITH_UNROLL(4)
            for (int j = 0; j < width; j += 2) {
              Swmmac1250Vec bv;
              std::memcpy(&bv, right[group] + 4 * (start + j), sizeof(bv));
              acc[j + 0] = swmmac1250_fma(x, bv, first, acc[j + 0]);
              acc[j + 1] = swmmac1250_fma(x, bv, first4, acc[j + 1]);
            }
            Swmmac1250Vec y;
            std::memcpy(&y, left[2 * group + 1] + row, sizeof(y));
            const Gfx1250PackedWords second = (metadata >> 2) & 3u;
            const Gfx1250PackedWords second4 = second + 4u;
            metadata >>= 4;
            RJ_ARITH_UNROLL(4)
            for (int j = 0; j < width; j += 2) {
              Swmmac1250Vec bv;
              std::memcpy(&bv, right[group] + 4 * (start + j), sizeof(bv));
              acc[j + 0] = swmmac1250_fma(y, bv, second, acc[j + 0]);
              acc[j + 1] = swmmac1250_fma(y, bv, second4, acc[j + 1]);
            }
          } else {
            Swmmac1250Vec x, y;
            int word = swmmac1250_input_index<Bits, K / 8>(0, 2 * group) / per_word;
            swmmac1250_read_a<Bits, BfA>(a[word / 32] + word % 32 + row, group, x, y);
            const Gfx1250PackedWords first = metadata & 3u;
            const Gfx1250PackedWords second = (metadata >> 2) & 3u;
            metadata >>= 4;
            RJ_ARITH_UNROLL(4)
            for (int j = 0; j < width; j += 2) {
              int bword = swmmac1250_input_index<Bits, K / 4>(0, 4 * group) / per_word;
              const uint32_t *lo = b[bword / 32] + bword % 32 + start + j;
              const uint32_t *hi = b[bword / 32 + 1] + bword % 32 + start + j;
              Swmmac1250Vec bv = swmmac1250_read_b<Bits, BfB>(lo, hi);
              RJ_ARITH_UNROLL(2)
              for (int c = 0; c < 2; c++) {
                acc[j + c] =
                    swmmac1250_fma(x, bv, first + static_cast<uint32_t>(4 * c), acc[j + c]);
                acc[j + c] =
                    swmmac1250_fma(y, bv, second + static_cast<uint32_t>(4 * c), acc[j + c]);
              }
            }
          }
        }
      }
      for (int j = 0; j < width; j++) {
        nan |= unsigned(_mm256_movemask_ps(_mm256_cmp_ps(acc[j], acc[j], _CMP_UNORD_Q)));
        std::memcpy(columns[start + j] + row, &acc[j], sizeof(acc[j]));
      }
    }
  }
  swmmac1250_transpose(result, columns);
  return nan != 0;
#endif
}

template <int Bits, bool Bf> inline bool swmmac1250_raw_is_nan(uint32_t raw) {
  if constexpr (Bits == 8) {
    return Bf ? (raw & 0x7fu) > 0x7cu : (raw & 0x7fu) == 0x7fu;
  } else {
    return Bf ? (raw & 0x7fffu) > 0x7f80u : (raw & 0x7fffu) > 0x7c00u;
  }
}

template <int Bits, bool Bf> inline uint32_t swmmac1250_decode_nan_bits(uint32_t raw) {
  if constexpr (Bits == 8) {
    return as_bits(mfma_decode_fp8<Bf, false>(uint8_t(raw))) | 0x00400000U;
  } else if constexpr (Bf) {
    return (raw << 16) | 0x00400000U;
  } else {
    return as_bits(f16_to_float(uint16_t(raw))) | 0x00400000U;
  }
}

// Test 16 contiguous B VGPR lanes (columns 0..15) for raw NaN in one SIMD pass.
template <int Bits, bool Bf>
inline uint32_t swmmac1250_b_nan_mask16(const uint32_t *b_lanes, int shift) {
  constexpr int mag_mask = Bits == 8 ? 0x7f : 0x7fff;
  constexpr int thresh = Bits == 8 ? (Bf ? 0x7c : 0x7e) : (Bf ? 0x7f80 : 0x7c00);
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  const __m512i w = _mm512_srlv_epi32(_mm512_loadu_si512(b_lanes), _mm512_set1_epi32(shift));
  const __m512i mag = _mm512_and_si512(w, _mm512_set1_epi32(mag_mask));
  return uint32_t(_mm512_cmpgt_epi32_mask(mag, _mm512_set1_epi32(thresh)));
#else
  const __m128i sh = _mm_cvtsi32_si128(shift);
  const __m256i mask = _mm256_set1_epi32(mag_mask);
  const __m256i lim = _mm256_set1_epi32(thresh);
  const __m256i lo = _mm256_and_si256(
      _mm256_srl_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_lanes)), sh), mask);
  const __m256i hi = _mm256_and_si256(
      _mm256_srl_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_lanes + 8)), sh),
      mask);
  const uint32_t mlo =
      uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(lo, lim))));
  const uint32_t mhi =
      uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(hi, lim))));
  return mlo | (mhi << 8);
#endif
}

// Once an accumulator becomes NaN, only a later selected NaN in A or B can
// change its payload. At the same term, A has priority over B. A NaN made
// solely by arithmetic has the matrix FMA's canonical negative encoding.
// Scan only rows with NaN outputs and test all 16 B columns in parallel.
template <int Bits, bool BfA, bool BfB>
[[gnu::noinline]] void swmmac1250_repair_inputs(const uint32_t *const *a, const uint32_t *const *b,
                                                const uint32_t *const *index,
                                                const float initial[16][16], float result[16][16]) {
  constexpr int K = Bits == 16 ? 64 : 128;
  constexpr int per_word = 32 / Bits;
  constexpr uint32_t elem_mask = (1u << Bits) - 1u;
  for (int row = 0; row < 16; ++row) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
    const __m512 r = _mm512_load_ps(result[row]);
    uint32_t nan_cols = uint32_t(_mm512_cmp_ps_mask(r, r, _CMP_UNORD_Q));
#else
    const __m256 r0 = _mm256_load_ps(result[row]);
    const __m256 r1 = _mm256_load_ps(result[row] + 8);
    uint32_t nan_cols = uint32_t(_mm256_movemask_ps(_mm256_cmp_ps(r0, r0, _CMP_UNORD_Q))) |
                        (uint32_t(_mm256_movemask_ps(_mm256_cmp_ps(r1, r1, _CMP_UNORD_Q))) << 8);
#endif
    if (!nan_cols) {
      continue;
    }
    for (int t = K / 2 - 1; t >= 0 && nan_cols; --t) {
      const int a_idx = swmmac1250_input_index<Bits, K / 8>(row, t);
      const uint32_t a_raw =
          (a[a_idx / (32 * per_word)][(a_idx / per_word) % 32] >> (Bits * (a_idx % per_word))) &
          elem_mask;
      if (swmmac1250_raw_is_nan<Bits, BfA>(a_raw)) [[unlikely]] {
        const float val = as_float(swmmac1250_decode_nan_bits<Bits, BfA>(a_raw));
        while (nan_cols) {
          const int col = __builtin_ctz(nan_cols);
          result[row][col] = val;
          nan_cols &= nan_cols - 1u;
        }
        break;
      }
      const int lane = row + 16 * (t / (K / 4)), local = t % (K / 4);
      const int k = 4 * (t / 2) + int((index[local / 16][lane] >> (2 * (local % 16))) & 3u);
      const int b_base = swmmac1250_input_index<Bits, K / 4>(0, k);
      const uint32_t *b_lanes = b[b_base / (32 * per_word)] + (b_base / per_word) % 32;
      const int b_shift = Bits * (b_base % per_word);
      uint32_t hit = swmmac1250_b_nan_mask16<Bits, BfB>(b_lanes, b_shift) & nan_cols;
      if (hit) [[unlikely]] {
        nan_cols &= ~hit;
        while (hit) {
          const int col = __builtin_ctz(hit);
          hit &= hit - 1u;
          const uint32_t b_raw = (b_lanes[col] >> b_shift) & elem_mask;
          result[row][col] = as_float(swmmac1250_decode_nan_bits<Bits, BfB>(b_raw));
        }
      }
    }
    while (nan_cols) {
      const int col = __builtin_ctz(nan_cols);
      nan_cols &= nan_cols - 1u;
      uint32_t bits = as_bits(initial[row][col]);
      if ((bits & 0x7fffffffU) <= 0x7f800000U) {
        bits = 0xffc00000U;
      }
      result[row][col] = as_float(bits | 0x00400000U);
    }
  }
}

template <int Bits, bool BfA, bool BfB, int Form>
void swmmac1250_float_vector(uint64_t flags, uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b, const uint32_t *const *index) {
  constexpr auto form = Form == 1   ? Wmma1250Form::F16
                        : Form == 2 ? Wmma1250Form::Bf16
                        : Form == 3 ? Wmma1250Form::Bf16F32
                                    : Wmma1250Form::F32F16;
  alignas(64) float result[16][16];
  gfx1250_load_sparse_accumulators<form>(d, 0, result);
  if (swmmac1250_reduce_simd<Bits, BfA, BfB>(a, b, index, result)) {
    alignas(64) float initial[16][16];
    gfx1250_load_sparse_accumulators<form>(d, 0, initial);
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
