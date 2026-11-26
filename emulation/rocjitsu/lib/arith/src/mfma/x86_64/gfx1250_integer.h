// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed gfx1250 Wave32 byte matrix kernels. Dense vectors span output columns. Sparse vectors span
// rows to select B bytes with a byte shuffle.
#pragma once

#include "common/bits.h"
#include "common/compiler.h"
#include "mfma/sparse_integer.h"
#if defined(RJ_ARITH_TARGET_X86_64_V4)
#include "common/x86_64/alu_x86_64_v4.h"
#else
#include "common/x86_64/alu_x86_64_v3.h"
#endif

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::gfx1250_i8 {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
constexpr int Width = 16;
using Native = __m512i;
#else
constexpr int Width = 8;
using Native = __m256i;
#endif
constexpr int Parts = 16 / Width;
using Words [[gnu::vector_size(Width * 4)]] = uint32_t;
using Halves [[gnu::vector_size(Width * 4)]] = uint16_t;
using SignedHalves [[gnu::vector_size(Width * 4)]] = int16_t;
using Tile = Native[16][Parts];

// Each quartet is one VGPR word per row of A or column of dense B.
inline const uint32_t *quartet(const uint32_t *const *source, int q) {
  const int word = matrix_i8_index<16, 64, 1, true>(0, 4 * q, 0, 0, true) / 4;
  return source[word / 32] + word % 32;
}
inline const uint32_t *sparse_b_quartet(const uint32_t *const *b, int q) {
  const int word = sparse_matrix_b_index<16, true>(0, 4 * q) / 4;
  return b[word / 32] + word % 32;
}
inline Native load(const uint32_t *p) {
  Native v;
  __builtin_memcpy(&v, p, sizeof(v));
  return v;
}
inline void store(uint32_t *p, Native v) { __builtin_memcpy(p, &v, sizeof(v)); }

template <bool Odd, bool Signed> inline Native extend(Native v) {
  auto words = bit_cast<Halves>(v);
  if constexpr (Signed) {
    if constexpr (!Odd) {
      words <<= 8;
    }
    return bit_cast<Native>(bit_cast<SignedHalves>(words) >> 8);
  } else if constexpr (Odd) {
    return bit_cast<Native>(words >> 8);
  } else {
    return bit_cast<Native>(words & 255u);
  }
}

#if defined(RJ_ARITH_TARGET_X86_64_V4)
// The 64-product byte dot fits in signed 24 bits for every signedness combination.
// Add C before publishing D in ascending row order so all whole-VGPR aliases hold.
[[gnu::always_inline]] inline void finish_regs(bool clamp, uint32_t *const *d,
                                               const uint32_t *const *c, __m512i rows[16]) {
  if (!clamp) {
    RJ_ARITH_UNROLL(16)
    for (int r = 0; r < 16; ++r) {
      rows[r] = _mm512_add_epi32(rows[r], _mm512_loadu_si512(c[r & 7] + ((r & 8) << 1)));
    }
  } else {
    RJ_ARITH_UNROLL(16)
    for (int r = 0; r < 16; ++r) {
      __m512i acc = _mm512_loadu_si512(c[r & 7] + ((r & 8) << 1));
#if defined(RJ_ARITH_TARGET_AVX512VNNI)
      rows[r] = saturate_signed_sum_24(acc, rows[r]);
#else
      __m512i sum = _mm512_add_epi32(acc, rows[r]);
      rows[r] = saturate_signed_sum(acc, rows[r], sum);
#endif
    }
  }
  RJ_ARITH_UNROLL(16)
  for (int r = 0; r < 16; ++r) {
    _mm512_storeu_si512(d[r & 7] + ((r & 8) << 1), rows[r]);
  }
}

// Dense A and B pair reg 0..7 lanes 0..15 with 0..15 and lanes 16..31 with 16..31.
template <bool Signed>
inline void unpack_dense_v4(const uint32_t *const *src, uint32_t even[16][16],
                            uint32_t odd[16][16]) {
  for (int reg = 0; reg < 8; ++reg) {
    const uint32_t *sp = src[reg];
    __m512i v0 = _mm512_loadu_si512(sp);
    __m512i v1 = _mm512_loadu_si512(sp + 16);
    _mm512_store_si512(even[2 * reg], extend<false, Signed>(v0));
    _mm512_store_si512(odd[2 * reg], extend<true, Signed>(v0));
    _mm512_store_si512(even[2 * reg + 1], extend<false, Signed>(v1));
    _mm512_store_si512(odd[2 * reg + 1], extend<true, Signed>(v1));
  }
}

// Two 8-row passes keep all 16 output rows in ZMM registers with zero accumulator spills.
[[gnu::noinline]] inline void
dense_v4_reduce(bool clamp, uint32_t *const *d, const uint32_t *const *c,
                const uint32_t left_even[16][16], const uint32_t left_odd[16][16],
                const uint32_t right_even[16][16], const uint32_t right_odd[16][16]) {
  __m512i rows[16]{};
  for (int q = 0; q < 16; ++q) {
    __m512i re = _mm512_load_si512(right_even[q]);
    __m512i ro = _mm512_load_si512(right_odd[q]);
    RJ_ARITH_UNROLL(8)
    for (int i = 0; i < 8; ++i) {
      __m512i e = _mm512_set1_epi32(bit_cast<int32_t>(left_even[q][i]));
      __m512i o = _mm512_set1_epi32(bit_cast<int32_t>(left_odd[q][i]));
      rows[i] = _mm512_add_epi32(rows[i], _mm512_madd_epi16(re, e));
      rows[i] = _mm512_add_epi32(rows[i], _mm512_madd_epi16(ro, o));
    }
  }
  for (int q = 0; q < 16; ++q) {
    __m512i re = _mm512_load_si512(right_even[q]);
    __m512i ro = _mm512_load_si512(right_odd[q]);
    RJ_ARITH_UNROLL(8)
    for (int i = 0; i < 8; ++i) {
      __m512i e = _mm512_set1_epi32(bit_cast<int32_t>(left_even[q][8 + i]));
      __m512i o = _mm512_set1_epi32(bit_cast<int32_t>(left_odd[q][8 + i]));
      rows[8 + i] = _mm512_add_epi32(rows[8 + i], _mm512_madd_epi16(re, e));
      rows[8 + i] = _mm512_add_epi32(rows[8 + i], _mm512_madd_epi16(ro, o));
    }
  }
  finish_regs(clamp, d, c, rows);
}

inline void dense_v4(uint64_t controls, bool clamp, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c) {
  alignas(64) uint32_t left_even[16][16], left_odd[16][16];
  alignas(64) uint32_t right_even[16][16], right_odd[16][16];
  if (controls & RJ_ARITH_WMMA_SIGNED_A) {
    unpack_dense_v4<true>(a, left_even, left_odd);
  } else {
    unpack_dense_v4<false>(a, left_even, left_odd);
  }
  if (controls & RJ_ARITH_WMMA_SIGNED_B) {
    unpack_dense_v4<true>(b, right_even, right_odd);
  } else {
    unpack_dense_v4<false>(b, right_even, right_odd);
  }
  dense_v4_reduce(clamp, d, c, left_even, left_odd, right_even, right_odd);
}

#else
template <bool Signed>
inline void unpack_dense_left_v3(const uint32_t *const *src, uint32_t even[16][16],
                                 uint32_t odd[16][16]) {
  for (int reg = 0; reg < 8; ++reg) {
    const uint32_t *sp = src[reg];
    for (int bank = 0; bank < 2; ++bank) {
      const int q = 2 * reg + bank;
      __m256i v0 = load(sp + 16 * bank), v1 = load(sp + 16 * bank + 8);
      store(even[q], extend<false, Signed>(v0));
      store(even[q] + 8, extend<false, Signed>(v1));
      store(odd[q], extend<true, Signed>(v0));
      store(odd[q] + 8, extend<true, Signed>(v1));
    }
  }
}

template <bool Signed>
inline void unpack_dense_right_v3(const uint32_t *const *src, Tile &even, Tile &odd) {
  for (int reg = 0; reg < 8; ++reg) {
    const uint32_t *sp = src[reg];
    for (int bank = 0; bank < 2; ++bank) {
      const int q = 2 * reg + bank;
      __m256i v0 = load(sp + 16 * bank), v1 = load(sp + 16 * bank + 8);
      even[q][0] = extend<false, Signed>(v0);
      even[q][1] = extend<false, Signed>(v1);
      odd[q][0] = extend<true, Signed>(v0);
      odd[q][1] = extend<true, Signed>(v1);
    }
  }
}

template <int Rows>
[[gnu::always_inline]] inline void
dense_v3_block(int r0, bool clamp, const uint32_t *const *c, const uint32_t left_even[16][16],
               const uint32_t left_odd[16][16], const Tile &right_even, const Tile &right_odd,
               Tile &result) {
  __m256i sums[Rows][2]{};
  for (int q = 0; q < 16; ++q) {
    __m256i re0 = right_even[q][0], re1 = right_even[q][1];
    RJ_ARITH_UNROLL(6)
    for (int i = 0; i < Rows; ++i) {
      __m256i e = _mm256_set1_epi32(bit_cast<int32_t>(left_even[q][r0 + i]));
      sums[i][0] = _mm256_add_epi32(sums[i][0], _mm256_madd_epi16(e, re0));
      sums[i][1] = _mm256_add_epi32(sums[i][1], _mm256_madd_epi16(e, re1));
    }
    __m256i ro0 = right_odd[q][0], ro1 = right_odd[q][1];
    RJ_ARITH_UNROLL(6)
    for (int i = 0; i < Rows; ++i) {
      __m256i o = _mm256_set1_epi32(bit_cast<int32_t>(left_odd[q][r0 + i]));
      sums[i][0] = _mm256_add_epi32(sums[i][0], _mm256_madd_epi16(o, ro0));
      sums[i][1] = _mm256_add_epi32(sums[i][1], _mm256_madd_epi16(o, ro1));
    }
  }
  // Fuse C-read and optional saturation into the buffered result before any D store.
  if (!clamp) {
    RJ_ARITH_UNROLL(6)
    for (int i = 0; i < Rows; ++i) {
      const int r = r0 + i;
      const uint32_t *cp = c[r & 7] + ((r & 8) << 1);
      result[r][0] = _mm256_add_epi32(sums[i][0], load(cp));
      result[r][1] = _mm256_add_epi32(sums[i][1], load(cp + 8));
    }
  } else {
    RJ_ARITH_UNROLL(6)
    for (int i = 0; i < Rows; ++i) {
      const int r = r0 + i;
      const uint32_t *cp = c[r & 7] + ((r & 8) << 1);
      __m256i acc0 = load(cp), acc1 = load(cp + 8);
      result[r][0] = saturate_signed_sum(acc0, sums[i][0], _mm256_add_epi32(acc0, sums[i][0]));
      result[r][1] = saturate_signed_sum(acc1, sums[i][1], _mm256_add_epi32(acc1, sums[i][1]));
    }
  }
}

[[gnu::noinline]] inline void dense_v3_reduce(bool clamp, uint32_t *const *d,
                                              const uint32_t *const *c,
                                              const uint32_t left_even[16][16],
                                              const uint32_t left_odd[16][16],
                                              const Tile &right_even, const Tile &right_odd) {
  Tile result;
  // 6+6+4 blocking uses 12 YMM accumulators + 4 scratch YMMs, reducing B reloads from 4x to 3x.
  dense_v3_block<6>(0, clamp, c, left_even, left_odd, right_even, right_odd, result);
  dense_v3_block<6>(6, clamp, c, left_even, left_odd, right_even, right_odd, result);
  dense_v3_block<4>(12, clamp, c, left_even, left_odd, right_even, right_odd, result);
  for (int r = 0; r < 16; ++r) {
    uint32_t *dp = d[r & 7] + ((r & 8) << 1);
    store(dp, result[r][0]);
    store(dp + 8, result[r][1]);
  }
}

inline void dense_v3(uint64_t controls, bool clamp, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c) {
  alignas(32) uint32_t left_even[16][16], left_odd[16][16];
  Tile right_even, right_odd;
  if (controls & RJ_ARITH_WMMA_SIGNED_A) {
    unpack_dense_left_v3<true>(a, left_even, left_odd);
  } else {
    unpack_dense_left_v3<false>(a, left_even, left_odd);
  }
  if (controls & RJ_ARITH_WMMA_SIGNED_B) {
    unpack_dense_right_v3<true>(b, right_even, right_odd);
  } else {
    unpack_dense_right_v3<false>(b, right_even, right_odd);
  }
  dense_v3_reduce(clamp, d, c, left_even, left_odd, right_even, right_odd);
}

#endif

#if defined(RJ_ARITH_TARGET_X86_64_V4)
[[gnu::always_inline]] inline void transpose(const Tile &columns, Tile &rows) {
  __m512i t[16], u[16];
  for (int i = 0; i < 8; ++i) {
    t[2 * i] = _mm512_unpacklo_epi32(columns[2 * i][0], columns[2 * i + 1][0]);
    t[2 * i + 1] = _mm512_unpackhi_epi32(columns[2 * i][0], columns[2 * i + 1][0]);
  }
  for (int g = 0; g < 4; ++g) {
    u[4 * g + 0] = _mm512_unpacklo_epi64(t[4 * g], t[4 * g + 2]);
    u[4 * g + 1] = _mm512_unpackhi_epi64(t[4 * g], t[4 * g + 2]);
    u[4 * g + 2] = _mm512_unpacklo_epi64(t[4 * g + 1], t[4 * g + 3]);
    u[4 * g + 3] = _mm512_unpackhi_epi64(t[4 * g + 1], t[4 * g + 3]);
  }
  for (int j = 0; j < 4; ++j) {
    __m512i v0 = _mm512_shuffle_i32x4(u[j], u[4 + j], 0x88);
    __m512i v1 = _mm512_shuffle_i32x4(u[j], u[4 + j], 0xDD);
    __m512i w0 = _mm512_shuffle_i32x4(u[8 + j], u[12 + j], 0x88);
    __m512i w1 = _mm512_shuffle_i32x4(u[8 + j], u[12 + j], 0xDD);
    rows[j][0] = _mm512_shuffle_i32x4(v0, w0, 0x88);
    rows[8 + j][0] = _mm512_shuffle_i32x4(v0, w0, 0xDD);
    rows[4 + j][0] = _mm512_shuffle_i32x4(v1, w1, 0x88);
    rows[12 + j][0] = _mm512_shuffle_i32x4(v1, w1, 0xDD);
  }
}

#else
[[gnu::always_inline]] inline void transpose(const Tile &columns, Tile &rows) {
  for (int h = 0; h < 2; ++h) {
    for (int ch = 0; ch < 2; ++ch) {
      __m256i t[8], u[8];
      for (int i = 0; i < 4; ++i) {
        t[2 * i] =
            _mm256_unpacklo_epi32(columns[8 * ch + 2 * i][h], columns[8 * ch + 2 * i + 1][h]);
        t[2 * i + 1] =
            _mm256_unpackhi_epi32(columns[8 * ch + 2 * i][h], columns[8 * ch + 2 * i + 1][h]);
      }
      for (int g = 0; g < 2; ++g) {
        u[4 * g + 0] = _mm256_unpacklo_epi64(t[4 * g], t[4 * g + 2]);
        u[4 * g + 1] = _mm256_unpackhi_epi64(t[4 * g], t[4 * g + 2]);
        u[4 * g + 2] = _mm256_unpacklo_epi64(t[4 * g + 1], t[4 * g + 3]);
        u[4 * g + 3] = _mm256_unpackhi_epi64(t[4 * g + 1], t[4 * g + 3]);
      }
      for (int j = 0; j < 4; ++j) {
        rows[8 * h + j][ch] = _mm256_permute2x128_si256(u[j], u[4 + j], 0x20);
        rows[8 * h + 4 + j][ch] = _mm256_permute2x128_si256(u[j], u[4 + j], 0x31);
      }
    }
  }
}

#endif

// The product sum fits in int32_t for all input signedness combinations. Add C
// only after the dot, with an optional final signed clamp, and read every source
// before publishing D. Row order is observable when destination sections alias.
[[gnu::always_inline]] inline void finish(bool clamp, uint32_t *const *d, const uint32_t *const *c,
                                          Tile &rows) {
  for (int r = 0; r < 16; ++r) {
    for (int p = 0; p < Parts; ++p) {
      auto acc = load(c[r % 8] + 16 * (r / 8) + Width * p);
      auto sum = bit_cast<Native>(bit_cast<Words>(acc) + bit_cast<Words>(rows[r][p]));
      rows[r][p] = clamp ? saturate_signed_sum(acc, rows[r][p], sum) : sum;
    }
  }
  for (int r = 0; r < 16; ++r) {
    for (int p = 0; p < Parts; ++p) {
      store(d[r % 8] + 16 * (r / 8) + Width * p, rows[r][p]);
    }
  }
}

inline Native add(Native a, Native b) {
  return bit_cast<Native>(bit_cast<Words>(a) + bit_cast<Words>(b));
}
inline Native broadcast(uint32_t v) { return bit_cast<Native>(Words{} + v); }
inline Native broadcast128(__m128i v) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  return _mm512_broadcast_i32x4(v);
#else
  return _mm256_broadcastsi128_si256(v);
#endif
}
inline Native shuffle_bytes(Native v, Native control) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  return _mm512_shuffle_epi8(v, control);
#else
  return _mm256_shuffle_epi8(v, control);
#endif
}
inline Native madd(Native a, Native b) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  return _mm512_madd_epi16(a, b);
#else
  return _mm256_madd_epi16(a, b);
#endif
}
// Expand two adjacent compressed A bytes per lane. A 32-bit B broadcast and
// per-row byte shuffle select and zero-extend the matching B bytes to halfwords.
// Flipping signed B to unsigned requires subtracting 128 times the A row sum.
inline void sparse(uint64_t controls, bool clamp, uint32_t *const *d, const uint32_t *const *a,
                   const uint32_t *const *b, const uint32_t *const *c,
                   const uint32_t *const *index) {
  const bool signed_a = controls & RJ_ARITH_WMMA_SIGNED_A,
             signed_b = controls & RJ_ARITH_WMMA_SIGNED_B;
  alignas(64) uint32_t right[32][16];
  Native left[32][Parts], pick[32][Parts], row_sum[Parts]{};
  auto bias = broadcast(signed_b ? 0x80808080u : 0);
  for (int pair = 0; pair < 32; ++pair) {
    for (int p = 0; p < Parts; ++p) {
      store(right[pair] + Width * p, load(sparse_b_quartet(b, pair) + Width * p) ^ bias);
    }
  }
  // A zeroed low byte followed by an arithmetic shift sign-extends a selected
  // byte. For unsigned A, place it directly in the low byte of the halfword.
  auto lo = broadcast128(_mm_setr_epi8(0, -1, 1, -1, 4, -1, 5, -1, 8, -1, 9, -1, 12, -1, 13, -1));
  auto hi = broadcast128(_mm_setr_epi8(2, -1, 3, -1, 6, -1, 7, -1, 10, -1, 11, -1, 14, -1, 15, -1));
  if (signed_a) {
    lo = bit_cast<Native>((bit_cast<Halves>(lo) << 8) | 255u);
    hi = bit_cast<Native>((bit_cast<Halves>(hi) << 8) | 255u);
  }
  for (int q = 0; q < 16; ++q) {
    for (int p = 0; p < Parts; ++p) {
      auto word = load(quartet(a, q) + Width * p), first = shuffle_bytes(word, lo),
           second = shuffle_bytes(word, hi);
      if (signed_a) {
        first = bit_cast<Native>(bit_cast<SignedHalves>(first) >> 8);
        second = bit_cast<Native>(bit_cast<SignedHalves>(second) >> 8);
      }
      left[2 * q][p] = first;
      left[2 * q + 1][p] = second;
      if (signed_b) {
        row_sum[p] = add(row_sum[p],
                         madd(bit_cast<Native>(bit_cast<Halves>(first) + bit_cast<Halves>(second)),
                              broadcast(0x00010001u)));
      }
    }
  }
  for (int group = 0; group < 4; ++group) {
    for (int p = 0; p < Parts; ++p) {
      auto metadata = bit_cast<Words>(load(index[group % 2] + 16 * (group / 2) + Width * p));
      for (int j = 0; j < 8; ++j) {
        auto field = metadata >> (4 * j);
        pick[8 * group + j][p] =
            bit_cast<Native>((field & 3u) | ((field << 14) & 0x30000u) | 0x80008000u);
      }
    }
  }
  constexpr int Columns = Width == 16 ? 16 : 4;
  Tile columns, rows;
  for (int tile = 0; tile < 16; tile += Columns) {
    Native sums[Columns][Parts];
    for (int i = 0; i < Columns; ++i) {
      for (int p = 0; p < Parts; ++p) {
        sums[i][p] = bit_cast<Native>(-(bit_cast<Words>(row_sum[p]) << 7));
      }
    }
    for (int pair = 0; pair < 32; ++pair) {
      for (int i = 0; i < Columns; ++i) {
        auto word = broadcast(right[pair][tile + i]);
        for (int p = 0; p < Parts; ++p) {
          sums[i][p] = add(sums[i][p], madd(shuffle_bytes(word, pick[pair][p]), left[pair][p]));
        }
      }
    }
    for (int i = 0; i < Columns; ++i) {
      for (int p = 0; p < Parts; ++p) {
        columns[tile + i][p] = sums[i][p];
      }
    }
  }
  transpose(columns, rows);
  finish(clamp, d, c, rows);
}
template <bool Sparse>
void compute(uint64_t controls, bool clamp, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c, const uint32_t *const *index) {
  if constexpr (Sparse) {
    sparse(controls, clamp, d, a, b, c, index);
  } else {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
    dense_v4(controls, clamp, d, a, b, c);
#else
    dense_v3(controls, clamp, d, a, b, c);
#endif
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::gfx1250_i8
