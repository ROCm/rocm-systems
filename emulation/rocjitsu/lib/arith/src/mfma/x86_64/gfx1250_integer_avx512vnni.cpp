// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 dense and sparse Wave32 IU8 using full-range, non-saturating VNNI.
#include "mfma/backends.h"
#include "mfma/x86_64/gfx1250_integer.h"

namespace rocjitsu::arith {
namespace {
using namespace gfx1250_i8;
// VPDPBUSD takes unsigned B and signed A. Flipping an unsigned A or signed B
// adds a correction from the opposite operand's sum. If both flip, subtract
// their cross term, 64 * 128 * 128, once per accumulator.
void dense_vnni(uint64_t controls, bool clamp, uint32_t *const *d, const uint32_t *const *a,
                const uint32_t *const *b, const uint32_t *const *c) {
  const bool unsigned_a = !(controls & RJ_ARITH_WMMA_SIGNED_A),
             signed_b = (controls & RJ_ARITH_WMMA_SIGNED_B) != 0;
  const __m512i a_bias = _mm512_set1_epi8(unsigned_a ? char(0x80) : char(0));
  const __m512i b_bias = _mm512_set1_epi8(signed_b ? char(0x80) : char(0));
  const __m512i ones = _mm512_set1_epi8(1);
  alignas(64) uint32_t left[16][16], row_adjust[16];
  __m512i right[16], row_sum = _mm512_setzero_si512(), column_sum = _mm512_setzero_si512();
  for (int q = 0; q < 16; ++q) {
    __m512i left_word = _mm512_xor_si512(_mm512_loadu_si512(quartet(a, q)), a_bias);
    _mm512_store_si512(left[q], left_word);
    right[q] = _mm512_xor_si512(_mm512_loadu_si512(quartet(b, q)), b_bias);
    if (signed_b) {
      row_sum = _mm512_dpbusd_epi32(row_sum, ones, left_word);
    }
    if (unsigned_a) {
      column_sum = _mm512_dpbusd_epi32(column_sum, right[q], ones);
    }
  }
  __m512i col_adjust = _mm512_slli_epi32(column_sum, 7);
  if (unsigned_a && signed_b) {
    col_adjust = _mm512_sub_epi32(col_adjust, _mm512_set1_epi32(1 << 20));
  }
  _mm512_store_si512(row_adjust,
                     _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_slli_epi32(row_sum, 7)));
  Tile rows;
  for (int r0 = 0; r0 < 16; r0 += 4) {
    __m512i acc[4];
    for (int i = 0; i < 4; ++i) {
      acc[i] =
          _mm512_add_epi32(col_adjust, _mm512_set1_epi32(bit_cast<int32_t>(row_adjust[r0 + i])));
    }
    for (int q = 0; q < 16; ++q) {
      for (int i = 0; i < 4; ++i) {
        acc[i] = _mm512_dpbusd_epi32(acc[i], right[q],
                                     _mm512_set1_epi32(bit_cast<int32_t>(left[q][r0 + i])));
      }
    }
    for (int i = 0; i < 4; ++i) {
      rows[r0 + i][0] = acc[i];
    }
  }
  finish(clamp, d, c, rows);
}

// Sparse VNNI on the row axis. For quartet q (t = 4q..4q+3) the B words of
// pairs 2q and 2q+1 are interleaved into one qword per column. Broadcasting it and
// shuffling with the per-row control [s0, s1, 4+s2, 4+s3] yields the four
// selected B bytes of every row, which feed one VPDPBUSD against the left words.
// Operand order follows A signedness. Equal source signedness requires flipping
// B and correcting by 128 times the A row sum.
template <bool SignedA>
void sparse_vnni(uint64_t controls, bool clamp, uint32_t *const *d, const uint32_t *const *a,
                 const uint32_t *const *b, const uint32_t *const *c, const uint32_t *const *index) {
  const bool signed_b = (controls & RJ_ARITH_WMMA_SIGNED_B) != 0, bias_b = (SignedA == signed_b);
  const __m512i b_bias = _mm512_set1_epi8(bias_b ? char(0x80) : char(0));
  const __m512i ones = _mm512_set1_epi8(1);
  alignas(64) uint64_t paired_b[16][16];
  for (int q = 0; q < 16; ++q) {
    __m512i w0 = _mm512_xor_si512(_mm512_loadu_si512(sparse_b_quartet(b, 2 * q)), b_bias);
    __m512i w1 = _mm512_xor_si512(_mm512_loadu_si512(sparse_b_quartet(b, 2 * q + 1)), b_bias);
    _mm512_store_si512(paired_b[q], _mm512_unpacklo_epi32(w0, w1));
    _mm512_store_si512(paired_b[q] + 8, _mm512_unpackhi_epi32(w0, w1));
  }
  __m512i controls_by_quartet[16];
  const __m512i lane =
      _mm512_broadcast_i32x4(_mm_setr_epi8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12));
  // Replicate one metadata byte into each dword, split its four selectors,
  // and add four to the high pair to select from the second B word.
  const __m512i shifts = _mm512_set1_epi32(0x00040000),
                selector_bits = _mm512_set1_epi32(0x0C030C03);
  const __m512i second_quartet = _mm512_set1_epi32(0x10040000),
                multiplier = _mm512_set1_epi16(0x4001);
  for (int g = 0; g < 4; ++g) {
    __m512i m = _mm512_loadu_si512(index[g & 1] + 16 * (g >> 1));
    for (int j = 0; j < 4; ++j) {
      __m512i v = _mm512_shuffle_epi8(m, _mm512_add_epi8(lane, _mm512_set1_epi8(char(j))));
      v = _mm512_ternarylogic_epi32(_mm512_srlv_epi16(v, shifts), selector_bits, second_quartet,
                                    0xEA);
      controls_by_quartet[4 * g + j] = _mm512_maddubs_epi16(v, multiplier);
    }
  }
  __m512i row_sum = _mm512_setzero_si512();
  if (bias_b) {
    for (int q = 0; q < 16; ++q) {
      __m512i left_word = _mm512_loadu_si512(quartet(a, q));
      row_sum = SignedA ? _mm512_dpbusd_epi32(row_sum, ones, left_word)
                        : _mm512_dpbusd_epi32(row_sum, left_word, ones);
    }
  }
  row_sum = _mm512_slli_epi32(row_sum, 7);
  if constexpr (SignedA) {
    row_sum = _mm512_sub_epi32(_mm512_setzero_si512(), row_sum);
  }
  Tile columns, rows;
  for (int col = 0; col < 16; ++col) {
    columns[col][0] = row_sum;
  }
  for (int q = 0; q < 16; ++q) {
    const __m512i left_word = _mm512_loadu_si512(quartet(a, q)), control = controls_by_quartet[q];
    for (int col = 0; col < 16; ++col) {
      // Unpacking above interleaves dwords separately in each 128-bit lane.
      const int slot = ((col & 2) ? 8 : 0) + 2 * (col >> 2) + (col & 1);
      __m512i t =
          _mm512_shuffle_epi8(_mm512_set1_epi64(bit_cast<int64_t>(paired_b[q][slot])), control);
      columns[col][0] = SignedA ? _mm512_dpbusd_epi32(columns[col][0], t, left_word)
                                : _mm512_dpbusd_epi32(columns[col][0], left_word, t);
    }
  }
  transpose(columns, rows);
  finish(clamp, d, c, rows);
}

} // namespace
template <bool Sparse>
void gfx1250_integer_avx512vnni(uint64_t controls, bool clamp, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c, const uint32_t *const *index) {
  if constexpr (!Sparse) {
    dense_vnni(controls, clamp, d, a, b, c);
  } else if (controls & RJ_ARITH_WMMA_SIGNED_A) {
    sparse_vnni<true>(controls, clamp, d, a, b, c, index);
  } else {
    sparse_vnni<false>(controls, clamp, d, a, b, c, index);
  }
}
template Gfx1250IntegerBackend gfx1250_integer_avx512vnni<false>;
template Gfx1250IntegerBackend gfx1250_integer_avx512vnni<true>;
} // namespace rocjitsu::arith
