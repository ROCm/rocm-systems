// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 dense and sparse Wave32 IU8 using full-range, non-saturating VNNI.
#include "mfma/backends.h"
#include "mfma/x86_64/gfx1250_integer.h"

namespace rocjitsu::arith {
namespace {
using namespace gfx1250_i8;

// Add each lane of row_adj to the corresponding row accumulator in registers,
// avoiding a 64-byte store followed by 16 4-byte broadcasts (STLF stall).
template <int G>
[[gnu::always_inline]] inline void add_row_adjust_group(__m512i acc[16], __m512i row_adj) {
  __m128i q = _mm512_extracti32x4_epi32(row_adj, G);
  acc[4 * G + 0] = _mm512_add_epi32(acc[4 * G + 0], _mm512_broadcastd_epi32(q));
  acc[4 * G + 1] =
      _mm512_add_epi32(acc[4 * G + 1], _mm512_broadcastd_epi32(_mm_shuffle_epi32(q, 0x55)));
  acc[4 * G + 2] =
      _mm512_add_epi32(acc[4 * G + 2], _mm512_broadcastd_epi32(_mm_shuffle_epi32(q, 0xAA)));
  acc[4 * G + 3] =
      _mm512_add_epi32(acc[4 * G + 3], _mm512_broadcastd_epi32(_mm_shuffle_epi32(q, 0xFF)));
}

[[gnu::always_inline]] inline void add_row_adjust(__m512i acc[16], __m512i row_adj) {
  add_row_adjust_group<0>(acc, row_adj);
  add_row_adjust_group<1>(acc, row_adj);
  add_row_adjust_group<2>(acc, row_adj);
  add_row_adjust_group<3>(acc, row_adj);
}

// VPDPBUSD takes unsigned bytes in operand 1 and signed bytes in operand 2 (m32bcst).
// - s8 x u8: zero bias, A uses {1to16} embedded broadcast directly.
// - u8 x s8: zero bias, operand swap (A in operand 1, B in operand 2).
// - s8 x s8: bias B in registers (s8 -> u8), A keeps {1to16}, subtract 128 * row_sum(A).
// - u8 x u8: bias A into a 1 KB stack buffer (u8 -> s8) so A uses {1to16}, add 128 * col_sum(B).
// Dense A and B pair reg 0..7 lanes 0..15 with 0..15 and lanes 16..31 with 16..31.
template <bool SignedA, bool SignedB>
[[gnu::noinline]] void dense_vnni(uint64_t, bool clamp, uint32_t *const *d,
                                  const uint32_t *const *a, const uint32_t *const *b,
                                  const uint32_t *const *c, const uint32_t *const *) {
  constexpr bool BiasA = (!SignedA && !SignedB);
  constexpr bool BiasB = (SignedA && SignedB);
  constexpr bool SwapOperands = (!SignedA && SignedB);

  const __m512i bias80 = _mm512_set1_epi8(char(0x80));
  const __m512i ones = _mm512_set1_epi8(1);
  alignas(64) uint32_t left_buf[16][16];
  if constexpr (BiasA) {
    for (int reg = 0; reg < 8; ++reg) {
      const uint32_t *ap = a[reg];
      _mm512_store_si512(left_buf[2 * reg], _mm512_xor_si512(_mm512_loadu_si512(ap), bias80));
      _mm512_store_si512(left_buf[2 * reg + 1],
                         _mm512_xor_si512(_mm512_loadu_si512(ap + 16), bias80));
    }
  }

  __m512i acc[16]{};
  __m512i adj_sum = _mm512_setzero_si512();

  for (int reg = 0; reg < 8; ++reg) {
    const uint32_t *ap_base = a[reg];
    const uint32_t *bp_base = b[reg];
    RJ_ARITH_UNROLL(2)
    for (int bank = 0; bank < 2; ++bank) {
      const uint32_t *a_ptr = BiasA ? left_buf[2 * reg + bank] : (ap_base + (bank << 4));
      __m512i right = _mm512_loadu_si512(bp_base + (bank << 4));
      if constexpr (BiasA) {
        adj_sum = _mm512_dpbusd_epi32(adj_sum, right, ones);
      } else if constexpr (BiasB) {
        right = _mm512_xor_si512(right, bias80);
        adj_sum = _mm512_dpbusd_epi32(adj_sum, ones, _mm512_loadu_si512(a_ptr));
      }
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < 16; ++r) {
        __m512i left_b = _mm512_set1_epi32(bit_cast<int32_t>(a_ptr[r]));
        if constexpr (SwapOperands) {
          acc[r] = _mm512_dpbusd_epi32(acc[r], left_b, right);
        } else {
          acc[r] = _mm512_dpbusd_epi32(acc[r], right, left_b);
        }
      }
    }
  }

  if constexpr (BiasA) {
    __m512i col_adj = _mm512_slli_epi32(adj_sum, 7);
    RJ_ARITH_UNROLL(16)
    for (int r = 0; r < 16; ++r) {
      acc[r] = _mm512_add_epi32(acc[r], col_adj);
    }
  } else if constexpr (BiasB) {
    __m512i row_adj = _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_slli_epi32(adj_sum, 7));
    add_row_adjust(acc, row_adj);
  }
  finish_regs(clamp, d, c, acc);
}

// Transpose 16 column-major ZMM registers into 16 row-major ZMM registers in place.
[[gnu::always_inline]] inline void transpose_regs_vnni(const __m512i columns[16],
                                                       __m512i rows[16]) {
  __m512i t[16], u[16];
  RJ_ARITH_UNROLL(8)
  for (int i = 0; i < 8; ++i) {
    t[2 * i] = _mm512_unpacklo_epi32(columns[2 * i], columns[2 * i + 1]);
    t[2 * i + 1] = _mm512_unpackhi_epi32(columns[2 * i], columns[2 * i + 1]);
  }
  RJ_ARITH_UNROLL(4)
  for (int g = 0; g < 4; ++g) {
    u[4 * g + 0] = _mm512_unpacklo_epi64(t[4 * g], t[4 * g + 2]);
    u[4 * g + 1] = _mm512_unpackhi_epi64(t[4 * g], t[4 * g + 2]);
    u[4 * g + 2] = _mm512_unpacklo_epi64(t[4 * g + 1], t[4 * g + 3]);
    u[4 * g + 3] = _mm512_unpackhi_epi64(t[4 * g + 1], t[4 * g + 3]);
  }
  RJ_ARITH_UNROLL(4)
  for (int j = 0; j < 4; ++j) {
    __m512i v0 = _mm512_shuffle_i32x4(u[j], u[4 + j], 0x88);
    __m512i v1 = _mm512_shuffle_i32x4(u[j], u[4 + j], 0xDD);
    __m512i w0 = _mm512_shuffle_i32x4(u[8 + j], u[12 + j], 0x88);
    __m512i w1 = _mm512_shuffle_i32x4(u[8 + j], u[12 + j], 0xDD);
    rows[j] = _mm512_shuffle_i32x4(v0, w0, 0x88);
    rows[8 + j] = _mm512_shuffle_i32x4(v0, w0, 0xDD);
    rows[4 + j] = _mm512_shuffle_i32x4(v1, w1, 0x88);
    rows[12 + j] = _mm512_shuffle_i32x4(v1, w1, 0xDD);
  }
}

// Sparse VNNI on the row axis. For quartet q (t = 4q..4q+3) the B words of
// pairs 2q and 2q+1 are interleaved into one qword per column. Broadcasting it
// from L1 and shuffling with the per-row control [s0, s1, 4+s2, 4+s3] yields
// the four selected B bytes of every row, feeding one VPDPBUSD against A.
// Operand order follows A signedness; equal signedness biases B and corrects
// by +-128 times the A row sum.
template <bool SignedA, bool BiasB>
[[gnu::noinline]] void sparse_vnni(uint64_t, bool clamp, uint32_t *const *d,
                                   const uint32_t *const *a, const uint32_t *const *b,
                                   const uint32_t *const *c, const uint32_t *const *index) {
  alignas(64) uint64_t paired_b[16][16];
  const __m512i b_bias = _mm512_set1_epi8(char(0x80));
  for (int half = 0; half < 2; ++half) {
    for (int sub = 0; sub < 4; ++sub) {
      const uint32_t *b0p = b[(half << 3) | (sub << 1)];
      const uint32_t *b1p = b[(half << 3) | (sub << 1) | 1];
      const int q0 = (half << 3) | sub;
      const int q1 = q0 | 4;
      __m512i w00 = _mm512_loadu_si512(b0p);
      __m512i w01 = _mm512_loadu_si512(b1p);
      __m512i w10 = _mm512_loadu_si512(b0p + 16);
      __m512i w11 = _mm512_loadu_si512(b1p + 16);
      if constexpr (BiasB) {
        w00 = _mm512_xor_si512(w00, b_bias);
        w01 = _mm512_xor_si512(w01, b_bias);
        w10 = _mm512_xor_si512(w10, b_bias);
        w11 = _mm512_xor_si512(w11, b_bias);
      }
      _mm512_store_si512(paired_b[q0], _mm512_unpacklo_epi32(w00, w01));
      _mm512_store_si512(paired_b[q0] + 8, _mm512_unpackhi_epi32(w00, w01));
      _mm512_store_si512(paired_b[q1], _mm512_unpacklo_epi32(w10, w11));
      _mm512_store_si512(paired_b[q1] + 8, _mm512_unpackhi_epi32(w10, w11));
    }
  }

  const __m512i ones = _mm512_set1_epi8(1);
  const __m512i lane =
      _mm512_broadcast_i32x4(_mm_setr_epi8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12));
  const __m512i shifts = _mm512_set1_epi32(0x00040000);
  const __m512i selector_bits = _mm512_set1_epi32(0x0C030C03);
  const __m512i second_quartet = _mm512_set1_epi32(0x10040000);
  const __m512i multiplier = _mm512_set1_epi16(0x4001);

  __m512i columns[16]{};
  __m512i row_sum = _mm512_setzero_si512();

  for (int g = 0; g < 4; ++g) {
    __m512i m = _mm512_loadu_si512(index[g & 1] + ((g & 2) << 3));
    for (int sub = 0; sub < 4; ++sub) {
      const int q = (g << 2) | sub;
      __m512i v = _mm512_shuffle_epi8(m, lane);
      m = _mm512_srli_epi32(m, 8);
      v = _mm512_ternarylogic_epi32(_mm512_srlv_epi16(v, shifts), selector_bits, second_quartet,
                                    0xEA);
      const __m512i control = _mm512_maddubs_epi16(v, multiplier);
      const __m512i left_word = _mm512_loadu_si512(a[((g & 2) << 1) | sub] + ((g & 1) << 4));
      if constexpr (BiasB) {
        row_sum = SignedA ? _mm512_dpbusd_epi32(row_sum, ones, left_word)
                          : _mm512_dpbusd_epi32(row_sum, left_word, ones);
      }
      const uint64_t *pb = paired_b[q];
      RJ_ARITH_UNROLL(16)
      for (int col = 0; col < 16; ++col) {
        // Unpacking above interleaves dwords separately in each 128-bit lane.
        const int slot = ((col & 2) ? 8 : 0) + 2 * (col >> 2) + (col & 1);
        __m512i t = _mm512_shuffle_epi8(_mm512_set1_epi64(bit_cast<int64_t>(pb[slot])), control);
        columns[col] = SignedA ? _mm512_dpbusd_epi32(columns[col], t, left_word)
                               : _mm512_dpbusd_epi32(columns[col], left_word, t);
      }
    }
  }

  if constexpr (BiasB) {
    row_sum = _mm512_slli_epi32(row_sum, 7);
    if constexpr (SignedA) {
      row_sum = _mm512_sub_epi32(_mm512_setzero_si512(), row_sum);
    }
    RJ_ARITH_UNROLL(16)
    for (int col = 0; col < 16; ++col) {
      columns[col] = _mm512_add_epi32(columns[col], row_sum);
    }
  }

  __m512i rows[16];
  transpose_regs_vnni(columns, rows);
  finish_regs(clamp, d, c, rows);
}

} // namespace

template <bool Sparse>
void gfx1250_integer_avx512vnni(uint64_t controls, bool clamp, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c, const uint32_t *const *index) {
  const unsigned sign_key = unsigned(controls & (RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B));
  if constexpr (!Sparse) {
    switch (sign_key) {
    case 0:
      return dense_vnni<false, false>(controls, clamp, d, a, b, c, index);
    case RJ_ARITH_WMMA_SIGNED_A:
      return dense_vnni<true, false>(controls, clamp, d, a, b, c, index);
    case RJ_ARITH_WMMA_SIGNED_B:
      return dense_vnni<false, true>(controls, clamp, d, a, b, c, index);
    default:
      return dense_vnni<true, true>(controls, clamp, d, a, b, c, index);
    }
  } else {
    switch (sign_key) {
    case 0:
      return sparse_vnni<false, true>(controls, clamp, d, a, b, c, index);
    case RJ_ARITH_WMMA_SIGNED_A:
      return sparse_vnni<true, false>(controls, clamp, d, a, b, c, index);
    case RJ_ARITH_WMMA_SIGNED_B:
      return sparse_vnni<false, false>(controls, clamp, d, a, b, c, index);
    default:
      return sparse_vnni<true, true>(controls, clamp, d, a, b, c, index);
    }
  }
}
template Gfx1250IntegerBackend gfx1250_integer_avx512vnni<false>;
template Gfx1250IntegerBackend gfx1250_integer_avx512vnni<true>;
} // namespace rocjitsu::arith
