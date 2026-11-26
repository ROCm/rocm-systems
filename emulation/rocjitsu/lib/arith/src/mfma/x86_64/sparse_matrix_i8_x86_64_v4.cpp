// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v4 sparse byte matrix entry points.
#include "common/compiler.h"
#include "mfma/backends.h"
#include "mfma/sparse_integer.h"
#include "mfma/x86_64/sparse_matrix_i8_common.h"
#include <immintrin.h>

namespace rocjitsu::arith {
namespace {
using namespace RJ_ARITH_IMPL_NAMESPACE;

// Zero-extending byte-pair shuffle table: for selector s = (s1 << 2) | s0,
// places unsigned bytes (ub[s0], ub[s1]) into 16-bit lanes [uint16(ub[s0]), uint16(ub[s1])].
struct SmfmacPairZextShufTable {
  alignas(64) uint8_t row[16][64];
};

constexpr SmfmacPairZextShufTable make_smfmac_pair_zext_shuf() {
  SmfmacPairZextShufTable tbl{};
  for (int s = 0; s < 16; ++s) {
    const uint8_t s0 = static_cast<uint8_t>(s & 3);
    const uint8_t s1 = static_cast<uint8_t>((s >> 2) & 3);
    for (int sub = 0; sub < 4; ++sub) {
      for (int dw = 0; dw < 4; ++dw) {
        const uint8_t base = static_cast<uint8_t>(4 * dw);
        const int off = sub * 16 + dw * 4;
        tbl.row[s][off + 0] = static_cast<uint8_t>(base + s0);
        tbl.row[s][off + 1] = 0x80u;
        tbl.row[s][off + 2] = static_cast<uint8_t>(base + s1);
        tbl.row[s][off + 3] = 0x80u;
      }
    }
  }
  return tbl;
}

inline constexpr SmfmacPairZextShufTable kSmfmacPairZextShuf = make_smfmac_pair_zext_shuf();

// Signed overflow flips the sign of the wrapped sum, which then selects the limit.
inline __m512i saturate_signed_sum_sparse_v4(__m512i acc, __m512i dot, __m512i sum) {
  const __mmask16 ovf = _mm512_movepi32_mask(_mm512_ternarylogic_epi32(acc, dot, sum, 0x42));
  return _mm512_mask_xor_epi32(sum, ovf, _mm512_srai_epi32(sum, 31), _mm512_set1_epi32(INT32_MIN));
}

// Compute 8 rows x 32 columns (both 16-col chunks) using pure AVX-512BW (x86-64-v4).
// Biasing B bytes by 0x80 lets a single zero-extending vpshufb produce uint16 pairs in [0, 255]
// for exact vpmaddwd against sign-extended int16 A pairs, corrected once per pass by -128 *
// row_sum(A).
template <int K, bool Clamp, bool ClampPerPass>
[[gnu::noinline]] void smfmac_i8_tile8x32_v4(int row_base, int ctrl_shift, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *index,
                                             const uint32_t *const *d, uint32_t (*result)[64]) {
  constexpr int num_pairs = K / 8;
  constexpr int num_passes = ClampPerPass ? (K / 32) : 1;
  constexpr int pairs_per_pass = num_pairs / num_passes;

  const __m256i shuf_lo = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(-1, 0, -1, 1, -1, 4, -1, 5, -1, 8, -1, 9, -1, 12, -1, 13));
  const __m256i shuf_hi = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(-1, 2, -1, 3, -1, 6, -1, 7, -1, 10, -1, 11, -1, 14, -1, 15));

  alignas(32) int32_t a_lo[num_pairs][8];
  alignas(32) int32_t a_hi[num_pairs][8];
  __m256i bias256[num_passes];
  for (int pass = 0; pass < num_passes; ++pass) {
    __m256i row_sum16 = _mm256_setzero_si256();
    for (int lp = 0; lp < pairs_per_pass; ++lp) {
      const int p = pass * pairs_per_pass + lp;
      const SmfmacI8PairLoc loc = smfmac_i8_pair_loc<32, K>(p);
      const __m256i aw = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(a[loc.a_reg] + loc.a_lane + row_base));
      const __m256i lo = _mm256_srai_epi16(_mm256_shuffle_epi8(aw, shuf_lo), 8);
      const __m256i hi = _mm256_srai_epi16(_mm256_shuffle_epi8(aw, shuf_hi), 8);
      _mm256_store_si256(reinterpret_cast<__m256i *>(a_lo[p]), lo);
      _mm256_store_si256(reinterpret_cast<__m256i *>(a_hi[p]), hi);
      row_sum16 = _mm256_add_epi16(row_sum16, _mm256_add_epi16(lo, hi));
    }
    const __m256i row_sum = _mm256_madd_epi16(row_sum16, _mm256_set1_epi16(1));
    bias256[pass] = _mm256_slli_epi32(_mm256_sub_epi32(_mm256_setzero_si256(), row_sum), 7);
  }

  const __m512i sign_mask = _mm512_set1_epi8(-128);
  __m512i cur0[8], cur1[8];
  if constexpr (ClampPerPass) {
    RJ_ARITH_UNROLL(8)
    for (int r = 0; r < 8; ++r) {
      const int row = row_base + r;
      const int out_reg = (row / 8) * 4 + (row & 3);
      const int out_lane = ((row >> 2) & 1) * 32;
      cur0[r] = _mm512_loadu_si512(d[out_reg] + out_lane);
      cur1[r] = _mm512_loadu_si512(d[out_reg] + out_lane + 16);
    }
  }

  for (int pass = 0; pass < num_passes; ++pass) {
    __m512i acc0[8], acc1[8];
    const __m128i lo4 = _mm256_castsi256_si128(bias256[pass]);
    const __m128i hi4 = _mm256_extracti128_si256(bias256[pass], 1);
    acc0[0] = acc1[0] = _mm512_broadcastd_epi32(lo4);
    acc0[1] = acc1[1] = _mm512_broadcastd_epi32(_mm_shuffle_epi32(lo4, 0x55));
    acc0[2] = acc1[2] = _mm512_broadcastd_epi32(_mm_shuffle_epi32(lo4, 0xaa));
    acc0[3] = acc1[3] = _mm512_broadcastd_epi32(_mm_shuffle_epi32(lo4, 0xff));
    acc0[4] = acc1[4] = _mm512_broadcastd_epi32(hi4);
    acc0[5] = acc1[5] = _mm512_broadcastd_epi32(_mm_shuffle_epi32(hi4, 0x55));
    acc0[6] = acc1[6] = _mm512_broadcastd_epi32(_mm_shuffle_epi32(hi4, 0xaa));
    acc0[7] = acc1[7] = _mm512_broadcastd_epi32(_mm_shuffle_epi32(hi4, 0xff));

    for (int lp = 0; lp < pairs_per_pass; ++lp) {
      const int p = pass * pairs_per_pass + lp;
      const SmfmacI8PairLoc loc = smfmac_i8_pair_loc<32, K>(p);
      const __m512i ub0_c0 =
          _mm512_xor_si512(_mm512_loadu_si512(b[loc.b_reg] + loc.b_lane), sign_mask);
      const __m512i ub1_c0 =
          _mm512_xor_si512(_mm512_loadu_si512(b[loc.b_reg + 1] + loc.b_lane), sign_mask);
      const __m512i ub0_c1 =
          _mm512_xor_si512(_mm512_loadu_si512(b[loc.b_reg] + loc.b_lane + 16), sign_mask);
      const __m512i ub1_c1 =
          _mm512_xor_si512(_mm512_loadu_si512(b[loc.b_reg + 1] + loc.b_lane + 16), sign_mask);
      const uint32_t *ip = index[0] + loc.a_lane + row_base;
      const int shift = ctrl_shift + loc.nib_shift;
      RJ_ARITH_UNROLL(8)
      for (int r = 0; r < 8; ++r) {
        const uint32_t sel = (ip[r] >> shift) & 0xffu;
        const __m512i m0 = _mm512_load_si512(kSmfmacPairZextShuf.row[sel & 15u]);
        const __m512i m1 = _mm512_load_si512(kSmfmacPairZextShuf.row[sel >> 4]);
        const __m512i av0 = _mm512_set1_epi32(a_lo[p][r]);
        const __m512i av1 = _mm512_set1_epi32(a_hi[p][r]);
        const __m512i p0_c0 = _mm512_madd_epi16(_mm512_shuffle_epi8(ub0_c0, m0), av0);
        const __m512i p1_c0 = _mm512_madd_epi16(_mm512_shuffle_epi8(ub1_c0, m1), av1);
        acc0[r] = _mm512_add_epi32(acc0[r], _mm512_add_epi32(p0_c0, p1_c0));
        const __m512i p0_c1 = _mm512_madd_epi16(_mm512_shuffle_epi8(ub0_c1, m0), av0);
        const __m512i p1_c1 = _mm512_madd_epi16(_mm512_shuffle_epi8(ub1_c1, m1), av1);
        acc1[r] = _mm512_add_epi32(acc1[r], _mm512_add_epi32(p0_c1, p1_c1));
      }
    }

    if constexpr (ClampPerPass) {
      RJ_ARITH_UNROLL(8)
      for (int r = 0; r < 8; ++r) {
        const __m512i sum0 = _mm512_add_epi32(cur0[r], acc0[r]);
        const __m512i sum1 = _mm512_add_epi32(cur1[r], acc1[r]);
        cur0[r] = saturate_signed_sum_sparse_v4(cur0[r], acc0[r], sum0);
        cur1[r] = saturate_signed_sum_sparse_v4(cur1[r], acc1[r], sum1);
      }
    } else {
      RJ_ARITH_UNROLL(8)
      for (int r = 0; r < 8; ++r) {
        const int row = row_base + r;
        const int out_reg = (row / 8) * 4 + (row & 3);
        const int out_lane = ((row >> 2) & 1) * 32;
        const __m512i init0 = _mm512_loadu_si512(d[out_reg] + out_lane);
        const __m512i init1 = _mm512_loadu_si512(d[out_reg] + out_lane + 16);
        const __m512i sum0 = _mm512_add_epi32(init0, acc0[r]);
        const __m512i sum1 = _mm512_add_epi32(init1, acc1[r]);
        if constexpr (Clamp) {
          _mm512_store_si512(&result[out_reg][out_lane],
                             saturate_signed_sum_sparse_v4(init0, acc0[r], sum0));
          _mm512_store_si512(&result[out_reg][out_lane + 16],
                             saturate_signed_sum_sparse_v4(init1, acc1[r], sum1));
        } else {
          _mm512_store_si512(&result[out_reg][out_lane], sum0);
          _mm512_store_si512(&result[out_reg][out_lane + 16], sum1);
        }
      }
    }
  }

  if constexpr (ClampPerPass) {
    RJ_ARITH_UNROLL(8)
    for (int r = 0; r < 8; ++r) {
      const int row = row_base + r;
      const int out_reg = (row / 8) * 4 + (row & 3);
      const int out_lane = ((row >> 2) & 1) * 32;
      _mm512_store_si512(&result[out_reg][out_lane], cur0[r]);
      _mm512_store_si512(&result[out_reg][out_lane + 16], cur1[r]);
    }
  }
}

template <int K, bool Clamp, bool ClampPerPass>
void smfmac_i8_m32_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                      const uint32_t *const *b, const uint32_t *const *index) {
  constexpr int output_regs = 16;
  const int ctrl_shift =
      (32 * K < 2048 && (controls & 3) == 0) ? 16 * static_cast<int>((controls >> 3) & 1) : 0;
  alignas(64) uint32_t result[output_regs][64];
  for (int row_base = 0; row_base < 32; row_base += 8) {
    smfmac_i8_tile8x32_v4<K, Clamp, ClampPerPass>(row_base, ctrl_shift, a, b, index, d, result);
  }
  for (int reg = 0; reg < output_regs; ++reg) {
    for (int lane = 0; lane < 64; lane += 16) {
      _mm512_storeu_si512(d[reg] + lane, _mm512_load_si512(&result[reg][lane]));
    }
  }
}

} // namespace

// Compute all tied accumulator outputs in this ISA domain.
template <int M, int K, bool Wmma>
void sparse_matrix_i8_x86_64_v4(uint64_t controls, bool clamp, bool clamp_per_pass,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *index) {
  if constexpr (!Wmma && M == 32) {
    if (!clamp) {
      smfmac_i8_m32_v4<K, false, false>(controls, d, a, b, index);
      return;
    }
    if constexpr (K == 32) {
      smfmac_i8_m32_v4<32, true, false>(controls, d, a, b, index);
      return;
    }
    if constexpr (K == 64) {
      if (clamp_per_pass) {
        smfmac_i8_m32_v4<64, true, true>(controls, d, a, b, index);
        return;
      }
      smfmac_i8_m32_v4<64, true, false>(controls, d, a, b, index);
      return;
    }
  }
  RJ_ARITH_IMPL_NAMESPACE::sparse_matrix_i8_kernel<M, K, Wmma>(controls, clamp, clamp_per_pass, d,
                                                               a, b, index);
}
template void sparse_matrix_i8_x86_64_v4<32, 32, false>(uint64_t controls, bool clamp,
                                                        bool clamp_per_pass, uint32_t *const *d,
                                                        const uint32_t *const *a,
                                                        const uint32_t *const *b,
                                                        const uint32_t *const *index);
template void sparse_matrix_i8_x86_64_v4<32, 64, false>(uint64_t controls, bool clamp,
                                                        bool clamp_per_pass, uint32_t *const *d,
                                                        const uint32_t *const *a,
                                                        const uint32_t *const *b,
                                                        const uint32_t *const *index);
} // namespace rocjitsu::arith
