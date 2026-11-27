// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-isolated x86-64-v4 dense byte matrix entry points.
#include "common/compiler.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "mfma/backends.h"
#include "mfma/integer.h"
#include "mfma/x86_64/batch4.h"
#include "mfma/x86_64/matrix_i8_k4.h"
namespace rocjitsu::arith {
namespace {
using namespace RJ_ARITH_IMPL_NAMESPACE;

// Fast AVX-512 signed 32-bit addition saturation using vpternlogd + vpmovd2m.
inline __m512i saturate_signed_sum_fast(__m512i x, __m512i y, __m512i sum) {
  const __m512i ovf = _mm512_ternarylogic_epi32(x, y, sum, 0x42);
  const __m512i limit = _mm512_xor_si512(_mm512_srai_epi32(x, 31), _mm512_set1_epi32(0x7fffffff));
  return _mm512_mask_mov_epi32(sum, _mm512_movepi32_mask(ovf), limit);
}

inline __m512i load_a_block16_v4(const uint32_t *src, int a_base, int r_mask) {
  if (__builtin_expect(r_mask == 15, 1)) {
    return _mm512_loadu_si512(src + a_base);
  }
  if (r_mask == 7) {
    const __m256i v8 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + a_base));
    return _mm512_inserti64x4(_mm512_castsi256_si512(v8), v8, 1);
  }
  return _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + a_base)));
}

// Each lane holds four signed bytes. Multiply alternating byte pairs as signed
// halfwords so their two-product sums retain the full 32-bit range.
void batch4_i8(uint64_t controls, bool clamp, uint32_t *const *d, const uint32_t *a,
               const uint32_t *b, const uint32_t *const *c) {
  __m512i result[4][4];
  for (int lane = 0; lane < 64; lane += 16) {
    int first_a = mfma_a_lane(lane, controls);
    auto groups = mfma_batch4_groups(lane, first_a, controls);
    auto left = _mm512_loadu_si512(a + (first_a & ~15));
    auto right = _mm512_loadu_si512(b + mfma_b_lane(lane, controls));
    auto even = [](__m512i bytes) { return _mm512_srai_epi16(_mm512_slli_epi16(bytes, 8), 8); };
    auto b0 = even(right), b1 = _mm512_srai_epi16(right, 8);
    RJ_ARITH_UNROLL(4)
    for (int row = 0; row < 4; ++row) {
      auto x = _mm512_permutexvar_epi32(_mm512_add_epi32(groups, _mm512_set1_epi32(row)), left);
      auto dot = _mm512_add_epi32(_mm512_madd_epi16(even(x), b0),
                                  _mm512_madd_epi16(_mm512_srai_epi16(x, 8), b1));
      auto initial = _mm512_loadu_si512(c[row] + lane);
      auto sum = _mm512_add_epi32(initial, dot);
      result[row][lane / 16] = clamp ? saturate_signed_sum(initial, dot, sum) : sum;
    }
  }
  for (int row = 0; row < 4; ++row) {
    for (int lane = 0; lane < 64; lane += 16) {
      _mm512_storeu_si512(d[row] + lane, result[row][lane / 16]);
    }
  }
}

// Each lane holds four signed bytes. Multiply alternating byte pairs as signed
// halfwords so their two-product sums retain the full 32-bit range.
template <bool Clamp>
void batch4_i8_unbroadcast(int blgp, uint32_t *const *d, const uint32_t *a, const uint32_t *b,
                           const uint32_t *const *c) {
  // BLGP selects a half-wave or quarter-wave, rotates by 16 lanes, or leaves B in place.
  const int lane_mask = blgp >= 4 ? 15 : blgp == 1 || blgp == 2 ? 31 : 63;
  const int lane_offset = blgp == 3 ? 16 : 0;
  const int lane_base = blgp >= 4 ? (blgp - 4) * 16 : blgp == 2 ? 32 : 0;
  __m512i result[4][4];
  auto even = [](__m512i bytes) { return _mm512_srai_epi16(_mm512_slli_epi16(bytes, 8), 8); };
  for (int lane = 0; lane < 64; lane += 16) {
    const __m512i left = _mm512_loadu_si512(a + lane);
    const __m512i right = _mm512_loadu_si512(b + ((lane + lane_offset) & lane_mask) + lane_base);
    const __m512i a0 = even(left), a1 = _mm512_srai_epi16(left, 8);
    const __m512i b0 = even(right), b1 = _mm512_srai_epi16(right, 8);
    auto accumulate = [&]<int Row>() {
      constexpr _MM_PERM_ENUM row_shuffle = static_cast<_MM_PERM_ENUM>(0x55 * Row);
      const __m512i lo = _mm512_shuffle_epi32(a0, row_shuffle);
      const __m512i hi = _mm512_shuffle_epi32(a1, row_shuffle);
      const __m512i dot = _mm512_add_epi32(_mm512_madd_epi16(lo, b0), _mm512_madd_epi16(hi, b1));
      const __m512i initial = _mm512_loadu_si512(c[Row] + lane);
      const __m512i sum = _mm512_add_epi32(initial, dot);
      if constexpr (Clamp) {
        result[Row][lane / 16] = saturate_signed_sum_fast(initial, dot, sum);
      } else {
        result[Row][lane / 16] = sum;
      }
    };
    accumulate.template operator()<0>();
    accumulate.template operator()<1>();
    accumulate.template operator()<2>();
    accumulate.template operator()<3>();
  }
  for (int row = 0; row < 4; ++row) {
    for (int lane = 0; lane < 64; lane += 16) {
      _mm512_storeu_si512(d[row] + lane, result[row][lane / 16]);
    }
  }
}

template <int M, int K, int Batch, bool Clamp, bool ClampPerPass = false>
void matrix_i8_dense_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int per_lane = K * M * Batch / 64;
  constexpr int k_groups = K / per_lane;
  constexpr int regs_per_lane = per_lane / 4;
  constexpr int num_q = k_groups * regs_per_lane;
  constexpr int row_blocks = M / 16;
  constexpr int col_chunks = M / 16;
  constexpr int output_regs = M * M * Batch / 64;

  const int cbsz = static_cast<int>(controls & 7);
  const int r_mask = cbsz == 3 ? 7 : cbsz == 4 ? 3 : 15;
  auto even = [](__m512i bytes) { return _mm512_srai_epi16(_mm512_slli_epi16(bytes, 8), 8); };

  alignas(64) uint32_t result[output_regs][64];
  alignas(64) int32_t a_even[num_q][16];
  alignas(64) int32_t a_odd[num_q][16];

  for (int batch = 0; batch < Batch; ++batch) {
    for (int rb = 0; rb < row_blocks; ++rb) {
      int qi = 0;
      for (int kg = 0; kg < k_groups; ++kg) {
        const int a_base = mfma_a_lane(batch * M + kg * M * Batch + rb * 16, controls);
        for (int reg = 0; reg < regs_per_lane; ++reg, ++qi) {
          const __m512i aw = load_a_block16_v4(a[reg], a_base, r_mask);
          _mm512_store_si512(a_even[qi], even(aw));
          _mm512_store_si512(a_odd[qi], _mm512_srai_epi16(aw, 8));
        }
      }

      for (int ch = 0; ch < col_chunks; ++ch) {
        // Two 8-row passes keep all 8 row accumulators in ZMMs with zero spills.
        for (int half = 0; half < 2; ++half) {
          const int r0 = half * 8;
          __m512i acc[8];
          RJ_ARITH_UNROLL(8)
          for (int r = 0; r < 8; ++r) {
            acc[r] = _mm512_setzero_si512();
          }
          int q = 0;
          for (int kg = 0; kg < k_groups; ++kg) {
            const int b_base = mfma_b_lane(batch * M + kg * M * Batch + ch * 16, controls);
            for (int reg = 0; reg < regs_per_lane; ++reg, ++q) {
              const __m512i bw = _mm512_loadu_si512(b[reg] + b_base);
              const __m512i b0 = even(bw);
              const __m512i b1 = _mm512_srai_epi16(bw, 8);
              const int32_t *ae = &a_even[q][r0];
              const int32_t *ao = &a_odd[q][r0];
              RJ_ARITH_UNROLL(8)
              for (int r = 0; r < 8; ++r) {
                const __m512i p0 = _mm512_madd_epi16(b0, _mm512_set1_epi32(ae[r]));
                const __m512i p1 = _mm512_madd_epi16(b1, _mm512_set1_epi32(ao[r]));
                acc[r] = _mm512_add_epi32(acc[r], _mm512_add_epi32(p0, p1));
              }
              if constexpr (ClampPerPass) {
                if ((q & 3) == 3) {
                  // gfx950 saturates after each 16-element (4-word) K pass.
                  RJ_ARITH_UNROLL(8)
                  for (int r = 0; r < 8; ++r) {
                    const int row = rb * 16 + r0 + r;
                    const int out_reg = batch * (M * M / 64) + (row / (256 / M)) * 4 + (row & 3);
                    const int out_lane = ((row >> 2) % (64 / M)) * M + ch * 16;
                    const __m512i initial = q == 3 ? _mm512_loadu_si512(c[out_reg] + out_lane)
                                                   : _mm512_load_si512(&result[out_reg][out_lane]);
                    const __m512i sum = _mm512_add_epi32(initial, acc[r]);
                    _mm512_store_si512(&result[out_reg][out_lane],
                                       saturate_signed_sum_fast(initial, acc[r], sum));
                    acc[r] = _mm512_setzero_si512();
                  }
                }
              }
            }
          }
          if constexpr (!ClampPerPass) {
            RJ_ARITH_UNROLL(8)
            for (int r = 0; r < 8; ++r) {
              const int row = rb * 16 + r0 + r;
              const int out_reg = batch * (M * M / 64) + (row / (256 / M)) * 4 + (row & 3);
              const int out_lane = ((row >> 2) % (64 / M)) * M + ch * 16;
              const __m512i initial = _mm512_loadu_si512(c[out_reg] + out_lane);
              const __m512i sum = _mm512_add_epi32(initial, acc[r]);
              __m512i out_v;
              if constexpr (Clamp) {
                out_v = saturate_signed_sum_fast(initial, acc[r], sum);
              } else {
                out_v = sum;
              }
              _mm512_store_si512(&result[out_reg][out_lane], out_v);
            }
          }
        }
      }
    }
  }
  for (int reg = 0; reg < output_regs; ++reg) {
    for (int lane = 0; lane < 64; lane += 16) {
      _mm512_storeu_si512(d[reg] + lane, _mm512_load_si512(&result[reg][lane]));
    }
  }
}
} // namespace
// Evaluate the full matrix using this translation unit's SIMD reduction.
template <int M, int K, int Batch, bool Wmma, int ABlockLanes>
void matrix_i8_x86_64_v4(uint64_t controls, bool clamp_overflow, bool clamp_per_pass,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  // Gfx90a uses separate A-lane groups; keep the older single-batch shapes
  // on the generic reducer as well.
  if constexpr (Wmma || ABlockLanes != 0 || (Batch == 1 && (K == 8 || (M == 16 && K == 16)))) {
    RJ_ARITH_IMPL_NAMESPACE::matrix_i8_kernel<M, K, Batch, Wmma, ABlockLanes>(
        controls, clamp_overflow, clamp_per_pass, d, a, b, c);
  } else if constexpr (M == 4) {
    static_assert(K == 4 && Batch == 16);
    if ((controls & 7u) != 0) {
      batch4_i8(controls, clamp_overflow, d, a[0], b[0], c);
    } else if (clamp_overflow) {
      batch4_i8_unbroadcast<true>(static_cast<int>((controls >> 7) & 7), d, a[0], b[0], c);
    } else {
      batch4_i8_unbroadcast<false>(static_cast<int>((controls >> 7) & 7), d, a[0], b[0], c);
    }
    return;
  } else if constexpr (K == 4) {
    if (clamp_overflow) {
      mfma_i8_k4::reduce<M, Batch, true>(controls, d, a[0], b[0], c);
    } else {
      mfma_i8_k4::reduce<M, Batch, false>(controls, d, a[0], b[0], c);
    }
  } else {
    if constexpr (K > 16) {
      if (clamp_overflow && clamp_per_pass) {
        matrix_i8_dense_v4<M, K, Batch, true, true>(controls, d, a, b, c);
        return;
      }
    }
    if (clamp_overflow) {
      matrix_i8_dense_v4<M, K, Batch, true, false>(controls, d, a, b, c);
    } else {
      matrix_i8_dense_v4<M, K, Batch, false, false>(controls, d, a, b, c);
    }
  }
}
template void matrix_i8_x86_64_v4<32, 4, 2, false>(uint64_t controls, bool clamp_overflow,
                                                   bool clamp_per_pass, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v4<16, 4, 4, false>(uint64_t controls, bool clamp_overflow,
                                                   bool clamp_per_pass, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v4<4, 4, 16, false>(uint64_t controls, bool clamp_overflow,
                                                   bool clamp_per_pass, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v4<32, 16, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    bool clamp_per_pass, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v4<16, 32, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    bool clamp_per_pass, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v4<16, 64, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    bool clamp_per_pass, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v4<32, 32, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    bool clamp_per_pass, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v4<16, 64, 1, true>(uint64_t controls, bool clamp_overflow,
                                                   bool clamp_per_pass, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

// Older full-wave shapes and gfx90a adjacent-block broadcasts share the reduction.
template void matrix_i8_x86_64_v4<32, 4, 2, false, 32>(uint64_t, bool, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v4<16, 4, 4, false, 16>(uint64_t, bool, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v4<4, 4, 16, false, 4>(uint64_t, bool, bool, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *);
template void matrix_i8_x86_64_v4<32, 8, 1, false, 0>(uint64_t, bool, bool, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *);
template void matrix_i8_x86_64_v4<32, 8, 1, false, 32>(uint64_t, bool, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v4<16, 16, 1, false, 0>(uint64_t, bool, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v4<16, 16, 1, false, 16>(uint64_t, bool, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
} // namespace rocjitsu::arith
