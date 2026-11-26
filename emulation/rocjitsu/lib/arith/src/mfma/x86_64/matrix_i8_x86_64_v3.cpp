// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-isolated x86-64-v3 dense byte matrix entry points.
#include "common/compiler.h"
#include "mfma/backends.h"
#include "mfma/integer.h"
#include "mfma/x86_64/matrix_i8_k4.h"
#include <immintrin.h>

namespace rocjitsu::arith {
namespace {
using namespace RJ_ARITH_IMPL_NAMESPACE;

inline __m256i saturate_signed_sum_fast_v3(__m256i x, __m256i y, __m256i sum) {
  const __m256i ovf = _mm256_and_si256(_mm256_xor_si256(x, sum), _mm256_xor_si256(y, sum));
  const __m256i limit = _mm256_xor_si256(_mm256_srai_epi32(x, 31), _mm256_set1_epi32(0x7fffffff));
  return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(sum), _mm256_castsi256_ps(limit),
                                              _mm256_castsi256_ps(ovf)));
}

inline __m256i even_bytes_v3(__m256i bytes) {
  return _mm256_srai_epi16(_mm256_slli_epi16(bytes, 8), 8);
}

inline __m256i load_a_block8_v3(const uint32_t *src, int a_base, bool dup4) {
  if (__builtin_expect(!dup4, 1)) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + a_base));
  }
  const __m128i v4 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + a_base));
  return _mm256_inserti128_si256(_mm256_castsi128_si256(v4), v4, 1);
}

template <bool Clamp>
void batch4_i8_v3(uint64_t controls, uint32_t *const *d, const uint32_t *a, const uint32_t *b,
                  const uint32_t *const *c) {
  __m256i result[4][8];
  for (int lane = 0; lane < 64; lane += 8) {
    const __m128i a0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + mfma_a_lane(lane, controls)));
    const __m128i a1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + mfma_a_lane(lane + 4, controls)));
    const __m256i left = _mm256_inserti128_si256(_mm256_castsi128_si256(a0), a1, 1);
    const __m256i right =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + mfma_b_lane(lane, controls)));
    const __m256i b0 = even_bytes_v3(right);
    const __m256i b1 = _mm256_srai_epi16(right, 8);
    const __m256i l0 = even_bytes_v3(left);
    const __m256i l1 = _mm256_srai_epi16(left, 8);

    auto do_row = [&]<int Row, int ShufImm>() {
      const __m256i dot =
          _mm256_add_epi32(_mm256_madd_epi16(_mm256_shuffle_epi32(l0, ShufImm), b0),
                           _mm256_madd_epi16(_mm256_shuffle_epi32(l1, ShufImm), b1));
      const __m256i initial = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[Row] + lane));
      const __m256i sum = _mm256_add_epi32(initial, dot);
      if constexpr (Clamp) {
        result[Row][lane / 8] = saturate_signed_sum_fast_v3(initial, dot, sum);
      } else {
        result[Row][lane / 8] = sum;
      }
    };
    do_row.template operator()<0, 0x00>();
    do_row.template operator()<1, 0x55>();
    do_row.template operator()<2, 0xaa>();
    do_row.template operator()<3, 0xff>();
  }
  for (int row = 0; row < 4; ++row) {
    for (int lane = 0; lane < 64; lane += 8) {
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(d[row] + lane), result[row][lane / 8]);
    }
  }
}

template <int M, int K, int Batch, bool Clamp, bool ClampPerPass = false>
void matrix_i8_dense_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int per_lane = K * M * Batch / 64;
  constexpr int k_groups = K / per_lane;
  constexpr int regs_per_lane = per_lane / 4;
  constexpr int num_q = k_groups * regs_per_lane;
  constexpr int row_blocks8 = M / 8;
  constexpr int col_chunks = M / 8;
  constexpr int output_regs = M * M * Batch / 64;

  const bool dup4 = static_cast<int>(controls & 7) == 4;

  alignas(32) uint32_t result[output_regs][64];
  alignas(32) int32_t a_even[num_q][8];
  alignas(32) int32_t a_odd[num_q][8];

  for (int batch = 0; batch < Batch; ++batch) {
    for (int rb = 0; rb < row_blocks8; ++rb) {
      int qi = 0;
      for (int kg = 0; kg < k_groups; ++kg) {
        const int a_base = mfma_a_lane(batch * M + kg * M * Batch + rb * 8, controls);
        for (int reg = 0; reg < regs_per_lane; ++reg, ++qi) {
          const __m256i aw = load_a_block8_v3(a[reg], a_base, dup4);
          _mm256_store_si256(reinterpret_cast<__m256i *>(a_even[qi]), even_bytes_v3(aw));
          _mm256_store_si256(reinterpret_cast<__m256i *>(a_odd[qi]), _mm256_srai_epi16(aw, 8));
        }
      }

      for (int ch = 0; ch < col_chunks; ++ch) {
        // Two 4-row passes keep all 4 row accumulators + B pairs in YMMs with zero spills.
        for (int half = 0; half < 2; ++half) {
          const int r0 = half * 4;
          __m256i acc[4];
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            acc[r] = _mm256_setzero_si256();
          }
          int q = 0;
          for (int kg = 0; kg < k_groups; ++kg) {
            const int b_base = mfma_b_lane(batch * M + kg * M * Batch + ch * 8, controls);
            for (int reg = 0; reg < regs_per_lane; ++reg, ++q) {
              const __m256i bw =
                  _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b[reg] + b_base));
              const __m256i b0 = even_bytes_v3(bw);
              const __m256i b1 = _mm256_srai_epi16(bw, 8);
              const int32_t *ae = &a_even[q][r0];
              const int32_t *ao = &a_odd[q][r0];
              RJ_ARITH_UNROLL(4)
              for (int r = 0; r < 4; ++r) {
                const __m256i p0 = _mm256_madd_epi16(b0, _mm256_set1_epi32(ae[r]));
                const __m256i p1 = _mm256_madd_epi16(b1, _mm256_set1_epi32(ao[r]));
                acc[r] = _mm256_add_epi32(acc[r], _mm256_add_epi32(p0, p1));
              }
              if constexpr (ClampPerPass) {
                if ((q & 3) == 3) {
                  // gfx950 saturates after each 16-element (4-word) K pass.
                  RJ_ARITH_UNROLL(4)
                  for (int r = 0; r < 4; ++r) {
                    const int row = rb * 8 + r0 + r;
                    const int out_reg = batch * (M * M / 64) + (row / (256 / M)) * 4 + (row & 3);
                    const int out_lane = ((row >> 2) % (64 / M)) * M + ch * 8;
                    const __m256i initial =
                        q == 3 ? _mm256_loadu_si256(
                                     reinterpret_cast<const __m256i *>(c[out_reg] + out_lane))
                               : _mm256_load_si256(
                                     reinterpret_cast<const __m256i *>(&result[out_reg][out_lane]));
                    const __m256i sum = _mm256_add_epi32(initial, acc[r]);
                    _mm256_store_si256(reinterpret_cast<__m256i *>(&result[out_reg][out_lane]),
                                       saturate_signed_sum_fast_v3(initial, acc[r], sum));
                    acc[r] = _mm256_setzero_si256();
                  }
                }
              }
            }
          }
          if constexpr (!ClampPerPass) {
            RJ_ARITH_UNROLL(4)
            for (int r = 0; r < 4; ++r) {
              const int row = rb * 8 + r0 + r;
              const int out_reg = batch * (M * M / 64) + (row / (256 / M)) * 4 + (row & 3);
              const int out_lane = ((row >> 2) % (64 / M)) * M + ch * 8;
              const __m256i initial =
                  _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[out_reg] + out_lane));
              const __m256i sum = _mm256_add_epi32(initial, acc[r]);
              __m256i out_v;
              if constexpr (Clamp) {
                out_v = saturate_signed_sum_fast_v3(initial, acc[r], sum);
              } else {
                out_v = sum;
              }
              _mm256_store_si256(reinterpret_cast<__m256i *>(&result[out_reg][out_lane]), out_v);
            }
          }
        }
      }
    }
  }
  for (int reg = 0; reg < output_regs; ++reg) {
    for (int lane = 0; lane < 64; lane += 8) {
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(d[reg] + lane),
                          _mm256_load_si256(reinterpret_cast<const __m256i *>(&result[reg][lane])));
    }
  }
}
} // namespace

// Evaluate the full matrix using this translation unit's SIMD reduction.
template <int M, int K, int Batch, bool Wmma, int ABlockLanes>
void matrix_i8_x86_64_v3(uint64_t controls, bool clamp_overflow, bool clamp_per_pass,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  // Gfx90a uses separate A-lane groups; keep the older single-batch shapes
  // on the generic reducer as well.
  if constexpr (Wmma || ABlockLanes != 0 || (Batch == 1 && (K == 8 || (M == 16 && K == 16)))) {
    RJ_ARITH_IMPL_NAMESPACE::matrix_i8_kernel<M, K, Batch, Wmma, ABlockLanes>(
        controls, clamp_overflow, clamp_per_pass, d, a, b, c);
  } else if constexpr (M == 4) {
    static_assert(K == 4 && Batch == 16);
    if (clamp_overflow) {
      batch4_i8_v3<true>(controls, d, a[0], b[0], c);
    } else {
      batch4_i8_v3<false>(controls, d, a[0], b[0], c);
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
        matrix_i8_dense_v3<M, K, Batch, true, true>(controls, d, a, b, c);
        return;
      }
    }
    if (clamp_overflow) {
      matrix_i8_dense_v3<M, K, Batch, true, false>(controls, d, a, b, c);
    } else {
      matrix_i8_dense_v3<M, K, Batch, false, false>(controls, d, a, b, c);
    }
  }
}
template void matrix_i8_x86_64_v3<32, 4, 2, false>(uint64_t controls, bool clamp_overflow,
                                                   bool clamp_per_pass, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v3<16, 4, 4, false>(uint64_t controls, bool clamp_overflow,
                                                   bool clamp_per_pass, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v3<4, 4, 16, false>(uint64_t controls, bool clamp_overflow,
                                                   bool clamp_per_pass, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v3<32, 16, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    bool clamp_per_pass, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v3<16, 32, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    bool clamp_per_pass, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v3<16, 64, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    bool clamp_per_pass, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v3<32, 32, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    bool clamp_per_pass, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v3<16, 64, 1, true>(uint64_t controls, bool clamp_overflow,
                                                   bool clamp_per_pass, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

// Older full-wave shapes and gfx90a adjacent-block broadcasts share the reduction.
template void matrix_i8_x86_64_v3<32, 4, 2, false, 32>(uint64_t, bool, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v3<16, 4, 4, false, 16>(uint64_t, bool, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v3<4, 4, 16, false, 4>(uint64_t, bool, bool, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *);
template void matrix_i8_x86_64_v3<32, 8, 1, false, 0>(uint64_t, bool, bool, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *);
template void matrix_i8_x86_64_v3<32, 8, 1, false, 32>(uint64_t, bool, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v3<16, 16, 1, false, 0>(uint64_t, bool, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v3<16, 16, 1, false, 16>(uint64_t, bool, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
} // namespace rocjitsu::arith
