// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v3 sparse byte matrix entry points.
#include "common/compiler.h"
#include "mfma/backends.h"
#include "mfma/sparse_integer.h"
#include "mfma/x86_64/sparse_matrix_i8_common.h"
#include <immintrin.h>

namespace rocjitsu::arith {
namespace {
using namespace RJ_ARITH_IMPL_NAMESPACE;

inline __m256i saturate_signed_sum_sparse_v3(__m256i x, __m256i y, __m256i sum) {
  const __m256i ovf = _mm256_and_si256(_mm256_xor_si256(x, sum), _mm256_xor_si256(y, sum));
  const __m256i limit = _mm256_xor_si256(_mm256_srai_epi32(x, 31), _mm256_set1_epi32(0x7fffffff));
  return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(sum), _mm256_castsi256_ps(limit),
                                              _mm256_castsi256_ps(ovf)));
}

inline __m256i even_bytes_sparse_v3(__m256i bytes) {
  return _mm256_srai_epi16(_mm256_slli_epi16(bytes, 8), 8);
}

template <int M, int K, bool Clamp, bool ClampPerPass = false>
void smfmac_i8_kernel_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *index) {
  constexpr int num_pairs = K / 8;
  constexpr int row_blocks8 = M / 8;
  constexpr int col_chunks = M / 8;
  constexpr int output_regs = M * M / 64;

  const int ctrl_shift =
      (M * K < 2048 && (controls & 3) == 0) ? 16 * static_cast<int>((controls >> 3) & 1) : 0;

  alignas(32) uint32_t result[output_regs][64];
  alignas(32) int32_t a_even[num_pairs][8];
  alignas(32) int32_t a_odd[num_pairs][8];
  alignas(32) uint8_t sels[num_pairs][8];

  for (int rb = 0; rb < row_blocks8; ++rb) {
    for (int p = 0; p < num_pairs; ++p) {
      const SmfmacI8PairLoc loc = smfmac_i8_pair_loc<M, K>(p);
      const __m256i aw =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[loc.a_reg] + loc.a_lane + rb * 8));
      _mm256_store_si256(reinterpret_cast<__m256i *>(a_even[p]), even_bytes_sparse_v3(aw));
      _mm256_store_si256(reinterpret_cast<__m256i *>(a_odd[p]), _mm256_srai_epi16(aw, 8));
      const uint32_t *ip = index[0] + loc.a_lane + rb * 8;
      const int shift = ctrl_shift + loc.nib_shift;
      RJ_ARITH_UNROLL(8)
      for (int r = 0; r < 8; ++r) {
        sels[p][r] = static_cast<uint8_t>((ip[r] >> shift) & 0xffu);
      }
    }

    for (int ch = 0; ch < col_chunks; ++ch) {
      for (int half = 0; half < 2; ++half) {
        const int r0 = half * 4;
        __m256i acc[4];
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < 4; ++r) {
          acc[r] = _mm256_setzero_si256();
        }
        for (int step = 0; step < num_pairs; ++step) {
          // On gfx950, 16-column SMFMAC orders hardware passes by half-wave
          // (pairs {0,1,4,5}, {2,3,6,7}, ...), while 32-column SMFMAC is sequential.
          const int p =
              (ClampPerPass && M == 16)
                  ? (8 * (step / 8) + 2 * ((step / 4) & 1) + 4 * ((step >> 1) & 1) + (step & 1))
                  : step;
          const SmfmacI8PairLoc loc = smfmac_i8_pair_loc<M, K>(p);
          const __m256i b0 = _mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(b[loc.b_reg] + loc.b_lane + ch * 8));
          const __m256i b1 = _mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(b[loc.b_reg + 1] + loc.b_lane + ch * 8));
          const int32_t *ae = &a_even[p][r0];
          const int32_t *ao = &a_odd[p][r0];
          const uint8_t *sp = &sels[p][r0];
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            const uint32_t sel = sp[r];
            const __m256i s0 =
                _mm256_shuffle_epi8(b0, _mm256_load_si256(reinterpret_cast<const __m256i *>(
                                            kSmfmacPairShuf.row[sel & 15u])));
            const __m256i s1 =
                _mm256_shuffle_epi8(b1, _mm256_load_si256(reinterpret_cast<const __m256i *>(
                                            kSmfmacPairShuf.row[sel >> 4])));
            const __m256i picked = _mm256_blend_epi16(s0, s1, 0xaa);
            const __m256i p0 =
                _mm256_madd_epi16(even_bytes_sparse_v3(picked), _mm256_set1_epi32(ae[r]));
            const __m256i p1 =
                _mm256_madd_epi16(_mm256_srai_epi16(picked, 8), _mm256_set1_epi32(ao[r]));
            acc[r] = _mm256_add_epi32(acc[r], _mm256_add_epi32(p0, p1));
          }
          if constexpr (ClampPerPass) {
            if ((step & 3) == 3) {
              // gfx950 saturates after each 32-element (4-pair) K pass.
              RJ_ARITH_UNROLL(4)
              for (int r = 0; r < 4; ++r) {
                const int row = rb * 8 + r0 + r;
                const int out_reg = (row / (256 / M)) * 4 + (row & 3);
                const int out_lane = ((row >> 2) % (64 / M)) * M + ch * 8;
                const __m256i initial =
                    step == 3 ? _mm256_loadu_si256(
                                    reinterpret_cast<const __m256i *>(d[out_reg] + out_lane))
                              : _mm256_load_si256(
                                    reinterpret_cast<const __m256i *>(&result[out_reg][out_lane]));
                const __m256i sum = _mm256_add_epi32(initial, acc[r]);
                _mm256_store_si256(reinterpret_cast<__m256i *>(&result[out_reg][out_lane]),
                                   saturate_signed_sum_sparse_v3(initial, acc[r], sum));
                acc[r] = _mm256_setzero_si256();
              }
            }
          }
        }
        if constexpr (!ClampPerPass) {
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            const int row = rb * 8 + r0 + r;
            const int out_reg = (row / (256 / M)) * 4 + (row & 3);
            const int out_lane = ((row >> 2) % (64 / M)) * M + ch * 8;
            const __m256i initial =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d[out_reg] + out_lane));
            const __m256i sum = _mm256_add_epi32(initial, acc[r]);
            __m256i out_v;
            if constexpr (Clamp) {
              out_v = saturate_signed_sum_sparse_v3(initial, acc[r], sum);
            } else {
              out_v = sum;
            }
            _mm256_store_si256(reinterpret_cast<__m256i *>(&result[out_reg][out_lane]), out_v);
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

// Compute all tied accumulator outputs in this ISA domain.
template <int M, int K, bool Wmma>
void sparse_matrix_i8_x86_64_v3(uint64_t controls, bool clamp, bool clamp_per_pass,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *index) {
  if constexpr (Wmma) {
    RJ_ARITH_IMPL_NAMESPACE::sparse_matrix_i8_kernel<M, K, Wmma>(controls, clamp, false, d, a, b,
                                                                 index);
    return;
  } else {
    if constexpr (K > 32) {
      if (clamp && clamp_per_pass) {
        smfmac_i8_kernel_v3<M, K, true, true>(controls, d, a, b, index);
        return;
      }
    }
    if (clamp) {
      smfmac_i8_kernel_v3<M, K, true, false>(controls, d, a, b, index);
    } else {
      smfmac_i8_kernel_v3<M, K, false, false>(controls, d, a, b, index);
    }
  }
}

template void sparse_matrix_i8_x86_64_v3<16, 64, false>(uint64_t controls, bool clamp,
                                                        bool clamp_per_pass, uint32_t *const *d,
                                                        const uint32_t *const *a,
                                                        const uint32_t *const *b,
                                                        const uint32_t *const *index);
template void sparse_matrix_i8_x86_64_v3<32, 32, false>(uint64_t controls, bool clamp,
                                                        bool clamp_per_pass, uint32_t *const *d,
                                                        const uint32_t *const *a,
                                                        const uint32_t *const *b,
                                                        const uint32_t *const *index);
template void sparse_matrix_i8_x86_64_v3<16, 128, false>(uint64_t controls, bool clamp,
                                                         bool clamp_per_pass, uint32_t *const *d,
                                                         const uint32_t *const *a,
                                                         const uint32_t *const *b,
                                                         const uint32_t *const *index);
template void sparse_matrix_i8_x86_64_v3<32, 64, false>(uint64_t controls, bool clamp,
                                                        bool clamp_per_pass, uint32_t *const *d,
                                                        const uint32_t *const *a,
                                                        const uint32_t *const *b,
                                                        const uint32_t *const *index);
template void sparse_matrix_i8_x86_64_v3<16, 128, true>(uint64_t controls, bool clamp,
                                                        bool clamp_per_pass, uint32_t *const *d,
                                                        const uint32_t *const *a,
                                                        const uint32_t *const *b,
                                                        const uint32_t *const *index);
} // namespace rocjitsu::arith
