// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared AVX-512 BF16 pair-dot (vdpbf16ps) MFMA reduction over [pair_plane][64] operands.
#pragma once
#include "common/compiler.h"
#include "common/x86_64/bf16_broadcast.h"
#include "mfma/layout.h"
#include "mfma/x86_64/batch4.h"
#include "mfma/x86_64/reduction.h"
#include <cstring>
#include <immintrin.h>

#if !defined(RJ_ARITH_TARGET_X86_64_V4_BF16)
#error "mfma/x86_64/bf16_dot.h requires AVX-512 BF16 (-mavx512bf16)"
#endif

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Detect any nonzero FP32 subnormal in 16 lanes (vdpbf16ps flushes subnormal C inputs).
inline __mmask16 mfma_f32_subnormal_mask(__m512i bits) {
  const __m512i mag = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff));
  return _mm512_cmplt_epu32_mask(_mm512_sub_epi32(mag, _mm512_set1_epi32(1)),
                                 _mm512_set1_epi32(0x007fffff));
}

// Check all C VGPRs for nonzero FP32 subnormals before any destination writes.
template <int Registers> inline bool mfma_c_has_subnormal(const uint32_t *const *c) {
  const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
  const __m512i one = _mm512_set1_epi32(1);
  __m512i minimum = _mm512_set1_epi32(-1);
  for (int reg = 0; reg < Registers; ++reg) {
    for (int lane = 0; lane < 64; lane += 16) {
      const __m512i mag = _mm512_and_si512(_mm512_loadu_si512(c[reg] + lane), abs_mask);
      minimum = _mm512_min_epu32(minimum, _mm512_sub_epi32(mag, one));
    }
  }
  return _mm512_cmplt_epu32_mask(minimum, _mm512_set1_epi32(0x007fffff)) != 0;
}

// Reduce M=4, Batch=16 (4x4x4_16b) using vdpbf16ps in 16 ZMM accumulators.
template <int K, int ABlockLanes>
bool mfma_reduce_bf16_batch4(uint64_t controls, uint32_t *const *d, const uint32_t (*left)[64],
                             const uint32_t (*right)[64], const uint32_t *const *c,
                             uint64_t flags) {
  const bool clamp_overflow = (flags & RJ_ARITH_FP16_OVFL) != 0;
  const bool preserve_compatibility =
      (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  constexpr int pair_planes = K / 2;
  __m512 acc[4][4];
  __mmask16 subnormal = 0;
  for (int g = 0; g < 4; ++g) {
    for (int row = 0; row < 4; ++row) {
      const __m512i bits = _mm512_loadu_si512(c[row] + 16 * g);
      if (preserve_compatibility) {
        subnormal |= mfma_f32_subnormal_mask(bits);
      }
      acc[g][row] = _mm512_castsi512_ps(bits);
    }
  }
  if (__builtin_expect(subnormal != 0, 0)) {
    return false;
  }
  const __m512i off1 = _mm512_set1_epi32(1);
  const __m512i off2 = _mm512_set1_epi32(2);
  const __m512i off3 = _mm512_set1_epi32(3);
  for (int g = 0; g < 4; ++g) {
    const int lane = 16 * g;
    const int first_a = mfma_a_lane(lane, controls, ABlockLanes);
    const int first_b = mfma_b_lane(lane, controls);
    const __m512i groups = mfma_batch4_groups(lane, first_a, controls, ABlockLanes);
    const __m512i idx1 = _mm512_add_epi32(groups, off1);
    const __m512i idx2 = _mm512_add_epi32(groups, off2);
    const __m512i idx3 = _mm512_add_epi32(groups, off3);
    const int a_chunk = first_a & ~15;
    RJ_ARITH_UNROLL(4)
    for (int p = 0; p < pair_planes; ++p) {
      const __m512i a_vec = _mm512_load_si512(left[p] + a_chunk);
      const __m512i b_vec = _mm512_load_si512(right[p] + first_b);
      acc[g][0] = _mm512_dpbf16_ps(acc[g][0], bit_cast<__m512bh>(b_vec),
                                   bit_cast<__m512bh>(_mm512_permutexvar_epi32(groups, a_vec)));
      acc[g][1] = _mm512_dpbf16_ps(acc[g][1], bit_cast<__m512bh>(b_vec),
                                   bit_cast<__m512bh>(_mm512_permutexvar_epi32(idx1, a_vec)));
      acc[g][2] = _mm512_dpbf16_ps(acc[g][2], bit_cast<__m512bh>(b_vec),
                                   bit_cast<__m512bh>(_mm512_permutexvar_epi32(idx2, a_vec)));
      acc[g][3] = _mm512_dpbf16_ps(acc[g][3], bit_cast<__m512bh>(b_vec),
                                   bit_cast<__m512bh>(_mm512_permutexvar_epi32(idx3, a_vec)));
    }
  }
  if (clamp_overflow) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
    const __m512i inf_bits = _mm512_set1_epi32(0x7f800000);
    __mmask16 any_inf = 0;
    for (int g = 0; g < 4; ++g) {
      for (int row = 0; row < 4; ++row) {
        any_inf |= _mm512_cmpeq_epi32_mask(
            _mm512_and_si512(_mm512_castps_si512(acc[g][row]), abs_mask), inf_bits);
      }
    }
    if (__builtin_expect(any_inf != 0, 0)) {
      return false;
    }
  }
  for (int row = 0; row < 4; ++row) {
    for (int g = 0; g < 4; ++g) {
      _mm512_storeu_ps(reinterpret_cast<float *>(d[row] + 16 * g), acc[g][row]);
    }
  }
  return true;
}

// Reduce M=16, Batch=1 directly in 16 ZMM accumulators using vdpbf16ps.
template <int K, int ABlockLanes>
bool mfma_reduce_bf16_16x16(uint64_t controls, uint32_t *const *d, const uint32_t (*left)[64],
                            const uint32_t (*right)[64], const uint32_t *const *c, uint64_t flags) {
  const bool clamp_overflow = (flags & RJ_ARITH_FP16_OVFL) != 0;
  const bool preserve_compatibility =
      (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  constexpr int pair_planes = K / 8;
  __m512 acc[16];
  __mmask16 subnormal = 0;
  RJ_ARITH_UNROLL(16)
  for (int row = 0; row < 16; ++row) {
    const __m512i bits = _mm512_loadu_si512(c[row % 4] + (row / 4) * 16);
    if (preserve_compatibility) {
      subnormal |= mfma_f32_subnormal_mask(bits);
    }
    acc[row] = _mm512_castsi512_ps(bits);
  }
  if (__builtin_expect(subnormal != 0, 0)) {
    return false;
  }
  if (__builtin_expect(controls == 0, 1)) {
    for (int g = 0; g < 4; ++g) {
      const int base = g * 16;
      RJ_ARITH_UNROLL(4)
      for (int p = 0; p < pair_planes; ++p) {
        const uint32_t *a_ptr = left[p] + base;
        const __m512i b_vec = _mm512_load_si512(right[p] + base);
        RJ_ARITH_UNROLL(16)
        for (int row = 0; row < 16; ++row) {
          acc[row] = _mm512_dpbf16_ps(acc[row], bit_cast<__m512bh>(b_vec),
                                      bit_cast<__m512bh>(broadcast_bf16_pair(&a_ptr[row])));
        }
      }
    }
  } else {
    const int cbsz = int(controls & 7);
    // Block broadcasts preserve each row within the selected matrix block.
    const int row_mask = ABlockLanes == 0 && cbsz && cbsz <= 4 ? (64 >> cbsz) - 1 : 63;
    for (int g = 0; g < 4; ++g) {
      const int base = g * 16;
      const int a_base = mfma_a_lane(base, controls, ABlockLanes);
      const int b_base = mfma_b_lane(base, controls);
      for (int p = 0; p < pair_planes; ++p) {
        const uint32_t *a_ptr = left[p] + a_base;
        const __m512i b_vec = _mm512_load_si512(right[p] + b_base);
        RJ_ARITH_UNROLL(16)
        for (int row = 0; row < 16; ++row) {
          acc[row] =
              _mm512_dpbf16_ps(acc[row], bit_cast<__m512bh>(b_vec),
                               bit_cast<__m512bh>(broadcast_bf16_pair(&a_ptr[row & row_mask])));
        }
      }
    }
  }
  if (clamp_overflow) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
    const __m512i inf_bits = _mm512_set1_epi32(0x7f800000);
    __mmask16 any_inf = 0;
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      any_inf |= _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_castps_si512(acc[row]), abs_mask),
                                         inf_bits);
    }
    if (__builtin_expect(any_inf != 0, 0)) {
      return false;
    }
  }
  RJ_ARITH_UNROLL(16)
  for (int row = 0; row < 16; ++row) {
    _mm512_storeu_ps(reinterpret_cast<float *>(d[row % 4] + (row / 4) * 16), acc[row]);
  }
  return true;
}

// Reduce general M>=16 shapes (32x32xK, 32x32x4_2b, 16x16x4_4b) with 8/16-row vdpbf16ps blocking.
template <int M, int K, int Batch, int ABlockLanes>
bool mfma_reduce_bf16_tiled(uint64_t controls, uint32_t *const *d, const uint32_t (*left)[64],
                            const uint32_t (*right)[64], const uint32_t *const *c, uint64_t flags) {
  const bool clamp_overflow = (flags & RJ_ARITH_FP16_OVFL) != 0;
  const bool preserve_compatibility =
      (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  constexpr int registers = M * M * Batch / 64;
  constexpr int per_lane = K / (64 / (M * Batch));
  constexpr int pair_planes = per_lane / 2;
  constexpr int groups = 64 / (M * Batch);
  constexpr int row_tile = M == 16 ? 16 : 8;
  constexpr int col_vecs = M / 16;
  if (preserve_compatibility && __builtin_expect(mfma_c_has_subnormal<registers>(c), 0)) {
    return false;
  }
  const bool direct = !clamp_overflow && mfma_direct_store_safe<registers>(d, c);
  alignas(64) float result[registers][64];
  const int cbsz = int(controls & 7);
  // Block broadcasts preserve each row within the selected matrix block.
  const int row_mask = ABlockLanes == 0 && cbsz && cbsz <= 4 ? (64 >> cbsz) - 1 : 63;
  const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
  const __m512i inf_bits = _mm512_set1_epi32(0x7f800000);
  __mmask16 any_inf = 0;
  for (int batch = 0; batch < Batch; ++batch) {
    for (int rb = 0; rb < M; rb += row_tile) {
      __m512 acc[row_tile][col_vecs];
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < row_tile; ++r) {
        const MfmaWord out = mfma_output<M, Batch>(rb + r, 0, batch);
        for (int p = 0; p < col_vecs; ++p) {
          acc[r][p] =
              _mm512_loadu_ps(reinterpret_cast<const float *>(c[out.reg] + out.lane + 16 * p));
        }
      }
      if (__builtin_expect(controls == 0, 1)) {
        for (int g = 0; g < groups; ++g) {
          const int base = batch * M + g * M * Batch;
          RJ_ARITH_UNROLL(4)
          for (int k = 0; k < pair_planes; ++k) {
            const uint32_t *a_ptr = left[k] + base + rb;
            __m512i b_vec[col_vecs];
            for (int p = 0; p < col_vecs; ++p) {
              b_vec[p] = _mm512_load_si512(right[k] + base + 16 * p);
            }
            RJ_ARITH_UNROLL(16)
            for (int r = 0; r < row_tile; ++r) {
              const __m512i a_bcast = broadcast_bf16_pair(&a_ptr[r]);
              for (int p = 0; p < col_vecs; ++p) {
                acc[r][p] = _mm512_dpbf16_ps(acc[r][p], bit_cast<__m512bh>(b_vec[p]),
                                             bit_cast<__m512bh>(a_bcast));
              }
            }
          }
        }
      } else {
        for (int g = 0; g < groups; ++g) {
          const int base = batch * M + g * M * Batch;
          const int a_base = mfma_a_lane(base, controls, ABlockLanes);
          int b_base[col_vecs];
          for (int p = 0; p < col_vecs; ++p) {
            b_base[p] = mfma_b_lane(base + 16 * p, controls);
          }
          for (int k = 0; k < pair_planes; ++k) {
            const uint32_t *a_ptr = left[k] + a_base;
            __m512i b_vec[col_vecs];
            for (int p = 0; p < col_vecs; ++p) {
              b_vec[p] = _mm512_load_si512(right[k] + b_base[p]);
            }
            RJ_ARITH_UNROLL(16)
            for (int r = 0; r < row_tile; ++r) {
              const __m512i a_bcast = broadcast_bf16_pair(&a_ptr[(rb + r) & row_mask]);
              for (int p = 0; p < col_vecs; ++p) {
                acc[r][p] = _mm512_dpbf16_ps(acc[r][p], bit_cast<__m512bh>(b_vec[p]),
                                             bit_cast<__m512bh>(a_bcast));
              }
            }
          }
        }
      }
      if (__builtin_expect(direct, 1)) {
        RJ_ARITH_UNROLL(16)
        for (int r = 0; r < row_tile; ++r) {
          const MfmaWord out = mfma_output<M, Batch>(rb + r, 0, batch);
          for (int p = 0; p < col_vecs; ++p) {
            _mm512_storeu_ps(reinterpret_cast<float *>(d[out.reg] + out.lane + 16 * p), acc[r][p]);
          }
        }
      } else {
        RJ_ARITH_UNROLL(16)
        for (int r = 0; r < row_tile; ++r) {
          const MfmaWord out = mfma_output<M, Batch>(rb + r, 0, batch);
          for (int p = 0; p < col_vecs; ++p) {
            if (clamp_overflow) {
              any_inf |= _mm512_cmpeq_epi32_mask(
                  _mm512_and_si512(_mm512_castps_si512(acc[r][p]), abs_mask), inf_bits);
            }
            _mm512_store_ps(result[out.reg] + out.lane + 16 * p, acc[r][p]);
          }
        }
      }
    }
  }
  if (__builtin_expect(direct, 1)) {
    return true;
  }
  if (__builtin_expect(clamp_overflow && any_inf != 0, 0)) {
    return false;
  }
  for (int reg = 0; reg < registers; ++reg) {
    for (int lane = 0; lane < 64; lane += 16) {
      _mm512_storeu_ps(reinterpret_cast<float *>(d[reg] + lane),
                       _mm512_load_ps(result[reg] + lane));
    }
  }
  return true;
}

// Dispatch BF16 pair-dot MFMA reduction; returns false (with d untouched) if fallback is needed.
template <int M, int K, int Batch, int ABlockLanes = 0>
bool mfma_reduce_bf16_pairs(uint64_t controls, uint32_t *const *d, const uint32_t (*left)[64],
                            const uint32_t (*right)[64], const uint32_t *const *c, uint64_t flags) {
  if constexpr (M == 4 && Batch == 16) {
    return mfma_reduce_bf16_batch4<K, ABlockLanes>(controls, d, left, right, c, flags);
  } else if constexpr (M == 16 && Batch == 1) {
    return mfma_reduce_bf16_16x16<K, ABlockLanes>(controls, d, left, right, c, flags);
  } else {
    return mfma_reduce_bf16_tiled<M, K, Batch, ABlockLanes>(controls, d, left, right, c, flags);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
