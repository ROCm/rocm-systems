// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-isolated AVX-512 VNNI dense signed-byte MFMA entry points.
#include "common/compiler.h"
#include "mfma/backends.h"
#include "mfma/layout.h"
#include <immintrin.h>

namespace rocjitsu::arith {
namespace {
using namespace RJ_ARITH_IMPL_NAMESPACE;

// Saturating addition of a signed 24-bit dot product to a signed 32-bit accumulator.
// All dense MFMA i8 shapes have K <= 64, so |dot| <= 64 * 128 * 128 = 1048576 < 2^23.
inline __m512i saturate_signed_sum_24(__m512i initial, __m512i dot) {
  const __m512i shuf = _mm512_set4_epi32(0x0e0d800c, 0x0a098008, 0x06058004, 0x02018000);
  const __m512i weights = _mm512_set1_epi32(0x01000001);
  return _mm512_dpwssds_epi32(initial, _mm512_shuffle_epi8(dot, shuf), weights);
}

// Broadcast each lane of `v` into local register array `out[0..15]`.
inline void broadcast_16_lanes(__m512i v, __m512i (&out)[16]) {
  auto expand4 = [&](int base, __m128i lane4) {
    out[base + 0] = _mm512_broadcastd_epi32(lane4);
    out[base + 1] = _mm512_broadcastd_epi32(_mm_shuffle_epi32(lane4, 0x55));
    out[base + 2] = _mm512_broadcastd_epi32(_mm_shuffle_epi32(lane4, 0xaa));
    out[base + 3] = _mm512_broadcastd_epi32(_mm_shuffle_epi32(lane4, 0xff));
  };
  expand4(0, _mm512_castsi512_si128(v));
  expand4(4, _mm512_extracti32x4_epi32(v, 1));
  expand4(8, _mm512_extracti32x4_epi32(v, 2));
  expand4(12, _mm512_extracti32x4_epi32(v, 3));
}

// Load 16 lanes of A at `a_base` applying MFMA CBSZ/ABID broadcast controls.
inline __m512i load_a_block16(const uint32_t *src, int a_base, int r_mask) {
  if (__builtin_expect(r_mask == 15, 1)) {
    return _mm512_loadu_si512(src + a_base);
  }
  if (r_mask == 7) {
    const __m256i v8 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + a_base));
    return _mm512_inserti64x4(_mm512_castsi256_si512(v8), v8, 1);
  }
  return _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + a_base)));
}

// Compute a 16-row x 16-col tile of signed-byte MFMA dot products in 16 local ZMMs,
// add C (with optional signed saturation), and store to `result` (or directly to `d` when M==16,
// Batch==1).
template <int M, int K, int Batch, bool Clamp>
inline void eval_tile16x16_vnni(int batch, int rb, int ch, int r_mask, uint64_t controls,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c, __m512i bias_vec,
                                uint32_t *const *d_direct, uint32_t (*result_buf)[64]) {
  constexpr int per_lane = K * M * Batch / 64;
  constexpr int k_groups = K / per_lane;
  constexpr int regs_per_lane = per_lane / 4;

  const __m512i sign_mask = _mm512_set1_epi8(-128);
  __m512i acc[16];
  broadcast_16_lanes(bias_vec, acc);

  if (__builtin_expect(r_mask == 15, 1)) {
    for (int kg = 0; kg < k_groups; ++kg) {
      const int lane_base = batch * M + kg * M * Batch;
      const int a_base = mfma_a_lane(lane_base + rb * 16, controls);
      const int b_base = mfma_b_lane(lane_base + ch * 16, controls);
      for (int reg = 0; reg < regs_per_lane; ++reg) {
        const __m512i ub = _mm512_xor_si512(_mm512_loadu_si512(b[reg] + b_base), sign_mask);
        const uint32_t *ap = a[reg] + a_base;
        RJ_ARITH_UNROLL(16)
        for (int r = 0; r < 16; ++r) {
          acc[r] = _mm512_dpbusd_epi32(acc[r], ub, _mm512_set1_epi32(static_cast<int>(ap[r])));
        }
      }
    }
  } else {
    for (int kg = 0; kg < k_groups; ++kg) {
      const int lane_base = batch * M + kg * M * Batch;
      const int a_base = mfma_a_lane(lane_base + rb * 16, controls);
      const int b_base = mfma_b_lane(lane_base + ch * 16, controls);
      for (int reg = 0; reg < regs_per_lane; ++reg) {
        const __m512i ub = _mm512_xor_si512(_mm512_loadu_si512(b[reg] + b_base), sign_mask);
        const uint32_t *ap = a[reg] + a_base;
        RJ_ARITH_UNROLL(16)
        for (int r = 0; r < 16; ++r) {
          acc[r] =
              _mm512_dpbusd_epi32(acc[r], ub, _mm512_set1_epi32(static_cast<int>(ap[r & r_mask])));
        }
      }
    }
  }

  if constexpr (M == 16 && Batch == 1) {
    // Consume all 16 C vectors before any D store so arbitrary aliasing (d==a/b/c) is exact.
    RJ_ARITH_UNROLL(16)
    for (int r = 0; r < 16; ++r) {
      const int out_reg = r & 3;
      const int out_lane = (r >> 2) * 16;
      const __m512i initial = _mm512_loadu_si512(c[out_reg] + out_lane);
      if constexpr (Clamp) {
        acc[r] = saturate_signed_sum_24(initial, acc[r]);
      } else {
        acc[r] = _mm512_add_epi32(initial, acc[r]);
      }
    }
    RJ_ARITH_UNROLL(16)
    for (int r = 0; r < 16; ++r) {
      const int out_reg = r & 3;
      const int out_lane = (r >> 2) * 16;
      _mm512_storeu_si512(d_direct[out_reg] + out_lane, acc[r]);
    }
  } else {
    RJ_ARITH_UNROLL(16)
    for (int r = 0; r < 16; ++r) {
      const int row = rb * 16 + r;
      const int out_reg = batch * (M * M / 64) + (row / (256 / M)) * 4 + (row & 3);
      const int out_lane = ((row >> 2) % (64 / M)) * M + ch * 16;
      const __m512i initial = _mm512_loadu_si512(c[out_reg] + out_lane);
      __m512i out_v;
      if constexpr (Clamp) {
        out_v = saturate_signed_sum_24(initial, acc[r]);
      } else {
        out_v = _mm512_add_epi32(initial, acc[r]);
      }
      _mm512_store_si512(&result_buf[out_reg][out_lane], out_v);
    }
  }
}

template <int M, int K, int Batch, bool Clamp>
void matrix_i8_dense_vnni(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                          const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int per_lane = K * M * Batch / 64;
  constexpr int k_groups = K / per_lane;
  constexpr int regs_per_lane = per_lane / 4;
  constexpr int row_blocks = M / 16;
  constexpr int col_chunks = M / 16;
  constexpr int output_regs = M * M * Batch / 64;

  const int cbsz = static_cast<int>(controls & 7);
  const int r_mask = cbsz == 3 ? 7 : cbsz == 4 ? 3 : 15;
  const __m512i zero = _mm512_setzero_si512();
  const __m512i ones = _mm512_set1_epi8(1);

  if constexpr (M == 16 && Batch == 1) {
    __m512i row_sum = zero;
    for (int kg = 0; kg < k_groups; ++kg) {
      const int a_base = mfma_a_lane(kg * 16, controls);
      for (int reg = 0; reg < regs_per_lane; ++reg) {
        row_sum = _mm512_dpbusd_epi32(row_sum, ones, load_a_block16(a[reg], a_base, r_mask));
      }
    }
    const __m512i bias_vec = _mm512_slli_epi32(_mm512_sub_epi32(zero, row_sum), 7);
    eval_tile16x16_vnni<M, K, Batch, Clamp>(0, 0, 0, r_mask, controls, a, b, c, bias_vec, d,
                                            nullptr);
    return;
  }

  alignas(64) uint32_t result[output_regs][64];
  for (int batch = 0; batch < Batch; ++batch) {
    for (int rb = 0; rb < row_blocks; ++rb) {
      __m512i row_sum = zero;
      for (int kg = 0; kg < k_groups; ++kg) {
        const int a_base = mfma_a_lane(batch * M + kg * M * Batch + rb * 16, controls);
        for (int reg = 0; reg < regs_per_lane; ++reg) {
          row_sum = _mm512_dpbusd_epi32(row_sum, ones, load_a_block16(a[reg], a_base, r_mask));
        }
      }
      const __m512i bias_vec = _mm512_slli_epi32(_mm512_sub_epi32(zero, row_sum), 7);
      for (int ch = 0; ch < col_chunks; ++ch) {
        eval_tile16x16_vnni<M, K, Batch, Clamp>(batch, rb, ch, r_mask, controls, a, b, c, bias_vec,
                                                nullptr, result);
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

template <int M, int K, int Batch>
void matrix_i8_avx512vnni(uint64_t controls, bool clamp_overflow, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
  if (clamp_overflow) {
    matrix_i8_dense_vnni<M, K, Batch, true>(controls, d, a, b, c);
  } else {
    matrix_i8_dense_vnni<M, K, Batch, false>(controls, d, a, b, c);
  }
}

template void matrix_i8_avx512vnni<32, 16, 1>(uint64_t, bool, uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *);
template void matrix_i8_avx512vnni<16, 32, 1>(uint64_t, bool, uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *);
template void matrix_i8_avx512vnni<16, 64, 1>(uint64_t, bool, uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *);
template void matrix_i8_avx512vnni<32, 32, 1>(uint64_t, bool, uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *);
} // namespace rocjitsu::arith
