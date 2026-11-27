// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Row-streaming K=4 signed-byte MFMA shared by the x86-64-v3 and x86-64-v4 domains.
#pragma once
#include "common/bits.h"
#include "common/compiler.h"
#include "mfma/layout.h"
#include "mfma/x86_64/reduction.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::mfma_i8_k4 {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
constexpr int Width = 16;
using Native = __m512i;
inline Native madd(Native a, Native b) { return _mm512_madd_epi16(a, b); }
// Signed overflow flips the sign of the wrapped sum, which then selects the limit.
inline Native saturate(Native acc, Native dot, Native sum) {
  const __mmask16 overflow = _mm512_movepi32_mask(_mm512_ternarylogic_epi32(acc, dot, sum, 0x42));
  return _mm512_mask_xor_epi32(sum, overflow, _mm512_srai_epi32(sum, 31),
                               _mm512_set1_epi32(INT32_MIN));
}
#else
constexpr int Width = 8;
using Native = __m256i;
inline Native madd(Native a, Native b) { return _mm256_madd_epi16(a, b); }
// The sum wraps below acc exactly when a positive dot overflows, so that
// compare yields both the overflow sign (against the dot sign) and the limit.
inline Native saturate(Native acc, Native dot, Native sum) {
  const __m256i below = _mm256_cmpgt_epi32(acc, sum);
  const __m256i overflow = _mm256_xor_si256(below, dot);
  const __m256i limit = _mm256_xor_si256(below, _mm256_set1_epi32(INT32_MIN));
  return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(sum), _mm256_castsi256_ps(limit),
                                              _mm256_castsi256_ps(overflow)));
}
#endif
using Words [[gnu::vector_size(Width * 4)]] = uint32_t;
using Halves [[gnu::vector_size(Width * 4)]] = uint16_t;
using SignedHalves [[gnu::vector_size(Width * 4)]] = int16_t;

inline Native load(const uint32_t *p) {
  Native v;
  __builtin_memcpy(&v, p, sizeof(v));
  return v;
}
inline void store(uint32_t *p, Native v) { __builtin_memcpy(p, &v, sizeof(v)); }
inline Native add(Native a, Native b) {
  return bit_cast<Native>(bit_cast<Words>(a) + bit_cast<Words>(b));
}
inline Native broadcast(uint32_t v) { return bit_cast<Native>(Words{} + v); }
// Sign-extend bytes 0 and 2, or bytes 1 and 3, of every lane to halfwords.
inline Native even_bytes(Native v) {
  return bit_cast<Native>(bit_cast<SignedHalves>(bit_cast<Halves>(v) << 8) >> 8);
}
inline Native odd_bytes(Native v) { return bit_cast<Native>(bit_cast<SignedHalves>(v) >> 8); }

// One B word per column leaves nothing to accumulate across K, so every output
// vector is a single pass: hold both halfword planes of B for a batch in
// registers, broadcast the two A halfword pairs once per row for all column
// chunks, and stream D one VGPR at a time. A and B are read in full before the
// first D store, so D may alias either source. Any accumulator overlap that an
// ascending in-place store could reach reads from a staged copy of C instead.
template <int M, int Batch, bool Clamp>
void reduce(uint64_t controls, uint32_t *const *d, const uint32_t *a, const uint32_t *b,
            const uint32_t *const *c) {
  constexpr int chunks = M / Width;
  constexpr int regs_per_batch = M * M / 64;
  constexpr int registers = regs_per_batch * Batch;
  constexpr int rows_per_reg = 64 / M;
  static_assert(M * Batch == 64);

  alignas(64) uint32_t a_even[64], a_odd[64];
  RJ_ARITH_UNROLL(8)
  for (int lane = 0; lane < 64; lane += Width) {
    const Native word = load(a + lane);
    store(a_even + lane, even_bytes(word));
    store(a_odd + lane, odd_bytes(word));
  }
  // CBSZ repeats one block of A lanes across the wave. Resolve it here so the
  // row loop indexes logical lanes with constant offsets.
  const uint32_t *left_even = a_even, *left_odd = a_odd;
  alignas(64) uint32_t repeated_even[64], repeated_odd[64];
  const int cbsz = static_cast<int>(controls & 7);
  if (cbsz && cbsz <= 4) {
    const int block_mask = (64 >> cbsz) - 1;
    const int block = mfma_a_lane(0, controls);
    for (int lane = 0; lane < 64; lane += 4) {
      const int source = block + (lane & block_mask);
      __builtin_memcpy(repeated_even + lane, a_even + source, 4 * sizeof(uint32_t));
      __builtin_memcpy(repeated_odd + lane, a_odd + source, 4 * sizeof(uint32_t));
    }
    left_even = repeated_even;
    left_odd = repeated_odd;
  }
  Native b_even[Batch * chunks], b_odd[Batch * chunks];
  RJ_ARITH_UNROLL(8)
  for (int i = 0; i < Batch * chunks; ++i) {
    const Native word = load(b + mfma_b_lane(i * Width, controls));
    b_even[i] = even_bytes(word);
    b_odd[i] = odd_bytes(word);
  }

  alignas(64) uint32_t staged[registers][64];
  const uint32_t *staged_regs[registers];
  if (!mfma_ascending_store_safe<registers>(d, c)) {
    for (int reg = 0; reg < registers; ++reg) {
      __builtin_memcpy(staged[reg], c[reg], sizeof(staged[reg]));
      staged_regs[reg] = staged[reg];
    }
    c = staged_regs;
  }
  for (int batch = 0; batch < Batch; ++batch) {
    Native right_even[chunks], right_odd[chunks];
    RJ_ARITH_UNROLL(4)
    for (int ch = 0; ch < chunks; ++ch) {
      right_even[ch] = b_even[batch * chunks + ch];
      right_odd[ch] = b_odd[batch * chunks + ch];
    }
    for (int reg = 0; reg < regs_per_batch; ++reg) {
      const int out_reg = batch * regs_per_batch + reg;
      const int first_row = batch * M + (256 / M) * (reg / 4) + reg % 4;
      const uint32_t *const initial = c[out_reg];
      uint32_t *const target = d[out_reg];
      RJ_ARITH_UNROLL(4)
      for (int j = 0; j < rows_per_reg; ++j) {
        const Native even = broadcast(left_even[first_row + 4 * j]);
        const Native odd = broadcast(left_odd[first_row + 4 * j]);
        RJ_ARITH_UNROLL(4)
        for (int ch = 0; ch < chunks; ++ch) {
          const int offset = j * M + ch * Width;
          const Native dot = add(madd(right_even[ch], even), madd(right_odd[ch], odd));
          const Native acc = load(initial + offset);
          const Native sum = add(acc, dot);
          if constexpr (Clamp) {
            store(target + offset, saturate(acc, dot, sum));
          } else {
            store(target + offset, sum);
          }
        }
      }
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::mfma_i8_k4
