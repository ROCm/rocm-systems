// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1201 sparse FP8/BF8 with FP32 output using native BF16 pair dots.
#include "wmma/sparse_layout.h"
#include "wmma/swmmac_float.h"
#include "wmma/x86_64/swmmac_vector.h"
#include "wmma/x86_64/wmma_loose_x86_64_v4.h"

namespace rocjitsu::arith {

template <bool Bf8A, bool Bf8B, int Wave>
bool swmmac_fp8_x86_64_v4_bf16(uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *index) {
  // DPBF16 always uses round-to-nearest. The FMA path follows host controls.
  if ((_mm_getcsr() & _MM_ROUND_MASK) != _MM_ROUND_NEAREST) {
    return false;
  }
  __m512 acc[16];
  __mmask16 subnormal = 0;
  for (int row = 0; row < 16; ++row) {
    auto bits = _mm512_loadu_si512(d[row % 8] + 16 * (row / 8));
    acc[row] = _mm512_castsi512_ps(bits);
    subnormal |= wmma_f32_subnormal(bits);
  }
  if (subnormal) {
    return false;
  }

  alignas(64) uint32_t left[8][16], selected[8][16], right[8][16 * 16];
  const int key = SwmmacIndex<8, 32, Wave>::shift(mode);
  const auto low_mask = _mm512_set1_epi32(0xffff);
  for (int pair = 0; pair < 8; ++pair) {
    int ck = 2 * pair;
    auto af = swmmac_a_field<8, Wave>(0, ck);
    auto words = _mm512_loadu_si512(a[af.reg] + af.lane);
    _mm512_store_si512(left[pair], wmma_fp8_bf16_pairs<Bf8A>(words, (ck / 2) % 2));
    auto metadata = sparse_x86::load(index[0] + af.lane);
    auto nibble = (metadata >> (key + 2 * swmmac_index_slot<8, Wave>(ck))) & 15u;
    _mm512_store_si512(selected[pair], bit_cast<__m512i>(nibble << 4));

    // Each B word contains four consecutive K values. Form all 16 selector pairs
    // (hi << 2 | lo), with lo in the high half and hi in the low half for DPBF16.
    auto bf = swmmac_b_field<8, Wave>(0, 4 * pair);
    words = _mm512_loadu_si512(b[bf.reg] + bf.lane);
    auto b01 = wmma_fp8_bf16_pairs<Bf8B>(words, 0), b23 = wmma_fp8_bf16_pairs<Bf8B>(words, 1);
    const __m512i lo_in_hi[4] = {
        _mm512_andnot_si512(low_mask, b01),
        _mm512_slli_epi32(b01, 16),
        _mm512_andnot_si512(low_mask, b23),
        _mm512_slli_epi32(b23, 16),
    };
    const __m512i hi_in_lo[4] = {
        _mm512_srli_epi32(b01, 16),
        _mm512_and_si512(low_mask, b01),
        _mm512_srli_epi32(b23, 16),
        _mm512_and_si512(low_mask, b23),
    };
    for (int hi = 0; hi < 4; ++hi) {
      for (int lo = 0; lo < 4; ++lo) {
        _mm512_store_si512(right[pair] + (hi * 4 + lo) * 16,
                           _mm512_or_si512(lo_in_hi[lo], hi_in_lo[hi]));
      }
    }
  }
  for (int pair = 0; pair < 8; ++pair) {
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      auto lhs = _mm512_broadcastd_epi32(_mm_loadu_si32(&left[pair][row]));
      auto rhs = _mm512_loadu_si512(right[pair] + selected[pair][row]);
      acc[row] = _mm512_dpbf16_ps(acc[row], (__m512bh)lhs, (__m512bh)rhs);
    }
  }
  // All input reads precede the public dispatcher's ordered destination stores.
  for (int reg = 0; reg < 8; ++reg) {
    for (int group = 0; group < 2; ++group) {
      _mm512_storeu_si512(d[reg] + 16 * group, _mm512_castps_si512(acc[reg + 8 * group]));
    }
  }
  return true;
}

template bool swmmac_fp8_x86_64_v4_bf16<false, false, 32>(uint32_t, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template bool swmmac_fp8_x86_64_v4_bf16<false, false, 64>(uint32_t, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template bool swmmac_fp8_x86_64_v4_bf16<false, true, 32>(uint32_t, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template bool swmmac_fp8_x86_64_v4_bf16<false, true, 64>(uint32_t, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template bool swmmac_fp8_x86_64_v4_bf16<true, false, 32>(uint32_t, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template bool swmmac_fp8_x86_64_v4_bf16<true, false, 64>(uint32_t, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template bool swmmac_fp8_x86_64_v4_bf16<true, true, 32>(uint32_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template bool swmmac_fp8_x86_64_v4_bf16<true, true, 64>(uint32_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
} // namespace rocjitsu::arith
