// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1201 sparse FP8/BF8 with FP32 output using native BF16 pair dots.
#include "wmma/sparse_layout.h"
#include "wmma/swmmac_float.h"
#include "wmma/x86_64/swmmac_vector.h"
#include "wmma/x86_64/wmma_loose_x86_64_v4.h"

namespace rocjitsu::arith {
namespace {

template <bool Bf8> inline __m512i wmma_fp8_bf16_pairs(__m512i words, int half) {
  __m256i packed = _mm512_cvtepi32_epi16(_mm512_srli_epi32(words, half * 16));
  __m512i raw = _mm512_cvtepu8_epi16(packed);
  __m512i magnitude = _mm512_and_si512(raw, _mm512_set1_epi16(127));
  // Rebase normal exponents directly in the BF16 encoding.
  __m512i encoded = _mm512_add_epi16(_mm512_slli_epi16(magnitude, Bf8 ? 5 : 4),
                                     _mm512_set1_epi16((Bf8 ? 112 : 120) << 7));
  // Positive subnormal FP8 encodings indexed by their fraction.
  alignas(64) static constexpr uint16_t Bf8Subnormals[32] = {0, 0x3780, 0x3800, 0x3840};
  alignas(64) static constexpr uint16_t Fp8Subnormals[32] = {0,      0x3b00, 0x3b80, 0x3bc0,
                                                             0x3c00, 0x3c20, 0x3c40, 0x3c60};
  __m512i subnormal_values = _mm512_load_si512(Bf8 ? Bf8Subnormals : Fp8Subnormals);
  encoded = _mm512_mask_permutexvar_epi16(
      encoded, _mm512_cmplt_epi16_mask(magnitude, _mm512_set1_epi16(Bf8 ? 4 : 8)), magnitude,
      subnormal_values);
  if constexpr (Bf8) {
    encoded = _mm512_mask_set1_epi16(
        encoded, _mm512_cmpeq_epi16_mask(magnitude, _mm512_set1_epi16(124)), 0x7f80);
  }
  encoded = _mm512_mask_set1_epi16(
      encoded, _mm512_cmpgt_epi16_mask(magnitude, _mm512_set1_epi16(Bf8 ? 124 : 126)), 0x7fc0);
  encoded = _mm512_or_si512(
      encoded, _mm512_and_si512(_mm512_slli_epi16(raw, 8), _mm512_set1_epi16(INT16_MIN)));
  return _mm512_rol_epi32(encoded, 16);
}

} // namespace

template <bool Bf8A, bool Bf8B, int Wave>
void swmmac_fp8_x86_64_v4_bf16(uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *index) {
  __m512 acc[16];
  for (int row = 0; row < 16; ++row) {
    auto bits = _mm512_loadu_si512(d[row % 8] + 16 * (row / 8));
    acc[row] = _mm512_castsi512_ps(bits);
  }

  alignas(64) uint32_t left[8][16], selected[8][16], right[8][6 * 16];
  const int key = SwmmacIndex<8, 32, Wave>::shift(mode);
  const auto low_mask = _mm512_set1_epi32(0xffff);
  for (int pair = 0; pair < 8; ++pair) {
    int ck = 2 * pair;
    auto af = swmmac_a_field<8, Wave>(0, ck);
    auto words = _mm512_loadu_si512(a[af.reg] + af.lane);
    _mm512_store_si512(left[pair], wmma_fp8_bf16_pairs<Bf8A>(words, (ck / 2) % 2));
    auto metadata = sparse_x86::load(index[0] + af.lane);
    auto nibble = (metadata >> (key + 2 * swmmac_index_slot<8, Wave>(ck))) & 15u;
    _mm512_store_si512(selected[pair], bit_cast<__m512i>(sparse_x86::pair_offset<16>(nibble)));

    // Each B word contains four consecutive K values. Form the six legal pairs
    // in metadata order, with the earlier K in the high half for DPBF16.
    auto bf = swmmac_b_field<8, Wave>(0, 4 * pair);
    words = _mm512_loadu_si512(b[bf.reg] + bf.lane);
    auto b01 = wmma_fp8_bf16_pairs<Bf8B>(words, 0), b23 = wmma_fp8_bf16_pairs<Bf8B>(words, 1);
    auto b0 = _mm512_andnot_si512(low_mask, b01), b1 = _mm512_slli_epi32(b01, 16);
    auto b2 = _mm512_srli_epi32(b23, 16), b3 = _mm512_and_si512(low_mask, b23);
    _mm512_store_si512(right[pair], b01);
    _mm512_store_si512(right[pair] + 16, _mm512_or_si512(b0, b2));
    _mm512_store_si512(right[pair] + 32, _mm512_or_si512(b1, b2));
    _mm512_store_si512(right[pair] + 48, _mm512_or_si512(b0, b3));
    _mm512_store_si512(right[pair] + 64, _mm512_or_si512(b1, b3));
    _mm512_store_si512(right[pair] + 80, b23);
  }
  for (int pair = 0; pair < 8; ++pair) {
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      auto lhs = _mm512_broadcastd_epi32(_mm_loadu_si32(&left[pair][row]));
      auto rhs = _mm512_loadu_si512(right[pair] + selected[pair][row]);
      acc[row] = _mm512_dpbf16_ps(acc[row], bit_cast<__m512bh>(lhs), bit_cast<__m512bh>(rhs));
    }
  }
  // All input reads precede the public dispatcher's ordered destination stores.
  for (int reg = 0; reg < 8; ++reg) {
    for (int group = 0; group < 2; ++group) {
      _mm512_storeu_si512(d[reg] + 16 * group, _mm512_castps_si512(acc[reg + 8 * group]));
    }
  }
}

template void swmmac_fp8_x86_64_v4_bf16<false, false, 32>(uint32_t, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void swmmac_fp8_x86_64_v4_bf16<false, false, 64>(uint32_t, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void swmmac_fp8_x86_64_v4_bf16<false, true, 32>(uint32_t, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_fp8_x86_64_v4_bf16<false, true, 64>(uint32_t, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_fp8_x86_64_v4_bf16<true, false, 32>(uint32_t, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_fp8_x86_64_v4_bf16<true, false, 64>(uint32_t, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_fp8_x86_64_v4_bf16<true, true, 32>(uint32_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_fp8_x86_64_v4_bf16<true, true, 64>(uint32_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
} // namespace rocjitsu::arith
