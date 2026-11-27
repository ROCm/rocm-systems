// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 BF16 pair-dot backend (vdpbf16ps) for loose CDNA3/CDNA4 BF16 MFMA.
#include "mfma/backends.h"
#include "mfma/x86_64/bf16_dot.h"

namespace rocjitsu::arith {

// Stage packed BF16 sources while checking that all nonzero finite factors have biased
// exponents in [96, 159] (no subnormal inputs, no product underflow/overflow).
// Returns false with d unchanged when inputs require the general x86-64-v4 FMA backend.
template <int M, int K, int Batch>
bool mfma_half_x86_64_v4_bf16(uint64_t controls, bool clamp_overflow, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  constexpr int per_lane = K / (64 / (M * Batch));
  constexpr int regs = per_lane / 2;
  alignas(64) uint32_t left[regs][64], right[regs][64];
  __m512i out_of_range = _mm512_setzero_si512();
  const __m512i one16 = _mm512_set1_epi16(1);
  const __m512i finite_limit = _mm512_set1_epi16(short(0xfeff));
  const __m512i low_exp = _mm512_set1_epi16(96 << 8);
  auto check_range = [&](__m512i words) {
    const __m512i mag = _mm512_slli_epi16(words, 1);
    const __mmask32 nonzero_finite =
        _mm512_cmplt_epu16_mask(_mm512_sub_epi16(mag, one16), finite_limit);
    return _mm512_maskz_sub_epi16(nonzero_finite, mag, low_exp);
  };
  for (int reg = 0; reg < regs; ++reg) {
    for (int lane = 0; lane < 64; lane += 16) {
      const __m512i aw = _mm512_loadu_si512(a[reg] + lane);
      const __m512i bw = _mm512_loadu_si512(b[reg] + lane);
      out_of_range =
          _mm512_ternarylogic_epi32(out_of_range, check_range(aw), check_range(bw), 0xfe);
      // vdpbf16ps consumes the high half first; swap halves to preserve guest K order.
      _mm512_store_si512(left[reg] + lane, _mm512_rol_epi32(aw, 16));
      _mm512_store_si512(right[reg] + lane, _mm512_rol_epi32(bw, 16));
    }
  }
  if (__builtin_expect(_mm512_test_epi16_mask(out_of_range, _mm512_set1_epi16(short(0xc000))) != 0,
                       0)) {
    return false;
  }
  return mfma_reduce_bf16_pairs<M, K, Batch>(controls, d, left, right, c, clamp_overflow);
}

template bool mfma_half_x86_64_v4_bf16<4, 4, 16>(uint64_t, bool, uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *,
                                                 const uint32_t *const *);
template bool mfma_half_x86_64_v4_bf16<32, 8, 1>(uint64_t, bool, uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *,
                                                 const uint32_t *const *);
template bool mfma_half_x86_64_v4_bf16<16, 16, 1>(uint64_t, bool, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template bool mfma_half_x86_64_v4_bf16<16, 32, 1>(uint64_t, bool, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template bool mfma_half_x86_64_v4_bf16<32, 16, 1>(uint64_t, bool, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);

} // namespace rocjitsu::arith
