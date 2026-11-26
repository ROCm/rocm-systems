// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 BF16 pair-dot backend for loose gfx11/gfx12 BF16 WMMA, including
// both wave sizes and FP32 or packed BF16 results. Fuses fast-path range and
// subnormal screening with single-pass operand loading and tail-calls the
// asymmetric exponent path out of line so the common path stays spill-free.

#include "common/compiler.h"
#include "common/simd.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/x86_64/wmma_loose_x86_64_v4.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Return the smallest unsigned 16-bit exponent across all packed halfwords.
int min_exponent(__m512i value) {
  value = _mm512_min_epu16(value, _mm512_srli_epi32(value, 16));
  return int(_mm512_reduce_min_epu32(_mm512_and_si512(value, _mm512_set1_epi32(0xffff))));
}

// Return the largest unsigned 16-bit exponent across all packed halfwords.
int max_exponent(__m512i value) {
  value = _mm512_max_epu16(value, _mm512_srli_epi32(value, 16));
  return int(_mm512_reduce_max_epu32(_mm512_and_si512(value, _mm512_set1_epi32(0xffff))));
}

// Cold-path check and execution for asymmetric A/B exponent ranges when the
// fast symmetric [96, 159] check fails. Tail-called so the fast path makes no
// calls while vector accumulators are live.
template <int Lanes, bool Rdna3, bool Packed>
[[gnu::noinline, gnu::cold]] bool
wmma_bf16_asymmetric_fallback(uint32_t instruction_flags, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  int minimum[2], maximum[2];
  for (int operand = 0; operand < 2; ++operand) {
    const uint32_t *const *v = operand ? b : a;
    __m512i min_exp = _mm512_set1_epi16(255), max_exp = _mm512_setzero_si512();
    for (int reg = 0; reg < (Rdna3 ? 8 : 128 / Lanes); ++reg) {
      for (int lane = 0; lane < (Rdna3 ? 16 : Lanes); lane += 16) {
        __m512i magnitude =
            _mm512_and_si512(_mm512_loadu_si512(v[reg] + lane), _mm512_set1_epi16(0x7fff));
        __mmask32 zero = _mm512_cmpeq_epi16_mask(magnitude, _mm512_setzero_si512());
        __mmask32 tiny = _mm512_cmp_epu16_mask(magnitude, _mm512_set1_epi16(0x80), _MM_CMPINT_LT);
        __mmask32 special =
            _mm512_cmp_epu16_mask(magnitude, _mm512_set1_epi16(0x7f80), _MM_CMPINT_GE);
        // DPBF16 flushes subnormal factors even under the default host environment.
        if (tiny & ~zero) {
          return false;
        }
        __m512i exponent = _mm512_srli_epi16(magnitude, 7);
        max_exp = _mm512_max_epu16(max_exp, _mm512_maskz_mov_epi16(~special, exponent));
        // Zero factors do not lower the minimum nonzero exponent.
        min_exp = _mm512_min_epu16(
            min_exp, _mm512_mask_mov_epi16(exponent, zero | special, _mm512_set1_epi16(255)));
      }
    }
    minimum[operand] = min_exponent(min_exp);
    maximum[operand] = max_exponent(max_exp);
  }
  if (minimum[0] + minimum[1] < 128 || maximum[0] + maximum[1] > 380) {
    return false;
  }

  const __m512i a_sign =
      _mm512_set1_epi32((instruction_flags & RJ_ARITH_WMMA_NEG_LO_A ? 0x8000 : 0) |
                        (instruction_flags & RJ_ARITH_WMMA_NEG_HI_A ? INT32_MIN : 0));
  const __m512i b_sign =
      _mm512_set1_epi32((instruction_flags & RJ_ARITH_WMMA_NEG_LO_B ? 0x8000 : 0) |
                        (instruction_flags & RJ_ARITH_WMMA_NEG_HI_B ? INT32_MIN : 0));
  alignas(64) uint32_t left[8][16];
  alignas(64) uint32_t right[8][16];
  for (int step = 0; step < 8; ++step) {
    const int pair =
        Packed && !Rdna3 && Lanes == 32 ? rocjitsu::arith::wmma_gfx12_w32_k(2 * step) / 2 : step;
    const rocjitsu::arith::WmmaRegisterSection src =
        rocjitsu::arith::wmma_half_input<Lanes, Rdna3>(2 * pair);
    _mm512_store_si512(
        left[step],
        _mm512_rol_epi32(_mm512_xor_si512(_mm512_loadu_si512(a[src.reg] + src.lane), a_sign), 16));
    _mm512_store_si512(
        right[step],
        _mm512_rol_epi32(_mm512_xor_si512(_mm512_loadu_si512(b[src.reg] + src.lane), b_sign), 16));
  }
  __m512 sums[16];
  rocjitsu::arith::wmma_load_acc<true, Lanes, Rdna3, Packed>(sums, instruction_flags, c);
  rocjitsu::arith::wmma_bf16_pairs<Packed, Rdna3>(sums, left, right);
  rocjitsu::arith::wmma_store_acc<true, Lanes, Rdna3, Packed, false>(sums, instruction_flags, d);
  return true;
}

} // namespace

namespace rocjitsu::arith {

// Try loose BF16 WMMA with validated instruction flags. Return true after writing D,
// or false with D unchanged when inputs require another backend. D may alias inputs.
template <int Lanes, bool Rdna3, bool Packed>
bool wmma_x86_64_v4_bf16(uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c) {
  const __m512i a_sign =
      _mm512_set1_epi32((instruction_flags & RJ_ARITH_WMMA_NEG_LO_A ? 0x8000 : 0) |
                        (instruction_flags & RJ_ARITH_WMMA_NEG_HI_A ? INT32_MIN : 0));
  const __m512i b_sign =
      _mm512_set1_epi32((instruction_flags & RJ_ARITH_WMMA_NEG_LO_B ? 0x8000 : 0) |
                        (instruction_flags & RJ_ARITH_WMMA_NEG_HI_B ? INT32_MIN : 0));

  // Biased exponents in [96, 159] meet both product bounds; zeros and nonfinite
  // factors impose no exponent bound. Screen A and B in the same pass that loads
  // and swaps their halves for DPBF16's high-half-first order.
  const __m512i ones = _mm512_set1_epi16(1);
  const __m512i finite_limit = _mm512_set1_epi16(short(0xfeff));
  const __m512i low_bound = _mm512_set1_epi16(96 << 8);
  auto range = [&](__m512i words) {
    __m512i magnitude = _mm512_slli_epi16(words, 1);
    __mmask32 nonzero_finite =
        _mm512_cmplt_epu16_mask(_mm512_sub_epi16(magnitude, ones), finite_limit);
    return _mm512_maskz_sub_epi16(nonzero_finite, magnitude, low_bound);
  };

  alignas(64) uint32_t left[8][16];
  alignas(64) uint32_t right[8][16];
  __m512i out_of_range = _mm512_setzero_si512();
  for (int step = 0; step < 8; ++step) {
    const int pair = Packed && !Rdna3 && Lanes == 32 ? wmma_gfx12_w32_k(2 * step) / 2 : step;
    const rocjitsu::arith::WmmaRegisterSection src = wmma_half_input<Lanes, Rdna3>(2 * pair);
    // Swapping halves and flipping sign bits preserve the multiset of 16-bit
    // magnitudes (range() shifts out bit 15), so range() consumes aw_rot/bw_rot
    // directly without extending raw aw/bw live ranges.
    const __m512i aw_rot =
        _mm512_rol_epi32(_mm512_xor_si512(_mm512_loadu_si512(a[src.reg] + src.lane), a_sign), 16);
    const __m512i bw_rot =
        _mm512_rol_epi32(_mm512_xor_si512(_mm512_loadu_si512(b[src.reg] + src.lane), b_sign), 16);
    _mm512_store_si512(left[step], aw_rot);
    _mm512_store_si512(right[step], bw_rot);
    out_of_range = _mm512_ternarylogic_epi32(out_of_range, range(aw_rot), range(bw_rot), 0xfe);
  }
  const bool range_fail =
      _mm512_test_epi16_mask(out_of_range, _mm512_set1_epi16(short(0xc000))) != 0;

  // Load C once while tracking the minimum nonzero magnitude to detect any
  // subnormal input accumulator (DPBF16 flushes subnormal accumulators).
  __m512 sums[16];
  const __m512i min_c_mag = wmma_load_acc<true, Lanes, Rdna3, Packed, /*TrackSubnormal=*/true>(
      sums, instruction_flags, c);
  if (_mm512_cmplt_epu32_mask(min_c_mag, _mm512_set1_epi32(0x007fffff))) {
    return false;
  }
  if (range_fail) {
    return wmma_bf16_asymmetric_fallback<Lanes, Rdna3, Packed>(instruction_flags, d, a, b, c);
  }

  wmma_bf16_pairs<Packed, Rdna3>(sums, left, right);
  wmma_store_acc<true, Lanes, Rdna3, Packed, false>(sums, instruction_flags, d);
  return true;
}

template bool wmma_x86_64_v4_bf16<32, false, false>(uint32_t, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template bool wmma_x86_64_v4_bf16<32, false, true>(uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template bool wmma_x86_64_v4_bf16<32, true, false>(uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template bool wmma_x86_64_v4_bf16<32, true, true>(uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template bool wmma_x86_64_v4_bf16<64, false, false>(uint32_t, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template bool wmma_x86_64_v4_bf16<64, false, true>(uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template bool wmma_x86_64_v4_bf16<64, true, false>(uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template bool wmma_x86_64_v4_bf16<64, true, true>(uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);

} // namespace rocjitsu::arith
