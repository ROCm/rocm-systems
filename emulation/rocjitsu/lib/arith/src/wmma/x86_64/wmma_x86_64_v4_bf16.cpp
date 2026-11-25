// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 BF16 pair-dot backend for loose gfx11/gfx12 BF16 WMMA, including
// both wave sizes and FP32 or packed BF16 results. Guards operand ranges and
// denormals, declining inputs that require the general arithmetic path.

#include "common/compiler.h"
#include "common/simd.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/x86_64/wmma_loose_x86_64_v4.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Biased exponents in [96,159] meet both product bounds. Zeros and nonfinite
// factors impose no exponent bound. A subnormal accumulator still needs the
// general backend. Check the frequent range before computing per-input extrema.
template <int Lanes, bool Rdna3, bool Packed>
static bool bf16_fast_range(uint32_t flags, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  auto out_of_range = _mm512_setzero_si512();
  // Removing the sign puts the exponent in bits 8..15. After subtracting the
  // lower bound, bits 14..15 are clear throughout the accepted interval.
  auto range = [](__m512i words) {
    auto magnitude = _mm512_slli_epi16(words, 1);
    auto nonzero_finite = _mm512_cmplt_epu16_mask(_mm512_sub_epi16(magnitude, _mm512_set1_epi16(1)),
                                                  _mm512_set1_epi16(short(0xfeff)));
    return _mm512_maskz_sub_epi16(nonzero_finite, magnitude, _mm512_set1_epi16(96 << 8));
  };
  for (int reg = 0; reg < (Rdna3 ? 8 : 128 / Lanes); ++reg) {
    // RDNA3 replicates both factors across 16-lane groups.
    for (int lane = 0; lane < (Rdna3 ? 16 : Lanes); lane += 16) {
      auto x = range(_mm512_loadu_si512(a[reg] + lane));
      auto y = range(_mm512_loadu_si512(b[reg] + lane));
      out_of_range = _mm512_ternarylogic_epi32(out_of_range, x, y, 0xfe);
    }
  }
  // Subtracting one maps zero to the unsigned maximum and makes the lowest
  // nonzero magnitude sufficient to detect any subnormal.
  auto minimum = _mm512_set1_epi32(-1);
  constexpr int Registers = 256 / Lanes / (Packed && !Rdna3 ? 2 : 1);
  for (int reg = 0; reg < Registers; ++reg) {
    for (int lane = 0; lane < Lanes; lane += 16) {
      auto words = _mm512_loadu_si512(c[reg] + lane);
      if constexpr (Packed) {
        if constexpr (Rdna3) {
          words =
              _mm512_srl_epi32(words, _mm_cvtsi32_si128(flags & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0));
          words = _mm512_and_si512(words, _mm512_set1_epi32(0xffff));
        }
        auto mag = _mm512_and_si512(words, _mm512_set1_epi16(0x7fff));
        minimum = _mm512_min_epu16(minimum, _mm512_sub_epi16(mag, _mm512_set1_epi16(1)));
      } else {
        auto mag = _mm512_and_si512(words, _mm512_set1_epi32(INT32_MAX));
        minimum = _mm512_min_epu32(minimum, _mm512_sub_epi32(mag, _mm512_set1_epi32(1)));
      }
    }
  }
  bool subnormal = Packed ? bool(_mm512_cmplt_epu16_mask(minimum, _mm512_set1_epi16(127)))
                          : bool(_mm512_cmplt_epu32_mask(minimum, _mm512_set1_epi32(0x007fffff)));
  return !(_mm512_test_epi16_mask(out_of_range, _mm512_set1_epi16(short(0xc000))) | subnormal);
}

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

} // namespace

namespace rocjitsu::arith {

// Return whether operand magnitudes meet this loose pair-dot backend's
// range/denormal restrictions. Inputs and modifiers must use the selected layout.
template <int Lanes, bool Rdna3, bool Packed>
static bool supports_bf16_dot(uint32_t instruction_flags, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  // The common exponent interval [64,190] meets both pair-product bounds.
  // Use the full per-operand extrema check for asymmetric operand ranges.
  auto outside_common_range = [](__m512i words) {
    auto mag = _mm512_and_si512(words, _mm512_set1_epi16(0x7fff));
    auto offset = _mm512_sub_epi16(mag, _mm512_set1_epi16(64 << 7));
    return _mm512_cmp_epu16_mask(offset, _mm512_set1_epi16((127 << 7) - 1), _MM_CMPINT_GT) &
           _mm512_cmpneq_epi16_mask(mag, _mm512_setzero_si512());
  };
  __mmask32 outside = 0;
  for (int pair = 0; pair < 8; ++pair) {
    auto src = wmma_half_input<Lanes, Rdna3>(2 * pair);
    outside |= outside_common_range(_mm512_loadu_si512(a[src.reg] + src.lane));
    outside |= outside_common_range(_mm512_loadu_si512(b[src.reg] + src.lane));
  }
  if (outside) {
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

    // Require conservative exponent bounds on every possible product.
    if (minimum[0] + minimum[1] < 128 || maximum[0] + maximum[1] > 380) {
      return false;
    }
  }
  constexpr int Registers = 256 / Lanes / (Packed && !Rdna3 ? 2 : 1);
  for (int reg = 0; reg < Registers; ++reg) {
    for (int lane = 0; lane < Lanes; lane += 16) {
      auto words = _mm512_loadu_si512(c[reg] + lane);
      if constexpr (Packed) {
        if constexpr (Rdna3) {
          words = _mm512_srl_epi32(
              words, _mm_cvtsi32_si128(instruction_flags & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0));
          words = _mm512_and_si512(words, _mm512_set1_epi32(0xffff));
        }
        auto magnitude = _mm512_and_si512(words, _mm512_set1_epi16(0x7fff));
        auto tiny = _mm512_cmp_epu16_mask(magnitude, _mm512_set1_epi16(0x80), _MM_CMPINT_LT);
        if (tiny & _mm512_cmpneq_epi16_mask(magnitude, _mm512_setzero_si512())) {
          return false;
        }
      } else {
        auto magnitude = _mm512_and_si512(words, _mm512_set1_epi32(INT32_MAX));
        auto tiny = _mm512_cmp_epu32_mask(magnitude, _mm512_set1_epi32(0x00800000), _MM_CMPINT_LT);
        if (tiny & _mm512_cmpneq_epi32_mask(magnitude, _mm512_setzero_si512())) {
          return false;
        }
      }
    }
  }
  return true;
}

// Compute loose BF16 A*B+C after supports_bf16_dot accepts the inputs.
// Modifiers must be validated. D may
// overlap sources; packed gfx11 output preserves the unselected halfword.
template <int Lanes, bool Rdna3, bool Packed>
static void wmma_bf16(uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                      const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t a_sign = (instruction_flags & RJ_ARITH_WMMA_NEG_LO_A ? 0x8000U : 0) |
                          (instruction_flags & RJ_ARITH_WMMA_NEG_HI_A ? 0x80000000U : 0);
  auto b_sign = _mm512_set1_epi32((instruction_flags & RJ_ARITH_WMMA_NEG_LO_B ? 0x8000 : 0) |
                                  (instruction_flags & RJ_ARITH_WMMA_NEG_HI_B ? INT32_MIN : 0));
  // Loose arithmetic permits DPBF16 to flush subnormal intermediate sums.
  // DPBF16 consumes the high half first, with fixed DAZ/FTZ. Swap both
  // factors' halves to match the loose low-then-high order.
  alignas(64) uint32_t left[8][16];
  __m512i right[8];
  for (int step = 0; step < 8; ++step) {
    int pair = Packed && !Rdna3 && Lanes == 32 ? wmma_gfx12_w32_k(2 * step) / 2 : step;
    auto src = wmma_half_input<Lanes, Rdna3>(2 * pair);
    _mm512_store_si512(left[step],
                       _mm512_rol_epi32(_mm512_xor_si512(_mm512_loadu_si512(a[src.reg] + src.lane),
                                                         _mm512_set1_epi32(a_sign)),
                                        16));
    right[step] =
        _mm512_rol_epi32(_mm512_xor_si512(_mm512_loadu_si512(b[src.reg] + src.lane), b_sign), 16);
  }
  __m512 sums[16];
  wmma_load_acc<true, Lanes, Rdna3, Packed>(sums, instruction_flags, c);
  wmma_bf16_pairs<Packed, Rdna3>(sums, left, right);
  wmma_store_acc<true, Lanes, Rdna3, Packed, false>(sums, instruction_flags, d);
}

// Try loose BF16 WMMA with validated instruction flags. Return true after writing D,
// or false with D unchanged when inputs require another backend. D may alias inputs.
template <int Lanes, bool Rdna3, bool Packed>
bool wmma_x86_64_v4_bf16(uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c) {
  if (!bf16_fast_range<Lanes, Rdna3, Packed>(instruction_flags, a, b, c) &&
      !supports_bf16_dot<Lanes, Rdna3, Packed>(instruction_flags, a, b, c)) {
    return false;
  }
  wmma_bf16<Lanes, Rdna3, Packed>(instruction_flags, d, a, b, c);
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
