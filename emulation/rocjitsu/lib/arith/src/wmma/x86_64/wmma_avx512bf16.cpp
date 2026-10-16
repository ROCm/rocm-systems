// SPDX-License-Identifier: MIT

// AVX-512 BF16 pair-dot backend for loose gfx11/gfx12 BF16 WMMA, including
// both wave sizes and FP32 or packed BF16 results. Guards operand ranges and
// denormals, declining inputs that require the general arithmetic path.

#include "common/simd.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Return the smallest unsigned 16-bit exponent across all packed halfwords.
int min_exponent(__m512i value) {
  value = _mm512_min_epu16(value, _mm512_srli_epi32(value, 16));
  return int(_mm512_reduce_min_epu32(_mm512_and_si512(value, _mm512_set1_epi32(65535))));
}

// Return the largest unsigned 16-bit exponent across all packed halfwords.
int max_exponent(__m512i value) {
  value = _mm512_max_epu16(value, _mm512_srli_epi32(value, 16));
  return int(_mm512_reduce_max_epu32(_mm512_and_si512(value, _mm512_set1_epi32(65535))));
}

} // namespace

namespace rocjitsu::arith {

// Return whether operand magnitudes meet this loose pair-dot backend's
// range/denormal restrictions. Inputs and modifiers must use the selected layout.
template <int Lanes, bool Rdna3, bool Packed>
static bool supports_bf16_dot(uint32_t mode, const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  int minimum[2], maximum[2];
  for (int operand = 0; operand < 2; ++operand) {
    const uint32_t *const *v = operand ? b : a;
    __m512i min_exp = _mm512_set1_epi16(255), max_exp = _mm512_setzero_si512();
    for (int reg = 0; reg < (Rdna3 ? 8 : 128 / Lanes); ++reg) {
      for (int lane = 0; lane < Lanes; lane += 16) {
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

  // Preserve the previous conservative bounds on every possible product.
  if (minimum[0] + minimum[1] < 128 || maximum[0] + maximum[1] > 380) {
    return false;
  }
  constexpr int Registers = 256 / Lanes / (Packed && !Rdna3 ? 2 : 1);
  for (int reg = 0; reg < Registers; ++reg) {
    for (int lane = 0; lane < Lanes; lane += 16) {
      auto words = _mm512_loadu_si512(c[reg] + lane);
      if constexpr (Packed) {
        if constexpr (Rdna3) {
          words =
              _mm512_srl_epi32(words, _mm_cvtsi32_si128(mode & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0));
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
// Modifiers must be validated; Modified=false requires no modifiers. D may
// overlap sources; packed gfx11 output preserves the unselected halfword.
template <int Lanes, bool Rdna3, bool Packed, bool Modified>
static void wmma_bf16(uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                      const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int Groups = Rdna3 ? Lanes / 16 : 1;
  constexpr int Registers = 256 / Lanes;
  constexpr int DestRegisters = Packed && !Rdna3 ? Registers / 2 : Registers;
  const uint32_t a_sign = (mode & RJ_ARITH_WMMA_NEG_LO_A ? 0x8000U : 0) |
                          (mode & RJ_ARITH_WMMA_NEG_HI_A ? 0x80000000U : 0);
  auto b_sign = _mm512_set1_epi32((mode & RJ_ARITH_WMMA_NEG_LO_B ? 0x8000 : 0) |
                                  (mode & RJ_ARITH_WMMA_NEG_HI_B ? INT32_MIN : 0));
  // DPBF16 consumes the high half first, with fixed DAZ/FTZ. Swap both
  // factors' halves to match the loose low-then-high order, not the exact GPU model.
  __m512i right[Groups][8];
  for (int group = 0; group < Groups; ++group) {
    for (int pair = 0; pair < 8; ++pair) {
      int reg = Rdna3 ? pair : pair % (Registers / 2);
      int lane = 16 * (Rdna3 ? group : pair / (Registers / 2));
      right[group][pair] =
          _mm512_rol_epi32(_mm512_xor_si512(_mm512_loadu_si512(b[reg] + lane), b_sign), 16);
    }
  }
  uint32_t result[DestRegisters][Lanes];
  if constexpr (Packed && !Rdna3) {
    for (auto &reg : result) {
      for (auto &word : reg) {
        word = 0;
      }
    }
  }
  for (int row = 0; row < 16; ++row) {
    int group = Rdna3 ? row % Groups : 0;
    int reg = Rdna3 ? row / Groups : row % Registers;
    int lane = Rdna3 ? 16 * group : 16 * (row / 8) + (Lanes == 64 ? 32 * ((row / 4) % 2) : 0);
    int dest = Packed && !Rdna3 ? reg / 2 : reg;
    int shift = Rdna3 ? (mode & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0) : 16 * (reg % 2);
    auto bits = _mm512_loadu_si512(c[dest] + lane);
    if constexpr (Packed) {
      bits = _mm512_slli_epi32(_mm512_srl_epi32(bits, _mm_cvtsi32_si128(shift)), 16);
    }
    if constexpr (Modified) {
      bits = _mm512_and_si512(bits, _mm512_set1_epi32(mode & RJ_ARITH_WMMA_ABS_C ? INT32_MAX : -1));
      bits = _mm512_xor_si512(bits, _mm512_set1_epi32(mode & RJ_ARITH_WMMA_NEG_C ? INT32_MIN : 0));
    }
    auto acc = _mm512_castsi512_ps(bits);
// Keep pair addresses constant and the decoded B vectors in registers.
#pragma GCC unroll 8
    for (int step = 0; step < 8; ++step) {
      int pair = Packed && !Rdna3 && Lanes == 32
                     ? (step & 1) | ((step & 2) << 1) | ((step & 4) >> 1)
                     : step;
      int ar = Rdna3 ? pair : pair % (Registers / 2);
      int al = row + 16 * (Rdna3 ? group : pair / (Registers / 2));
      uint32_t aw = a[ar][al] ^ a_sign;
      auto left = _mm512_set1_epi32(int((aw << 16) | (aw >> 16)));
      acc = _mm512_dpbf16_ps(acc, (__m512bh)left, (__m512bh)right[group][pair]);
      // Packed GPU WMMA truncates BF16 after each DOT2 (gfx11) or DOT4
      // (gfx12). VCVTNEPS2BF16 rounds instead and is not interchangeable.
      if constexpr (Packed) {
        if (Rdna3 || step % 2) {
          acc = _mm512_castsi512_ps(
              _mm512_and_si512(_mm512_castps_si512(acc), _mm512_set1_epi32(-65536)));
        }
      }
    }
    bits = _mm512_castps_si512(acc);
    if constexpr (Packed) {
      bits = _mm512_sll_epi32(_mm512_srli_epi32(bits, 16), _mm_cvtsi32_si128(shift));
      auto old = _mm512_loadu_si512((Rdna3 ? d[dest] : result[dest]) + lane);
      bits =
          _mm512_or_si512(bits, _mm512_andnot_si512(_mm512_set1_epi32(int(0xffffU << shift)), old));
    }
    _mm512_storeu_si512(result[dest] + lane, bits);
  }
  // Stage every source/preserved-half read before writing potentially aliased D.
  for (int reg = 0; reg < DestRegisters; ++reg) {
    for (int lane = 0; lane < Lanes; lane += 16) {
      _mm512_storeu_si512(d[reg] + lane, _mm512_loadu_si512(result[reg] + lane));
    }
  }
}

// Try loose BF16 WMMA with validated modifiers. Return true after writing D,
// or false with D unchanged when inputs require another backend. D may alias inputs.
template <int Lanes, bool Rdna3, bool Packed>
bool wmma_avx512bf16(uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c) {
  if (!supports_bf16_dot<Lanes, Rdna3, Packed>(mode, a, b, c)) {
    return false;
  }
  if (mode) {
    wmma_bf16<Lanes, Rdna3, Packed, true>(mode, d, a, b, c);
  } else {
    wmma_bf16<Lanes, Rdna3, Packed, false>(0, d, a, b, c);
  }
  return true;
}

template bool wmma_avx512bf16<32, false, false>(uint32_t, uint32_t *const *,
                                                const uint32_t *const *, const uint32_t *const *,
                                                const uint32_t *const *);
template bool wmma_avx512bf16<32, false, true>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                               const uint32_t *const *, const uint32_t *const *);
template bool wmma_avx512bf16<32, true, false>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                               const uint32_t *const *, const uint32_t *const *);
template bool wmma_avx512bf16<32, true, true>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *);
template bool wmma_avx512bf16<64, false, false>(uint32_t, uint32_t *const *,
                                                const uint32_t *const *, const uint32_t *const *,
                                                const uint32_t *const *);
template bool wmma_avx512bf16<64, false, true>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                               const uint32_t *const *, const uint32_t *const *);
template bool wmma_avx512bf16<64, true, false>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                               const uint32_t *const *, const uint32_t *const *);
template bool wmma_avx512bf16<64, true, true>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *);

} // namespace rocjitsu::arith
