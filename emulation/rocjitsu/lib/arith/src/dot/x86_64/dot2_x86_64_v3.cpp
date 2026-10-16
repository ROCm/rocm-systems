// SPDX-License-Identifier: MIT

// AVX2 loose FP16/BF16 DOT2 using widened FP32 arithmetic.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Widen selected FP16/BF16 halfwords and apply the supplied sign flip.
template <bool Bf16> __m256 widen(__m256i words, int shift, uint32_t negate) {
  auto halves =
      _mm256_and_si256(_mm256_srl_epi32(words, _mm_cvtsi32_si128(shift)), _mm256_set1_epi32(65535));
  halves = _mm256_xor_si256(halves, _mm256_set1_epi32(negate ? 0x8000 : 0));
  if constexpr (Bf16) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(halves, 16));
  } else {
    return _mm256_cvtph_ps(
        _mm_packus_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1)));
  }
}

// Accumulate loose FP16/BF16 DOT2 products for the selected lanes. Requires validated
// controls; preserve inactive destination lanes.
template <bool Bf16>
void dot(uint32_t exec_mask, uint32_t modifiers, uint32_t *d, const uint32_t *a, const uint32_t *b,
         const uint32_t *c) {
  const int a0 = modifiers & RJ_ARITH_DOT_LO_A_HIGH ? 16 : 0;
  const int b0 = modifiers & RJ_ARITH_DOT_LO_B_HIGH ? 16 : 0;
  const int a1 = modifiers & RJ_ARITH_DOT_HI_A_LOW ? 0 : 16;
  const int b1 = modifiers & RJ_ARITH_DOT_HI_B_LOW ? 0 : 16;
  const auto neg_c = _mm256_set1_epi32(modifiers & RJ_ARITH_DOT_NEG_C ? INT32_MIN : 0);
  for (int lane = 0; lane < 32; lane += 8) {
    auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto acc = _mm256_castsi256_ps(
        _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane)), neg_c));
    acc = _mm256_fmadd_ps(widen<Bf16>(va, a0, modifiers & RJ_ARITH_DOT_NEG_LO_A),
                          widen<Bf16>(vb, b0, modifiers & RJ_ARITH_DOT_NEG_LO_B), acc);
    acc = _mm256_fmadd_ps(widen<Bf16>(va, a1, modifiers & RJ_ARITH_DOT_NEG_HI_A),
                          widen<Bf16>(vb, b1, modifiers & RJ_ARITH_DOT_NEG_HI_B), acc);
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask,
                           _mm256_castps_si256(acc));
  }
}

} // namespace

// Accumulate loose FP16/BF16 DOT2 products. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void dot2_x86_64_v3(bool bf16, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                    const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  if (bf16) {
    dot<true>(exec_mask, modifiers, d, a, b, c);
  } else {
    dot<false>(exec_mask, modifiers, d, a, b, c);
  }
}

} // namespace rocjitsu::arith
