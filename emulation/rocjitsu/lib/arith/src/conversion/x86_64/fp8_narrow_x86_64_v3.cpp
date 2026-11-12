// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16/FP32 narrowing to FP8/BF8 with nearest or seeded stochastic rounding.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/fp8_narrow_impl.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

using rocjitsu::arith::Fp8NarrowPolicy;

// Select yes where signed a > b, and no otherwise.
__m256i select_greater(__m256i a, __m256i b, __m256i yes, __m256i no) {
  return _mm256_blendv_epi8(no, yes, _mm256_cmpgt_epi32(a, b));
}

// Narrow FP32 bits to FP8/BF8 using nearest or seeded stochastic rounding and optional
// saturation.
template <bool Bf8, bool Stochastic>
__m256i narrow(__m256i raw, __m256i seed, bool saturate, Fp8NarrowPolicy policy) {
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  bool fnuz = policy == Fp8NarrowPolicy::Fnuz;
  bool unsigned_format = policy == Fp8NarrowPolicy::E5M3;
  bool rdna = policy == Fp8NarrowPolicy::Gfx12;
  int terminal = fnuz ? 128 : unsigned_format ? 255 : Bf8 ? 124 : 127;
  int bias = unsigned_format ? 15 : (Bf8 ? 15 : 7) + int(fnuz);
  auto one = _mm256_set1_epi32(1), zero = _mm256_setzero_si256();
  auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi32(0x7fffffff));
  auto exponent = _mm256_sub_epi32(_mm256_srli_epi32(magnitude, 23), _mm256_set1_epi32(127 - bias));
  auto significand = _mm256_or_si256(_mm256_and_si256(magnitude, _mm256_set1_epi32(0x7fffff)),
                                     _mm256_set1_epi32(0x800000));
  auto alignment = _mm256_max_epi32(_mm256_sub_epi32(one, exponent), zero);
  __m256i result;
  if constexpr (Stochastic) {
    if (rdna) {
      significand = _mm256_srlv_epi32(significand, alignment);
      result = _mm256_srli_epi32(
          _mm256_add_epi32(significand, _mm256_srli_epi32(seed, 9 + fraction_bits)),
          23 - fraction_bits);
    } else {
      auto shift = _mm256_add_epi32(alignment, _mm256_set1_epi32(23 - fraction_bits));
      auto random = _mm256_srlv_epi32(seed, _mm256_sub_epi32(_mm256_set1_epi32(32), shift));
      result = _mm256_srlv_epi32(_mm256_add_epi32(significand, random), shift);
      result = select_greater(shift, _mm256_set1_epi32(24), zero, result);
    }
  } else {
    auto shift = _mm256_min_epi32(
        _mm256_add_epi32(alignment, _mm256_set1_epi32(23 - fraction_bits)), _mm256_set1_epi32(31));
    auto half = _mm256_sllv_epi32(one, _mm256_sub_epi32(shift, one));
    auto odd = _mm256_and_si256(_mm256_srlv_epi32(significand, shift), one);
    auto rounded =
        _mm256_add_epi32(significand, _mm256_add_epi32(_mm256_sub_epi32(half, one), odd));
    result = _mm256_srlv_epi32(rounded, shift);
  }
  auto normal_exponent =
      _mm256_slli_epi32(_mm256_max_epi32(_mm256_sub_epi32(exponent, one), zero), fraction_bits);
  result = _mm256_add_epi32(result, normal_exponent);
  auto limit = _mm256_set1_epi32(terminal);
  if (saturate) {
    auto finite_limit = _mm256_set1_epi32(terminal - 1);
    limit = rdna ? select_greater(_mm256_set1_epi32(0x7f800000), magnitude, finite_limit, limit)
                 : finite_limit;
  }
  result = _mm256_min_epu32(result, limit);
  auto sign =
      unsigned_format ? zero : _mm256_and_si256(_mm256_srli_epi32(raw, 24), _mm256_set1_epi32(128));
  if (fnuz) {
    sign = select_greater(result, zero, sign, zero);
    auto signed_result = _mm256_or_si256(result, sign);
    result = select_greater(result, _mm256_set1_epi32(127), _mm256_set1_epi32(128), signed_result);
  } else {
    result = _mm256_or_si256(result, sign);
  }
  auto nan = fnuz              ? _mm256_set1_epi32(128)
             : unsigned_format ? _mm256_set1_epi32(255)
             : rdna            ? _mm256_set1_epi32(Bf8 ? 0xfe : 0xff)
                               : _mm256_or_si256(sign, _mm256_set1_epi32(127));
  return select_greater(magnitude, _mm256_set1_epi32(0x7f800000), nan, result);
}

} // namespace

namespace rocjitsu::arith {

// Narrow FP16/FP32 to FP8/BF8. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Bf8, bool Stochastic, bool Half>
void fp8_narrow_x86_64_v3(uint32_t exec_mask, uint32_t mode, bool saturate, uint32_t *d,
                          const uint32_t *a, const uint32_t *b, Fp8NarrowPolicy policy) {
  unsigned shift = Stochastic ? ((mode >> 16) & 3) * 8 : mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  auto count = _mm_cvtsi32_si128(int(shift));
  uint32_t selected_bits = (Stochastic ? 255U : 65535U) << shift;
  auto selected = _mm256_set1_epi32(int(selected_bits));
  auto keep_a = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1);
  auto flip_a = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  auto keep_b = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_B ? INT32_MAX : -1);
  auto flip_b = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_B ? INT32_MIN : 0);
  for (int lane = 0; lane < 32; lane += 8) {
    auto av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    __m256i bv;
    if constexpr (Half && !Stochastic) {
      bv = _mm256_setzero_si256();
    } else {
      bv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    }
    if constexpr (Half) {
      // Native half widening may quiet NaNs; narrowing only depends on their sign.
      auto widen = [](__m256i raw) {
        raw = _mm256_and_si256(raw, _mm256_set1_epi32(0xffff));
        auto packed =
            _mm_packus_epi32(_mm256_castsi256_si128(raw), _mm256_extracti128_si256(raw, 1));
        return _mm256_castps_si256(_mm256_cvtph_ps(packed));
      };
      if constexpr (!Stochastic) {
        bv = widen(_mm256_srli_epi32(av, 16));
      }
      if (Stochastic && (mode & RJ_ARITH_ALU_HIGH_A)) {
        av = _mm256_srli_epi32(av, 16);
      }
      av = widen(av);
    }
    auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    av = _mm256_xor_si256(_mm256_and_si256(av, keep_a), flip_a);
    auto value = narrow<Bf8, Stochastic>(av, bv, saturate, policy);
    if constexpr (!Stochastic) {
      bv = _mm256_xor_si256(_mm256_and_si256(bv, keep_b), flip_b);
      auto high = narrow<Bf8, false>(bv, _mm256_setzero_si256(), saturate, policy);
      value = _mm256_or_si256(value, _mm256_slli_epi32(high, 8));
    }
    value = _mm256_or_si256(_mm256_andnot_si256(selected, old), _mm256_sll_epi32(value, count));
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, value);
  }
}

template void fp8_narrow_x86_64_v3<false, false, false>(uint32_t, uint32_t, bool, uint32_t *,
                                                        const uint32_t *, const uint32_t *,
                                                        Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v3<false, false, true>(uint32_t, uint32_t, bool, uint32_t *,
                                                       const uint32_t *, const uint32_t *,
                                                       Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v3<false, true, false>(uint32_t, uint32_t, bool, uint32_t *,
                                                       const uint32_t *, const uint32_t *,
                                                       Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v3<false, true, true>(uint32_t, uint32_t, bool, uint32_t *,
                                                      const uint32_t *, const uint32_t *,
                                                      Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v3<true, false, false>(uint32_t, uint32_t, bool, uint32_t *,
                                                       const uint32_t *, const uint32_t *,
                                                       Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v3<true, false, true>(uint32_t, uint32_t, bool, uint32_t *,
                                                      const uint32_t *, const uint32_t *,
                                                      Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v3<true, true, false>(uint32_t, uint32_t, bool, uint32_t *,
                                                      const uint32_t *, const uint32_t *,
                                                      Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v3<true, true, true>(uint32_t, uint32_t, bool, uint32_t *,
                                                     const uint32_t *, const uint32_t *,
                                                     Fp8NarrowPolicy);

} // namespace rocjitsu::arith
