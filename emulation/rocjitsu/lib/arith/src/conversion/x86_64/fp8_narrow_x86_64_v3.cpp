// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16/FP32 narrowing to FP8/BF8 with nearest or seeded stochastic rounding.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/x86_64/fp8_narrow_x86_64_v3.h"
#include "conversion/fp8_narrow_impl.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <immintrin.h>
#include <stdint.h>

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
    auto value = x86_64_v3::fp8_narrow_detail::narrow<Bf8, Stochastic>(av, bv, saturate, policy);
    if constexpr (!Stochastic) {
      bv = _mm256_xor_si256(_mm256_and_si256(bv, keep_b), flip_b);
      auto high = x86_64_v3::fp8_narrow_detail::narrow<Bf8, false>(bv, _mm256_setzero_si256(),
                                                                   saturate, policy);
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
