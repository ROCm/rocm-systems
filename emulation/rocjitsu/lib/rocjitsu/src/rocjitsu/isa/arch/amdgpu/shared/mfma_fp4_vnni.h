// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/mfma_fp4_avx512.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mfma_fp4_dispatch.h"

#include <cstddef>
#include <cstdint>

#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__FMA__)
namespace rocjitsu::amdgpu {

inline void mfma_fp4_stage_vnni(const uint32_t *packed_a, const uint32_t *packed_b, uint32_t reg,
                                uint32_t *a, uint32_t *b) {
  const __m256i positive = _mm256_setr_epi32(12, 13, 14, 15, 16, 18, 20, 24);
  const __m256i negative = _mm256_setr_epi32(12, 11, 10, 9, 8, 6, 4, 0);
  const __m256i shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  const __m512i lut =
      _mm512_setr_epi32(0, 1, 2, 3, 4, 6, 8, 12, 0, 255, 254, 253, 252, 250, 248, 244);
  for (uint32_t block = 0; block < 4; ++block) {
    for (uint32_t row = 0; row < 16; ++row) {
      auto indices = _mm256_srlv_epi32(_mm256_set1_epi32(packed_a[block * 16 + row]), shifts);
      auto values = _mm256_permutex2var_epi32(positive, indices, negative);
      _mm_storel_epi64(reinterpret_cast<__m128i *>(a + row * 32 + block * 8 + reg * 2),
                       _mm256_cvtepi32_epi8(values));
    }
    const auto words = _mm512_loadu_si512(packed_b + block * 16);
    [&]<std::size_t... Group>(std::index_sequence<Group...>) {
      (
          [&] {
            auto p0 = _mm512_permutexvar_epi32(_mm512_srli_epi32(words, Group * 16), lut);
            auto p1 = _mm512_slli_epi32(
                _mm512_permutexvar_epi32(_mm512_srli_epi32(words, Group * 16 + 4), lut), 8);
            auto p2 = _mm512_slli_epi32(
                _mm512_permutexvar_epi32(_mm512_srli_epi32(words, Group * 16 + 8), lut), 16);
            auto p3 = _mm512_slli_epi32(
                _mm512_permutexvar_epi32(_mm512_srli_epi32(words, Group * 16 + 12), lut), 24);
            _mm512_storeu_si512(b + (block * 8 + reg * 2 + Group) * 16,
                                _mm512_or_si512(_mm512_or_si512(p0, p1), _mm512_or_si512(p2, p3)));
          }(),
          ...);
    }(std::make_index_sequence<2>{});
  }
}

// Four exact integer products per lane. A holds 2*FP4+12 as unsigned bytes;
// B holds 2*FP4 as signed bytes. Subtract 12*sum(B) before scaling by 1/4.
// A biased 32-term sum has magnitude at most 9216; after correction it is at
// most 4608. Both integer accumulation and conversion to FP32 are exact.
// Preconditions: host RNE and VNNI support. NormalScales additionally requires
// the common normal-scale gate.
// Keep the four scaled block additions in their original order.
template <bool NormalScales = true>
__attribute__((target("avx512vnni"))) inline void
mfma_fp4_16x16x128_vnni(const uint32_t *a, const uint32_t *b, float *c, const uint8_t *a_scales,
                        const uint8_t *b_scales) {
  for (uint32_t block = 0; block < 4; ++block) {
    __m512i sums[16];
    for (auto &sum : sums)
      sum = _mm512_setzero_si512();
    __m512i correction = _mm512_setzero_si512();
    for (uint32_t k = block * 8; k < (block + 1) * 8; ++k) {
      const auto bv = _mm512_loadu_si512(b + k * 16);
      correction = _mm512_dpbusd_epi32(correction, _mm512_set1_epi32(0x0c0c0c0c), bv);
      [&]<std::size_t... Row>(std::index_sequence<Row...>)
          __attribute__((always_inline, target("avx512vnni"))) {
        ((sums[Row] = _mm512_dpbusd_epi32(sums[Row], _mm512_set1_epi32(a[Row * 32 + k]), bv)), ...);
      }
      (std::make_index_sequence<16>{});
    }
    const auto bscale = _mm512_cvtepu8_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(b_scales + block * 16)));
    [&]<std::size_t... Row>(std::index_sequence<Row...>) __attribute__((always_inline)) {
      (
          [&] {
            auto value = _mm512_cvtepi32_ps(_mm512_sub_epi32(sums[Row], correction));
            __m512 scaled;
            if constexpr (NormalScales) {
              auto shift = _mm512_slli_epi32(
                  _mm512_add_epi32(bscale,
                                   _mm512_set1_epi32(int(a_scales[block * 16 + Row]) - 256)),
                  23);
              auto bits = _mm512_castps_si512(value);
              auto mask = _mm512_cmp_ps_mask(value, _mm512_setzero_ps(), _CMP_NEQ_OQ);
              scaled = _mm512_castsi512_ps(_mm512_mask_add_epi32(bits, mask, bits, shift));
            } else {
              scaled = mfma_fp4_scale_avx512(_mm512_mul_ps(value, _mm512_set1_ps(0.25f)),
                                             a_scales[block * 16 + Row], b_scales + block * 16);
            }
            _mm512_storeu_ps(c + Row * 16,
                             mfma_fp4_add_avx512(_mm512_loadu_ps(c + Row * 16), scaled));
          }(),
          ...);
    }
    (std::make_index_sequence<16>{});
  }
}

} // namespace rocjitsu::amdgpu
#endif
