// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 wide integer arithmetic; stage both words before publishing aliases.
#include "integer/integer64.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>

namespace rocjitsu::arith {
namespace {
// Widen consecutive VGPR words into 64-bit CPU lanes.
__m256i words(const uint32_t *ptr) {
  return _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr)));
}
// Combine low/high VGPR planes into complete 64-bit CPU lanes.
__m256i pair(const uint32_t *const *ptr, int lane) {
  return _mm256_or_si256(words(ptr[0] + lane), _mm256_slli_epi64(words(ptr[1] + lane), 32));
}
} // namespace

// Evaluate validated controls, retaining inactive lanes and arbitrary register aliases.
template <Integer64 Op>
void integer64_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  uint32_t staged[2][32];
  for (int lane = 0; lane < 32; lane += 4) {
    auto x = pair(a, lane);
    __m256i y;
    if constexpr (Op == Integer64::ShiftAdd) {
      y = words(b[0] + lane);
    } else {
      y = pair(b, lane);
    }
    __m256i result;
    auto bias = _mm256_set1_epi64x(INT64_MIN);
    auto x_biased = _mm256_xor_si256(x, bias), y_biased = _mm256_xor_si256(y, bias);
    if constexpr (Op == Integer64::Add) {
      result = _mm256_add_epi64(x, y);
      if (mode & RJ_ARITH_ALU_CLAMP) {
        auto overflow = _mm256_cmpgt_epi64(x_biased, _mm256_xor_si256(result, bias));
        result = _mm256_or_si256(result, overflow);
      }
    } else if constexpr (Op == Integer64::Sub) {
      result = _mm256_sub_epi64(x, y);
      if (mode & RJ_ARITH_ALU_CLAMP) {
        result = _mm256_andnot_si256(_mm256_cmpgt_epi64(y_biased, x_biased), result);
      }
    } else if constexpr (Op == Integer64::Mul) {
      // The high-high product is beyond bit 63 and cannot affect the wrapped result.
      auto cross = _mm256_add_epi64(_mm256_mul_epu32(x, _mm256_srli_epi64(y, 32)),
                                    _mm256_mul_epu32(_mm256_srli_epi64(x, 32), y));
      result = _mm256_add_epi64(_mm256_mul_epu32(x, y), _mm256_slli_epi64(cross, 32));
    } else if constexpr (Op == Integer64::ShiftAdd) {
      auto shift = _mm256_and_si256(y, _mm256_set1_epi64x(7));
      shift = _mm256_andnot_si256(_mm256_cmpgt_epi64(shift, _mm256_set1_epi64x(4)), shift);
      result = _mm256_add_epi64(_mm256_sllv_epi64(x, shift), pair(c, lane));
    } else {
      constexpr bool signed_order = Op == Integer64::MinI || Op == Integer64::MaxI;
      constexpr bool maximum = Op == Integer64::MaxU || Op == Integer64::MaxI;
      auto greater =
          signed_order ? _mm256_cmpgt_epi64(x, y) : _mm256_cmpgt_epi64(x_biased, y_biased);
      result = maximum ? _mm256_blendv_epi8(y, x, greater) : _mm256_blendv_epi8(x, y, greater);
    }
    auto output = _mm256_permutevar8x32_epi32(result, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(staged[0] + lane), _mm256_castsi256_si128(output));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(staged[1] + lane),
                     _mm256_extracti128_si256(output, 1));
  }
  // Publish high words last when destination registers coincide.
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < 32; lane += 8) {
      auto result = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(staged[reg] + lane));
      auto mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[reg] + lane), mask, result);
    }
  }
}

template void integer64_x86_64_v3<Integer64::Add>(uint32_t, uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void integer64_x86_64_v3<Integer64::Sub>(uint32_t, uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void integer64_x86_64_v3<Integer64::Mul>(uint32_t, uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void integer64_x86_64_v3<Integer64::MinU>(uint32_t, uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void integer64_x86_64_v3<Integer64::MaxU>(uint32_t, uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void integer64_x86_64_v3<Integer64::MinI>(uint32_t, uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void integer64_x86_64_v3<Integer64::MaxI>(uint32_t, uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void integer64_x86_64_v3<Integer64::ShiftAdd>(uint32_t, uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);

} // namespace rocjitsu::arith
