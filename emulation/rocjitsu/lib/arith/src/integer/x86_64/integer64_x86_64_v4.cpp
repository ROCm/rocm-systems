// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 wide integer arithmetic; stage both words before publishing aliases.
#include "integer/integer64.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>

namespace rocjitsu::arith {
namespace {
// Widen consecutive VGPR words into 64-bit CPU lanes.
__m512i words(const uint32_t *ptr) {
  return _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr)));
}
// Combine low/high VGPR planes into complete 64-bit CPU lanes.
__m512i pair(const uint32_t *const *ptr, int lane) {
  return _mm512_or_si512(words(ptr[0] + lane), _mm512_slli_epi64(words(ptr[1] + lane), 32));
}
} // namespace

// Evaluate validated controls, retaining inactive lanes and arbitrary register aliases.
template <Integer64 Op>
void integer64_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  uint32_t staged[2][32];
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = pair(a, lane);
    __m512i y;
    if constexpr (Op == Integer64::ShiftAdd) {
      y = words(b[0] + lane);
    } else {
      y = pair(b, lane);
    }
    __m512i result;
    if constexpr (Op == Integer64::Add) {
      result = _mm512_add_epi64(x, y);
      if (mode & RJ_ARITH_ALU_CLAMP) {
        result = _mm512_mask_set1_epi64(result, _mm512_cmplt_epu64_mask(result, x), -1);
      }
    } else if constexpr (Op == Integer64::Sub) {
      result = _mm512_sub_epi64(x, y);
      if (mode & RJ_ARITH_ALU_CLAMP) {
        result = _mm512_mask_set1_epi64(result, _mm512_cmplt_epu64_mask(x, y), 0);
      }
    } else if constexpr (Op == Integer64::Mul) {
      result = _mm512_mullo_epi64(x, y);
    } else if constexpr (Op == Integer64::ShiftAdd) {
      auto shift = _mm512_and_si512(y, _mm512_set1_epi64(7));
      shift = _mm512_maskz_mov_epi64(_mm512_cmple_epu64_mask(shift, _mm512_set1_epi64(4)), shift);
      result = _mm512_add_epi64(_mm512_sllv_epi64(x, shift), pair(c, lane));
    } else if constexpr (Op == Integer64::MinU) {
      result = _mm512_min_epu64(x, y);
    } else if constexpr (Op == Integer64::MaxU) {
      result = _mm512_max_epu64(x, y);
    } else if constexpr (Op == Integer64::MinI) {
      result = _mm512_min_epi64(x, y);
    } else {
      result = _mm512_max_epi64(x, y);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(staged[0] + lane),
                        _mm512_cvtepi64_epi32(result));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(staged[1] + lane),
                        _mm512_cvtepi64_epi32(_mm512_srli_epi64(result, 32)));
  }
  // Publish high words last when destination registers coincide.
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < 32; lane += 16) {
      auto result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(staged[reg] + lane));
      _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane), result);
    }
  }
}

template void integer64_x86_64_v4<Integer64::Add>(uint32_t, uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void integer64_x86_64_v4<Integer64::Sub>(uint32_t, uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void integer64_x86_64_v4<Integer64::Mul>(uint32_t, uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void integer64_x86_64_v4<Integer64::MinU>(uint32_t, uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void integer64_x86_64_v4<Integer64::MaxU>(uint32_t, uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void integer64_x86_64_v4<Integer64::MinI>(uint32_t, uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void integer64_x86_64_v4<Integer64::MaxI>(uint32_t, uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void integer64_x86_64_v4<Integer64::ShiftAdd>(uint32_t, uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);

} // namespace rocjitsu::arith
