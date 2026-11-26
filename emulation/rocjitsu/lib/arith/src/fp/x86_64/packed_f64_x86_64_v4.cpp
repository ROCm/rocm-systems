// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v4 packed FP64 selection fixups and alias-safe masked result publication.
#include "fp/packed_f64.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
namespace rocjitsu::arith {
// Finish both packed elements before stores; arithmetic itself stays in existing FP64 backends.
void publish_packed_f64_x86_64_v4(bool selection, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                  uint32_t *const *d, uint32_t *const *result,
                                  const uint32_t *const *a, const uint32_t *const *b) {
  if (selection || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    for (int reg = 0; reg < 4; reg += 2) {
      for (int lane = 0; lane < 32; lane += 16) {
        auto low = _mm512_loadu_si512(result[reg] + lane),
             high = _mm512_loadu_si512(result[reg + 1] + lane);
        if (selection && !(mode & RJ_ARITH_PK_CLAMP)) {
          // Legacy packed number selection canonicalizes two NaNs rather than keeping a payload.
          auto isnan = [&](const uint32_t *lo, const uint32_t *hi) {
            auto ah =
                _mm512_and_si512(_mm512_loadu_si512(hi + lane), _mm512_set1_epi32(0x7fffffff));
            return __mmask16(
                _mm512_cmpgt_epi32_mask(ah, _mm512_set1_epi32(0x7ff00000)) |
                (_mm512_cmpeq_epi32_mask(ah, _mm512_set1_epi32(0x7ff00000)) &
                 _mm512_cmpneq_epi32_mask(_mm512_loadu_si512(lo + lane), _mm512_setzero_si512())));
          };
          auto both = __mmask16(isnan(a[reg], a[reg + 1]) & isnan(b[reg], b[reg + 1]));
          low = _mm512_mask_mov_epi32(low, both, _mm512_setzero_si512());
          high = _mm512_mask_mov_epi32(high, both, _mm512_set1_epi32(0x7ff80000));
        }
        if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
          auto tiny = _mm512_cmpeq_epi32_mask(_mm512_and_si512(high, _mm512_set1_epi32(0x7ff00000)),
                                              _mm512_setzero_si512());
          low = _mm512_mask_mov_epi32(low, tiny, _mm512_setzero_si512());
          high = _mm512_mask_and_epi32(high, tiny, high, _mm512_set1_epi32(int32_t(0x80000000U)));
        }
        _mm512_storeu_si512(result[reg] + lane, low);
        _mm512_storeu_si512(result[reg + 1] + lane, high);
      }
    }
  }
  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < 32; lane += 16) {
      _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane),
                               _mm512_loadu_si512(result[reg] + lane));
    }
  }
}
} // namespace rocjitsu::arith
