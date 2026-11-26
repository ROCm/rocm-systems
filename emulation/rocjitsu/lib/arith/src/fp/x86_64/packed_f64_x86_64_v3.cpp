// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v3 packed FP64 selection fixups and alias-safe masked result publication.
#include "fp/packed_f64.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
namespace rocjitsu::arith {
// Finish both packed elements before stores; arithmetic itself stays in existing FP64 backends.
void publish_packed_f64_x86_64_v3(bool selection, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                  uint32_t *const *d, uint32_t *const *result,
                                  const uint32_t *const *a, const uint32_t *const *b) {
  if (selection || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    for (int reg = 0; reg < 4; reg += 2) {
      for (int lane = 0; lane < 32; lane += 8) {
        auto low = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)),
             high = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg + 1] + lane));
        if (selection && !(mode & RJ_ARITH_PK_CLAMP)) {
          // Legacy packed number selection canonicalizes two NaNs rather than keeping a payload.
          auto isnan = [&](const uint32_t *lo, const uint32_t *hi) {
            auto ah =
                _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(hi + lane)),
                                 _mm256_set1_epi32(0x7fffffff));
            auto nonzero = _mm256_xor_si256(
                _mm256_cmpeq_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(lo + lane)),
                                   _mm256_setzero_si256()),
                _mm256_set1_epi32(-1));
            return _mm256_or_si256(
                _mm256_cmpgt_epi32(ah, _mm256_set1_epi32(0x7ff00000)),
                _mm256_and_si256(_mm256_cmpeq_epi32(ah, _mm256_set1_epi32(0x7ff00000)), nonzero));
          };
          auto both = _mm256_and_si256(isnan(a[reg], a[reg + 1]), isnan(b[reg], b[reg + 1]));
          low = _mm256_andnot_si256(both, low);
          high = _mm256_blendv_epi8(high, _mm256_set1_epi32(0x7ff80000), both);
        }
        if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
          auto tiny = _mm256_cmpeq_epi32(_mm256_and_si256(high, _mm256_set1_epi32(0x7ff00000)),
                                         _mm256_setzero_si256());
          low = _mm256_andnot_si256(tiny, low);
          high = _mm256_blendv_epi8(
              high, _mm256_and_si256(high, _mm256_set1_epi32(int32_t(0x80000000U))), tiny);
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[reg] + lane), low);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[reg + 1] + lane), high);
      }
    }
  }
  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < 32; lane += 8) {
      auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                      _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(
          reinterpret_cast<int *>(d[reg] + lane), active,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
    }
  }
}
} // namespace rocjitsu::arith
