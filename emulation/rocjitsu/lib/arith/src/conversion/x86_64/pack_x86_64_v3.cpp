// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 saturating INT16-to-U8 packing and selection of two FP16 source halfwords.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/pack.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Pack saturated bytes or selected FP16 halves. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Saturate>
void pack_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                    const uint32_t *a, const uint32_t *b, bool quiet_nan) {
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    __m256i result;
    if constexpr (Saturate) {
      auto lo = _mm256_srai_epi32(_mm256_slli_epi32(x, 16), 16);
      auto hi = _mm256_srai_epi32(x, 16);
      lo = _mm256_min_epi32(_mm256_max_epi32(lo, _mm256_setzero_si256()), _mm256_set1_epi32(255));
      hi = _mm256_min_epi32(_mm256_max_epi32(hi, _mm256_setzero_si256()), _mm256_set1_epi32(255));
      auto packed = _mm256_or_si256(lo, _mm256_slli_epi32(hi, 8));
      auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      unsigned shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
      auto selected = _mm256_sll_epi32(packed, _mm_cvtsi32_si128(shift));
      result = _mm256_or_si256(_mm256_andnot_si256(_mm256_set1_epi32(int(65535u << shift)), old),
                               selected);
    } else {
      auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
      x = _mm256_srl_epi32(x, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
      y = _mm256_srl_epi32(y, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0));
      x = _mm256_and_si256(x, _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 32767 : 65535));
      y = _mm256_and_si256(y, _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_B ? 32767 : 65535));
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        auto flush = [](__m256i bits) {
          auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x400),
                                         _mm256_and_si256(bits, _mm256_set1_epi32(0x7fff)));
          return _mm256_blendv_epi8(bits, _mm256_and_si256(bits, _mm256_set1_epi32(0x8000)), tiny);
        };
        x = flush(x);
        y = flush(y);
      }
      if (quiet_nan) {
        x = _mm256_or_si256(
            x, _mm256_and_si256(_mm256_cmpgt_epi32(_mm256_and_si256(x, _mm256_set1_epi32(32767)),
                                                   _mm256_set1_epi32(0x7c00)),
                                _mm256_set1_epi32(0x200)));
        y = _mm256_or_si256(
            y, _mm256_and_si256(_mm256_cmpgt_epi32(_mm256_and_si256(y, _mm256_set1_epi32(32767)),
                                                   _mm256_set1_epi32(0x7c00)),
                                _mm256_set1_epi32(0x200)));
      }
      x = _mm256_xor_si256(x, _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? 32768 : 0));
      y = _mm256_xor_si256(y, _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_B ? 32768 : 0));
      result = _mm256_or_si256(x, _mm256_slli_epi32(y, 16));
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void pack_x86_64_v3<true>(uint64_t, uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                   const uint32_t *, bool);
template void pack_x86_64_v3<false>(uint64_t, uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                    const uint32_t *, bool);

} // namespace rocjitsu::arith
