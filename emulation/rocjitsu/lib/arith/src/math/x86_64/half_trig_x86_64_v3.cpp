// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16 sine/cosine with guest input/output modifiers.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/half_x86_64_v3.h"
#include "math/trig.h"
#include "math/x86_64/trig_x86_64_v3.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Apply half-precision output scaling and clamp to a rounded trig result.
__m256i output(__m256i bits, uint32_t mode) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fff));
  unsigned scale = (mode >> 6) & 3;
  if (scale) {
    __m256i scaled;
    if (scale == 3) {
      const __m256i rounded = _mm256_add_epi32(
          _mm256_srli_epi32(magnitude, 1),
          _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_and_si256(magnitude, _mm256_set1_epi32(3)),
                                              _mm256_set1_epi32(3)),
                           _mm256_set1_epi32(1)));
      scaled = _mm256_blendv_epi8(_mm256_sub_epi32(magnitude, _mm256_set1_epi32(0x400)), rounded,
                                  _mm256_cmpgt_epi32(_mm256_set1_epi32(0x800), magnitude));
    } else {
      scaled = _mm256_add_epi32(magnitude, _mm256_set1_epi32(int(scale * 0x400)));
    }
    const __m256i tiny = _mm256_or_si256(_mm256_cmpgt_epi32(_mm256_set1_epi32(0x400), magnitude),
                                         _mm256_cmpgt_epi32(_mm256_set1_epi32(0x400), scaled));
    scaled = _mm256_andnot_si256(
        tiny, _mm256_or_si256(scaled, _mm256_and_si256(bits, _mm256_set1_epi32(0x8000))));
    bits =
        _mm256_blendv_epi8(bits, scaled, _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7c00), magnitude));
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    const __m256i negative = _mm256_cmpgt_epi32(bits, _mm256_set1_epi32(0x7fff));
    const __m256i nan = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7c00));
    bits = _mm256_blendv_epi8(_mm256_min_epu32(bits, _mm256_set1_epi32(0x3c00)), zero,
                              _mm256_or_si256(negative, nan));
  }
  return bits;
}

// Evaluate FP16 sine/cosine for the selected lanes. Requires validated controls; preserve
// inactive destination lanes.
template <bool Cosine> void run(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const int a_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  const int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const __m256i keep = _mm256_set1_epi32(d_shift ? 0xffff : -65536);
  for (int lane = 0; lane < 32; lane += 8) {
    const __m256 input = half_input<false>(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), a_shift, mode);
    const __m256 value = trig_value<Cosine>(_mm256_castps_si256(input));
    __m256i result = output(half_narrow<false>(value, false), mode);
    result = _mm256_sll_epi32(result, _mm_cvtsi32_si128(d_shift));
    const __m256i previous = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    result = _mm256_or_si256(result, _mm256_and_si256(previous, keep));
    const __m256i lane_exec_mask =
        _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                          _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

} // namespace

// Evaluate FP16 sine/cosine. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void half_trig_x86_64_v3(bool cosine, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a) {
  if (cosine) {
    run<true>(exec_mask, mode, d, a);
  } else {
    run<false>(exec_mask, mode, d, a);
  }
}

} // namespace rocjitsu::arith
