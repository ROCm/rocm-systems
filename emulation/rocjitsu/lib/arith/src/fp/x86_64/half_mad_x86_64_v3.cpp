// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 half MAD with an FP32 intermediate and vector output modifiers.

#include "fp/half_mad.h"
#include "fp/x86_64/half_mad_output_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate a wave in eight-lane chunks, preserving masked lanes and the opposite half.
void half_mad_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int lane = 0; lane < 32; lane += 8) {
    auto load = [&](const uint32_t *p, uint32_t m) {
      return half_input(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p + lane)),
                        m & RJ_ARITH_ALU_HIGH_A ? 16 : 0, m);
    };
    auto x = load(a, mode), y = load(b, mode >> 1), z = load(c, mode >> 2);
    // Half products are exact in FP32; FMA retains the required FP32 sum rounding.
    auto result = finish_half_mad(_mm256_fmadd_ps(x, y, z), mode, flags);
    result = _mm256_sll_epi32(result, _mm_cvtsi32_si128(shift));
    auto previous = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    result = _mm256_or_si256(result,
                             _mm256_and_si256(previous, _mm256_set1_epi32(shift ? 65535 : -65536)));
    auto mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                  _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), mask, result);
  }
}

} // namespace rocjitsu::arith
