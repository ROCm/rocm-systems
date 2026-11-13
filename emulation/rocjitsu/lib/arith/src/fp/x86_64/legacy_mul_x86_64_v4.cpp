// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX512 inherited multiply with instruction-local guest rounding.
// Shares AVX2-width finalization; never changes host floating-point controls.
#include "fp/x86_64/legacy_mul_x86_64.h"

namespace rocjitsu::arith {

// Execute one Wave32 block; source/destination aliases and inactive lanes are valid.
void legacy_mul_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a, const uint32_t *b) {
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  for (int lane = 0; lane < 32; lane += 16) {
    auto xl = legacy_mul_input(a + lane, mode, flags);
    auto xh = legacy_mul_input(a + lane + 8, mode, flags);
    auto yl = legacy_mul_input(b + lane, mode >> 1, flags);
    auto yh = legacy_mul_input(b + lane + 8, mode >> 1, flags);
    auto x = _mm512_insertf32x8(_mm512_castps256_ps512(xl), xh, 1);
    auto y = _mm512_insertf32x8(_mm512_castps256_ps512(yl), yh, 1);
    __m512 value;
    switch (rounding) {
    case 1:
      value = _mm512_mul_round_ps(x, y, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
      break;
    case 2:
      value = _mm512_mul_round_ps(x, y, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
      break;
    case 3:
      value = _mm512_mul_round_ps(x, y, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      break;
    default:
      value = _mm512_mul_round_ps(x, y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      break;
    }
    auto low =
        legacy_mul_finish(_mm512_castps512_ps256(value), xl, yl, flags, mode, a + lane, b + lane);
    auto high = legacy_mul_finish(_mm512_extractf32x8_ps(value, 1), xh, yh, flags, mode,
                                  a + lane + 8, b + lane + 8);
    auto result = _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

} // namespace rocjitsu::arith
