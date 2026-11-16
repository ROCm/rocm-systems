// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// V4 mixed MAD with an FP32 FMA intermediate and directed half narrowing.
#include "fp/x86_64/mixed_mad_x86_64.h"

namespace rocjitsu::arith {

// Execute one Wave32 block, preserving inactive lanes and the other output half.
void mixed_mad_x86_64_v4(MixedFma destination, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                         uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = mixed_mad_input(a + lane, mode);
    auto y = mixed_mad_input(b + lane, mode >> 1);
    auto z = mixed_mad_input(c + lane, mode >> 2);
    auto xh = mixed_mad_input(a + lane + 8, mode);
    auto yh = mixed_mad_input(b + lane + 8, mode >> 1);
    auto zh = mixed_mad_input(c + lane + 8, mode >> 2);
    auto wide_x = _mm512_insertf32x8(_mm512_castps256_ps512(x), xh, 1);
    auto wide_y = _mm512_insertf32x8(_mm512_castps256_ps512(y), yh, 1);
    auto wide_z = _mm512_insertf32x8(_mm512_castps256_ps512(z), zh, 1);
    auto value = _mm512_fmadd_ps(wide_x, wide_y, wide_z);
    auto low = mixed_mad_finish(destination, flags, mode, _mm512_castps512_ps256(value), x, y, z,
                                a + lane, b + lane, c + lane);
    auto high = mixed_mad_finish(destination, flags, mode, _mm512_extractf32x8_ps(value, 1), xh, yh,
                                 zh, a + lane + 8, b + lane + 8, c + lane + 8);
    low = mixed_mad_merge(destination, low, d + lane);
    high = mixed_mad_merge(destination, high, d + lane + 8);
    auto result = _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

} // namespace rocjitsu::arith
