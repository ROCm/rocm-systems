// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// V3 mixed MAD with an FP32 FMA intermediate and directed half narrowing.
#include "fp/x86_64/mixed_mad_x86_64.h"

namespace rocjitsu::arith {

// Execute one Wave32 block, preserving inactive lanes and the other output half.
void mixed_mad_x86_64_v3(MixedFma destination, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                         uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = mixed_mad_input(a + lane, mode);
    auto y = mixed_mad_input(b + lane, mode >> 1);
    auto z = mixed_mad_input(c + lane, mode >> 2);
    auto value = _mm256_fmadd_ps(x, y, z);
    auto result =
        mixed_mad_finish(destination, flags, mode, value, x, y, z, a + lane, b + lane, c + lane);
    result = mixed_mad_merge(destination, result, d + lane);
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

} // namespace rocjitsu::arith
