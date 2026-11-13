// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// AVX2 Loose TANH core; baseline code owns modifiers and masked publication.
#include "math/x86_64/tanh_loose.h"
#include "math/x86_64/tanh_loose_impl.h"
namespace rocjitsu::arith {
// Evaluate prepared FP32 lanes without reading or changing the host FP controls.
void tanh_loose_x86_64_v3(uint32_t *d, const uint32_t *a) {
  for (int lane = 0; lane < 32; lane += 8) {
    const auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    const auto result = tanh_vector<F8, I8>((I8)raw);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + lane), (__m256i)result);
  }
}
} // namespace rocjitsu::arith
