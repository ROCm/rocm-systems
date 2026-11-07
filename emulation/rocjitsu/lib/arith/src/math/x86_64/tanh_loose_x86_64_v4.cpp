// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// AVX-512 Loose TANH core; baseline code owns modifiers and masked publication.
#include "math/x86_64/tanh_loose.h"
#include "math/x86_64/tanh_loose_impl.h"
namespace rocjitsu::arith {
// Evaluate prepared FP32 lanes without reading or changing the host FP controls.
void tanh_loose_x86_64_v4(uint32_t *d, const uint32_t *a) {
  for (int lane = 0; lane < 32; lane += 16) {
    const auto raw = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    const auto result = tanh_vector<F16, I16>((I16)raw);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(d + lane), (__m512i)result);
  }
}
} // namespace rocjitsu::arith
