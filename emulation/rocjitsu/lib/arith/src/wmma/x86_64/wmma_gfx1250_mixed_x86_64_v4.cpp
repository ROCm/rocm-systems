// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v4 mixed-format gfx1250 WMMA backend.
#include "wmma/gfx1250_backends.h"
#include "wmma/gfx1250_mixed.h"
namespace rocjitsu::arith {
// Decode both operands and reduce columns at this translation unit's SIMD width.
template <int M>
void wmma_gfx1250_mixed_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  gfx1250_mixed_kernel<M>(controls, d, a, b, c);
}
template void wmma_gfx1250_mixed_x86_64_v4<16>(uint64_t controls, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b,
                                               const uint32_t *const *c);
template void wmma_gfx1250_mixed_x86_64_v4<32>(uint64_t controls, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b,
                                               const uint32_t *const *c);
// Apply block scaling after the ordered fused dot.
template <int M, int Block>
void wmma_gfx1250_scaled_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c,
                                   const uint32_t *const *scale_a, const uint32_t *const *scale_b) {
  gfx1250_mixed_kernel<M, Block>(controls, d, a, b, c, scale_a, scale_b);
}
template void wmma_gfx1250_scaled_x86_64_v4<16, 16>(
    uint64_t controls, uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
    const uint32_t *const *c, const uint32_t *const *scale_a, const uint32_t *const *scale_b);
template void wmma_gfx1250_scaled_x86_64_v4<16, 32>(
    uint64_t controls, uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
    const uint32_t *const *c, const uint32_t *const *scale_a, const uint32_t *const *scale_b);
template void wmma_gfx1250_scaled_x86_64_v4<32, 16>(
    uint64_t controls, uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
    const uint32_t *const *c, const uint32_t *const *scale_a, const uint32_t *const *scale_b);
template void wmma_gfx1250_scaled_x86_64_v4<32, 32>(
    uint64_t controls, uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
    const uint32_t *const *c, const uint32_t *const *scale_a, const uint32_t *const *scale_b);
} // namespace rocjitsu::arith
