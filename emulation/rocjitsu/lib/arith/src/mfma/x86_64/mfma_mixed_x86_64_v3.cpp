// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v3 mixed-format MFMA column-wise fused reduction.
#include "mfma/backends.h"
#include "mfma/mixed.h"
namespace rocjitsu::arith {
// Decode independently packed operands and stage all outputs before publication.
template <int M, int K>
void mfma_mixed_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                          const uint32_t *const *b, const uint32_t *const *c) {
  mfma_mixed_kernel<M, K>(controls, d, a, b, c);
}
template void mfma_mixed_x86_64_v3<16, 128>(uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *c);
template void mfma_mixed_x86_64_v3<32, 64>(uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c);
// Evaluate fused block dots with inherited combined-exponent scaling.
template <int M, int K>
void mfma_scaled_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                           const uint32_t *const *b, const uint32_t *const *c,
                           const uint32_t *scale_a, const uint32_t *scale_b, bool clamp_overflow) {
  mfma_mixed_kernel<M, K, true>(controls, d, a, b, c, scale_a, scale_b, clamp_overflow);
}
template void mfma_scaled_x86_64_v3<16, 128>(uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c, const uint32_t *scale_a,
                                             const uint32_t *scale_b, bool clamp_overflow);
template void mfma_scaled_x86_64_v3<32, 64>(uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *c, const uint32_t *scale_a,
                                            const uint32_t *scale_b, bool clamp_overflow);
} // namespace rocjitsu::arith
