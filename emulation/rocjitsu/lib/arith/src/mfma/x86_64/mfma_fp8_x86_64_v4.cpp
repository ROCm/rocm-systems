// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v4 FP8/FNUZ MFMA entry points with isolated SIMD decoding and reduction.
#include "mfma/backends.h"
#include "mfma/x86_64/fp8.h"

namespace rocjitsu::arith {
// Evaluate a complete matrix with the selected source formats and ISA.
template <int M, int K, bool Bf8A, bool Bf8B, bool Fnuz>
void mfma_fp8_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c) {
  mfma_fp8_kernel<M, K, Bf8A, Bf8B, Fnuz>(controls, d, a, b, c);
}
template void mfma_fp8_x86_64_v4<16, 32, false, false, true>(uint64_t controls, uint32_t *const *d,
                                                             const uint32_t *const *a,
                                                             const uint32_t *const *b,
                                                             const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<16, 32, false, false, false>(uint64_t controls, uint32_t *const *d,
                                                              const uint32_t *const *a,
                                                              const uint32_t *const *b,
                                                              const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<16, 32, false, true, true>(uint64_t controls, uint32_t *const *d,
                                                            const uint32_t *const *a,
                                                            const uint32_t *const *b,
                                                            const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<16, 32, false, true, false>(uint64_t controls, uint32_t *const *d,
                                                             const uint32_t *const *a,
                                                             const uint32_t *const *b,
                                                             const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<16, 32, true, false, true>(uint64_t controls, uint32_t *const *d,
                                                            const uint32_t *const *a,
                                                            const uint32_t *const *b,
                                                            const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<16, 32, true, false, false>(uint64_t controls, uint32_t *const *d,
                                                             const uint32_t *const *a,
                                                             const uint32_t *const *b,
                                                             const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<16, 32, true, true, true>(uint64_t controls, uint32_t *const *d,
                                                           const uint32_t *const *a,
                                                           const uint32_t *const *b,
                                                           const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<16, 32, true, true, false>(uint64_t controls, uint32_t *const *d,
                                                            const uint32_t *const *a,
                                                            const uint32_t *const *b,
                                                            const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<32, 16, false, false, true>(uint64_t controls, uint32_t *const *d,
                                                             const uint32_t *const *a,
                                                             const uint32_t *const *b,
                                                             const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<32, 16, false, false, false>(uint64_t controls, uint32_t *const *d,
                                                              const uint32_t *const *a,
                                                              const uint32_t *const *b,
                                                              const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<32, 16, false, true, true>(uint64_t controls, uint32_t *const *d,
                                                            const uint32_t *const *a,
                                                            const uint32_t *const *b,
                                                            const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<32, 16, false, true, false>(uint64_t controls, uint32_t *const *d,
                                                             const uint32_t *const *a,
                                                             const uint32_t *const *b,
                                                             const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<32, 16, true, false, true>(uint64_t controls, uint32_t *const *d,
                                                            const uint32_t *const *a,
                                                            const uint32_t *const *b,
                                                            const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<32, 16, true, false, false>(uint64_t controls, uint32_t *const *d,
                                                             const uint32_t *const *a,
                                                             const uint32_t *const *b,
                                                             const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<32, 16, true, true, true>(uint64_t controls, uint32_t *const *d,
                                                           const uint32_t *const *a,
                                                           const uint32_t *const *b,
                                                           const uint32_t *const *c);
template void mfma_fp8_x86_64_v4<32, 16, true, true, false>(uint64_t controls, uint32_t *const *d,
                                                            const uint32_t *const *a,
                                                            const uint32_t *const *b,
                                                            const uint32_t *const *c);
} // namespace rocjitsu::arith
