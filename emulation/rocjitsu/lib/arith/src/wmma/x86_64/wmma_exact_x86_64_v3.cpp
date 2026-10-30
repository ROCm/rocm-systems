// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 instantiations of the shared bit-exact FP16/BF16 WMMA algorithm.
// Selects eight-column vectors for both wave sizes, gfx11/gfx12 layouts, and
// FP32 results; the algorithm lives in wmma_exact_x86_64.h.

#include "common/simd.h"
#include "wmma/x86_64/wmma_exact_x86_64.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Compute empirical bit-exact FP16/BF16 A*B+C for the selected gfx layout
// and wave size. Requires validated instruction flags; D may alias inputs.
template <bool Bf16, int Lanes, bool Rdna3>
void wmma_exact_x86_64_v3(uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                          const uint32_t *const *b, const uint32_t *const *c) {
  wmma_exact<Bf16, Lanes, Rdna3, __m256i>(instruction_flags, d, a, b, c);
}

template void wmma_exact_x86_64_v3<false, 32, false>(uint32_t, uint32_t *const *,
                                                     const uint32_t *const *,
                                                     const uint32_t *const *,
                                                     const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 32, true>(uint32_t, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 64, false>(uint32_t, uint32_t *const *,
                                                     const uint32_t *const *,
                                                     const uint32_t *const *,
                                                     const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 64, true>(uint32_t, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 32, false>(uint32_t, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 32, true>(uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 64, false>(uint32_t, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 64, true>(uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);

} // namespace rocjitsu::arith
