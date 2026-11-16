// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86_64_v3 FP16/BF16 MFMA. Stage packed sources once, then FMA across output columns.
#include "mfma/backends.h"
#include "mfma/x86_64/mfma_half.h"

namespace rocjitsu::arith {

// Evaluate a Wave64 MFMA with the caller-selected ISA implementation.
template <int M, int K, int Batch, bool Bf16>
void mfma_half_x86_64_v3(uint64_t controls, bool clamp_overflow, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  mfma_half_kernel<M, K, Batch, Bf16>(controls, clamp_overflow, d, a, b, c);
}

template void mfma_half_x86_64_v3<32, 4, 2, false>(uint64_t, bool, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void mfma_half_x86_64_v3<16, 4, 4, false>(uint64_t, bool, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void mfma_half_x86_64_v3<4, 4, 16, false>(uint64_t, bool, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void mfma_half_x86_64_v3<32, 8, 1, false>(uint64_t, bool, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void mfma_half_x86_64_v3<16, 16, 1, false>(uint64_t, bool, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void mfma_half_x86_64_v3<16, 32, 1, false>(uint64_t, bool, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void mfma_half_x86_64_v3<32, 16, 1, false>(uint64_t, bool, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void mfma_half_x86_64_v3<32, 4, 2, true>(uint64_t, bool, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void mfma_half_x86_64_v3<16, 4, 4, true>(uint64_t, bool, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void mfma_half_x86_64_v3<4, 4, 16, true>(uint64_t, bool, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void mfma_half_x86_64_v3<32, 8, 1, true>(uint64_t, bool, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void mfma_half_x86_64_v3<16, 16, 1, true>(uint64_t, bool, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void mfma_half_x86_64_v3<16, 32, 1, true>(uint64_t, bool, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void mfma_half_x86_64_v3<32, 16, 1, true>(uint64_t, bool, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);

} // namespace rocjitsu::arith
