// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86_64_v4 FP16/BF16 MFMA. Stage packed sources once, then FMA across output columns.
#include "mfma/backends.h"
#include "mfma/x86_64/mfma_half.h"

namespace rocjitsu::arith {

// Evaluate a Wave64 MFMA with the caller-selected ISA implementation.
template <int M, int K, int Batch, bool Bf16, MfmaF32Model Model>
void mfma_half_x86_64_v4(uint64_t controls, uint64_t flags, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  mfma_half_kernel<M, K, Batch, Bf16, Model>(controls, flags, d, a, b, c);
}

template void mfma_half_x86_64_v4<32, 4, 2, false>(uint64_t, uint64_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 4, 4, false>(uint64_t, uint64_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void mfma_half_x86_64_v4<4, 4, 16, false>(uint64_t, uint64_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void mfma_half_x86_64_v4<32, 8, 1, false>(uint64_t, uint64_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 16, 1, false>(uint64_t, uint64_t, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 32, 1, false>(uint64_t, uint64_t, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void mfma_half_x86_64_v4<32, 16, 1, false>(uint64_t, uint64_t, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void mfma_half_x86_64_v4<32, 4, 2, true>(uint64_t, uint64_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 4, 4, true>(uint64_t, uint64_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void mfma_half_x86_64_v4<4, 4, 16, true>(uint64_t, uint64_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void mfma_half_x86_64_v4<32, 8, 1, true>(uint64_t, uint64_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 16, 1, true>(uint64_t, uint64_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 32, 1, true>(uint64_t, uint64_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void mfma_half_x86_64_v4<32, 16, 1, true>(uint64_t, uint64_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);

template void mfma_half_x86_64_v4<32, 2, 2, true, MfmaF32Model::Gfx908>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 2, 4, true, MfmaF32Model::Gfx908>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<4, 2, 16, true, MfmaF32Model::Gfx908>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<32, 4, 1, true, MfmaF32Model::Gfx908>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 8, 1, true, MfmaF32Model::Gfx908>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<32, 4, 2, false, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                         uint32_t *const *,
                                                                         const uint32_t *const *,
                                                                         const uint32_t *const *,
                                                                         const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 4, 4, false, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                         uint32_t *const *,
                                                                         const uint32_t *const *,
                                                                         const uint32_t *const *,
                                                                         const uint32_t *const *);
template void mfma_half_x86_64_v4<4, 4, 16, false, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                         uint32_t *const *,
                                                                         const uint32_t *const *,
                                                                         const uint32_t *const *,
                                                                         const uint32_t *const *);
template void mfma_half_x86_64_v4<32, 8, 1, false, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                         uint32_t *const *,
                                                                         const uint32_t *const *,
                                                                         const uint32_t *const *,
                                                                         const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 16, 1, false, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                          uint32_t *const *,
                                                                          const uint32_t *const *,
                                                                          const uint32_t *const *,
                                                                          const uint32_t *const *);
template void mfma_half_x86_64_v4<32, 2, 2, true, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 2, 4, true, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<4, 2, 16, true, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<32, 4, 1, true, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 8, 1, true, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<32, 4, 2, true, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 4, 4, true, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<4, 4, 16, true, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<32, 8, 1, true, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_half_x86_64_v4<16, 16, 1, true, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                         uint32_t *const *,
                                                                         const uint32_t *const *,
                                                                         const uint32_t *const *,
                                                                         const uint32_t *const *);

} // namespace rocjitsu::arith
