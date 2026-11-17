// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v3 sparse floating matrix decoding and ordered fused reduction.
#include "mfma/backends.h"
#include "mfma/sparse_float.h"
namespace rocjitsu::arith {
// Evaluate all outputs before publishing stores.
template <int M, int K, int Bits, bool BfA, bool BfB, bool Fnuz>
void smfmac_float_x86_64_v3(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *index, bool clamp_overflow) {
  smfmac_float_kernel<M, K, Bits, BfA, BfB, Fnuz>(d, a, b, index, clamp_overflow);
}
template void smfmac_float_x86_64_v3<16, 32, 16, false, false, false>(uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      bool);
template void smfmac_float_x86_64_v3<16, 32, 16, true, true, false>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 16, 16, false, false, false>(uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      bool);
template void smfmac_float_x86_64_v3<32, 16, 16, true, true, false>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<16, 64, 16, false, false, false>(uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      bool);
template void smfmac_float_x86_64_v3<16, 64, 16, true, true, false>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 32, 16, false, false, false>(uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      bool);
template void smfmac_float_x86_64_v3<32, 32, 16, true, true, false>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<16, 64, 8, false, false, true>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<16, 64, 8, false, false, false>(uint32_t *const *,
                                                                     const uint32_t *const *,
                                                                     const uint32_t *const *,
                                                                     const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<16, 64, 8, false, true, true>(uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<16, 64, 8, false, true, false>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<16, 64, 8, true, false, true>(uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<16, 64, 8, true, false, false>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<16, 64, 8, true, true, true>(uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<16, 64, 8, true, true, false>(uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 32, 8, false, false, true>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 32, 8, false, false, false>(uint32_t *const *,
                                                                     const uint32_t *const *,
                                                                     const uint32_t *const *,
                                                                     const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 32, 8, false, true, true>(uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 32, 8, false, true, false>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 32, 8, true, false, true>(uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 32, 8, true, false, false>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 32, 8, true, true, true>(uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 32, 8, true, true, false>(uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<16, 128, 8, false, false, false>(uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      bool);
template void smfmac_float_x86_64_v3<16, 128, 8, false, true, false>(uint32_t *const *,
                                                                     const uint32_t *const *,
                                                                     const uint32_t *const *,
                                                                     const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<16, 128, 8, true, false, false>(uint32_t *const *,
                                                                     const uint32_t *const *,
                                                                     const uint32_t *const *,
                                                                     const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<16, 128, 8, true, true, false>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 64, 8, false, false, false>(uint32_t *const *,
                                                                     const uint32_t *const *,
                                                                     const uint32_t *const *,
                                                                     const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 64, 8, false, true, false>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 64, 8, true, false, false>(uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *const *, bool);
template void smfmac_float_x86_64_v3<32, 64, 8, true, true, false>(uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *, bool);
} // namespace rocjitsu::arith
