// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v3 gfx1250 sparse floating matrix kernels.
#include "wmma/gfx1250_backends.h"
#include "wmma/gfx1250_sparse_float.h"
namespace rocjitsu::arith {
// Evaluate all outputs before publishing stores.
template <int Bits, bool BfA, bool BfB, int Form>
void swmmac_float_x86_64_v3(uint64_t flags, uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b, const uint32_t *const *index) {
  swmmac1250_float_kernel<Bits, BfA, BfB, Form>(flags, d, a, b, index);
}
template void swmmac_float_x86_64_v3<16, false, false, 0>(uint64_t, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void swmmac_float_x86_64_v3<16, true, true, 0>(uint64_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_float_x86_64_v3<16, false, false, 1>(uint64_t, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void swmmac_float_x86_64_v3<16, true, true, 2>(uint64_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_float_x86_64_v3<16, true, true, 3>(uint64_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_float_x86_64_v3<8, false, false, 0>(uint64_t, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_float_x86_64_v3<8, false, false, 1>(uint64_t, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_float_x86_64_v3<8, false, true, 0>(uint64_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_float_x86_64_v3<8, false, true, 1>(uint64_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_float_x86_64_v3<8, true, false, 0>(uint64_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_float_x86_64_v3<8, true, false, 1>(uint64_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_float_x86_64_v3<8, true, true, 0>(uint64_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void swmmac_float_x86_64_v3<8, true, true, 1>(uint64_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
} // namespace rocjitsu::arith
