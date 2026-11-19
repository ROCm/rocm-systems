// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v3 FP64 dense matrix reduction for Wave64 MFMA and Wave32 WMMA.
#include "common/dense_f64.h"
#include "common/dense_f64_backends.h"

namespace rocjitsu::arith {
// Compute every batch before stores; compile-time layout selects the VGPR mapping.
template <int M, int Batch, int Lanes>
void dense_f64_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c) {
  RJ_ARITH_IMPL_NAMESPACE::dense_f64_kernel<M, Batch, Lanes>(controls, d, a, b, c);
}
template void dense_f64_x86_64_v3<16, 1, 64>(uint64_t, uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *);
template void dense_f64_x86_64_v3<4, 4, 64>(uint64_t, uint32_t *const *, const uint32_t *const *,
                                            const uint32_t *const *, const uint32_t *const *);
template void dense_f64_x86_64_v3<16, 1, 32>(uint64_t, uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *);
} // namespace rocjitsu::arith
